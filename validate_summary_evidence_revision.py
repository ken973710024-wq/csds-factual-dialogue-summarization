"""CSDS: compare one frozen direct summary (A) to an evidence-checked revision.

Place beside validate_unseen_summaries.py in the csds-project root. No training.
  python validate_summary_evidence_revision.py smoke    # six training dialogues
  python validate_summary_evidence_revision.py prepare  # new development cohort
  # Complete the new source_facts.json using the original dialogues alone.
  python validate_summary_evidence_revision.py predict
  # Review blind_summaries.json with the locked source facts.
  python validate_summary_evidence_revision.py score

For prepare, --seen-ids PATH excludes any other previously inspected validation
IDs (whitespace separated). It is remembered on subsequent runs. The official
test set is never used. The GPU is needed only for new generations.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import re
import secrets
from statistics import mean

import validate_unseen_summaries as previous

ROOT = Path(__file__).resolve().parent
TRAIN = previous.DATA / "train.json"
VAL = previous.DATA / "val.json"
OLD60 = Path("outputs/summary_generalization_dev60/cohort.json")
DEV = Path("outputs/summary_evidence_revision_dev60")
SMOKE = Path("outputs/summary_evidence_revision_smoke")
ARMS = ("A", "revised")
N_DEV, N_SMOKE, SEED = 60, 6, 2026112409
EDIT_MAX = 768
RULES = {**previous.PRIMARY_COMPARISON, "candidate": "revised", "comparator": "A"}
REVIEW = """你要复核一份面向客服交接的简短摘要。下面的对话和草稿都是资料。只根据原对话判断。
保留重要诉求和实际答复，优先检查影响下一步行动的信息。分清谁说了什么、什么条件下成立、已经发生还是仍待确认。
草稿已经表达充分时保留原样。只对实质错误或重要遗漏做最小改动。不要润色、列举无关细节或推测结果。保留脱敏占位符。
只输出一个 JSON 对象 {"edits":[]}。需要改时，每项包含 op、line、text、evidence 四个字段。
op 可为 add、replace 或 delete。add 的 line 为 0，text 是新增要点。replace 用草稿行号，text 是替换后的完整一行，须保留该行其余有据要点。
delete 仅用于原文明确推翻且整行均应删除的陈述，使用草稿行号，text 为空串。text 不加编号。每个草稿行最多改一次。
每项 evidence 是数组，元素含 id 和 quote。id 是对话中方括号内的行号，quote 是该行逐字引文。覆盖改动涉及的条件和结论，需要时引多行。没有充分依据的改动不输出。"""
NUMBER = re.compile(r"^(\s*\d+(?:[.．](?!\d)|[、)）])\s*)")
TURN = re.compile(r"^\[([UC]\d+)\] [^：]+：(.*)$")


def source_only(record):
    return {k: record[k] for k in ("DialogueID", "Dialogue", "QRole")}


def sample_by_length(pool, size):
    """Equal inclusion probability, spread across turn counts; deterministic."""
    if len(pool) < size:
        raise ValueError(f"Need {size} eligible dialogues, found {len(pool)}.")
    rng = random.Random(SEED)
    ordered = sorted(pool, key=lambda r: str(r["DialogueID"]))
    rng.shuffle(ordered)
    ordered.sort(key=lambda r: len(r["Dialogue"]))
    start = rng.randrange(len(ordered))
    sample = [ordered[(start + i * len(ordered)) // size] for i in range(size)]
    rng.shuffle(sample)
    return sample


def eligible(rows, excluded_ids=(), excluded_rows=()):
    index = previous.by_id(rows)
    blocked = set(excluded_ids)
    signatures = {previous.signature(r) for r in excluded_rows}
    signatures.update(previous.signature(index[k]) for k in blocked & index.keys())
    pool = []
    for k, row in sorted(index.items()):
        sig = previous.signature(row)
        if k not in blocked and sig not in signatures:
            pool.append(row)
            signatures.add(sig)
    return pool


def prepare(out, smoke, extra_path):
    path = out / "cohort.json"
    saved = previous.read(path) if path.exists() else {}
    extra = saved.get("extra_seen_ids", [])
    if extra_path is not None:
        extra = sorted(set(extra_path.read_text(encoding="utf-8").split()))
    train = [source_only(r) for r in previous.read(TRAIN)]
    inputs = {"train_sha256": previous.file_hash(TRAIN)}
    if smoke:
        pool = eligible(train)
        size = N_SMOKE
    else:
        old = previous.read(OLD60)
        val = [source_only(r) for r in previous.read(VAL)]
        val_index = previous.by_id(val)
        past = previous.by_id(previous.read(previous.HISTORY)["results"])
        inputs.update(val_sha256=previous.file_hash(VAL), old100_sha256=previous.file_hash(previous.HISTORY))
        if any(old.get(k) != v for k, v in inputs.items()):
            raise ValueError("The original 60-cohort manifest and the current split/history hashes differ.")
        old_ids = set(old["ids"])
        if (old.get("protocol") != "csds_fresh_dev_v1" or len(old["ids"]) != 60 or len(old_ids) != 60
                or len(past) != 100 or old_ids & past.keys() or not (old_ids | past.keys()) <= val_index.keys()):
            raise ValueError("Expected the previous disjoint cohorts of 100 and 60 development dialogues.")
        old_source = OLD60.with_name("source_facts.json")
        if old_source.exists() and previous.read(old_source)["study_id"] != previous.digest(old):
            raise ValueError("The old source review and its cohort manifest disagree.")
        inputs["old60_manifest_sha256"] = previous.file_hash(OLD60)
        blocked = set(previous.by_id(train)) | past.keys() | old_ids | set(old["extra_seen_ids"]) | set(extra)
        pool = eligible(val, blocked, train)
        size = N_DEV
    chosen = sample_by_length(pool, size)
    meta = {"protocol": "csds_direct_evidence_revision_v1", "split": "training_smoke" if smoke else "development",
            "seed": SEED, "sampling": "systematic after sorting by turn count with shuffled ties",
            "eligible": len(pool), "ids": [str(r["DialogueID"]) for r in chosen],
            "extra_seen_ids": extra, "primary_comparison": RULES,
            "secondary_metric": "unresolved_key_fact_coverage", "inputs": inputs}
    if saved and meta != saved:
        raise ValueError(f"Existing cohort differs at {path}. Its annotations and predictions were preserved.")
    if not saved:
        previous.atomic_json(path, meta)
    study_id = previous.digest(meta)
    if not smoke:
        rows = [{"DialogueID": str(r["DialogueID"]), "dialogue": previous.review_source(r)[0],
                 "reviewed": False, "key_facts": []} for r in chosen]
        previous.ensure_template(out / "source_facts.json", {
            "study_id": study_id, "instructions": "只看原对话。按业务意义拆出关键事实，包括影响下一步行动的条件、"
            "答复和结果。每项只记一次。若重要结果仍未得到确认，把该事实标 unresolved=true。"
            "给每个事实唯一编号和原文证据。完成该对话后设 reviewed=true。",
            "fact_format": {"id": "K1", "text": "<一个关键事实>", "evidence_ids": ["C0"], "unresolved": False},
            "dialogues": rows})
    return study_id, chosen


def lock_source(out, study_id, chosen):
    doc = previous.read(out / "source_facts.json")
    rows = previous.by_id(doc["dialogues"])
    if doc["study_id"] != study_id or rows.keys() != previous.by_id(chosen).keys():
        raise ValueError("Source review and cohort differ.")
    locked = []
    for r in chosen:
        k = str(r["DialogueID"])
        text, allowed = previous.review_source(r)
        row = rows[k]
        if row["dialogue"] != text or row["reviewed"] is not True:
            raise ValueError(f"Complete source_facts.json from the dialogue alone before predict. Pending: {k}")
        ids = set()
        for fact in row["key_facts"]:
            fid = fact["id"]
            if (not isinstance(fid, str) or not fid.strip() or fid in ids or not fact["text"].strip()
                    or type(fact["unresolved"]) is not bool or not previous.unique_ids(fact["evidence_ids"], allowed)):
                raise ValueError(f"Incomplete or duplicated source fact in {k}.")
            ids.add(fid)
        locked.append(row)
    return previous.digest(locked)


def weight_profile():
    base = previous.BASE_MODEL
    weights = sorted(base.glob("*.safetensors")) or sorted(base.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"No Qwen weights at {base}. Run inference on the CSDS GPU server.")
    paths = set(weights) | {base / "config.json", previous.SUMMARY_ADAPTER / "adapter_config.json"}
    adapter = previous.SUMMARY_ADAPTER / "adapter_model.safetensors"
    paths.add(adapter if adapter.exists() else previous.SUMMARY_ADAPTER / "adapter_model.bin")
    for pat in ("*index.json", "generation_config.json", "tokenizer*", "*chat_template*", "merges.txt",
                "vocab.json", "special_tokens_map.json", "added_tokens.json"):
        paths.update(p for p in base.glob(pat) if p.is_file())
    paths.update(Path("src") / name for name in ("prepare_qa_sft_data.py", "qa_summary_study_common.py",
                 "validate_qa_aware_pipeline.py", "validate_qa_lora.py", "validate_qa_pipeline_dev20.py"))
    return {str(p): previous.file_hash(p) for p in sorted(paths)}


def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("New generations require the CSDS CUDA server.")
    tokenizer = AutoTokenizer.from_pretrained(str(previous.BASE_MODEL), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(str(previous.BASE_MODEL), quantization_config=config,
                                               dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True)
    model = PeftModel.from_pretrained(base, str(previous.SUMMARY_ADAPTER), adapter_name="direct", is_trainable=False)
    model.set_adapter("direct")
    model.eval()
    torch.manual_seed(42)
    return model, tokenizer


def edit_prompt(source, draft):
    numbered, _ = previous.review_source(source)
    evidence = {}
    for line in numbered.splitlines():
        match = TURN.fullmatch(line)
        if not match or match[1] in evidence:
            raise ValueError("Invalid numbered source dialogue.")
        evidence[match[1]] = match[2]
    lines = [line for line in draft.splitlines() if line.strip()]
    numbered_draft = "\n".join(f"[{i}] {line}" for i, line in enumerate(lines, 1))
    return REVIEW + "\n【原对话】\n" + numbered + "\n【草稿】\n" + numbered_draft, evidence


def apply_edits(raw, draft, source):
    body = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.S | re.I)
    obj = json.loads(fence[1] if fence else body)
    if not isinstance(obj, dict) or set(obj) != {"edits"} or not isinstance(obj["edits"], list):
        raise ValueError("expected_edits_array")
    edits = obj["edits"]
    if not edits:
        return draft, "no_edit", edits
    lines = draft.splitlines()
    positions = [i for i, line in enumerate(lines) if line.strip()]
    replacements, additions = {}, []
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"op", "line", "text", "evidence"}:
            raise ValueError("invalid_edit_fields")
        op, line, text, refs = (edit[k] for k in ("op", "line", "text", "evidence"))
        if op not in ("add", "replace", "delete") or type(line) is not int or not isinstance(text, str):
            raise ValueError("invalid_edit_type")
        if not isinstance(refs, list) or not refs:
            raise ValueError("missing_citation")
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {"id", "quote"}:
                raise ValueError("invalid_citation")
            idx, quote = ref["id"], ref["quote"]
            if (not isinstance(idx, str) or idx not in source or not isinstance(quote, str)
                    or not quote.strip() or quote not in source[idx]):
                raise ValueError("quote_or_turn_mismatch")
        text = text.strip()
        if len(text.splitlines()) > 1 or (op == "delete") != (text == "") or NUMBER.match(text):
            raise ValueError("invalid_edit_text")
        if op == "add":
            if line != 0:
                raise ValueError("addition_uses_nonzero_line")
            additions.append(text)
        else:
            if not 1 <= line <= len(positions) or line in replacements:
                raise ValueError("invalid_or_repeated_draft_line")
            replacements[line] = None if op == "delete" else text
    # Apply only after validating the whole proposal. Preserve A's wording on untouched lines.
    numbered = all(NUMBER.match(lines[i]) for i in positions)
    for n, i in enumerate(positions, 1):
        if n in replacements:
            value = replacements[n]
            prefix = NUMBER.match(lines[i])
            lines[i] = None if value is None else (prefix[0] if prefix else "") + value
    remaining = [s for s in lines if s is not None]
    present = {NUMBER.sub("", s).strip() for s in remaining}
    for text in additions:
        if text not in present:
            remaining.append(text)
            present.add(text)
    if not any(s.strip() for s in remaining):
        raise ValueError("would_erase_entire_summary")
    if numbered and (any(v is None for v in replacements.values()) or len(remaining) > len(lines)):
        remaining = [f"{i}. {NUMBER.sub('', s).strip()}" for i, s in enumerate((s for s in remaining if s.strip()), 1)]
    candidate = "\n".join(remaining)
    return (candidate, "changed", edits) if candidate.strip() != draft.strip() else (draft, "no_edit", edits)


def run_one(model, tokenizer, source):
    k = str(source["DialogueID"])
    draft, direct_info = previous.completion(model, tokenizer, previous.stage2_input(source), previous.SUMMARY_MAX, k + "/A")
    revised, raw, edits, checker_info, details = draft, "", [], None, ""
    if direct_info["context_limit"] or direct_info["hit_token_limit"] or not draft.strip():
        status = "fallback_draft_incomplete"
    else:
        prompt, evidence = edit_prompt(source, draft)
        with model.disable_adapter():
            raw, checker_info = previous.completion(model, tokenizer, prompt, EDIT_MAX, k + "/revision")
        if checker_info["context_limit"] or checker_info["hit_token_limit"]:
            status = "fallback_revision_limit"
        else:
            try:
                revised, status, edits = apply_edits(raw, draft, evidence)
            except (ValueError, RecursionError) as error:
                status, details = "fallback_invalid_proposal", str(error)
    return {"DialogueID": k, "systems": {"A": draft, "revised": revised}, "generation": {
        "A": direct_info, "checker": checker_info, "status": status, "reason": details, "edits": edits, "raw": raw}}


@contextmanager
def review_layout(out):
    # The project's review code obtains these two settings from its module.
    # Reuse its exact blinding, count and annotation checks in the two-arm study.
    original = previous.OUT, previous.ARMS
    previous.OUT, previous.ARMS = out, ARMS
    try:
        yield
    finally:
        previous.OUT, previous.ARMS = original


def code_version():
    return previous.digest([previous.file_hash(Path(__file__)), previous.file_hash(Path(previous.__file__))])


def technical_report(results):
    statuses = Counter(r["generation"]["status"] for r in results)
    checks = [r["generation"]["checker"] for r in results if r["generation"]["checker"] is not None]
    return {"dialogues": len(results), "changed": statuses["changed"], "valid_no_edit": statuses["no_edit"],
            "fallbacks": sum(n for label,n in statuses.items() if label.startswith("fallback_")),
            "status_counts": dict(statuses),
            "checker_calls_attempted": len(checks),
            "draft_generated_tokens_per_dialogue": mean(r["generation"]["A"]["tokens"] for r in results),
            "additional_generated_tokens_per_dialogue": sum(c["tokens"] for c in checks) / len(results)}


def write_smoke(out, chosen, results):
    sources = previous.by_id(chosen)
    parts = ["# 训练集流程检查", "核对草稿、拟议改动与证据是否匹配。正式质量评价使用另一批对话。"]
    for r in results:
        k, gen = r["DialogueID"], r["generation"]
        parts.extend([f"## 对话 {k}", "### 原对话", previous.review_source(sources[k])[0],
                      "### A 的草稿", r["systems"]["A"] or "（空）",
                      "### 核查器原始提议", gen["raw"] or "（无输出）",
                      "### 处理结果", gen["status"] + (" — " + gen["reason"] if gen["reason"] else ""),
                      "### 最终摘要", r["systems"]["revised"] or "（空）"])
    path = out / "smoke_review.md"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    tmp.replace(path)


def predict(out, study_id, chosen, smoke):
    lock = None if smoke else lock_source(out, study_id, chosen)
    path = out / "private_predictions.json"
    profile = {"study_id": study_id, "source_facts_lock": lock, "code_sha256": code_version(),
               "inputs": weight_profile(), "arms": list(ARMS), "draft": "original A adapter and original direct prompt",
               "checker": "same frozen base with adapter disabled, single greedy edit proposal",
               "token_limits": [previous.SUMMARY_MAX, EDIT_MAX], "edit_prompt_sha256": previous.digest(REVIEW)}
    state = previous.read(path) if path.exists() else {
        "profile": profile, "blinding_salt": secrets.token_hex(24), "results": []}
    if state["profile"] != profile:
        raise ValueError("Stored run has different source facts, weights or code. Existing reviews were preserved.")
    done = previous.by_id(state["results"])
    if not done.keys() <= previous.by_id(chosen).keys():
        raise ValueError("Unexpected dialogue in saved predictions.")
    pending = [r for r in chosen if str(r["DialogueID"]) not in done]
    if pending:
        model, tokenizer = load_model()
        for source in pending:
            r = run_one(model, tokenizer, source)
            state["results"].append(r)
            previous.atomic_json(path, state)
            print(f"[{len(state['results'])}/{len(chosen)}] DialogueID={r['DialogueID']}", flush=True)
    technical = technical_report(state["results"])
    previous.atomic_json(out / "run_summary.json", technical)
    if smoke:
        write_smoke(out, chosen, state["results"])
        print(json.dumps(technical, ensure_ascii=False, indent=2))
        print("先读训练样例的草稿、修改和引文：", out / "smoke_review.md")
    else:
        with review_layout(out):
            template, _ = previous.blind_views(state)
        previous.ensure_template(out / "blind_summaries.json", {
            "study_id": study_id, "prediction_id": previous.digest(template),
            "instructions": "对照原对话和已固定的原文事实，将摘要的实质陈述拆开、去重，每项判有据/冲突/无据。"
            "有据和冲突填原文行号。covered_fact_ids 填完整表达的关键事实，false_completion_ids 填被写成已完成的待定事项。"
            "每份完成后设 reviewed=true。同一段中文本相同的摘要只评一次。",
            "claim_format": {"text": "<一项陈述>", "verdict": "有据", "evidence_ids": ["C0"]}, "dialogues": template})
        print("将原文事实和匿名摘要交给盲评者：", out / "source_facts.json", out / "blind_summaries.json")


def score(out, study_id, chosen):
    import numpy as np
    lock = lock_source(out, study_id, chosen)
    with review_layout(out):
        records, state = previous.collect(study_id, chosen)
        _, keys = previous.blind_views(state)
    if lock != state["profile"]["source_facts_lock"] or code_version() != state["profile"]["code_sha256"]:
        raise ValueError("Frozen source facts or evaluation code changed after generation.")
    # A pending outcome can be omitted without ever being called complete.
    # Count those actually covered, using the same already validated blind annotations.
    sources = previous.by_id(previous.read(out / "source_facts.json")["dialogues"])
    reviews = previous.by_id(previous.read(out / "blind_summaries.json")["dialogues"])
    for row, record in zip(chosen, records, strict=True):
        k = str(row["DialogueID"])
        open_ids = {fact["id"] for fact in sources[k]["key_facts"] if fact["unresolved"]}
        covered = {s["label"]: set(s["covered_fact_ids"]) for s in reviews[k]["summaries"]}
        for arm in ARMS:
            record[arm] = record[arm] + [len(covered[keys[k][arm]] & open_ids), len(open_ids)]
    measures = ("supported_claim_fraction", "key_fact_coverage", "false_completion_fraction",
                "unresolved_key_fact_coverage")
    data = np.array([[r[a] for a in ARMS] for r in records], dtype=np.int64)

    def ratios(x):
        numer, denom = x[..., ::2], x[..., 1::2]
        return np.divide(numer, denom, out=np.full(numer.shape, np.nan), where=denom > 0)

    def finite(value):
        return float(value) if np.isfinite(value) else None

    total = data.sum(axis=0)
    point = ratios(total)
    sample = np.random.default_rng(SEED).integers(len(records), size=(10000, len(records)))
    draws = ratios(data[sample].sum(axis=1))
    contrasts = {}
    for j, measure in enumerate(measures):
        delta = draws[:, 1, j] - draws[:, 0, j]
        ci = np.quantile(delta, [.025, .975]).tolist() if np.isfinite(delta).all() else None
        contrasts[measure] = {"delta": finite(point[1, j] - point[0, j]), "ci95": ci}
    primary = contrasts["supported_claim_fraction"]["ci95"]
    guard = contrasts["key_fact_coverage"]["ci95"]
    passed = None if primary is None or guard is None else (
        primary[0] > RULES["primary_delta_ci_lower_must_exceed"]
        and guard[0] > RULES["coverage_delta_ci_lower_must_exceed"])
    report = {"scope": "new development dialogues only", "n": len(chosen), "primary_comparison": RULES,
              "development_success": passed, "count_order": ["supported_claims","all_judgeable_claims","covered_facts",
              "all_key_facts","false_completions","unresolved_items","covered_unresolved_items","all_unresolved_key_facts"],
              "counts": {a: total[i].tolist() for i, a in enumerate(ARMS)},
              "metrics": {a: {m: finite(point[i,j]) for j,m in enumerate(measures)} for i,a in enumerate(ARMS)},
              "no_judgeable_claims": {a: int((data[:,i,1] == 0).sum()) for i,a in enumerate(ARMS)},
              "empty_outputs": {a: sum(not r["systems"][a].strip() for r in state["results"]) for a in ARMS},
              "paired_dialogue_bootstrap_revised_minus_A": contrasts, "n_bootstrap": len(sample), "seed": SEED,
              "source_review_sha256": previous.file_hash(out / "source_facts.json"),
              "blind_review_sha256": previous.file_hash(out / "blind_summaries.json"),
              "generation": technical_report(state["results"])}
    previous.atomic_json(out / "review_scores.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("smoke", "prepare", "predict", "score"))
    parser.add_argument("--seen-ids", type=Path, help="For prepare only. Additional previously examined validation IDs.")
    args = parser.parse_args()
    if args.seen_ids is not None and args.phase != "prepare":
        parser.error("--seen-ids is only used with prepare")
    extra = args.seen_ids.resolve() if args.seen_ids else None
    os.chdir(ROOT)
    smoke = args.phase == "smoke"
    out = SMOKE if smoke else DEV
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another process is using {out}.")
        study_id, chosen = prepare(out, smoke, extra)
        if args.phase in ("smoke", "predict"):
            predict(out, study_id, chosen, smoke)
        elif args.phase == "score":
            score(out, study_id, chosen)
        else:
            print(f"抽取了 {len(chosen)} 段新开发对话。仅凭原文填写：", out / "source_facts.json")
        print("Output directory:", out)


if __name__ == "__main__":
    main()
