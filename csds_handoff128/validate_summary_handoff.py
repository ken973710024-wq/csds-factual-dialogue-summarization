"""Draw a fresh development cohort, predict with frozen adapters, and score blinded reviews."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import secrets

import handoff_common as h
import validate_summary_evidence_revision as previous_revision
from qa_summary_study_common import stage2_input
from validate_qa_aware_pipeline import generate

OUT = Path("outputs/summary_handoff_pilot128_dev60")
ARMS = ("original", "handoff", "A")
N = 60
SEED = 20261126
RULES = {
    "candidate": "handoff", "primary_comparator": "original", "secondary_comparator": "A",
    "min_lower_ci_supported_delta": 0.0, "min_lower_ci_coverage_delta": -0.02,
    "secondary_endpoints": ["false_completion_fraction", "unresolved_key_fact_coverage", "generated_tokens"],
    "interval": "pooled fact counts; paired resampling of whole dialogues, 10000 replicates, two-sided 95%",
}


@contextmanager
def review_layout():
    old = h.review.OUT, h.review.ARMS
    h.review.OUT, h.review.ARMS = OUT, ARMS
    try:
        yield
    finally:
        h.review.OUT, h.review.ARMS = old


def select_fresh(train, val, seen, size=N):
    train_ids = set(h.review.by_id(train))
    val_ids = set(h.review.by_id(val))
    pool = previous_revision.eligible(val, train_ids | set(seen), train)
    if len(pool) < size:
        raise ValueError(f"开发集中仅余 {len(pool)} 段不同的可用原文，需 {size} 段")
    rng = random.Random(SEED)
    rows = sorted(pool, key=lambda r: str(r["DialogueID"]))
    rng.shuffle(rows)
    rows.sort(key=lambda r: len(r["Dialogue"]))
    start = rng.randrange(len(rows))
    chosen = [rows[(start + i * len(rows)) // size] for i in range(size)]
    rng.shuffle(chosen)
    return chosen, len(pool), val_ids


def prepare(extra_file=None):
    info, _ = h.read_candidates()
    train_file, val_file = (Path("data/CSDS") / name for name in ("train.json", "val.json"))
    # Strip existing summary / QA annotations before sampling and serialising any dialogue.
    train = [previous_revision.source_only(r) for r in h.review.read(train_file)]
    val = [previous_revision.source_only(r) for r in h.review.read(val_file)]
    path = OUT / "cohort.json"
    saved = h.review.read(path) if path.exists() else {}
    extra = (sorted(set(extra_file.read_text(encoding="utf-8").split()))
             if extra_file is not None else saved.get("extra_seen_ids", []))
    past_ids = set(info["previously_seen_dev_ids"])
    chosen, eligible, val_ids = select_fresh(train, val, past_ids | set(extra))
    if not past_ids <= val_ids:
        raise ValueError("历史开发名单与本地开发集的 ID 不一致")
    meta = {
        "protocol": "csds_handoff128_fresh_dev_v1", "split": "development", "seed": SEED, "n": N,
        "eligible_dialogues": eligible, "known_dev_exclusions": sorted(past_ids, key=int), "extra_seen_ids": extra,
        "train_sha256": h.review.file_hash(train_file), "val_sha256": h.review.file_hash(val_file),
        "ids": [str(r["DialogueID"]) for r in chosen], "comparison": RULES,
    }
    h.freeze(path, meta)
    study = h.review.digest(meta)
    h.review.ensure_template(OUT / "source_facts.json", {
        "study_id": study,
        "instructions": "只读原对话，列出接手客服必须知道的关键事实。一个事实一项，注明发言行号。"
            "重要的待定结果标 unresolved=true。说话者、条件或动作对象不同时分别核对。"
            "完成每段设置 reviewed=true。无实质业务事实可留空。不要查看模型输出。",
        "fact_format": {"id": "K1", "text": "一项关键事实", "evidence_ids": ["C0"], "unresolved": False},
        "dialogues": [{"DialogueID": str(r["DialogueID"]), "dialogue": h.review.review_source(r)[0],
                       "reviewed": False, "key_facts": []} for r in chosen],
    })
    return study, chosen


def fingerprint():
    config = h.review.read(h.OUT / "data/comparison_manifest.json")
    if (h.base_identity() != config["base_files"] or h.adapter_identity(h.SUMMARY_ADAPTER) != config["initial_adapter_files"]
            or config["prompt_sha256"] != h.text_sha(h.INSTRUCTION)):
        raise ValueError("基础模型、原始 A 或交接指令与训练时不符")
    weights = {"A": config["initial_adapter_files"]}
    for arm in h.ARMS:
        directory = h.OUT / arm
        final = directory / "final_adapter"
        proof = h.review.read(final / "completion.json")
        expected = {**config, "target_arm": arm}
        weights[arm] = h.adapter_identity(final)
        if (h.review.read(directory / "training_manifest.json") != expected
                or proof["training_config_sha256"] != h.review.digest(expected)
                or proof["optimizer_steps"] != config["optimizer_steps_per_arm"] or proof["adapter_files"] != weights[arm]):
            raise ValueError(f"{arm} 缺少与本次实验一致的训练记录")
    return {"adapters": weights, "training_conditions_sha256": h.review.digest(config)}


def evaluation_code():
    import validate_qa_aware_pipeline
    import prepare_qa_sft_data
    import qa_summary_study_common
    return {Path(p).name: h.review.file_hash(Path(p)) for p in
            (__file__, h.__file__, h.review.__file__, previous_revision.__file__, validate_qa_aware_pipeline.__file__,
             prepare_qa_sft_data.__file__, qa_summary_study_common.__file__)}


def predict(study, chosen):
    from peft import PeftModel
    lock = previous_revision.lock_source(OUT, study, chosen)
    path = OUT / "private_predictions.json"
    profile = {"study_id": study, "source_facts_lock": lock, "weights": fingerprint(), "code": evaluation_code(),
               "arms": list(ARMS), "decoding": "greedy", "max_new_tokens": h.DECODE_MAX}
    state = h.review.read(path) if path.exists() else {
        "profile": profile, "blinding_salt": secrets.token_hex(24), "results": []}
    if state["profile"] != profile:
        raise ValueError("原文事实、预测配置或权重已变化，旧预测已保留")
    done = h.review.by_id(state["results"])
    if not done.keys() <= h.review.by_id(chosen).keys():
        raise ValueError("预测含其他批次的对话")
    pending = [r for r in chosen if str(r["DialogueID"]) not in done]
    if pending:
        model = PeftModel.from_pretrained(h.load_base(), str(h.SUMMARY_ADAPTER), adapter_name="A", is_trainable=False)
        for arm in h.ARMS:
            model.load_adapter(str(h.OUT / arm / "final_adapter"), adapter_name=arm, is_trainable=False)
        tokenizer = h.get_tokenizer()
        model.eval()
        for r in pending:
            key = str(r["DialogueID"])
            prompt = h.make_prompt(h.review.review_source(r)[0])
            texts, lengths = {}, {}
            for arm in ARMS:
                model.set_adapter(arm)
                texts[arm], lengths[arm] = generate(model, tokenizer, stage2_input(r) if arm == "A" else prompt,
                                                      h.DECODE_MAX, f"{key}/{arm}")
            state["results"].append({"DialogueID": key, "systems": texts, "tokens": lengths})
            h.review.atomic_json(path, state)
            print(f"[{len(state['results'])}/{len(chosen)}] {key}", flush=True)
    with review_layout():
        blind, _ = h.review.blind_views(state)
    h.review.ensure_template(OUT / "blind_summaries.json", {
        "study_id": study, "prediction_id": h.review.digest(blind),
        "instructions": "逐份对照原文和已固定的关键事实。实质陈述拆开、去重，分别判有据、冲突或无据。"
            "有据与冲突提供原文行号。covered_fact_ids 只填完整表达的事实，false_completion_ids 填被写成已完成的待定事项。"
            "完整核对后设置 reviewed=true。相同输出只评一次。评审时只打开原文事实和匿名摘要。",
        "claim_format": {"text": "一项陈述", "verdict": "有据", "evidence_ids": ["C0"]}, "dialogues": blind,
    })
    print(OUT / "blind_summaries.json")


def score(study, chosen):
    import numpy as np

    lock = previous_revision.lock_source(OUT, study, chosen)
    with review_layout():
        records, state = h.review.collect(study, chosen)
        _, keys = h.review.blind_views(state)
    if state["profile"]["source_facts_lock"] != lock or state["profile"]["code"] != evaluation_code():
        raise ValueError("生成预测后原文事实或评价代码发生了变化")
    sources = h.review.by_id(h.review.read(OUT / "source_facts.json")["dialogues"])
    reviews = h.review.by_id(h.review.read(OUT / "blind_summaries.json")["dialogues"])
    for row, rec in zip(chosen, records, strict=True):
        k = str(row["DialogueID"])
        open_ids = {fact["id"] for fact in sources[k]["key_facts"] if fact["unresolved"]}
        covered = {r["label"]: set(r["covered_fact_ids"]) for r in reviews[k]["summaries"]}
        for arm in ARMS:
            rec[arm] = rec[arm] + [len(covered[keys[k][arm]] & open_ids), len(open_ids)]
    metrics = ("supported_claim_fraction", "key_fact_coverage", "false_completion_fraction", "unresolved_key_fact_coverage")
    data = np.array([[rec[a] for a in ARMS] for rec in records], dtype=np.int64)
    def ratios(x):
        n, d = x[..., ::2], x[..., 1::2]
        return np.divide(n, d, out=np.full(n.shape, np.nan), where=d > 0)
    def finite(x):
        return float(x) if np.isfinite(x) else None
    totals = data.sum(axis=0)
    point = ratios(totals)
    picks = np.random.default_rng(SEED).integers(len(records), size=(10000, len(records)))
    draws = ratios(data[picks].sum(axis=1))
    contrasts = {}
    i = ARMS.index("handoff")
    for baseline in ("original", "A"):
        j = ARMS.index(baseline)
        contrast = {}
        for k, metric in enumerate(metrics):
            values = draws[:, i, k] - draws[:, j, k]
            ci = np.quantile(values, [.025, .975]).tolist() if np.isfinite(values).all() else None
            contrast[metric] = {"delta": finite(point[i, k] - point[j, k]), "ci95": ci}
        accuracy = contrast["supported_claim_fraction"]["ci95"]
        coverage = contrast["key_fact_coverage"]["ci95"]
        contrast["meets_accuracy_and_coverage_rule"] = (None if accuracy is None or coverage is None else
            accuracy[0] > RULES["min_lower_ci_supported_delta"] and coverage[0] > RULES["min_lower_ci_coverage_delta"])
        contrasts["handoff_minus_" + baseline] = contrast
    report = {
        "scope": "fresh development dialogues", "n": len(records), "rules": RULES, "paired_differences": contrasts,
        "count_order": ["supported_claims","judged_claims","fully_covered_facts","key_facts",
                        "false_completions","unresolved_facts","covered_unresolved_facts","unresolved_facts"],
        "counts": {a: totals[i].tolist() for i,a in enumerate(ARMS)},
        "metrics": {a: {m: finite(point[i,j]) for j,m in enumerate(metrics)} for i,a in enumerate(ARMS)},
        "mean_generated_tokens": {a: float(np.mean([r["tokens"][a] for r in state["results"]])) for a in ARMS},
        "hit_generation_cap": {a: sum(r["tokens"][a] >= h.DECODE_MAX for r in state["results"]) for a in ARMS},
        "empty_outputs": {a: sum(not r["systems"][a].strip() for r in state["results"]) for a in ARMS},
        "with_no_judged_claims": {a: int((data[:,i,1] == 0).sum()) for i,a in enumerate(ARMS)},
        "changed_text_handoff_vs_original": sum(r["systems"]["handoff"] != r["systems"]["original"] for r in state["results"]),
        "bootstrap_replicates": len(picks), "seed": SEED, "source_facts_lock": lock,
        "blind_review_sha256": h.review.file_hash(OUT / "blind_summaries.json"),
    }
    h.review.atomic_json(OUT / "review_scores.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "predict", "score"))
    parser.add_argument("--seen-ids", type=Path, help="排除其他已看过的开发对话 ID，空白符分隔")
    args = parser.parse_args()
    if args.seen_ids and args.phase != "prepare":
        parser.error("--seen-ids 用于 prepare")
    extra = args.seen_ids.resolve() if args.seen_ids else None
    os.chdir(h.ROOT)
    with h.locked(OUT):
        study, chosen = prepare(extra)
        if args.phase == "prepare":
            print(f"已冻结 {len(chosen)} 段开发对话，请只读原文完成 {OUT/'source_facts.json'}", flush=True)
        elif args.phase == "predict":
            predict(study, chosen)
        else:
            score(study, chosen)


if __name__ == "__main__":
    main()

