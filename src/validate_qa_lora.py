#!/usr/bin/env python3
"""Validate a trained CSDS QA-LoRA against the SAME Qwen base model.

From /root/workspace/csds-project:
    python src/validate_qa_lora.py --check
    python src/validate_qa_lora.py --limit 5
    python src/validate_qa_lora.py --limit 50

No gold QA labels are inserted into either inference prompt; the exact stored
QA-SFT user prompt is reused, and only the LoRA adapter is toggled. Gold labels
are used exclusively for OFFLINE evaluation. Saved results resume atomically.

This measures the five fields actually trained: category, question, answer,
question_ids and evidence_ids. It does NOT score object/status labels, because
those fields were never supervised in qa_sft_train.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

DEFAULT_DEV = "data/qa_sft_dev.jsonl"
DEFAULT_ADAPTER = "outputs/qlora_qa_stage1/final_adapter"
DEFAULT_MODEL = "models/Qwen3-4B-Instruct-2507"
DEFAULT_OUT = "outputs/validate_qa_lora_dev50.json"
SEED = 20261122
FIELDS = {"category", "question", "answer", "question_ids", "evidence_ids"}


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_dev(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Development data not found: {path}")
    rows = []
    seen = set()
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not {"DialogueID", "prompt", "target"} <= row.keys():
            raise ValueError(f"Invalid dev record at line {ln}")
        d = str(row["DialogueID"])
        if d in seen:
            raise ValueError(f"Duplicate DialogueID={d} at line {ln}")
        seen.add(d)
        target = json.loads(row["target"])
        if not isinstance(target, list) or not target:
            raise ValueError(f"DialogueID={d}: target must be nonempty QA array")
        rows.append(row)
    if len(rows) < 50:
        raise ValueError(f"Expected >=50 dev dialogues; found {len(rows)}")
    return rows


def sample_dev(rows, n=50):
    # Fixed set of 50; --limit 5 processes the same first five on every run.
    if n > len(rows):
        raise ValueError(f"Cannot select {n} samples from {len(rows)}")
    indexes = sorted(random.Random(SEED).sample(range(len(rows)), n))
    return [(i, rows[i]) for i in indexes]


def numbered_speakers(prompt: str):
    # The training data has lines like [U0] 用户：... and [C0] 客服：...
    user_ids, agent_ids, excerpts = set(), set(), {}
    for line in prompt.splitlines():
        m = re.match(r"^\[(U\d+|C\d+)\]\s*(用户|客服)：(.*)$", line.strip())
        if not m:
            continue
        label, speaker, utterance = m.groups()
        if label[0] == "U" and speaker == "用户":
            user_ids.add(label)
        elif label[0] == "C" and speaker == "客服":
            agent_ids.add(label)
        else:
            raise ValueError(f"Speaker/id mismatch on {line!r}")
        if label in excerpts:
            raise ValueError(f"Duplicate utterance ID {label}")
        excerpts[label] = utterance
    if not user_ids and not agent_ids:
        raise ValueError("Cannot read numbered dialogue from training prompt")
    return user_ids, agent_ids, excerpts


def canonical_items(raw_text, prompt):
    """Parse generated QA and strictly check TRAINED schema + speaker-grounded IDs."""
    stripped = raw_text.strip()
    repaired_fence = False
    if stripped.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", stripped, flags=re.S | re.I)
        if match:
            stripped = match.group(1).strip()
            repaired_fence = True
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        return None, f"json_parse_error: {e.msg}", repaired_fence, False
    json_valid = True
    if not isinstance(parsed, list) or not parsed:
        return None, "not_nonempty_array", repaired_fence, json_valid
    users, agents, _ = numbered_speakers(prompt)
    cleaned = []
    for i, v in enumerate(parsed):
        if not isinstance(v, dict) or set(v) != FIELDS:
            return None, f"qa_{i}_wrong_fields", repaired_fence, json_valid
        if not all(isinstance(v[k], str) for k in ("category", "question", "answer")):
            return None, f"qa_{i}_wrong_string_type", repaired_fence, json_valid
        if not v["question"].strip() or not v["answer"].strip():
            return None, f"qa_{i}_empty_question_or_answer", repaired_fence, json_valid
        uids, cids = v["question_ids"], v["evidence_ids"]
        if not isinstance(uids, list) or not isinstance(cids, list):
            return None, f"qa_{i}_ids_not_lists", repaired_fence, json_valid
        if not cids or not all(isinstance(x, str) for x in uids + cids):
            return None, f"qa_{i}_bad_ids", repaired_fence, json_valid
        if len(set(uids)) != len(uids) or len(set(cids)) != len(cids):
            return None, f"qa_{i}_repeated_ids", repaired_fence, json_valid
        if any(uid not in users for uid in uids):
            return None, f"qa_{i}_invalid_user_ids", repaired_fence, json_valid
        if any(cid not in agents for cid in cids):
            return None, f"qa_{i}_invalid_agent_ids", repaired_fence, json_valid
        cleaned.append({
            "category": v["category"].strip(),
            "question": v["question"].strip(),
            "answer": v["answer"].strip(),
            "question_ids": uids,
            "evidence_ids": cids,
        })
    return cleaned, None, repaired_fence, json_valid


def lcs_f1(a: str, b: str):
    """Character-level LCS F1; cheap on short question/answer text."""
    a = re.sub(r"\s+", "", a)
    b = re.sub(r"\s+", "", b)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for ch in a:
        curr = [0] * (len(b) + 1)
        for j, target_ch in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if ch == target_ch else max(curr[j - 1], prev[j])
        prev = curr
    return 2.0 * prev[-1] / (len(a) + len(b))


def set_f1(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return None  # no evidence of question alignment, not perfect similarity
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def alignment_score(gold, pred):
    question_sim = lcs_f1(gold["question"], pred["question"])
    u_sim = set_f1(gold["question_ids"], pred["question_ids"])
    return question_sim if u_sim is None else 0.65 * question_sim + 0.35 * u_sim


def best_alignment(gold, pred, threshold=0.25):
    """Max-weight one-to-one matching, including unmatched cases, by gold QA mask.

    The match depends on question text and user evidence ONLY, not the answer,
    category or agent-evidence we're later assessing. Prevents answer leakage.
    """
    n = len(gold)
    if n > 14:
        raise ValueError(f"Gold count too high for DP matcher ({n}); revise matcher")
    # dp[used gold bitmask] = (weight, [(gold_idx, pred_idx, matchscore)])
    dp = {0: (0., [])}
    for j, p in enumerate(pred):
        new = dict(dp)  # allow unmatched predictions
        for mask, (score, pairs) in dp.items():
            for i, g in enumerate(gold):
                if mask & (1 << i):
                    continue
                sim = alignment_score(g, p)
                if sim < threshold:
                    continue
                nm = mask | (1 << i)
                ns = score + sim
                if nm not in new or ns > new[nm][0] + 1e-10:
                    new[nm] = (ns, pairs + [(i, j, round(sim, 6))])
        dp = new
    return max(dp.values(), key=lambda x: x[0])[1]


def score_case(gold, predicted):
    """Offline diagnostic proxies, not semantic/factuality gold evaluation."""
    pred = predicted or []
    pairs = best_alignment(gold, pred)
    used_g = {i for i, _, _ in pairs}
    used_p = {j for _, j, _ in pairs}
    question_sims = []
    answer_sims = []
    category_ok = 0
    category_n = 0
    e_tp = e_fp = e_fn = 0
    q_tp = q_fp = q_fn = 0
    detail = []
    for i, j, sim in pairs:
        g, p = gold[i], pred[j]
        qs = lcs_f1(g["question"], p["question"])
        ans = lcs_f1(g["answer"], p["answer"])
        question_sims.append(qs)
        answer_sims.append(ans)
        if g["category"]:  # exclude blank intent gold, not treat as true category
            category_n += 1
            category_ok += int(g["category"] == p["category"])
        gc, pc = set(g["evidence_ids"]), set(p["evidence_ids"])
        e_tp += len(gc & pc); e_fp += len(pc - gc); e_fn += len(gc - pc)
        gu, pu = set(g["question_ids"]), set(p["question_ids"])
        q_tp += len(gu & pu); q_fp += len(pu - gu); q_fn += len(gu - pu)
        detail.append({
            "gold_index": i, "pred_index": j, "matching_score": sim,
            "question_rouge_l_f1": round(qs, 4), "answer_rouge_l_f1": round(ans, 4),
            "category_equal": (g["category"] == p["category"] if g["category"] else None),
            "gold_question": g["question"], "pred_question": p["question"],
            "gold_answer": g["answer"], "pred_answer": p["answer"],
            "gold_evidence_ids": g["evidence_ids"], "pred_evidence_ids": p["evidence_ids"],
        })
    # Unmatched items contribute their evidence IDs as missing/spurious.
    for i, g in enumerate(gold):
        if i not in used_g:
            e_fn += len(set(g["evidence_ids"]))
            q_fn += len(set(g["question_ids"]))
    for j, p in enumerate(pred):
        if j not in used_p:
            e_fp += len(set(p["evidence_ids"]))
            q_fp += len(set(p["question_ids"]))

    return {
        "gold_qa_count": len(gold),
        "pred_qa_count": len(pred),
        "aligned_pairs": len(pairs),
        "count_exact": len(gold) == len(pred),
        "count_abs_error": abs(len(gold) - len(pred)),
        "matched_question_rouge_l_f1_mean": round(sum(question_sims)/len(question_sims), 6) if pairs else None,
        "matched_answer_rouge_l_f1_mean": round(sum(answer_sims)/len(answer_sims), 6) if pairs else None,
        "category_matches": category_ok,
        "category_total": category_n,
        "evidence_counts": {"tp": e_tp, "fp": e_fp, "fn": e_fn},
        "question_id_counts": {"tp": q_tp, "fp": q_fp, "fn": q_fn},
        "alignment": detail,
    }


def micro_f1(tp, fp, fn):
    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def aggregate(results):
    total = len(results)
    report = {"processed": total, "note": (
        "These are heuristic QA-alignment and character-overlap diagnostics against CSDS labels; "
        "NOT semantic faithfulness, actual object resolution, processing status accuracy or end-to-end summary quality."
    )}
    for key in ("base", "qa_lora"):
        good = [r for r in results if r.get(key, {}).get("schema_valid")]
        json_pass = sum(bool(r.get(key, {}).get("json_valid")) for r in results)
        schema_pass = len(good)
        scores = [r[key]["scores"] for r in results if "scores" in r.get(key, {})]
        if len(scores) != total:
            report[key] = {"error": "incomplete_results"}
            continue
        gt_total = sum(s["gold_qa_count"] for s in scores)
        pred_total = sum(s["pred_qa_count"] for s in scores)
        align_total = sum(s["aligned_pairs"] for s in scores)
        category_n = sum(s["category_total"] for s in scores)
        category_ok = sum(s["category_matches"] for s in scores)
        sum_count = sum(s["count_abs_error"] for s in scores)
        def count_metric(name):
            return micro_f1(*(sum(s[name][k] for s in scores) for k in ("tp", "fp", "fn")))
        qtexts = [s["matched_question_rouge_l_f1_mean"] for s in scores if s["matched_question_rouge_l_f1_mean"] is not None]
        atexts = [s["matched_answer_rouge_l_f1_mean"] for s in scores if s["matched_answer_rouge_l_f1_mean"] is not None]
        report[key] = {
            "json_valid": {"count": json_pass, "rate": round(json_pass/total, 4) if total else None},
            "schema_and_speaker_grounded_ids_valid": {"count": schema_pass, "rate": round(schema_pass/total, 4) if total else None},
            "qa_count_exact_rate": round(sum(s["count_exact"] for s in scores)/total, 4) if total else None,
            "qa_count_mae": round(sum_count/total, 4) if total else None,
            "predicted_qa_total": pred_total,
            "gold_qa_total": gt_total,
            "heuristic_qa_coverage_recall": round(align_total/gt_total, 4) if gt_total else None,
            "heuristic_qa_precision": round(align_total/pred_total, 4) if pred_total else None,
            "category_exact_accuracy_on_matched_labeled_qa": round(category_ok/category_n, 4) if category_n else None,
            "category_scored_pairs": category_n,
            "matched_question_char_rouge_l_mean_per_dialogue": round(sum(qtexts)/len(qtexts), 4) if qtexts else None,
            "matched_answer_char_rouge_l_mean_per_dialogue": round(sum(atexts)/len(atexts), 4) if atexts else None,
            "agent_evidence_micro": count_metric("evidence_counts"),
            "user_evidence_micro": count_metric("question_id_counts"),
            "errors": dict(Counter(r[key].get("error") for r in results if r.get(key, {}).get("error"))),
            "hit_max_new_tokens": sum(bool(r[key].get("hit_max_new_tokens")) for r in results),
        }
    return report


def save_report(base_file: Path, results):
    summary = base_file.with_name(base_file.stem + "_summary.json")
    review = base_file.with_name(base_file.stem + "_review.md")
    atomic_json(summary, aggregate(results))
    lines = [
        "# QA-LoRA vs base Qwen — QA extraction (development)", "",
        "**Reference QA is for offline evaluation only; neither inference prompt contains any gold QA.**", "",
        "**Automatic alignment & Rouge are approximate: visually inspect object, answer and speaker status.**", "",
    ]
    for r in results:
        lines += [f"## DialogueID {r['DialogueID']} (dev row {r['dev_row_index']})", "",
                  "### Original train-format prompt (identical for both models)", "",
                  "```text", r["prompt"], "```", "",
                  "### Annotated target (offline only)", "", "```json",
                  json.dumps(r["gold"], ensure_ascii=False, indent=2), "```", ""]
        for model_name in ("base", "qa_lora"):
            m = r.get(model_name)
            if not m:
                continue
            lines += [f"### {model_name} prediction", "", "```text", m.get("raw", ""), "```", "",
                      f"Schema valid: {m.get('schema_valid')} | Error: {m.get('error')} | Generated tokens: {m.get('generated_tokens')}", ""]
            if "scores" in m:
                s = m["scores"]
                lines += [f"QA count gold/pred: {s['gold_qa_count']}/{s['pred_qa_count']} | "
                          f"Heuristically matched: {s['aligned_pairs']} | "
                          f"Category exact on matched: {s['category_matches']}/{s['category_total']}", ""]
                for p in s["alignment"]:
                    lines += [f"- Match gold #{p['gold_index']+1} to predicted #{p['pred_index']+1} "
                              f"(score {p['matching_score']:.3f}); question F1={p['question_rouge_l_f1']:.3f}; "
                              f"answer F1={p['answer_rouge_l_f1']:.3f}"]
                lines += [""]
        lines += ["Manual review: Did the model merge distinct items/orders? Did it add unsupported facts? "
                  "Did it miss an answer or cite the wrong speaker?", "", "---", ""]
    review.parent.mkdir(parents=True, exist_ok=True)
    tmp = review.with_name(review.name + ".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(review)
    return summary, review


def generate(model, tokenizer, prompt, max_new_tokens):
    import torch
    messages = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_tensors="pt")
    # Model is loaded onto a single CUDA device, following the training setup.
    ids = ids.to("cuda:0")
    with torch.inference_mode():
        output = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    new_ids = output[0, ids.shape[-1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, len(new_ids), len(new_ids) == max_new_tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--dev", default=DEFAULT_DEV)
    ap.add_argument("--output", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--check", action="store_true", help="Check data/paths and show sample, no GPU")
    ap.add_argument("--evaluate-only", action="store_true", help="Rescore saved predictions, no GPU")
    args = ap.parse_args()
    if not 1 <= args.limit <= 50:
        ap.error("--limit must be between 1 and 50")
    if args.max_new_tokens < 128:
        ap.error("--max-new-tokens must be >=128")
    dev_path = Path(args.dev)
    rows = read_dev(dev_path)
    selected = sample_dev(rows, 50)[:args.limit]
    for idx, row in selected:
        u, c, _ = numbered_speakers(row["prompt"])
        target = json.loads(row["target"])
        for qa in target:
            if set(qa) != FIELDS or any(e not in c for e in qa["evidence_ids"]) or any(q not in u for q in qa["question_ids"]):
                raise ValueError(f"Bad dev gold QA labels in DialogueID={row['DialogueID']}")
    meta = {
        "version": "qa_sft_extraction_dev50_v1",
        "dev_sha256": sha256(dev_path),
        "seed": SEED,
        "dev_row_indices_fixed_50": [i for i, _ in sample_dev(rows, 50)],
        "model": args.model,
        "adapter": args.adapter,
        "max_new_tokens": args.max_new_tokens,
        "generation": "greedy; no sampling",
        "prompt": "EXACT qa_sft_dev.jsonl prompt (same as training)",
        "reference_qa_in_model_input": False,
        "gold_fields": sorted(FIELDS),
        "not_measured": ["processing_status", "object_identity", "semantic_factuality", "final_summary_quality"],
    }
    out_path = Path(args.output)
    if args.check:
        if not Path(args.model).is_dir():
            raise FileNotFoundError(f"Base model not found: {args.model}")
        if not (Path(args.adapter)/"adapter_config.json").is_file():
            raise FileNotFoundError(f"QA adapter not found: {args.adapter}/adapter_config.json")
        print(json.dumps({
            "status": "check_ok", "source_dev_dialogues": len(rows), "evaluating": len(selected),
            "sample_dialogue_ids": [row["DialogueID"] for _, row in selected[:5]],
            "dev_sha256": meta["dev_sha256"], "model": args.model,
            "adapter": args.adapter,
            "message": "No GPU inference performed; gold QA not included in prompts."
        }, ensure_ascii=False, indent=2))
        return
    existing = {"metadata": meta, "results": []}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing.get("metadata") != meta:
            raise ValueError("Existing output metadata mismatch (data/model/config); use a new --output path")
    done = {str(r["DialogueID"]): r for r in existing["results"]}
    if args.evaluate_only:
        if not done:
            raise ValueError(f"No completed saved results at {out_path}")
        for r in done.values():
            gold = r["gold"]
            for key in ("base", "qa_lora"):
                if key in r:
                    r[key]["scores"] = score_case(gold, r[key].get("parsed"))
        results = [r for idx, row in selected if (r := done.get(str(row["DialogueID"]))) is not None]
        summary, review = save_report(out_path, results)
        print(f"Rescored {len(results)} existing cases: {summary}, {review}")
        return
    if not Path(args.model).is_dir():
        raise FileNotFoundError(f"Base model not found: {args.model}")
    if not (Path(args.adapter)/"adapter_config.json").is_file():
        raise FileNotFoundError(f"QA adapter not found: {args.adapter}/adapter_config.json")
    pending = [(i, row) for i, row in selected if
               str(row["DialogueID"]) not in done or
               not all(k in done[str(row["DialogueID"])] for k in ("base", "qa_lora"))]
    if not pending:
        report_rows = [done[str(row["DialogueID"])] for _, row in selected]
        summary, review = save_report(out_path, report_rows)
        print(f"All {len(selected)} cases already completed. Results: {out_path}, {summary}, {review}")
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for model generation (RTX 4090 expected)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=config,
                                                 dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    print(f"Loaded base+QA-LoRA; pending {len(pending)} of {len(selected)} dialogues", flush=True)
    for k, (idx, row) in enumerate(pending, 1):
        d = str(row["DialogueID"])
        r = done.get(d, {
            "dev_row_index": idx, "DialogueID": row["DialogueID"],
            "prompt": row["prompt"], "gold": json.loads(row["target"]),
        })
        # Parse first, then persist after every model call, permitting resume.
        for name in ("base", "qa_lora"):
            if name in r:
                continue
            if name == "base":
                with model.disable_adapter():
                    raw, tok, hit = generate(model, tokenizer, row["prompt"], args.max_new_tokens)
            else:
                raw, tok, hit = generate(model, tokenizer, row["prompt"], args.max_new_tokens)
            parsed, err, fence, is_json = canonical_items(raw, row["prompt"])
            r[name] = {
                "raw": raw, "parsed": parsed, "json_valid": is_json,
                "schema_valid": parsed is not None, "error": err,
                "stripped_code_fence": fence, "generated_tokens": tok,
                "hit_max_new_tokens": hit, "scores": score_case(r["gold"], parsed),
            }
            done[d] = r
            existing["results"] = [done[str(x["DialogueID"])] for _, x in sample_dev(rows, 50) if str(x["DialogueID"]) in done]
            atomic_json(out_path, existing)
            print(f"  DialogueID={d} {name}: schema={parsed is not None}, "
                  f"QA={len(parsed or [])}, tokens={tok}, error={err}", flush=True)
        print(f"[{k}/{len(pending)}] complete DialogueID={d}", flush=True)
    results = [done[str(row["DialogueID"])] for _, row in selected]
    summary, review = save_report(out_path, results)
    print(f"Finished {len(results)} cases: {out_path}, {summary}, {review}", flush=True)
    print(json.dumps(aggregate(results), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
