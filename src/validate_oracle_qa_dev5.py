"""Experiment 3: five-case Oracle-QA diagnosis (same LoRA summary model).

In the original csds-project root:
    python src/validate_oracle_qa_dev5.py --check
    python src/validate_oracle_qa_dev5.py --show-prompts
    python src/validate_oracle_qa_dev5.py --limit 5

Requires:
  src/validate_qa_pipeline_dev20.py  (reuse EXACT generation/prompt/ROUGE)
  data/oracle_qa_dev5.json          (manually corrected QA from raw dialogue)
  outputs/validate_qa_pipeline_dev20.json (completed 20-case comparisons)
  models/Qwen3-4B-Instruct-2507/
  outputs/qlora_sft/final_adapter/

Controlled comparison:
   A: saved auto-QA -> LoRA summary (from 20-case checkpoint, no rerun)
   B: manually corrected QA -> SAME LoRA and SAME summary prompt/settings
Only the QA checklist changes. No training occurs. Development-only; this is
NOT an unbiased estimate of performance on new data and not a production oracle.
"""

import argparse
import hashlib
import json
from pathlib import Path

# Import the exact same summarisation prompt, preprocessing, scoring and decoding.
from validate_qa_pipeline_dev20 import (
    ADAPTER_DIR, MODEL_DIR, STAGE2_TOKENS, SUMMARY_RULES,
    generate, numbered_dialogue, parse_qa, score, save_json,
)

DEV20_FILE = Path("outputs/validate_qa_pipeline_dev20.json")
MANUAL_FILE = Path("data/oracle_qa_dev5.json")
OUTPUT_FILE = Path("outputs/validate_oracle_qa_dev5.json")
SUMMARY_FILE = Path("outputs/validate_oracle_qa_dev5_summary.json")
REVIEW_FILE = Path("outputs/validate_oracle_qa_dev5_review.md")
DIALOGUE_IDS = (1287, 1173, 8925, 2009, 2613)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_check():
    for path in (DEV20_FILE, MANUAL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; check the file upload location")
    data = json.loads(DEV20_FILE.read_text(encoding="utf-8"))
    source_rows = data.get("results", [])
    by_id = {r["DialogueID"]: r for r in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("Duplicate DialogueID in 20-case input")
    annotation = json.loads(MANUAL_FILE.read_text(encoding="utf-8"))
    annotated_rows = annotation.get("items", [])
    manual_by_id = {r["DialogueID"]: r for r in annotated_rows}
    if len(manual_by_id) != len(annotated_rows):
        raise ValueError("Duplicate DialogueID in human QA file")
    if set(manual_by_id) != set(DIALOGUE_IDS):
        raise ValueError(f"Manual QA file must contain exactly DialogueIDs {DIALOGUE_IDS}")
    selected = []
    for did in DIALOGUE_IDS:
        if did not in by_id:
            raise ValueError(f"DialogueID {did} was not present in original 20-case run")
        row = by_id[did]
        entry = manual_by_id[did]
        if row["index"] != entry["index"]:
            raise ValueError(f"Dialogue index mismatch for {did}")
        if not row.get("stage1_valid") or not row.get("stage1_qa"):
            raise ValueError(f"The auto-QA baseline for {did} is not valid")
        if not all(k in row for k in ("qa_guided_prediction", "qa_guided_scores", "direct_scores", "existing_sft_scores")):
            raise ValueError(f"Incomplete saved baseline for {did}")
        tagged, available = numbered_dialogue(row["dialogue"])
        if tagged != row["numbered_dialogue"]:
            raise ValueError(f"Numbered dialogue drift for {did}")
        qa, error = parse_qa(json.dumps(entry["qa"], ensure_ascii=False), available)
        if error:
            raise ValueError(f"Human QA for {did} invalid: {error}")
        selected.append((row, qa, entry["annotation_note"]))
    metadata = {
        "experiment": "oracle_qa_development_diagnostic_v1",
        "dialogue_ids": list(DIALOGUE_IDS),
        "original_dev20_sha256": sha256_file(DEV20_FILE),
        "manual_qa_sha256": sha256_file(MANUAL_FILE),
        "summary_prompt_version": data["metadata"]["summary_prompt_version"],
        "adapter": str(ADAPTER_DIR),
        "model": str(MODEL_DIR),
        "max_new_tokens": STAGE2_TOKENS,
        "sampling": "greedy; do_sample=False",
        "reference_or_reference_summary_used_in_model_input": False,
        "annotation_source": "Manually curated QA from original conversation turns only; diagnostic labels, not guaranteed gold",
        "comparison": "auto-QA LoRA (saved) vs manually corrected QA same LoRA same summary prompt",
    }
    return selected, metadata


def mean_scores(rows, key):
    fields = ("rouge_l_f1", "rouge_l_precision", "rouge_l_recall", "chars")
    return {x: round(sum(r[key][x] for r in rows) / len(rows), 4) for x in fields}


def summarise(rows):
    complete = [r for r in rows if "oracle_qa_scores" in r]
    summary = {"processed": len(rows), "fully_evaluated": len(complete), "planned": len(DIALOGUE_IDS)}
    if not complete:
        return summary
    for key, field in (
        ("existing_sft", "existing_sft_scores"),
        ("direct", "direct_scores"),
        ("auto_qa", "auto_qa_scores"),
        ("oracle_qa", "oracle_qa_scores"),
    ):
        summary[key] = mean_scores(complete, field)
    a, o = summary["auto_qa"], summary["oracle_qa"]
    summary["oracle_minus_auto"] = {
        "rouge_l_f1": round(o["rouge_l_f1"] - a["rouge_l_f1"], 4),
        "rouge_l_recall": round(o["rouge_l_recall"] - a["rouge_l_recall"], 4),
        "chars": round(o["chars"] - a["chars"], 2),
    }
    summary["oracle_f1_wins_vs_auto"] = sum(r["oracle_qa_scores"]["rouge_l_f1"] > r["auto_qa_scores"]["rouge_l_f1"] for r in complete)
    summary["oracle_recall_wins_vs_auto"] = sum(r["oracle_qa_scores"]["rouge_l_recall"] > r["auto_qa_scores"]["rouge_l_recall"] for r in complete)
    summary["note"] = "5 intentionally chosen, previously inspected development examples. Scores do not establish generalisation; inspect factuality and QA coverage manually."
    return summary


def format_qa(qa):
    lines = []
    for i, item in enumerate(qa, 1):
        cites = ", ".join(item["evidence_ids"])
        lines.append(f"{i}. **[{item['category']}] {item['question']}**")
        lines.append(f"   - 答复：{item['answer']}")
        lines.append(f"   - 状态：{item['status']}；证据：{cites}")
        lines.append("   - 原文：" + " / ".join(item["evidence_quotes"]))
    return "\n".join(lines)


def review_markdown(rows):
    parts = [
        "# Oracle QA diagnostic — 5 development cases",
        "",
        "The manually corrected QA was authored using original conversation text, not the reference summary. ",
        "This is a **diagnostic upper-bound approximation** and not a clean test-set evaluation.",
        "Compare auto vs corrected QA, then compare their resulting LoRA summaries for factuality.",
        "",
    ]
    for r in rows:
        parts.extend([
            f"## DialogueID {r['DialogueID']} (validation index {r['index']})", "",
            "### Original numbered dialogue", "", "```text", r["numbered_dialogue"], "```", "",
            "### Auto-extracted QA (Stage 1)", "", format_qa(r["auto_qa"]), "",
            "### Manually corrected QA", "", format_qa(r["manual_qa"]), "",
            "**Reason for correction:** " + r["annotation_note"], "",
            "### Standard SFT (Experiment 2)", "", "```text", r["existing_sft_prediction"], "```", "",
            "### Direct control (same Stage-2 prompt, no QA)", "", "```text", r["direct_prediction"], "```", "",
            "### Auto-QA guided (saved Stage-2 output)", "", "```text", r["auto_qa_prediction"], "```", "",
        ])
        if "oracle_qa_prediction" in r:
            parts.extend([
                "### Manually corrected-QA guided (new Stage-2 output)", "", "```text", r["oracle_qa_prediction"], "```", "",
                "### Character ROUGE-L F1 (not factuality)", "",
                f"- SFT = {r['existing_sft_scores']['rouge_l_f1']:.3f}",
                f"- Direct = {r['direct_scores']['rouge_l_f1']:.3f}",
                f"- Auto QA = {r['auto_qa_scores']['rouge_l_f1']:.3f}",
                f"- Corrected QA = {r['oracle_qa_scores']['rouge_l_f1']:.3f}", "",
            ])
        parts.extend(["**Manual factuality check:** Any missing business point? Any false completed action? Wrong object? Wrong processing state?", "", "---", ""])
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text("\n".join(parts), encoding="utf-8")


def to_result(row, qa, note):
    return {
        "DialogueID": row["DialogueID"],
        "index": row["index"],
        "dialogue": row["dialogue"],
        "numbered_dialogue": row["numbered_dialogue"],
        "annotation_note": note,
        "auto_qa": row["stage1_qa"],
        "manual_qa": qa,
        "existing_sft_prediction": row["existing_sft_prediction"],
        "existing_sft_scores": row["existing_sft_scores"],
        "direct_prediction": row["direct_prediction"],
        "direct_scores": row["direct_scores"],
        "auto_qa_prediction": row["qa_guided_prediction"],
        "auto_qa_scores": row["qa_guided_scores"],
        # The reference summary is not passed to the model; score() uses it offline only.
        "reference_for_offline_scoring": row["reference"],
    }


def main():
    parser = argparse.ArgumentParser(description="Manually corrected QA diagnostic: 5 fixed development cases")
    parser.add_argument("--check", action="store_true", help="Validate annotations/evidence with no GPU")
    parser.add_argument("--show-prompts", action="store_true", help="Show exact auto/corrected prompts with no GPU")
    parser.add_argument("--limit", type=int, default=5, help="Process first 1..5 cases, resumable")
    args = parser.parse_args()
    if not 1 <= args.limit <= len(DIALOGUE_IDS):
        parser.error("--limit must be in 1..5")

    selected, metadata = load_and_check()
    if args.check:
        for row, qa, _ in selected:
            print(f"DialogueID={row['DialogueID']} | auto_QA={len(row['stage1_qa'])} | corrected_QA={len(qa)} | evidence_ids valid")
        print("PASS: all five development dialogues and all corrected QA evidence IDs checked; no GPU used")
        return
    if args.show_prompts:
        row, qa, _ = selected[0]
        print("=== AUTO QA -> LoRA summary (previously executed; saved prediction) ===")
        print(SUMMARY_RULES + row["dialogue"] + "\n【QA 核对清单】\n" + json.dumps(row["stage1_qa"], ensure_ascii=False, indent=2))
        print("\n=== CORRECTED QA -> SAME LoRA summary (new inference) ===")
        print(SUMMARY_RULES + row["dialogue"] + "\n【QA 核对清单】\n" + json.dumps(qa, ensure_ascii=False, indent=2))
        return

    pending_target = selected[:args.limit]
    existing = {}
    if OUTPUT_FILE.exists():
        saved = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        if saved.get("metadata") != metadata:
            raise ValueError("Existing oracle results use a different source/annotation hash. Rename the old output before rerunning.")
        existing = {r["DialogueID"]: r for r in saved.get("results", [])}
        if len(existing) != len(saved.get("results", [])):
            raise ValueError("Duplicate records in oracle output")
    rows = []
    for row, qa, note in pending_target:
        did = row["DialogueID"]
        if did in existing and "oracle_qa_scores" in existing[did]:
            rows.append(existing[did])
        else:
            rows.append(to_result(row, qa, note))
    unfinished = [r for r in rows if "oracle_qa_scores" not in r]
    if unfinished:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: run the actual oracle summary inference on your 4090 server")
        for path in (MODEL_DIR / "config.json", ADAPTER_DIR / "adapter_model.safetensors"):
            if not path.is_file():
                raise FileNotFoundError(path)
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), quantization_config=quant,
            dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True,
        )
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=False)
        model.eval()
        torch.manual_seed(42)
        for item in rows:
            if "oracle_qa_scores" in item:
                continue
            context = json.dumps(item["manual_qa"], ensure_ascii=False, indent=2)
            content = SUMMARY_RULES + item["dialogue"] + "\n【QA 核对清单】\n" + context
            output, n_tokens = generate(model, tokenizer, content, STAGE2_TOKENS)
            item["oracle_qa_prediction"] = output
            item["oracle_qa_tokens"] = n_tokens
            item["oracle_qa_scores"] = score(output, "\n".join(item["reference_for_offline_scoring"]))
            save_json(OUTPUT_FILE, {"metadata": metadata, "results": rows})
            save_json(SUMMARY_FILE, summarise(rows))
            review_markdown(rows)
            print(
                f"DialogueID={item['DialogueID']} | auto_f1={item['auto_qa_scores']['rouge_l_f1']:.3f} "
                f"-> corrected_f1={item['oracle_qa_scores']['rouge_l_f1']:.3f}",
                flush=True,
            )
    save_json(OUTPUT_FILE, {"metadata": metadata, "results": rows})
    save_json(SUMMARY_FILE, summarise(rows))
    review_markdown(rows)
    print(json.dumps(summarise(rows), ensure_ascii=False, indent=2))
    print(f"\nResults: {OUTPUT_FILE}\nSummary: {SUMMARY_FILE}\nReview: {REVIEW_FILE}")


if __name__ == "__main__":
    main()
