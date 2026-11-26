"""Experiment 3: isolate Stage-2 faithful compression using five fixed, corrected-QA cases.

Run from the original csds-project root:
    python src/validate_stage2_faithful_compression_dev5.py --check
    python src/validate_stage2_faithful_compression_dev5.py --show-prompts
    python src/validate_stage2_faithful_compression_dev5.py --limit 1
    python src/validate_stage2_faithful_compression_dev5.py --limit 5

Controlled comparison:
  A: existing manually corrected QA + ORIGINAL Stage-2 summarisation prompt
     (saved output from validate_oracle_qa_dev5.py; NO recomputation)
  B: the EXACT SAME manually corrected QA + NEW faithful-compression prompt
     (one new LoRA inference per dialogue; same model, greedy decode, 384 tokens)

Development-only diagnostic. Neither QA nor reference summaries are used to train.
Reference summaries are NEVER included in the generation prompt. The five
hand-selected cases are not an unbiased test set.
"""

import argparse
import hashlib
import json
from pathlib import Path

from validate_oracle_qa_dev5 import (
    DEV20_FILE, MANUAL_FILE, DIALOGUE_IDS, load_and_check, sha256_file,
)
from validate_qa_pipeline_dev20 import (
    MODEL_DIR, ADAPTER_DIR, STAGE2_TOKENS, SUMMARY_RULES, generate, score, save_json,
)

PREVIOUS_OUTPUT = Path("outputs/validate_oracle_qa_dev5.json")
OUTPUT_FILE = Path("outputs/validate_stage2_faithful_compression_dev5.json")
SUMMARY_FILE = Path("outputs/validate_stage2_faithful_compression_dev5_summary.json")
REVIEW_FILE = Path("outputs/validate_stage2_faithful_compression_dev5_review.md")
PROMPT_FILE = Path("outputs/validate_stage2_faithful_compression_dev5_prompts.txt")

# Only intentional experimental change: replace Stage-2 instruction text.
# Keep dialogue text, QA JSON serialization, tokenizer/chat template, model,
# LoRA adapter and decoding exactly as in the earlier Oracle experiment.
COMPACT_RULES = """你是一名客服答复摘要员。根据原始对话和【QA 核对清单】生成准确、简洁的客服答复摘要。
【QA 核对清单】是辅助信息，不是绝对真相：如有冲突，以原始对话为准；不能补造原文没有的事实。

先在内部核对，再直接给最终摘要（不要输出核对过程）：
1. 按不同的订单、商品、服务对象分别核对。同一会话中换了商品时，绝不能混淆两款商品的答复；必要时用“第一款/第二款商品”区分。
2. 覆盖各个独立的重要客服结论以及客服主动给出的关键后续方案。重复追问和同义客服答复合并，不把寒暄、道谢、纯情绪当作业务结论。
3. 严格保留否定、条件、尚未确认、预计时限、处理中、承诺未来处理、已完成等事实区别。不能把“准备处理/正在处理”写成“已经完成”，不能把个别商品问题推广成整个商品类别。
4. 只保留对客服处理结果有用的信息；删除客套话、重复解释、无意义的商品链接和证据编号。用短句表达事实，不逐字复述 QA，也不把“原文没有说什么”的纠错说明照抄进摘要。
5. 简洁优先，但不能为了减少字数删除不同商品的相反结论或重要的下一步行动。一般每个独立业务结论一句；若两条紧密相关，可合成一句。不要为了固定句数或字数而牺牲事实。

只输出中文编号摘要，如“1. 客服表示……”。不要输出 JSON、问题清单、解释、证据、分析过程。
【原始对话】
"""


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_prompt(dialogue, qa, rules):
    # IMPORTANT: same formatting as validate_oracle_qa_dev5.py
    return rules + dialogue + "\n【QA 核对清单】\n" + json.dumps(qa, ensure_ascii=False, indent=2)


def load_inputs():
    selected, baseline_meta = load_and_check()  # Validate original 20-case data and every QA evidence ID
    if not PREVIOUS_OUTPUT.is_file():
        raise FileNotFoundError(f"Missing {PREVIOUS_OUTPUT}; run the completed Oracle QA experiment first")
    old_payload = json.loads(PREVIOUS_OUTPUT.read_text(encoding="utf-8"))
    old_meta = old_payload["metadata"]
    # Prevent comparing different runs/annotations with matching dialogue labels.
    for key in ("original_dev20_sha256", "manual_qa_sha256", "adapter", "model", "max_new_tokens", "sampling"):
        if old_meta.get(key) != baseline_meta.get(key):
            raise ValueError(f"Prior Oracle QA result/source mismatch on {key}; do not compare incompatible runs")
    if old_meta.get("summary_prompt_version") != baseline_meta.get("summary_prompt_version"):
        raise ValueError("Prior Oracle QA summary prompt version mismatch")
    if old_meta.get("reference_or_reference_summary_used_in_model_input") is not False:
        raise ValueError("Unexpected reference leakage in previous Oracle experiment")
    previous = old_payload.get("results", [])
    old_by_id = {row["DialogueID"]: row for row in previous}
    if len(old_by_id) != len(previous) or set(old_by_id) != set(DIALOGUE_IDS):
        raise ValueError("Expected exactly the five fixed completed Oracle QA records")

    out = []
    for src_row, corrected_qa, note in selected:
        did = src_row["DialogueID"]
        old = old_by_id[did]
        if not all(field in old for field in ("manual_qa", "oracle_qa_prediction", "oracle_qa_scores", "auto_qa_scores", "direct_scores", "existing_sft_scores")):
            raise ValueError(f"Missing previous Oracle result fields for {did}")
        if old["manual_qa"] != corrected_qa:
            raise ValueError(f"Oracle QA changed since last inference for {did}")
        if old["dialogue"] != src_row["dialogue"] or old["index"] != src_row["index"]:
            raise ValueError(f"Dialogue changed since previous experiment for {did}")
        if old["reference_for_offline_scoring"] != src_row["reference"]:
            raise ValueError(f"Reference changed since previous experiment for {did}")
        out.append({
            "DialogueID": did,
            "index": old["index"],
            "dialogue": old["dialogue"],
            "numbered_dialogue": old["numbered_dialogue"],
            "annotation_note": note,
            "manual_qa": corrected_qa,
            "reference_for_offline_scoring": old["reference_for_offline_scoring"],
            "existing_sft_prediction": old["existing_sft_prediction"],
            "existing_sft_scores": old["existing_sft_scores"],
            "direct_prediction": old["direct_prediction"],
            "direct_scores": old["direct_scores"],
            "auto_qa_prediction": old["auto_qa_prediction"],
            "auto_qa_scores": old["auto_qa_scores"],
            "oracle_old_prediction": old["oracle_qa_prediction"],
            "oracle_old_scores": old["oracle_qa_scores"],
        })

    meta = {
        "experiment": "stage2_faithful_compression_oracle_dev5_v1",
        "dialogue_ids": list(DIALOGUE_IDS),
        "old_oracle_file_sha256": sha256_file(PREVIOUS_OUTPUT),
        "dev20_file_sha256": sha256_file(DEV20_FILE),
        "manual_qa_file_sha256": sha256_file(MANUAL_FILE),
        "old_stage2_prompt_sha256": sha256_text(SUMMARY_RULES),
        "new_stage2_prompt_sha256": sha256_text(COMPACT_RULES),
        "qa_format": "json.dumps(qa, ensure_ascii=False, indent=2)",
        "model": str(MODEL_DIR),
        "adapter": str(ADAPTER_DIR),
        "max_new_tokens": STAGE2_TOKENS,
        "decoding": "greedy; do_sample=False",
        "reference_used_during_generation": False,
        "annotation_source": "manual QA curated from original dialogue only",
        "design": "Same manually corrected QA, same LoRA, same model/chat_template/decode; Stage-2 instruction changed ONLY",
    }
    return out, meta


def averages(rows, key):
    return {field: round(sum(r[key][field] for r in rows) / len(rows), 4)
            for field in ("rouge_l_f1", "rouge_l_precision", "rouge_l_recall", "chars")}


def summarise(rows):
    finished = [r for r in rows if "oracle_compact_scores" in r]
    result = {"processed": len(rows), "fully_evaluated": len(finished), "planned": len(DIALOGUE_IDS)}
    if not finished:
        return result
    for key, field in (("existing_sft", "existing_sft_scores"),
                       ("direct", "direct_scores"),
                       ("auto_qa", "auto_qa_scores"),
                       ("oracle_old", "oracle_old_scores"),
                       ("oracle_compact", "oracle_compact_scores")):
        result[key] = averages(finished, field)
    a, b = result["oracle_old"], result["oracle_compact"]
    result["compact_minus_original_oracle"] = {
        field: round(b[field] - a[field], 4)
        for field in ("rouge_l_f1", "rouge_l_precision", "rouge_l_recall", "chars")
    }
    result["compact_f1_wins"] = sum(r["oracle_compact_scores"]["rouge_l_f1"] > r["oracle_old_scores"]["rouge_l_f1"] for r in finished)
    result["compact_shorter_count"] = sum(r["oracle_compact_scores"]["chars"] < r["oracle_old_scores"]["chars"] for r in finished)
    result["note"] = "Character-level ROUGE-L and length are NOT factuality/coverage; manually inspect same QA against original dialogue. Dev5 chosen examples only."
    return result


def write_review(rows):
    lines = [
        "# Experiment 3 — Stage-2 faithful-compression prompt diagnostic (fixed Oracle QA)", "",
        "Control: same original conversation + same manually corrected QA + same LoRA.",
        "Only the Stage-2 instruction is changed. References are used for OFFLINE ROUGE scoring only.",
        "**Factuality must be checked against original dialogue, not inferred from ROUGE.**", "",
    ]
    for row in rows:
        lines += [f"## DialogueID {row['DialogueID']} (index {row['index']})", "",
                  "### Original dialogue", "", "```text", row["numbered_dialogue"], "```", "",
                  "### Fixed corrected QA", "", "```json",
                  json.dumps(row["manual_qa"], ensure_ascii=False, indent=2), "```", "",
                  "### Previous Oracle QA + old Stage-2 prompt", "", "```text", row["oracle_old_prediction"], "```", ""]
        if "oracle_compact_prediction" in row:
            a, b = row["oracle_old_scores"], row["oracle_compact_scores"]
            lines += ["### Same Oracle QA + new faithful-compression prompt", "", "```text",
                      row["oracle_compact_prediction"], "```", "",
                      f"- ROUGE-L F1: old={a['rouge_l_f1']:.3f}, new={b['rouge_l_f1']:.3f}",
                      f"- Characters: old={a['chars']}, new={b['chars']}", ""]
        lines += ["### Reference (offline scoring only)", "", "```text",
                  "\n".join(row["reference_for_offline_scoring"]), "```", "",
                  "### Manual check (please compare against original dialogue)", "",
                  "- [ ] Every independent important business conclusion remains present", 
                  "- [ ] Different products/orders have not been confused", 
                  "- [ ] No unsupported claim; no promised/pending action rewritten as completed", 
                  "- [ ] No significant duplicate, fluff, or verbose evidence copying", "",
                  "---", ""]
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Stage-2 faithful compression: same fixed 5 oracle QA, change only Stage-2 prompt")
    parser.add_argument("--check", action="store_true", help="Validate all saved inputs, IDs, hashes; no GPU")
    parser.add_argument("--show-prompts", action="store_true", help="Save and print old/new actual prompt contents; no GPU")
    parser.add_argument("--limit", type=int, default=5, help="Process first 1..5 cases; safely resumable")
    args = parser.parse_args()
    if not 1 <= args.limit <= len(DIALOGUE_IDS):
        parser.error("--limit must be 1..5")
    full_rows, metadata = load_inputs()
    if args.check:
        for row in full_rows:
            print(f"PASS DialogueID={row['DialogueID']}: same corrected QA={len(row['manual_qa'])} items, previous Oracle prediction available")
        print("PASS: five evidence-checked fixed QA; model/prompt settings ready (no GPU inference)")
        return
    if args.show_prompts:
        row = full_rows[0]
        old = build_prompt(row["dialogue"], row["manual_qa"], SUMMARY_RULES)
        new = build_prompt(row["dialogue"], row["manual_qa"], COMPACT_RULES)
        content = "=== OLD PROMPT (previous result) ===\n" + old + "\n\n=== NEW PROMPT (to generate) ===\n" + new + "\n"
        PROMPT_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROMPT_FILE.write_text(content, encoding="utf-8")
        print(content)
        print(f"\nSaved exact prompt contents to {PROMPT_FILE}")
        return

    # Validate that a resumed checkpoint was made from precisely these files.
    old_rows = {}
    if OUTPUT_FILE.exists():
        cached = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        if cached.get("metadata") != metadata:
            raise ValueError("Existing output from different input hashes or prompt; rename before retrying")
        old_rows = {r["DialogueID"]: r for r in cached.get("results", [])}
        if len(old_rows) != len(cached.get("results", [])):
            raise ValueError("Duplicate DialogueID in checkpoint")
    rows = []
    for current in full_rows[:args.limit]:
        did = current["DialogueID"]
        prev = old_rows.get(did)
        if prev and "oracle_compact_scores" in prev:
            rows.append(prev)
        else:
            rows.append(current)
    # If user ran --limit 5 then --limit 1, we do not silently delete saved work.
    if OUTPUT_FILE.exists() and any(k not in {r['DialogueID'] for r in rows} for k in old_rows):
        raise ValueError("Checkpoint already contains more examples than --limit. Run with a limit >= previous progress")

    unfinished = [r for r in rows if "oracle_compact_scores" not in r]
    if unfinished:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run inference on the RTX 4090 server")
        for path in (MODEL_DIR / "config.json", ADAPTER_DIR / "adapter_model.safetensors"):
            if not path.exists():
                raise FileNotFoundError(f"Missing model/adapter: {path}")
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), quantization_config=quant,
                                                    dtype=torch.bfloat16, device_map={"": 0},
                                                    local_files_only=True)
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=False)
        model.eval()
        torch.manual_seed(42)
        for row in rows:
            if "oracle_compact_scores" in row:
                print(f"Skip completed DialogueID={row['DialogueID']}")
                continue
            content = build_prompt(row["dialogue"], row["manual_qa"], COMPACT_RULES)
            text, tokens = generate(model, tokenizer, content, STAGE2_TOKENS)
            row["oracle_compact_prediction"] = text
            row["oracle_compact_tokens"] = tokens
            row["oracle_compact_scores"] = score(text, "\n".join(row["reference_for_offline_scoring"]))
            save_json(OUTPUT_FILE, {"metadata": metadata, "results": rows})
            save_json(SUMMARY_FILE, summarise(rows))
            write_review(rows)
            old = row["oracle_old_scores"]
            now = row["oracle_compact_scores"]
            print(f"DialogueID={row['DialogueID']} | old_F1={old['rouge_l_f1']:.3f} -> new_F1={now['rouge_l_f1']:.3f} | chars {old['chars']} -> {now['chars']}", flush=True)
    save_json(OUTPUT_FILE, {"metadata": metadata, "results": rows})
    save_json(SUMMARY_FILE, summarise(rows))
    write_review(rows)
    print(json.dumps(summarise(rows), ensure_ascii=False, indent=2))
    print(f"Outputs:\n {OUTPUT_FILE}\n {SUMMARY_FILE}\n {REVIEW_FILE}")


if __name__ == "__main__":
    main()
