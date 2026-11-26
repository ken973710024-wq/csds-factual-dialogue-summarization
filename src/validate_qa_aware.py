"""Experiment 3, stage B: QA-aware two-pass prompting (NO new training).

Run from /root/workspace/csds-project after uploading this file into src/:
  python src/validate_qa_aware.py --subset dev --limit 5
  python src/validate_qa_aware.py --subset dev
  python src/validate_qa_aware.py --subset holdout

Reuses the exact 100 examples and LoRA outputs in
  outputs/validate_base_vs_lora_100.json
Never feeds human references or gold QA annotations into the model.
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_DIR = Path("models/Qwen3-4B-Instruct-2507")
ADAPTER_DIR = Path("outputs/qlora_sft/final_adapter")
BASELINE_FILE = Path("outputs/validate_base_vs_lora_100.json")
OUTPUT_DIR = Path("outputs")
SPLIT_SEED = 20261122
DEV_SIZE = 20
EXTRACT_MAX_TOKENS = 768
SUMMARY_MAX_TOKENS = 384  # same limit as experiment 2

EXTRACT_PROMPT = """你是客服对话的结构化事实抽取器。暂时不要生成最终摘要。
逐项识别对话中所有重要的、互不重复的具体问题和客服主动给出的重要处理方案。
不能只按宽泛类别（如物流、售后）合并：同一类别中不同问题要分别记录。
按对话中的最终答复确认结果；存在前后矛盾时保留冲突，不要擅自解决。
每一项输出：
- category: 业务分类（如物流、发票、退款、售后、账号、商品、其他）
- question: 用户具体咨询的问题；如果是客服主动提供的方案，写“客服主动说明：...”
- answer: 客服明确提供的答复、关键条件和限制；没有明确答复写“未明确答复”
- status: 只能是“信息答复”“已完成”“承诺后续处理”“处理中”“未确认”之一
- evidence: 可追溯的客服原话摘录，不要编造
只根据对话事实填写；不要把用户要求或预计事项写成已完成。
仅返回 JSON 数组，每条记录为一个对象，不加 Markdown 或额外说明。

【原始对话】
"""

SUMMARY_PROMPT = """你是一名客服质检摘要员。请根据【原始对话】及【问答提取结果】生成最终摘要。
问答提取结果是模型生成的草稿，并非真实标签：必须与原始对话核对；冲突时以原始对话为准。
逐项覆盖所有重要且有客服实际答复的问题，以及客服主动给出的重要处理结果；
同一问题的重复表述应合并，不要漏掉不同问题的关键答复。
特别区分：已完成、承诺将处理、进行中、可能发生、尚未确认。
不要把客户诉求当作客服承诺，不要把尚未完成的操作当作已经完成。
只保留客服真实答复及必要条件、时间、金额；不要添加对话中没有的细节。
以简洁编号列表输出最终客服摘要；不输出类别、提取过程、证据列表或附注。

【原始对话】
"""


def clean_chars(text):
    text = re.sub(r"(?m)^\s*\d+[.、．]\s*", "", text)
    return list(re.sub(r"\s+", "", text))


def rouge_l_chars(prediction, reference):
    """Same char-level LCS ROUGE-L implementation as experiment 2."""
    a, b = clean_chars(prediction), clean_chars(reference)
    if not a or not b:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    prev = [0] * (len(b) + 1)
    for ch in a:
        curr = [0] * (len(b) + 1)
        for j, ref_ch in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if ch == ref_ch else max(prev[j], curr[j - 1])
        prev = curr
    length = prev[-1]
    p, r = length / len(a), length / len(b)
    return {"precision": p, "recall": r, "f1": (2 * p * r / (p + r)) if p + r else 0.0}


def score(prediction, reference, generated_tokens):
    s = rouge_l_chars(prediction, reference)
    return {
        "rouge_l_f1": round(s["f1"], 6),
        "rouge_l_precision": round(s["precision"], 6),
        "rouge_l_recall": round(s["recall"], 6),
        "chars": len(clean_chars(prediction)),
        "generated_tokens": generated_tokens,
        "hit_max_new_tokens": generated_tokens >= SUMMARY_MAX_TOKENS,
    }


def parse_qa_json(raw):
    """JSON may have code fences / surrounding text; failures are recorded."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    candidates = [raw]
    start, end = raw.find("["), raw.rfind("]")
    if start >= 0 and end > start:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return value
        except (ValueError, TypeError):
            pass
    return None


def generate(model, tokenizer, content, max_new_tokens):
    messages = [{"role": "user", "content": content}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    tokens = output[0, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(tokens, skip_special_tokens=True).strip()
    return decoded, len(tokens)


def write_json_atomic(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def mean(scores, key):
    return round(sum(x[key] for x in scores) / len(scores), 4)


def summarise(results):
    if not results:
        return {"evaluated": 0}
    old = [r["baseline_lora_scores"] for r in results]
    new = [r["qa_aware_scores"] for r in results]
    def describe(rows):
        return {
            "mean_char_rouge_l_f1": mean(rows, "rouge_l_f1"),
            "mean_char_rouge_l_precision": mean(rows, "rouge_l_precision"),
            "mean_char_rouge_l_recall": mean(rows, "rouge_l_recall"),
            "mean_chars": round(sum(r["chars"] for r in rows) / len(rows), 1),
            "hit_max_new_tokens": sum(r["hit_max_new_tokens"] for r in rows),
        }
    return {
        "evaluated": len(results),
        "standard_sft_lora": describe(old),
        "qa_aware_prompt_same_lora": describe(new),
        "qa_aware_f1_wins": sum(n["rouge_l_f1"] > b["rouge_l_f1"] for n, b in zip(new, old)),
        "qa_aware_recall_wins": sum(n["rouge_l_recall"] > b["rouge_l_recall"] for n, b in zip(new, old)),
        "stage1_json_parsed": sum(r["stage1_json_parsed"] for r in results),
        "note": "ROUGE is not factuality or semantic QA coverage; inspect dialogue and references manually.",
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: two-pass QA-aware prompting")
    parser.add_argument("--subset", choices=["dev", "holdout"], default="dev")
    parser.add_argument("--limit", type=int, default=0, help="evaluate first N samples of the subset; 0 = all")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if not BASELINE_FILE.is_file():
        raise FileNotFoundError(f"Missing {BASELINE_FILE}. Keep experiment 2's 100-example JSON in outputs/.")

    source = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    base_rows = source["results"]
    if len(base_rows) != 100 or len({r["index"] for r in base_rows}) != 100:
        raise RuntimeError("Expected exactly 100 unique entries in experiment 2 results")
    ids = sorted(r["index"] for r in base_rows)
    dev_ids = set(random.Random(SPLIT_SEED).sample(ids, DEV_SIZE))
    selected = [r for r in base_rows if (r["index"] in dev_ids) == (args.subset == "dev")]
    selected.sort(key=lambda r: r["index"])

    metadata = {
        "experiment": "Experiment 3B: two-pass QA-aware prompt, same LoRA as experiment 2",
        "subset": args.subset,
        "split_seed": SPLIT_SEED,
        "dev_size": DEV_SIZE,
        "selected_indices": [r["index"] for r in selected],
        "baseline_file": str(BASELINE_FILE),
        "adapter": str(ADAPTER_DIR),
        "stage1_max_new_tokens": EXTRACT_MAX_TOKENS,
        "stage2_max_new_tokens": SUMMARY_MAX_TOKENS,
        "reference_used_during_inference": False,
        "prompt_version": "qa_aware_two_pass_v1",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"validate_qa_aware_{args.subset}.json"
    summary_path = OUTPUT_DIR / f"validate_qa_aware_{args.subset}_summary.json"
    if out_path.exists():
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        if saved.get("metadata") != metadata:
            raise RuntimeError(f"Existing {out_path} has other settings; will not overwrite")
        results = saved["results"]
        done = {r["index"] for r in results}
        if len(done) != len(results) or not done.issubset(set(metadata["selected_indices"])):
            raise RuntimeError("Saved sample IDs are inconsistent")
    else:
        results, done = [], set()

    target = selected[:args.limit] if args.limit else selected
    pending = [r for r in target if r["index"] not in done]
    if not pending:
        print(json.dumps(summarise(results), ensure_ascii=False, indent=2))
        print("No pending samples.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found; run on the RTX 4090 cloud instance")
    for f in [MODEL_DIR / "config.json", ADAPTER_DIR / "adapter_model.safetensors"]:
        if not f.is_file():
            raise FileNotFoundError(f)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=False)
    model.eval()
    print(f"Experiment 3 | {args.subset} | completed {len(done)} | new {len(pending)}", flush=True)

    for row in pending:
        dialogue = row["dialogue"]
        stage1_raw, stage1_tokens = generate(
            model, tokenizer, EXTRACT_PROMPT + dialogue, EXTRACT_MAX_TOKENS
        )
        parsed = parse_qa_json(stage1_raw)
        qa_text = (
            json.dumps(parsed, ensure_ascii=False, indent=2) if parsed is not None
            else stage1_raw  # fallback: stage2 sees raw extraction and must check dialogue
        )
        final_prompt = (
            SUMMARY_PROMPT + dialogue + "\n\n【问答提取结果（待核对）】\n" + qa_text
        )
        summary, stage2_tokens = generate(model, tokenizer, final_prompt, SUMMARY_MAX_TOKENS)

        reference = "\n".join(row["reference"])  # scoring ONLY, never in the prompt
        old_scores = score(row["lora_prediction"], reference, row["lora"]["generated_tokens"])
        new_scores = score(summary, reference, stage2_tokens)
        result = {
            "index": row["index"],
            "DialogueID": row["DialogueID"],
            "QA_count": row["QA_count"],  # metadata ONLY, not model input
            "dialogue": dialogue,
            "reference": row["reference"],  # for after-inference evaluation
            "baseline_lora_prediction": row["lora_prediction"],
            "baseline_lora_scores": old_scores,
            "stage1_qa_raw": stage1_raw,
            "stage1_qa_parsed": parsed,
            "stage1_json_parsed": parsed is not None,
            "stage1_generated_tokens": stage1_tokens,
            "stage1_hit_max_new_tokens": stage1_tokens >= EXTRACT_MAX_TOKENS,
            "qa_aware_prediction": summary,
            "qa_aware_scores": new_scores,
        }
        results.append(result)
        done.add(row["index"])
        write_json_atomic(out_path, {"metadata": metadata, "results": results})
        write_json_atomic(summary_path, summarise(results))
        print(
            f"[{len(done)}/{len(selected)}] DialogueID={row['DialogueID']} "
            f"SFT={old_scores['rouge_l_f1']:.3f} QA={new_scores['rouge_l_f1']:.3f} "
            f"QA-JSON={'OK' if parsed is not None else 'FAILED'}",
            flush=True,
        )

    print("\nSummary:", json.dumps(summarise(results), ensure_ascii=False, indent=2))
    print(f"Predictions: {out_path}\nScores: {summary_path}")


if __name__ == "__main__":
    main()
