"""CSDS validation: 4-bit Base Qwen vs the SAME 4-bit Qwen + trained LoRA.

Run inside /root/workspace/csds-project:
    python src/validate_base_vs_lora.py --count 100

Uses an untouched subset of CSDS val.json, EXCLUDING the ten examples used
for earlier V1/V2 prompt development. No training and no parameter updates.
Saves predictions after each dialogue so an interrupted run can be resumed.
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
VAL_FILE = Path("data/CSDS/val.json")
PREVIOUS_10 = Path("outputs/baseline_v2_10.json")
MAX_NEW_TOKENS = 384
SEED = 20261121

# Must match the prompt used to prepare SFT training data and evaluate V2.
PROMPT = (
    "你是一名客服质检摘要员。"
    "请总结以下对话中客服实际给出的答复和处理结果。"
    "逐个识别用户咨询的问题，避免混淆不同订单和问题。"
    "严格区分已完成、承诺将处理、可能发生和尚未确认的事项。"
    "不要把用户的要求当成客服的承诺。"
    "不要把尚未完成的操作写成已经完成。"
    "只保留原始对话有依据的事实，"
    "包括重要条件、时间、金额和必要操作步骤。"
    "忽略问候、感谢、重复背景和无关链接。"
    "请简洁地逐条列出客服摘要。"
    "每个问题保留所有必要答复，但避免重复。\n\n"
)


def dialogue_to_text(sample):
    lines = []
    for turn in sample["Dialogue"]:
        speaker = "客服" if turn["speaker"] == "A" else sample["QRole"]
        lines.append(f"{speaker}：{turn['utterance'].replace(' ', '')}")
    return "\n".join(lines)


def clean_chars(text):
    """Character-level scoring, not a semantic correctness judgement."""
    text = re.sub(r"(?m)^\s*\d+[.、．]\s*", "", text)
    return list(re.sub(r"\s+", "", text))


def rouge_l_chars(prediction, reference):
    """LCS precision/recall/F1 on Chinese characters; no external packages."""
    a, b = clean_chars(prediction), clean_chars(reference)
    if not a or not b:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    # O(len(a)*len(b)) time, O(len(b)) memory
    prev = [0] * (len(b) + 1)
    for ch in a:
        curr = [0] * (len(b) + 1)
        for j, ref_ch in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if ch == ref_ch else max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[-1]
    p, r = lcs / len(a), lcs / len(b)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def generate(model, inputs, tokenizer):
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    ids = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(ids, skip_special_tokens=True), len(ids)


def summarise(results):
    def aggregate(which):
        rows = [r[which] for r in results]
        return {
            "mean_char_rouge_l_f1": round(sum(x["rouge_l_f1"] for x in rows) / len(rows), 4),
            "mean_char_rouge_l_precision": round(sum(x["rouge_l_precision"] for x in rows) / len(rows), 4),
            "mean_char_rouge_l_recall": round(sum(x["rouge_l_recall"] for x in rows) / len(rows), 4),
            "mean_chars": round(sum(x["chars"] for x in rows) / len(rows), 1),
            "hit_max_new_tokens": sum(x["hit_max_new_tokens"] for x in rows),
        }
    return {"evaluated": len(results), "base_4bit": aggregate("base"), "lora_4bit": aggregate("lora")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="new validation samples")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    torch.manual_seed(42)
    data = json.loads(VAL_FILE.read_text(encoding="utf-8"))
    prior = json.loads(PREVIOUS_10.read_text(encoding="utf-8"))
    excluded = {int(r["index"]) for r in prior}
    pool = [i for i in range(len(data)) if i not in excluded]
    if args.count > len(pool):
        parser.error(f"--count can be at most {len(pool)}")
    indices = sorted(random.Random(SEED).sample(pool, args.count))

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"validate_base_vs_lora_{args.count}.json"
    summary_file = out_dir / f"validate_base_vs_lora_{args.count}_summary.json"
    metadata = {
        "dataset": "CSDS val.json",
        "count": args.count,
        "seed": SEED,
        "excluded_prompt_development_indices": sorted(excluded),
        "indices": indices,
        "base_precision": "4-bit NF4, double quant, bfloat16 compute",
        "comparison": "same quantized Qwen with LoRA disabled vs enabled",
        "max_new_tokens": MAX_NEW_TOKENS,
        "score": "character-level ROUGE-L LCS F1 against joined AgentSumm; NOT factuality",
    }
    if out_file.exists():
        saved = json.loads(out_file.read_text(encoding="utf-8"))
        if saved.get("metadata") != metadata:
            raise RuntimeError(f"Existing results at {out_file} use different settings; do not overwrite")
        results = saved["results"]
        done = {r["index"] for r in results}
        if len(done) != len(results) or not done.issubset(set(indices)):
            raise RuntimeError("Existing result indices are inconsistent")
        print(f"Resuming: {len(results)}/{len(indices)} already complete", flush=True)
    else:
        results, done = [], set()

    if len(done) == len(indices):
        print("All already complete.")
        print(json.dumps(summarise(results), ensure_ascii=False, indent=2))
        return

    for path in [MODEL_DIR / "config.json", ADAPTER_DIR / "adapter_model.safetensors"]:
        if not path.is_file():
            raise FileNotFoundError(path)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        quantization_config=bnb,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=False)
    model.eval()
    print(f"Evaluating {len(indices)} samples, excluding {len(excluded)} development examples", flush=True)

    for idx in indices:
        if idx in done:
            continue
        sample = data[idx]
        dialogue = dialogue_to_text(sample)
        messages = [{"role": "user", "content": PROMPT + dialogue}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda:0")

        # For a fair comparison, keep the SAME 4-bit base weights for both runs.
        # Hugging Face PEFT documents this context manager for base-only inference.
        with model.disable_adapter():
            base_pred, base_token_count = generate(model, inputs, tokenizer)
        lora_pred, lora_token_count = generate(model, inputs, tokenizer)

        reference = "\n".join(sample["AgentSumm"])
        def score(text, token_count):
            r = rouge_l_chars(text, reference)
            return {
                "rouge_l_f1": round(r["f1"], 6),
                "rouge_l_precision": round(r["precision"], 6),
                "rouge_l_recall": round(r["recall"], 6),
                "chars": len(clean_chars(text)),
                "generated_tokens": token_count,
                "hit_max_new_tokens": token_count >= MAX_NEW_TOKENS,
            }
        row = {
            "index": idx,
            "DialogueID": sample["DialogueID"],
            "QA_count": len(sample["QA"]),
            "dialogue": dialogue,
            "reference": sample["AgentSumm"],
            "base_prediction": base_pred,
            "lora_prediction": lora_pred,
            "base": score(base_pred, base_token_count),
            "lora": score(lora_pred, lora_token_count),
        }
        results.append(row)
        done.add(idx)
        out_file.write_text(json.dumps({"metadata": metadata, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        summary_file.write_text(json.dumps(summarise(results), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{len(results)}/{len(indices)}] DialogueID={sample['DialogueID']} | "
              f"base={row['base']['rouge_l_f1']:.3f} LoRA={row['lora']['rouge_l_f1']:.3f}", flush=True)

    print("\nCompleted. Mean scores:", flush=True)
    print(json.dumps(summarise(results), ensure_ascii=False, indent=2), flush=True)
    print(f"Predictions: {out_file}\nSummary: {summary_file}", flush=True)


if __name__ == "__main__":
    main()
