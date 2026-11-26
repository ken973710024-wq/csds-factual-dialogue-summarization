"""Evaluate CSDS LoRA on the exact 10 dialogues used by Baseline V2.

Run from /root/workspace/csds-project:
    python src/eval_lora_10.py

No training is performed, and the original V2 results are preserved.
"""
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "models/Qwen3-4B-Instruct-2507"
ADAPTER = Path("outputs/qlora_sft/final_adapter")
BASELINE_FILE = Path("outputs/baseline_v2_10.json")
OUTPUT_FILE = Path("outputs/lora_vs_v2_10.json")

# Exactly the same instructions as src/baseline_v2.py.
PROMPT_V2 = (
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

if not (ADAPTER / "adapter_model.safetensors").is_file():
    raise FileNotFoundError(f"LoRA adapter not found: {ADAPTER}")
if not BASELINE_FILE.is_file():
    raise FileNotFoundError(f"Baseline V2 file not found: {BASELINE_FILE}")

records = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
assert len(records) == 10, f"Expected 10 baseline samples, found {len(records)}"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, dtype="auto", device_map="auto", local_files_only=True
)
model = PeftModel.from_pretrained(base_model, str(ADAPTER), is_trainable=False)
model.eval()

results = []
for n, record in enumerate(records, 1):
    messages = [{"role": "user", "content": PROMPT_V2 + record["dialogue"]}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([formatted], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output[0, inputs["input_ids"].shape[1]:]
    lora_prediction = tokenizer.decode(new_ids, skip_special_tokens=True)
    results.append({
        "index": record["index"],
        "DialogueID": record["DialogueID"],
        "QA_count": record["QA_count"],
        "dialogue": record["dialogue"],
        "reference": record["reference"],
        "baseline_v2_prediction": record["prediction"],
        "lora_prediction": lora_prediction,
    })
    OUTPUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{n}/10] DialogueID={record['DialogueID']} evaluated", flush=True)

print(f"Saved: {OUTPUT_FILE}")
