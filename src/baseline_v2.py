import json
import random
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 读取验证集
with open("data/CSDS/val.json", encoding="utf-8") as f:
    data = json.load(f)

# 与 V1 使用完全相同的 10 条对话
rng = random.Random(42)
indices = sorted(rng.sample(range(len(data)), 10))

# 2. 加载 Qwen（不训练）
model_path = "models/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    local_files_only=True
)

model.eval()

# 3. V2 Prompt：本次实验唯一主动改变的因素
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

results = []

output_path = Path("outputs/baseline_v2_10.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

# 4. 对同样的 10 条对话进行推理
for n, idx in enumerate(indices, 1):

    sample = data[idx]

    # 整理原始对话
    dialogue = []

    for turn in sample["Dialogue"]:
        speaker = (
            "客服"
            if turn["speaker"] == "A"
            else sample["QRole"]
        )

        utterance = turn["utterance"].replace(" ", "")
        dialogue.append(f"{speaker}：{utterance}")

    dialogue_text = "\n".join(dialogue)

    # 构造 Prompt
    messages = [{
        "role": "user",
        "content": PROMPT_V2 + dialogue_text
    }]

    # Tokenize
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    # 模型推理
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 只解码新生成的 Token
    new_tokens = output[0][inputs["input_ids"].shape[1]:]

    prediction = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    # 保存预测结果及人工参考摘要
    results.append({
        "index": idx,
        "DialogueID": sample["DialogueID"],
        "QA_count": len(sample["QA"]),
        "dialogue": dialogue_text,
        "prediction": prediction,
        "reference": sample["AgentSumm"]
    })

    # 每完成一条就保存一次
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"[{n}/10] DialogueID={sample['DialogueID']} completed",
        flush=True
    )

print("\nAll V2 predictions saved to:", output_path)
