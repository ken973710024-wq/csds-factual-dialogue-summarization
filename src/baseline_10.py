import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 固定抽取10条验证集对话
with open("data/CSDS/val.json", encoding="utf-8") as f:
    data = json.load(f)

rng = random.Random(42)
indices = sorted(rng.sample(range(len(data)), 10))

# 2. 只加载一次Qwen模型
model_path = "models/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    local_files_only=True
)
model.eval()

results = []

# 3. 对10条对话分别生成摘要
for n, idx in enumerate(indices, 1):
    sample = data[idx]

    dialogue = []
    for turn in sample["Dialogue"]:
        speaker = "客服" if turn["speaker"] == "A" else sample["QRole"]
        utterance = turn["utterance"].replace(" ", "")
        dialogue.append(f"{speaker}：{utterance}")

    dialogue_text = "\n".join(dialogue)

    messages = [{
        "role": "user",
        "content": (
            "请总结以下客服对话中客服给出的答复和处理结果。"
            "保留重要的条件、时间、金额和操作步骤。"
            "不要编造对话中不存在的信息。"
            "请用简洁的中文逐条列出摘要。\n\n"
            + dialogue_text
        )
    }]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        [text], return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(
        new_tokens, skip_special_tokens=True
    )

    results.append({
        "index": idx,
        "DialogueID": sample["DialogueID"],
        "QA_count": len(sample["QA"]),
        "dialogue": dialogue_text,
        "prediction": prediction,
        "reference": sample["AgentSumm"]
    })

    # 每完成一条就保存，避免中途退出丢失结果
    with open("outputs/baseline_10.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[{n}/10] DialogueID={sample['DialogueID']} completed", flush=True)

print("\nAll predictions saved to outputs/baseline_10.json")