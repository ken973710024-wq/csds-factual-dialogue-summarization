import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 读取验证集中的一条真实客服对话
with open("data/CSDS/val.json", encoding="utf-8") as f:
    data = json.load(f)

sample = data[0]

# 2. 将原始对话整理成模型能够理解的文本
dialogue = []
for turn in sample["Dialogue"]:
    speaker = "客服" if turn["speaker"] == "A" else sample["QRole"]
    utterance = turn["utterance"].replace(" ", "")
    dialogue.append(f"{speaker}：{utterance}")

dialogue_text = "\n".join(dialogue)

# 3. 构造摘要任务的 Prompt
messages = [
    {
        "role": "user",
        "content": (
            "请总结以下客服对话中客服给出的答复和处理结果。"
            "保留重要的条件、时间、金额和操作步骤。"
            "不要编造对话中不存在的信息。"
            "请用简洁的中文逐条列出摘要。\n\n"
            + dialogue_text
        )
    }
]

# 4. 从本地加载 Qwen 的 Tokenizer 和模型参数
model_path = "models/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(
    model_path, local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)
model.eval()

# 5. 将 Prompt 转换成 Token IDs
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Input tokens:", inputs["input_ids"].shape[1])

# 6. 让 Qwen 生成摘要
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

# 7. 生成完成之后，才读取人工参考摘要
print("\n========== 原始对话 ==========")
print(dialogue_text)

print("\n========== Qwen 生成摘要 ==========")
print(prediction)

print("\n========== 人工参考摘要 ==========")
for i, summary in enumerate(sample["AgentSumm"], 1):
    print(f"{i}. {summary}")
