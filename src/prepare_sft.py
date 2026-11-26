import json
from pathlib import Path
from transformers import AutoTokenizer

DATA_DIR = Path("data/CSDS")
OUT_DIR = Path("data")

# 与 Baseline V2 保持一致的任务指令
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

tokenizer = AutoTokenizer.from_pretrained(
    "models/Qwen3-4B-Instruct-2507",
    local_files_only=True
)

def prepare(split):
    with open(DATA_DIR / f"{split}.json", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    lengths = []

    for sample in raw_data:
        dialogue = []

        for turn in sample["Dialogue"]:
            speaker = (
                "客服" if turn["speaker"] == "A"
                else sample["QRole"]
            )
            text = turn["utterance"].replace(" ", "")
            dialogue.append(f"{speaker}：{text}")

        prompt = PROMPT + "\n".join(dialogue)

        target = "\n".join(
            f"{i}. {summary}"
            for i, summary in enumerate(sample["AgentSumm"], 1)
        )

        record = {
            "DialogueID": sample["DialogueID"],
            "prompt": prompt,
            "target": target
        }
        records.append(record)

        # 统计训练时需要的 Token 数量
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True
        )

        target_ids = tokenizer(
            target, add_special_tokens=False
        )["input_ids"]

        lengths.append(len(prompt_ids) + len(target_ids) + 1)

    out_path = OUT_DIR / f"{split}_sft.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lengths.sort()
    print(f"\n{split}: {len(records)} samples")
    print("Median tokens:", lengths[len(lengths) // 2])
    print("P95 tokens:", lengths[int(len(lengths) * 0.95)])
    print("Max tokens:", lengths[-1])
    print("Over 2048 tokens:", sum(x > 2048 for x in lengths))
    print("Saved:", out_path)

    if split == "train":
        print("\n===== Sample Prompt =====")
        print(records[0]["prompt"][:700])
        print("\n===== Sample Target =====")
        print(records[0]["target"])

prepare("train")
prepare("val")
