"""从项目根目录运行，将交接训练样例从已复核的 30 条扩展到 128 条的原文名单。"""

import fcntl
import json
from pathlib import Path
import random
import sys


PROJECT = Path.cwd()
sys.path.insert(0, str(PROJECT))
from validate_unseen_summaries import atomic_json, by_id, read, review_source, signature


SEED = 20261125
PILOT_SIZE = 128
SEED_FILE = Path(__file__).with_name("target_pairs_train30.jsonl")
OUTPUT = PROJECT / "outputs/summary_handoff_train_pilot128/next98_sources.json"


def prepare_packet(train_rows, seed_pairs):
    train = by_id(train_rows)
    seeds = by_id(seed_pairs)
    if len(seeds) != 30:
        raise ValueError("种子文件应包含本次抽查的 30 段对话。")

    seen_texts = set()
    for key, pair in seeds.items():
        if key not in train or review_source(train[key])[0] != pair["dialogue"]:
            raise ValueError(f"种子 {key} 与本地训练集原文不一致，请检查项目数据版本。")
        seen_texts.add(signature(train[key]))

    eligible = []
    for key in sorted(train):
        if key in seeds:
            continue
        signature_key = signature(train[key])
        if signature_key not in seen_texts:
            eligible.append(key)
            seen_texts.add(signature_key)

    selected = random.Random(SEED).sample(eligible, PILOT_SIZE - len(seeds))
    return {
        "split": "train",
        "seed": SEED,
        "pilot_size": PILOT_SIZE,
        "previous_ids": list(seeds),
        "sources": {key: review_source(train[key])[0] for key in selected},
    }


def main():
    train_rows = read(PROJECT / "data/CSDS/train.json")
    with SEED_FILE.open(encoding="utf-8") as stream:
        seed_pairs = [json.loads(line) for line in stream if line.strip()]
    packet = prepare_packet(train_rows, seed_pairs)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with (OUTPUT.parent / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not OUTPUT.exists():
            atomic_json(OUTPUT, packet)
        elif read(OUTPUT) != packet:
            raise ValueError(f"{OUTPUT} 已有不同内容，已保留原文件。")
    print(f"已有 {len(seed_pairs)} 段候选答案，另抽取 {len(packet['sources'])} 段训练原文。")
    print(OUTPUT)


if __name__ == "__main__":
    main()
