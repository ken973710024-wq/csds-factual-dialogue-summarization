#!/usr/bin/env python3
"""Convert existing CSDS human QA annotations to Stage-1 QA extraction SFT data.

Run from the csds-project root:
    python src/prepare_qa_sft_data.py
    python src/prepare_qa_sft_data.py --preview 2

No model/GPU/training required. Reads *only* data/CSDS/train.json.
Reserves a deterministic dialogue-level development split from the training set.
Does not invent object/status gold labels; category is CSDS 'intent' (not 'Topic').
Evidence is limited to real agent utterances (speaker=A).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from pathlib import Path

PROMPT_HEADER = """你是客服对话的结构化问答提取器。下面有带编号的客服对话：[U...] 是用户发言，[C...] 是客服发言。
仅依据对话，识别有明确客服答复的独立业务问题，把不同商品、订单、服务对象分开；同一问题的重复追问合并。
输出 JSON 数组，每项只包含以下五个字段，字段名完全一致：
- category：业务意图类别；
- question：用户业务问题或有明确客服答复的问题简述；
- answer：客服实际答复的忠实归纳，保留重要条件、否定、时间和数量；
- question_ids：支持该问题的用户发言编号（[U...]），没有可引用的用户发言时用空数组；
- evidence_ids：支持该答复的客服发言编号（[C...]），不得引用用户发言。
不得编造客服已完成的操作；不得混淆两个商品或不同订单。忽略寒暄、重复和无关聊天。
只输出 JSON 数组，不输出解释、Markdown 或最终客服摘要。
【带编号的客服对话】
"""


def utterance_text(turn):
    val = turn.get("utterance", "")
    if isinstance(val, list):
        val = " ".join(str(v) for v in val)
    return re.sub(r"\s+", "", str(val))


def numbered_lines(dialogue, dialog_id):
    lines, turn_to_label, speaker_of = [], {}, {}
    u = c = 0
    for idx, turn in enumerate(dialogue):
        if not isinstance(turn, dict):
            raise ValueError(f"DialogueID={dialog_id}: turn {idx} not a dictionary")
        src_id = str(turn.get("turn", idx))
        if src_id in turn_to_label:
            raise ValueError(f"DialogueID={dialog_id}: duplicate turn ID {src_id}")
        speaker = turn.get("speaker")
        if speaker == "A":
            label = f"C{c}"; c += 1
            role = "客服"
        else:
            label = f"U{u}"; u += 1
            role = "用户"
        turn_to_label[src_id] = label
        speaker_of[src_id] = speaker
        lines.append(f"[{label}] {role}：{utterance_text(turn)}")
    return "\n".join(lines), turn_to_label, speaker_of


def map_utterance_ids(ids, label_by_turn, expected_prefix, counters, name):
    if not isinstance(ids, list):
        counters[f"{name}_nonlist"] += 1
        return []
    output = []
    for src_id in ids:
        label = label_by_turn.get(str(src_id))
        if label is None:
            counters[f"{name}_unresolved"] += 1
        elif not label.startswith(expected_prefix):
            counters[f"{name}_wrong_speaker_removed"] += 1
        elif label not in output:
            output.append(label)
    return output


def build_examples(source, counts, intents):
    examples = []
    seen_dialogues = set()
    for idx, row in enumerate(source):
        d_id = row.get("DialogueID")
        if d_id is None or str(d_id) in seen_dialogues:
            raise ValueError(f"Missing or duplicated DialogueID at source index {idx}: {d_id!r}")
        seen_dialogues.add(str(d_id))
        turns = row.get("Dialogue")
        raw_qa = row.get("QA")
        if not isinstance(turns, list) or not isinstance(raw_qa, list):
            raise ValueError(f"DialogueID={d_id}: expected Dialogue and QA arrays")
        dialogue, by_turn, _ = numbered_lines(turns, d_id)
        target = []
        for qa in raw_qa:
            counts["qa_total"] += 1
            if not isinstance(qa, dict):
                counts["qa_malformed"] += 1
                continue
            question = qa.get("QueSumm")
            answer = qa.get("AnsSummLong")
            if not isinstance(question, str) or not question.strip():
                counts["qa_missing_question"] += 1
                continue
            if not isinstance(answer, str) or not answer.strip():
                counts["qa_missing_answer_skipped"] += 1
                continue
            u_ids = map_utterance_ids(
                qa.get("QueSummUttIDs"), by_turn, "U", counts, "question_ids")
            c_ids = map_utterance_ids(
                qa.get("AnsSummLongUttIDs"), by_turn, "C", counts, "answer_ids")
            if not c_ids:
                counts["qa_no_agent_evidence_skipped"] += 1
                continue
            intent = qa.get("intent")
            if not isinstance(intent, str):
                intent = ""
            intent = intent.strip()
            if intent:
                intents[intent] += 1
            else:
                counts["qa_missing_intent_retained_blank"] += 1
            if not u_ids:
                counts["qa_no_user_evidence_retained_empty"] += 1
            target.append({
                "category": intent,
                "question": question.strip(),
                "answer": answer.strip(),
                "question_ids": u_ids,
                "evidence_ids": c_ids,
            })
        if not target:
            counts["dialogues_no_usable_qa_skipped"] += 1
            continue
        # Use fully supervised target JSON. The original QA labels are never
        # included in the model input prompt.
        examples.append({
            "DialogueID": d_id,
            "prompt": PROMPT_HEADER + dialogue,
            "target": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            "qa_count": len(target),
        })
        counts["dialogues_retained"] += 1
        counts["qa_retained"] += len(target)
    return examples


def write_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in examples:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/CSDS/train.json")
    ap.add_argument("--train-output", default="data/qa_sft_train.jsonl")
    ap.add_argument("--dev-output", default="data/qa_sft_dev.jsonl")
    ap.add_argument("--report", default="outputs/prepare_qa_sft_report.json")
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--dev-ratio", type=float, default=0.05)
    ap.add_argument("--preview", type=int, default=1)
    args = ap.parse_args()
    if not 0 < args.dev_ratio < 0.5:
        ap.error("--dev-ratio must be between 0 and 0.5")
    source_path = Path(args.input)
    if not source_path.is_file():
        raise SystemExit(f"CSDS input not found: {source_path.resolve()}")
    raw_bytes = source_path.read_bytes()
    source = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(source, list):
        raise ValueError("Expected CSDS train.json to be a JSON list")
    counts, intents = collections.Counter(), collections.Counter()
    examples = build_examples(source, counts, intents)
    if len(examples) < 2:
        raise RuntimeError("Not enough usable dialogues for train/dev split")
    indices = list(range(len(examples)))
    random.Random(args.seed).shuffle(indices)
    dev_count = max(1, int(round(len(examples) * args.dev_ratio)))
    dev_indices = set(indices[:dev_count])
    train = [x for i, x in enumerate(examples) if i not in dev_indices]
    dev = [x for i, x in enumerate(examples) if i in dev_indices]
    assert set(x["DialogueID"] for x in train).isdisjoint(x["DialogueID"] for x in dev)
    assert sum(x["qa_count"] for x in examples) == counts["qa_retained"]
    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.dev_output), dev)
    report = {
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "conversion_version": "qa_sft_annotated_only_v1",
        "category_source": "QA.intent (NOT QA.Topic)",
        "train_dialogues": len(train),
        "dev_dialogues": len(dev),
        "train_qa_pairs": sum(x["qa_count"] for x in train),
        "dev_qa_pairs": sum(x["qa_count"] for x in dev),
        "counters": dict(counts),
        "top_intents": intents.most_common(20),
        "supervised_fields": ["category", "question", "answer", "question_ids", "evidence_ids"],
        "not_supervised": ["object_identity", "processing_status"],
        "split": "dialogue-level train-only 95/5 by seeded shuffle; CSDS val/test untouched",
        "prompt_version": "qa_sft_annotated_only_v1",
        "important_caveats": [
            "Some QA AnsSummLongUttIDs include user turns; those IDs were removed from customer-service evidence.",
            "Missing answers or agent evidence are excluded, not silently assigned invented labels.",
            "CSDS QA annotations are imperfect and may omit meaningful agent-initiated offers.",
            "Before training, inspect sequence lengths with the exact target tokenizer; never silently truncate targets.",
        ],
    }
    report_file = Path(args.report)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CSDS: {len(source)} source dialogues; {counts['qa_total']} annotated QA")
    print(f"QA-SFT: {len(train)} train dialogues + {len(dev)} dev dialogues")
    print(f"Retained QA pairs: {counts['qa_retained']} ; dropped no answer: {counts['qa_missing_answer_skipped']}; dropped no valid agent evidence: {counts['qa_no_agent_evidence_skipped']}")
    print(f"Filtered user IDs from agent evidence: {counts['answer_ids_wrong_speaker_removed']}")
    print(f"Category source is CSDS QA.intent; nonempty intent count: {sum(intents.values())}")
    print(f"Outputs: {args.train_output}, {args.dev_output}, {args.report}")
    for row in train[:max(0, args.preview)]:
        print(f"\nPREVIEW DialogueID={row['DialogueID']} | {row['qa_count']} QA")
        print("Prompt preview:\n", row["prompt"][:450])
        print("Target:\n", row["target"][:1100])


if __name__ == "__main__":
    main()
