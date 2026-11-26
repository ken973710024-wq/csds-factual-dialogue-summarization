#!/usr/bin/env python3
"""Inspect existing human QA labels in CSDS train.json before building QA-SFT data.

No model inference, no training, no mutation of the original dataset.
Usage: python src/check_csds_qa_annotations.py --input data/CSDS/train.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from statistics import mean

QA_FIELDS = (
    "QueSumm", "AnsSummShort", "AnsSummLong", "QASumm", "Topic",
    "QueSummUttIDs", "AnsSummShortUttIDs", "AnsSummLongUttIDs",
)

def populated(value):
    if value is None or value == "":
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    return True

def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError(f"Expected a JSON list, got {type(samples).__name__}")
    return samples

def text_of_turn(turn):
    value = turn.get("utterance", "")
    if isinstance(value, list):
        return " ".join(map(str, value))
    return str(value)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/CSDS/train.json")
    parser.add_argument("--output", default="outputs/check_csds_qa_annotations_report.json")
    parser.add_argument("--examples", type=int, default=2)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.is_file():
        raise SystemExit(f"Dataset not found: {path.resolve()}")
    samples = read_json(path)
    counts = collections.Counter()
    topics = collections.Counter()
    qa_per_dialogue = []
    label_validity = collections.Counter()
    preview = []
    missing = []

    for sample in samples:
        dialogue_id = sample.get("DialogueID")
        dialogue = sample.get("Dialogue") or []
        qa_list = sample.get("QA") or []
        if not isinstance(qa_list, list):
            raise ValueError(f"DialogueID={dialogue_id}: QA must be an array")
        qa_per_dialogue.append(len(qa_list))
        by_turn = {str(turn.get("turn")): turn for turn in dialogue if isinstance(turn, dict)}
        by_index = {str(i): turn for i, turn in enumerate(dialogue) if isinstance(turn, dict)}
        for qa in qa_list:
            counts["total_qa"] += 1
            for field in QA_FIELDS:
                counts[f"has_{field}"] += int(populated(qa.get(field)))
            topic = qa.get("Topic")
            if isinstance(topic, (dict, list)):
                topic = json.dumps(topic, ensure_ascii=False, sort_keys=True)
            if populated(topic):
                topics[str(topic)] += 1
            for field in ("QueSummUttIDs", "AnsSummLongUttIDs"):
                ids = qa.get(field) or []
                if not isinstance(ids, list):
                    label_validity[f"{field}_not_list"] += 1
                    continue
                label_validity[f"{field}_ids"] += len(ids)
                label_validity[f"{field}_match_turn"] += sum(str(i) in by_turn for i in ids)
                label_validity[f"{field}_match_index"] += sum(str(i) in by_index for i in ids)
            if not populated(qa.get("QueSumm")) or not populated(qa.get("AnsSummLong")):
                if len(missing) < 10:
                    missing.append({"DialogueID": dialogue_id, "QA": qa})

        if len(preview) < args.examples and qa_list:
            first = qa_list[0]
            evidence_ids = first.get("AnsSummLongUttIDs") or []
            evidence = []
            for id_value in evidence_ids[:8]:
                record = by_turn.get(str(id_value))
                evidence.append({
                    "utterance_id": id_value,
                    "found_by_turn_field": record is not None,
                    "speaker": record.get("speaker") if record else None,
                    "text": text_of_turn(record) if record else None,
                })
            preview.append({
                "DialogueID": dialogue_id,
                "dialogue_turns": len(dialogue),
                "qa_count": len(qa_list),
                "first_qa_as_stored": first,
                "first_qa_agent_evidence_preview": evidence,
                "example_stage1_target_using_only_available_labels": {
                    "category": first.get("Topic"),
                    "question": first.get("QueSumm"),
                    "answer": first.get("AnsSummLong"),
                    "evidence_turn_ids": first.get("AnsSummLongUttIDs"),
                },
            })

    total_qa = counts["total_qa"]
    report = {
        "source": str(path),
        "total_dialogues": len(samples),
        "total_qa_pairs": total_qa,
        "avg_qa_per_dialogue": round(mean(qa_per_dialogue), 3) if qa_per_dialogue else 0,
        "num_dialogues_with_no_QA": sum(x == 0 for x in qa_per_dialogue),
        "nonempty_qa_field_counts": {key: counts[f"has_{key}"] for key in QA_FIELDS},
        "nonempty_qa_field_rates": {
            key: round(counts[f"has_{key}"] / total_qa, 4) if total_qa else 0
            for key in QA_FIELDS
        },
        "utterance_id_resolution_checks": dict(label_validity),
        "top_topics": topics.most_common(15),
        "example_records": preview,
        "examples_missing_core_training_labels": missing,
        "important_note": (
            "CSDS provides QA summaries, topic and key-utterance IDs. "
            "Object identity and processing status are not declared standalone fields "
            "in the official annotation schema; do not invent their gold labels."
        ),
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"CSDS train: {len(samples)} dialogues, {total_qa} QA pairs")
    print(f"Mean QA/dialogue: {report['avg_qa_per_dialogue']}")
    for field in QA_FIELDS:
        print(f"  {field:24s}: {report['nonempty_qa_field_rates'][field]:.1%} non-empty")
    print("Evidence ID check:", dict(label_validity))
    for item in preview:
        target = item["example_stage1_target_using_only_available_labels"]
        print(f"\nDialogueID {item['DialogueID']}  ({item['qa_count']} QA pairs)")
        print(json.dumps(target, ensure_ascii=False, indent=2))
    print(f"\nReport saved: {dest}")

if __name__ == "__main__":
    main()
