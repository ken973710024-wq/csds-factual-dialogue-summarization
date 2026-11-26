#!/usr/bin/env python3
"""Preflight CSDS QA-SFT JSONL with the exact Qwen tokenizer/chat template.

Run from project root:
  python src/check_qa_sft_token_lengths.py

Only computes statistics; no CUDA, training or file modifications besides report.
"""
from __future__ import annotations
import argparse
import collections
import json
import math
from pathlib import Path
from transformers import AutoTokenizer


def qtile(values, percentage):
    if not values:
        return None
    sorted_values = sorted(values)
    return sorted_values[min(len(values)-1, math.ceil(len(values)*percentage/100)-1)]


def tokenize(tokenizer, row):
    prompt_messages = [{"role":"user","content":row["prompt"]}]
    full_messages = prompt_messages + [{"role":"assistant","content":row["target"]}]
    prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError(f"DialogueID={row['DialogueID']}: chat template prefix mismatch")
    return len(prompt), len(full)-len(prompt), len(full)


def inspect(path, tokenizer, thresholds):
    items, lengths, prompt_lengths, target_lengths = [], [], [], []
    qa_counts = collections.Counter()
    d_ids = set()
    with path.open(encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            row = json.loads(line)
            d_id = str(row['DialogueID'])
            if d_id in d_ids:
                raise ValueError(f'{path}:{line_num} duplicate DialogueID {d_id}')
            d_ids.add(d_id)
            qa = json.loads(row['target'])
            if not isinstance(qa, list) or not qa:
                raise ValueError(f'{path}:{line_num} target must be a nonempty QA list')
            if len(qa) != row['qa_count']:
                raise ValueError(f'{path}:{line_num} QA count mismatch')
            p, t, total = tokenize(tokenizer, row)
            if t == 0:
                raise ValueError(f'{path}:{line_num} empty assistant supervision')
            prompt_lengths.append(p)
            target_lengths.append(t)
            lengths.append(total)
            qa_counts[len(qa)] += 1
            items.append({"DialogueID":row['DialogueID'],"prompt_tokens":p,"target_tokens":t,"total_tokens":total,"qa_count":len(qa)})
    return {
        'file':str(path),'dialogues':len(items),
        'qa_pairs':sum(k*v for k,v in qa_counts.items()),
        'length_distribution':{
            'min':min(lengths) if lengths else None,
            'p50':qtile(lengths,50),'p90':qtile(lengths,90),'p95':qtile(lengths,95),
            'p99':qtile(lengths,99),'max':max(lengths) if lengths else None,
            'max_prompt':max(prompt_lengths) if prompt_lengths else None,
            'max_target':max(target_lengths) if target_lengths else None,
        },
        'over_limit':{str(n):{'dialogues':sum(x>n for x in lengths),'pct':round(100*sum(x>n for x in lengths)/len(lengths),2) if lengths else 0} for n in thresholds},
        'qa_per_dialogue':dict(sorted(qa_counts.items())),
        'top_longest':sorted(items,key=lambda x:x['total_tokens'],reverse=True)[:15],
        '_ids':d_ids
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',default='models/Qwen3-4B-Instruct-2507')
    ap.add_argument('--train',default='data/qa_sft_train.jsonl')
    ap.add_argument('--dev',default='data/qa_sft_dev.jsonl')
    ap.add_argument('--report',default='outputs/check_qa_sft_token_lengths_report.json')
    args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    train=inspect(Path(args.train),tok,[1536,2048,3072,4096])
    dev=inspect(Path(args.dev),tok,[1536,2048,3072,4096])
    if train['_ids'] & dev['_ids']:
        raise ValueError('Train and development DialogueIDs overlap!')
    train.pop('_ids'); dev.pop('_ids')
    report={"model":args.model,"chat_template":"apply_chat_template user(+ assistant), assistant-only loss", "train":train,"dev":dev,
            "note":"All counts include full assistant target and chat template. Never silently truncate QA labels."}
    output=Path(args.report); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for label,data in [('train',train),('dev',dev)]:
        print(label,'examples=',data['dialogues'],'pairs=',data['qa_pairs'], 'length:', data['length_distribution'])
        print('long cases:',data['over_limit'])
    print('Report:',output)

if __name__=='__main__':
    main()
