"""Shared, intentionally small helpers for the CSDS QA-aware Stage-2 study.

Run from the existing /root/workspace/csds-project. The original project's
PROMPT_HEADER, SUMMARY_RULES, QA validator and stage2_prompt are imported and
never recreated, to prevent subtle train/inference prompt drift.
"""
from __future__ import annotations
import hashlib
import json
import random
from pathlib import Path

from prepare_qa_sft_data import PROMPT_HEADER, numbered_lines
from validate_qa_lora import canonical_items, numbered_speakers
from validate_qa_aware_pipeline import stage2_prompt, dialogue_to_text
from validate_qa_pipeline_dev20 import SUMMARY_RULES, score

BASE_MODEL = Path('models/Qwen3-4B-Instruct-2507')
QA_ADAPTER = Path('outputs/qlora_qa_stage1/final_adapter')
CSDS_TRAIN = Path('data/CSDS/train.json')
SFT_TRAIN = Path('data/train_sft.jsonl')
CACHE = Path('data/qa_stage2_research/generated_qa_train.jsonl')
SEED = 20261122
DECODE_MAX = 768
SUMMARY_MAX = 384

def sha_bytes(value: bytes):
    return hashlib.sha256(value).hexdigest()

def sha_file(path):
    return sha_bytes(Path(path).read_bytes())

def sha_text(value):
    return sha_bytes(value.encode('utf-8'))

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)

def jsonl(path):
    with Path(path).open(encoding='utf-8') as file:
        for n, line in enumerate(file, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'{path}:{n}: malformed JSON: {e}') from e

def load_train():
    """Join by DialogueID: train gold summaries belong ONLY in target."""
    original = json.loads(CSDS_TRAIN.read_text(encoding='utf-8'))
    if not isinstance(original, list):
        raise ValueError(f'{CSDS_TRAIN} is not an array')
    by_id = {}
    for row in original:
        key = str(row['DialogueID'])
        if key in by_id:
            raise ValueError(f'Duplicate original train ID {key}')
        by_id[key] = row
    output, seen = [], set()
    for row in jsonl(SFT_TRAIN):
        key = str(row['DialogueID'])
        if key not in by_id or key in seen:
            raise ValueError(f'Unmatched / duplicated SFT train ID {key}')
        target = row.get('target')
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f'No train target for {key}')
        seen.add(key)
        source = by_id[key]
        if not isinstance(source.get('Dialogue'), list):
            raise ValueError(f'Invalid original Dialogue for {key}')
        output.append({'DialogueID': row['DialogueID'], 'source': source, 'target': target})
    if len(output) != len(original):
        raise ValueError(f'Train ID mismatch: original={len(original)} summary_SFT={len(output)}')
    return output

def select_rows(rows, sample_size):
    if sample_size <= 0 or sample_size > len(rows):
        raise ValueError(f'sample-size must be between 1 and {len(rows)}')
    indices = sorted(random.Random(SEED).sample(range(len(rows)), sample_size))
    return [rows[i] for i in indices]

def label_for_size(sample_size, n_all):
    return 'full' if sample_size == n_all else f'pilot{sample_size}'

def qa_input(source):
    numbered, _, _ = numbered_lines(source['Dialogue'], source['DialogueID'])
    return PROMPT_HEADER + numbered

def stage2_input(source, qa=None):
    dialogue = dialogue_to_text(source)
    if qa is None:
        return stage2_prompt(dialogue)
    numbered = qa_input(source)
    _, _, evidence_lines = numbered_speakers(numbered)
    return stage2_prompt(dialogue, qa, evidence_lines)

def cache_rows(path=CACHE):
    if not Path(path).exists():
        return {}
    done = {}
    for row in jsonl(path):
        key = str(row['DialogueID'])
        if key in done:
            raise ValueError(f'Duplicate QA cache key {key}')
        done[key] = row
    return done

def provenance():
    return {
        'train_sha256': sha_file(CSDS_TRAIN),
        'sft_sha256': sha_file(SFT_TRAIN),
        'qa_prompt_sha256': sha_text(PROMPT_HEADER),
        'qa_adapter_config_sha256': sha_file(QA_ADAPTER/'adapter_config.json'),
        'qa_adapter_weights_sha256': sha_file(QA_ADAPTER/'adapter_model.safetensors')
            if (QA_ADAPTER/'adapter_model.safetensors').is_file() else None,
        'stage2_prompt_sha256': sha_text(SUMMARY_RULES),
    }

def validate_cache_record(row, source, prov):
    key = str(source['DialogueID'])
    if str(row.get('DialogueID')) != key:
        raise ValueError(f'Wrong cache ID for {key}')
    if row.get('qa_prompt_sha256') != sha_text(qa_input(source)):
        raise ValueError(f'QA prompt mismatch in cached ID {key}')
    if row.get('provenance') != prov:
        raise ValueError(f'QA cache was generated using different model or train data: {key}')
    qa = row.get('qa')
    if qa is not None:
        if not isinstance(qa, list) or not qa:
            raise ValueError(f'Empty or invalid QA cache for {key}')
        _, agent_ids, _ = numbered_speakers(qa_input(source))
        for item in qa:
            if set(item) != {'category','question','answer','question_ids','evidence_ids'}:
                raise ValueError(f'Bad QA schema for {key}')
            if not set(item['evidence_ids']).issubset(agent_ids):
                raise ValueError(f'Wrong evidence IDs for {key}')
    return True
