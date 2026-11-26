#!/usr/bin/env python3
"""End-to-end CSDS QA-aware summarization: controlled A/B/C comparison.

Run from /root/workspace/csds-project:
    python src/validate_qa_aware_pipeline.py --check
    python src/validate_qa_aware_pipeline.py --limit 5
    python src/validate_qa_aware_pipeline.py --limit 20
    python src/validate_qa_aware_pipeline.py --limit 100

A: Qwen + original Summary-LoRA, original dialogue, common Stage-2 prompt.
B: SAME Stage-2 model/prompt + Base Qwen-extracted QA (same trained 5-field Stage-1 prompt).
C: SAME Stage-2 model/prompt + QA-LoRA-extracted QA (same Stage-1 prompt).

Also reports Experiment 2's *previously saved* standard-SFT prediction as
'historical_sft'; it is NOT equivalent to A's controlled prompt.

The first 20 are the already-used prompt-development dialogues. The final 80
were not used in the previous dev20 prompt exploration. Never tune prompts
on the 80 after running them. A future pristine CSDS test run is still needed.

Needs these ALREADY INSTALLED source files in src/:
  prepare_qa_sft_data.py (exact QA training prompt and utterance numbering)
  validate_qa_lora.py (exact trained 5-field strict QA parser)
  validate_qa_pipeline_dev20.py (frozen Stage-2 prompt and ROUGE-L scoring)

Requires existing Experiment-2 file outputs/validate_base_vs_lora_100.json,
CSDS official data/CSDS/val.json, both trained adapters, and base Qwen.
Never passes reference summaries or annotated QA to either model.
Does not change weights or perform any training. Atomic checkpoints after
EACH model generation permit safe restart (e.g. after cloud disconnect).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from prepare_qa_sft_data import PROMPT_HEADER, numbered_lines
from validate_qa_lora import canonical_items, numbered_speakers
from validate_qa_pipeline_dev20 import SUMMARY_RULES, score

MODEL_DIR = Path('models/Qwen3-4B-Instruct-2507')
QA_ADAPTER = Path('outputs/qlora_qa_stage1/final_adapter')
SUMMARY_ADAPTER = Path('outputs/qlora_sft/final_adapter')
BASELINE_FILE = Path('outputs/validate_base_vs_lora_100.json')
VAL_FILE = Path('data/CSDS/val.json')
OUTPUT = Path('outputs/validate_qa_aware_pipeline_100.json')
SUMMARY_OUTPUT = Path('outputs/validate_qa_aware_pipeline_100_summary.json')
REVIEW_OUTPUT = Path('outputs/validate_qa_aware_pipeline_100_review.md')

DEV20_SEED = 20261122
QA_MAX_NEW_TOKENS = 768  # same as QA-LoRA Stage-1 validation
SUMMARY_MAX_NEW_TOKENS = 384  # same as previous Stage-2 experiments
MAX_TOTAL_TOKENS = 8192  # guardrail, never truncate silently


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    return digest(path.read_bytes())


def text_sha(value: str) -> str:
    return digest(value.encode('utf-8'))


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def dialogue_to_text(sample: dict) -> str:
    """Exact reconstruction from Experiment 2 (only used for cross-check)."""
    lines = []
    for turn in sample['Dialogue']:
        speaker = '客服' if turn['speaker'] == 'A' else sample['QRole']
        lines.append(f"{speaker}：{turn['utterance'].replace(' ', '')}")
    return '\n'.join(lines)


def read_cohort() -> tuple[list[dict], list[int], dict]:
    for path in (BASELINE_FILE, VAL_FILE):
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}; this requires the SAME official validation cohort')
    source = json.loads(BASELINE_FILE.read_text(encoding='utf-8'))
    baseline_rows = source.get('results')
    metadata = source.get('metadata', {})
    if metadata.get('dataset') != 'CSDS val.json' or metadata.get('count') != 100:
        raise ValueError('Baseline source is not the expected 100 official CSDS val dialogues')
    if not isinstance(baseline_rows, list) or len(baseline_rows) != 100:
        raise ValueError('Baseline JSON must contain exactly 100 results')
    indices = [r['index'] for r in baseline_rows]
    if len(set(indices)) != 100 or sorted(indices) != metadata.get('indices'):
        raise ValueError('Baseline 100-case indices differ from original experiment metadata')
    original_val = json.loads(VAL_FILE.read_text(encoding='utf-8'))
    by_idx = {}
    for row in baseline_rows:
        idx = row['index']
        if not isinstance(idx, int) or idx < 0 or idx >= len(original_val):
            raise ValueError(f'Invalid validation index {idx!r}')
        orig = original_val[idx]
        if str(orig['DialogueID']) != str(row['DialogueID']):
            raise ValueError(f'Validation DialogueID mismatch for index {idx}')
        if row['dialogue'] != dialogue_to_text(orig):
            raise ValueError(f'Dialogue text differs from original CSDS validation at index {idx}')
        if row['reference'] != orig['AgentSumm']:
            raise ValueError(f'Gold summary mismatch at index {idx}')
        if not isinstance(row.get('lora_prediction'), str) or not isinstance(row.get('lora'), dict):
            raise ValueError(f'Historical SFT prediction missing at index {idx}')
        numbered, _, _ = numbered_lines(orig['Dialogue'], orig['DialogueID'])
        prompt = PROMPT_HEADER + numbered
        users, agents, extracts = numbered_speakers(prompt)
        if not agents:
            raise ValueError(f'No agent lines for index {idx}')
        by_idx[idx] = {
            'index': idx,
            'DialogueID': row['DialogueID'],
            'dialogue': row['dialogue'],
            'reference': '\n'.join(row['reference']),
            'historical_sft_prediction': row['lora_prediction'],
            'historical_sft_scores': row['lora'],
            'qa_prompt': prompt,
            'speaker_lines': extracts,
            'count_user_turns': len(users),
            'count_agent_turns': len(agents),
        }
    dev20 = sorted(random.Random(DEV20_SEED).sample(sorted(by_idx), 20))
    other80 = sorted(set(by_idx) - set(dev20))
    # Run development cases first to support safe progression --limit 5 -> 20 -> 100.
    ordered = [by_idx[i] for i in dev20 + other80]
    assert len(ordered) == 100 and len(other80) == 80
    return ordered, dev20, metadata


def experiment_metadata(dev20: list[int], baseline_meta: dict) -> dict:
    return {
        'version': 'qa_aware_pipeline_abc_v1',
        'official_validation_indices': baseline_meta['indices'],
        'first_20_indices_previously_used_for_prompt_development': dev20,
        'last_80_indices_untouched_during_dev20_prompt_exploration': True,
        'cohort_file': str(BASELINE_FILE),
        'cohort_sha256': file_sha(BASELINE_FILE),
        'official_val_file': str(VAL_FILE),
        'official_val_sha256': file_sha(VAL_FILE),
        'base_model': str(MODEL_DIR),
        'qa_adapter': str(QA_ADAPTER),
        'summary_adapter': str(SUMMARY_ADAPTER),
        'stage1_prompt_sha256': text_sha(PROMPT_HEADER),
        'stage2_prompt_sha256': text_sha(SUMMARY_RULES),
        'stage1_json_fields': ['category', 'question', 'answer', 'question_ids', 'evidence_ids'],
        'stage2_qa_context': 'same 5 QA fields plus verbatim agent evidence_quotes for both B and C',
        'stage2_summary_prompt': 'FROZEN SUMMARY_RULES from validate_qa_pipeline_dev20.py',
        'stage1_base_and_lora_prompt': 'EXACT PROMPT_HEADER from prepare_qa_sft_data.py',
        'stage1_max_new_tokens': QA_MAX_NEW_TOKENS,
        'stage2_max_new_tokens': SUMMARY_MAX_NEW_TOKENS,
        'max_total_tokens_guard': MAX_TOTAL_TOKENS,
        'generation': 'greedy, do_sample=False, 4-bit NF4/bf16 double-quant, single base model',
        'gold_reference_in_prompt': False,
        'gold_qa_in_prompt': False,
        'fallback': 'when invalid QA, use the SAME direct summary (count as failure, not drop case)',
        'historical_sft': 'existing saved Experiment 2 output (different original prompt; context only)',
        'known_caveats': [
            'Development 20 of this 100 previously used for prompt exploration; 80 are the restricted internal holdout.',
            'Official val set was used in earlier experiments; final claims need an untouched test set.',
            'ROUGE-L overlap is not semantic faithfulness, status or object accuracy.',
            'Stage 2 was trained directly on dialogue, not on QA-conditioned inputs.',
            'QA-LoRA was not trained with a separate status/object field.',
        ],
    }


def score_generated(pred: str, ref: str, generated_tokens: int, limit: int) -> dict:
    x = score(pred, ref)
    x['generated_tokens'] = generated_tokens
    x['hit_max_new_tokens'] = generated_tokens >= limit
    return x


def qa_as_context(items: list[dict], speaker_lines: dict[str, str]) -> str:
    """Add *verbatim* evidence quotes; no invented status or inferred claim."""
    enriched = []
    for item in items:
        q = dict(item)
        q['evidence_quotes'] = [speaker_lines[i] for i in item['evidence_ids']]
        enriched.append(q)
    return json.dumps(enriched, ensure_ascii=False, indent=2)


def stage2_prompt(dialogue: str, qa: list[dict] | None = None, speaker_lines: dict | None = None) -> str:
    content = SUMMARY_RULES + dialogue
    if qa is not None:
        if speaker_lines is None:
            raise ValueError('Evidence source lines required for QA-guided stage 2')
        content += '\n【QA 核对清单】\n' + qa_as_context(qa, speaker_lines)
    return content


def qa_result(raw: str, user_prompt: str, tokens: int, cap: int) -> dict:
    parsed, err, fence, json_valid = canonical_items(raw, user_prompt)
    return {
        'raw': raw,
        'parsed': parsed,
        'json_valid': json_valid,
        'schema_and_ids_valid': parsed is not None,
        'error': err,
        'code_fence_stripped': fence,
        'predicted_qa_count': len(parsed) if parsed is not None else None,
        'generated_tokens': tokens,
        'hit_max_new_tokens': tokens >= cap,
    }


def mean(rows: list[dict], key: str) -> dict | None:
    if not rows:
        return None
    fields = ['rouge_l_f1', 'rouge_l_precision', 'rouge_l_recall', 'chars']
    return {
        **{k: round(sum(x[key]['scores'][k] for x in rows) / len(rows), 4) for k in fields},
        'hit_max_new_tokens': sum(x[key]['scores'].get('hit_max_new_tokens', False) for x in rows),
    }


def paired_compare(rows: list[dict], left: str, right: str) -> dict | None:
    """right minus left; bootstrap interval of paired mean F1 delta."""
    if not rows:
        return None
    diffs = [r[right]['scores']['rouge_l_f1'] - r[left]['scores']['rouge_l_f1'] for r in rows]
    n = len(diffs)
    rand = random.Random(20261122)
    boots = sorted(sum(diffs[rand.randrange(n)] for _ in range(n)) / n for _ in range(2000))
    return {
        'f1_delta_right_minus_left': round(sum(diffs) / n, 4),
        'f1_paired_bootstrap_ci95': [round(boots[49], 4), round(boots[1949], 4)],
        'right_f1_wins': sum(v > 1e-10 for v in diffs),
        'left_f1_wins': sum(v < -1e-10 for v in diffs),
        'ties': sum(abs(v) <= 1e-10 for v in diffs),
    }


def aggregate(ordered_rows: list[dict], completed_by_id: dict[str, dict]) -> dict:
    def group(name: str, wanted: list[dict]) -> dict:
        completed = [completed_by_id[str(r['DialogueID'])] for r in wanted
                     if str(r['DialogueID']) in completed_by_id
                     and all(k in completed_by_id[str(r['DialogueID'])]
                             for k in ('direct', 'base_qa_guided', 'lora_qa_guided'))]
        valid_stage1 = {
            k: sum(bool(r.get(k, {}).get('schema_and_ids_valid')) for r in completed)
            for k in ('base_qa', 'lora_qa')
        }
        return {
            'name': name,
            'requested_cohort_size': len(wanted),
            'evaluated': len(completed),
            'valid_stage1_count': valid_stage1,
            'fallback_counts': {
                'base_qa_guided': sum(x['base_qa_guided'].get('fell_back_to_direct', False) for x in completed),
                'lora_qa_guided': sum(x['lora_qa_guided'].get('fell_back_to_direct', False) for x in completed),
            },
            'historical_sft': mean(completed, 'historical_sft'),
            'A_direct_common_prompt': mean(completed, 'direct'),
            'B_base_qa_guided': mean(completed, 'base_qa_guided'),
            'C_lora_qa_guided': mean(completed, 'lora_qa_guided'),
            'C_minus_A': paired_compare(completed, 'direct', 'lora_qa_guided'),
            'C_minus_B': paired_compare(completed, 'base_qa_guided', 'lora_qa_guided'),
            'B_minus_A': paired_compare(completed, 'direct', 'base_qa_guided'),
        }

    return {
        'warning': 'ROUGE-L is a character-overlap proxy, NOT proof of factual correctness; inspect review and holdout separately.',
        'all_100': group('all_100_includes_20_prompt_dev', ordered_rows),
        'dev_20': group('previously_used_prompt_development', ordered_rows[:20]),
        'holdout_80': group('held_out_from_20_case_prompt_development', ordered_rows[20:]),
    }


def save_outputs(metadata: dict, ordered: list[dict], done: dict[str, dict]) -> None:
    results = [done[str(x['DialogueID'])] for x in ordered if str(x['DialogueID']) in done]
    atomic_json(OUTPUT, {'metadata': metadata, 'results': results})
    # Summary counts only *fully completed* cases, not partially checkpointed ones.
    atomic_json(SUMMARY_OUTPUT, aggregate(ordered, done))
    lines = ['# QA-Aware Pipeline End-to-End Validation (100 CSDS official val cases)', '',
             'A: no QA / Summary-LoRA; B: Base Qwen QA / Summary-LoRA; C: QA-LoRA QA / Summary-LoRA.',
             'All three use the EXACT SAME Stage-2 base model, adapter, summarisation instructions, and greedy decoding.',
             'QA input to B and C uses the same EXACT five-field Stage-1 training prompt; only Stage-1 adapter differs.',
             'Historical SFT result uses a different original summary prompt: descriptive comparison ONLY.',
             '**Gold summaries below are for review/scoring ONLY. Not shown to any model.**', '',
             '**Automated ROUGE-L cannot establish factuality; verify timing, conditions, action status, and entity alignment.**', '',
             ]
    for r in results:
        lines += [f"## DialogueID {r['DialogueID']} — official val index {r['index']} ({r['split']})", '',
                  '### Original dialogue', '', '```text', r['dialogue'], '```', '',
                  '### Gold summary — offline ONLY', '', '```text', r['reference'], '```', '',
                  '### Historical Experiment 2 SFT (original training-time prompt)', '',
                  '```text', r['historical_sft']['prediction'], '```', '']
        if 'direct' in r:
            lines += ['### A — Common-prompt direct Summary-LoRA', '',
                      '```text', r['direct']['prediction'], '```', '']
        for k, title in [('base_qa', 'B Stage 1: Base Qwen QA'), ('lora_qa', 'C Stage 1: QA-LoRA QA')]:
            if k in r:
                case = r[k]
                lines += [f'### {title}', '',
                          f"Valid: {case['schema_and_ids_valid']}; error: {case['error']}; QA count: {case['predicted_qa_count']}", '',
                          '```json', json.dumps(case['parsed'], ensure_ascii=False, indent=2)
                          if case['parsed'] is not None else case['raw'], '```', '']
        for k, title in [('base_qa_guided', 'B — Base Qwen QA + Summary-LoRA'),
                         ('lora_qa_guided', 'C — QA-LoRA QA + Summary-LoRA')]:
            if k in r:
                lines += [f'### {title}', '',
                          ('**Fallback: Stage 1 invalid, reused direct output.**' if r[k].get('fell_back_to_direct') else ''),
                          '```text', r[k]['prediction'], '```', '']
        if all(k in r for k in ('direct', 'base_qa_guided', 'lora_qa_guided')):
            lines += [f"ROUGE-L F1: historical={r['historical_sft']['scores']['rouge_l_f1']:.3f}; "
                      f"A={r['direct']['scores']['rouge_l_f1']:.3f}; "
                      f"B={r['base_qa_guided']['scores']['rouge_l_f1']:.3f}; "
                      f"C={r['lora_qa_guided']['scores']['rouge_l_f1']:.3f}", '',
                      'Manual inspection: missing QA? duplicate QA? incorrect product/order? '
                      'incorrect status (completed vs promised)? missing times/conditions? copy artifacts?', '',
                      '---', '']
    REVIEW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_OUTPUT.write_text('\n'.join(lines), encoding='utf-8')


def generate(model, tokenizer, user_prompt: str, max_new_tokens: int, label: str) -> tuple[str, int]:
    import torch
    messages = [{'role': 'user', 'content': user_prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors='pt').to('cuda:0')
    total = inputs['input_ids'].shape[1] + max_new_tokens
    if total > MAX_TOTAL_TOKENS:
        raise ValueError(f'{label}: prompt+max_new_tokens={total} exceeds safety guard {MAX_TOTAL_TOKENS}; '
                         'No silent truncation. Investigate length before changing context guard.')
    with torch.inference_mode():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id)
    generated = out[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated)


def main():
    parser = argparse.ArgumentParser(description='Controlled end-to-end A/B/C QA-aware summary validation')
    parser.add_argument('--check', action='store_true', help='Verify all source paths and dataset alignment, without GPU')
    parser.add_argument('--show-prompts', action='store_true', help='Show actual A, B/C Stage-1 and Stage-2 templates; no GPU')
    parser.add_argument('--evaluate-only', action='store_true', help='Rescore/compile existing checkpoint without GPU')
    parser.add_argument('--limit', type=int, default=20, help='Fixed prefix: first 20 dev, then 80 holdout; 1..100')
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        parser.error('--limit must be from 1 to 100')

    ordered, dev20, baseline_meta = read_cohort()
    metadata = experiment_metadata(dev20, baseline_meta)
    ids = [str(r['DialogueID']) for r in ordered]
    if len(set(ids)) != 100:
        raise ValueError('Duplicate DialogueIDs in 100-case cohort')

    if args.check or args.show_prompts:
        print('DATA VERIFIED: 100 CSDS official validation dialogues match Experiment 2 exactly.')
        print(f'20 prompt-dev cases: {dev20}')
        print(f'80 held out from prior dev20: {[x["index"] for x in ordered[20:]]}')
        missing_files = []
        for path in (MODEL_DIR/'config.json', QA_ADAPTER/'adapter_config.json',
                     SUMMARY_ADAPTER/'adapter_config.json'):
            print(f'{"OK" if path.is_file() else "MISSING"}: {path}')
            if not path.is_file():
                missing_files.append(str(path))
        if missing_files:
            raise FileNotFoundError('Required model/adapter files are missing: ' + ', '.join(missing_files))
        if args.show_prompts:
            row = ordered[0]
            print('\n--- SHARED STAGE 1 (Base Qwen or QA-LoRA, same exact training prompt) ---')
            print(row['qa_prompt'])
            print('\n--- A STAGE 2 (Summary-LoRA direct) ---')
            print(stage2_prompt(row['dialogue']))
            print('\n--- B & C STAGE 2 template (same prompt; QA comes from their respective Stage 1) ---')
            print(stage2_prompt(row['dialogue']) + '\n【QA 核对清单】\n<5-field QA plus verbatim evidence quotes>')
        return

    done = {}
    if OUTPUT.exists():
        payload = json.loads(OUTPUT.read_text(encoding='utf-8'))
        if payload.get('metadata') != metadata:
            raise ValueError(f'{OUTPUT} is for a different dataset/model/prompt; refusing to mix experiment runs')
        saved = payload.get('results', [])
        done = {str(r['DialogueID']): r for r in saved}
        if len(done) != len(saved) or any(k not in ids for k in done):
            raise ValueError('Existing checkpoint has duplicate/out-of-cohort DialogueIDs')
        print(f'Checkpoint resume: {len(done)} cases with at least one saved step', flush=True)

    if args.evaluate_only:
        if not done:
            raise ValueError('Cannot evaluate-only with no checkpoint')
        save_outputs(metadata, ordered, done)
        print(json.dumps(aggregate(ordered, done), ensure_ascii=False, indent=2))
        return

    targets = ordered[:args.limit]
    pending = [r for r in targets if str(r['DialogueID']) not in done
               or not all(k in done[str(r['DialogueID'])] for k in
                          ('direct', 'base_qa', 'base_qa_guided', 'lora_qa', 'lora_qa_guided'))]
    if not pending:
        save_outputs(metadata, ordered, done)
        print(f'Already completed {len(targets)}/{len(targets)} requested cases.')
        print(json.dumps(aggregate(ordered, done), ensure_ascii=False, indent=2))
        return

    for path in (MODEL_DIR/'config.json', QA_ADAPTER/'adapter_config.json',
                 SUMMARY_ADAPTER/'adapter_config.json'):
        if not path.is_file():
            raise FileNotFoundError(path)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError('Need CUDA GPU for inference; run on the RTX 4090')
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type='nf4',
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR),
                                                quantization_config=bnb,
                                                dtype=torch.bfloat16,
                                                device_map={'': 0},
                                                local_files_only=True)
    # One shared Qwen base on GPU, both tiny adapters switchable.
    # Never merge adapters together or train anything in this script.
    model = PeftModel.from_pretrained(base, str(QA_ADAPTER),
                                     adapter_name='qa', is_trainable=False)
    model.load_adapter(str(SUMMARY_ADAPTER), adapter_name='summary', is_trainable=False)
    model.eval()
    torch.manual_seed(42)
    print(f'Loaded 4-bit Qwen with QA and Summary LoRA adapters. Pending {len(pending)}/{args.limit} cases.', flush=True)

    for k, row in enumerate(pending, 1):
        did = str(row['DialogueID'])
        record = done.get(did)
        if record is None:
            record = {
                'index': row['index'], 'DialogueID': row['DialogueID'],
                'split': 'dev20' if row['index'] in set(dev20) else 'holdout80',
                'dialogue': row['dialogue'],
                'reference': row['reference'],  # offline scoring and human review ONLY
                'historical_sft': {
                    'prediction': row['historical_sft_prediction'],
                    'scores': row['historical_sft_scores'],
                },
            }
        def checkpoint():
            done[did] = record
            # Store partial work after each call, to survive cloud interruptions.
            atomic_json(OUTPUT, {'metadata': metadata,
                                 'results': [done[str(r['DialogueID'])] for r in ordered
                                             if str(r['DialogueID']) in done]})

        # A. Direct summary with EXACT same Stage-2 rules, without QA.
        if 'direct' not in record:
            model.set_adapter('summary')
            prediction, n = generate(model, tokenizer, stage2_prompt(row['dialogue']),
                                     SUMMARY_MAX_NEW_TOKENS, 'A direct')
            record['direct'] = {
                'prediction': prediction, 'scores': score_generated(prediction, row['reference'], n,
                                                                     SUMMARY_MAX_NEW_TOKENS),
            }
            checkpoint()
            print(f"  {did} A direct={record['direct']['scores']['rouge_l_f1']:.3f}", flush=True)

        # B Stage 1. SAME *trained* Stage-1 5-field prompt, adapter DISABLED.
        if 'base_qa' not in record:
            model.set_adapter('qa')
            with model.disable_adapter():
                raw, n = generate(model, tokenizer, row['qa_prompt'], QA_MAX_NEW_TOKENS, 'B base QA')
            record['base_qa'] = qa_result(raw, row['qa_prompt'], n, QA_MAX_NEW_TOKENS)
            checkpoint()
            print(f"  {did} B QA valid={record['base_qa']['schema_and_ids_valid']} QA count={record['base_qa']['predicted_qa_count']}", flush=True)

        # B Stage 2. If QA invalid, explicitly reuse direct output (intention to treat).
        if 'base_qa_guided' not in record:
            qa = record['base_qa']['parsed']
            if qa is None:
                pred = record['direct']['prediction']
                n = record['direct']['scores']['generated_tokens']
                fallback = True
            else:
                model.set_adapter('summary')
                pred, n = generate(model, tokenizer,
                                   stage2_prompt(row['dialogue'], qa, row['speaker_lines']),
                                   SUMMARY_MAX_NEW_TOKENS, 'B summary with base QA')
                fallback = False
            record['base_qa_guided'] = {
                'prediction': pred, 'scores': score_generated(pred, row['reference'], n,
                                                              SUMMARY_MAX_NEW_TOKENS),
                'fell_back_to_direct': fallback,
            }
            checkpoint()
            print(f"  {did} B guided={record['base_qa_guided']['scores']['rouge_l_f1']:.3f}" +
                  (' (fallback)' if fallback else ''), flush=True)

        # C Stage 1. SAME 5-field prompt, but QA adapter enabled.
        if 'lora_qa' not in record:
            model.set_adapter('qa')
            raw, n = generate(model, tokenizer, row['qa_prompt'], QA_MAX_NEW_TOKENS, 'C QA-LoRA QA')
            record['lora_qa'] = qa_result(raw, row['qa_prompt'], n, QA_MAX_NEW_TOKENS)
            checkpoint()
            print(f"  {did} C QA valid={record['lora_qa']['schema_and_ids_valid']} QA count={record['lora_qa']['predicted_qa_count']}", flush=True)

        # C Stage 2.
        if 'lora_qa_guided' not in record:
            qa = record['lora_qa']['parsed']
            if qa is None:
                pred = record['direct']['prediction']
                n = record['direct']['scores']['generated_tokens']
                fallback = True
            else:
                model.set_adapter('summary')
                pred, n = generate(model, tokenizer,
                                   stage2_prompt(row['dialogue'], qa, row['speaker_lines']),
                                   SUMMARY_MAX_NEW_TOKENS, 'C summary with QA-LoRA QA')
                fallback = False
            record['lora_qa_guided'] = {
                'prediction': pred, 'scores': score_generated(pred, row['reference'], n,
                                                              SUMMARY_MAX_NEW_TOKENS),
                'fell_back_to_direct': fallback,
            }
            checkpoint()
            print(f"  {did} C guided={record['lora_qa_guided']['scores']['rouge_l_f1']:.3f}" +
                  (' (fallback)' if fallback else ''), flush=True)
        print(f"[{k}/{len(pending)}] index={row['index']} DialogueID={did} "
              f"A={record['direct']['scores']['rouge_l_f1']:.3f} "
              f"B={record['base_qa_guided']['scores']['rouge_l_f1']:.3f} "
              f"C={record['lora_qa_guided']['scores']['rouge_l_f1']:.3f}", flush=True)

    save_outputs(metadata, ordered, done)
    print('\n' + json.dumps(aggregate(ordered, done), ensure_ascii=False, indent=2), flush=True)
    print(f'\nDetailed: {OUTPUT}\nSummary: {SUMMARY_OUTPUT}\nReview: {REVIEW_OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
