#!/usr/bin/env python3
"""Batched, resumable Stage-1 QA generation for the CSDS Stage-2 F/G study.

Uses the *same* source data, prompt, QA-LoRA, greedy decoding, cache schema,
and provenance checks as prepare_generated_train_qa.py. The original script
MUST be stopped first: two writers must never append to the same cache.

Examples:
  python src/prepare_generated_train_qa_batched.py --sample-size 9101 --check
  python src/prepare_generated_train_qa_batched.py --sample-size 9101 --batch-size 4 --max-pending 24
  python src/prepare_generated_train_qa_batched.py --sample-size 9101 --batch-size 4
"""
import argparse
import json
from pathlib import Path

from qa_summary_study_common import (
    BASE_MODEL, QA_ADAPTER, CACHE, DECODE_MAX,
    load_train, select_rows, provenance, qa_input, sha_text,
    cache_rows, validate_cache_record,
)
from validate_qa_lora import canonical_items


def cut_at_eos(generated_ids, eos_ids):
    """Trim batch padding following EOS so token counts match single generation."""
    tokens = generated_ids.tolist() if hasattr(generated_ids, 'tolist') else list(generated_ids)
    for j, token in enumerate(tokens):
        if token in eos_ids:
            return tokens[: j + 1]
    return tokens


def make_groups(items, size):
    """Length grouping reduces padding; order is irrelevant because cache keys are DialogueIDs."""
    items = sorted(items, key=lambda obj: len(obj[2]))
    return [items[k:k + size] for k in range(0, len(items), size)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-size', type=int, default=9101)
    parser.add_argument('--limit', type=int, default=0,
                        help='Only consider first N sampled rows (0 = all).')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-pending', type=int, default=0,
                        help='Generate at most this many UN-CACHED rows (0 = all).')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_pending < 0:
        parser.error('--batch-size must be >= 1 and --max-pending >= 0')

    rows = load_train()
    selected = select_rows(rows, args.sample_size)
    if args.limit < 0 or args.limit > len(selected):
        parser.error('--limit must be from 0 to sample-size')
    if args.limit:
        selected = selected[:args.limit]

    prov = provenance()
    done = cache_rows()
    by_source = {str(row['DialogueID']): row['source'] for row in rows}
    for key, item in done.items():
        if key not in by_source:
            raise ValueError(f'Unknown cached DialogueID: {key}')
        validate_cache_record(item, by_source[key], prov)

    remaining = [r for r in selected if str(r['DialogueID']) not in done]
    print(f'Cache existing: {len(done)}, selected: {len(selected)}, pending: {len(remaining)}', flush=True)
    if args.max_pending:
        remaining = remaining[:args.max_pending]

    paths = (BASE_MODEL / 'config.json', QA_ADAPTER / 'adapter_config.json')
    for path in paths:
        print(('OK ' if path.is_file() else 'MISSING ') + str(path), flush=True)
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.check or not remaining:
        return

    # The prompt generation method is identical to the original script.
    items = [(row, qa_input(row['source'])) for row in remaining]

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')

    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # crucial for batch generation of decoder-only LMs

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL), quantization_config=quant,
        dtype=torch.bfloat16, device_map={'': 0}, local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, str(QA_ADAPTER), is_trainable=False)
    model.eval()
    torch.manual_seed(42)

    eos = tokenizer.eos_token_id
    eos_ids = set(int(x) for x in (eos if isinstance(eos, (list, tuple)) else [eos]) if x is not None)
    # User assistant's original model.generate() used eos_token_id from generation config;
    # honor that, including if it differs from tokenizer.eos_token_id.
    model_eos = model.generation_config.eos_token_id
    for e in (model_eos if isinstance(model_eos, (list, tuple)) else [model_eos]):
        if e is not None:
            eos_ids.add(int(e))
    if not eos_ids:
        raise ValueError('No EOS token configured')

    queued = []
    for row, prompt in items:
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        queued.append((row, prompt, rendered))

    groups = make_groups(queued, args.batch_size)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    processed = 0

    def infer_batch(group):
        rendered = [item[2] for item in group]
        tok = tokenizer(rendered, return_tensors='pt', padding=True)
        input_len = tok['input_ids'].shape[1]
        if input_len + DECODE_MAX > 8192:
            ids = ', '.join(str(item[0]['DialogueID']) for item in group)
            raise ValueError(f'QA context guard: max padded prompt + {DECODE_MAX} > 8192, IDs {ids}')
        tok = tok.to('cuda:0')
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    **tok, max_new_tokens=DECODE_MAX, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            if len(group) == 1:
                raise
            del tok
            torch.cuda.empty_cache()
            half = len(group) // 2
            return infer_batch(group[:half]) + infer_batch(group[half:])
        answer = []
        for k, (row, prompt, _) in enumerate(group):
            tokens = cut_at_eos(output_ids[k, input_len:], eos_ids)
            raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            qa, error, _, _ = canonical_items(raw, prompt)
            answer.append({
                'DialogueID': row['DialogueID'],
                'qa_prompt_sha256': sha_text(prompt),
                'provenance': prov,
                'qa': qa,
                'valid': qa is not None,
                'error': error,
                'generated_tokens': len(tokens),
                'hit_max_new_tokens': len(tokens) >= DECODE_MAX,
                'raw': raw if qa is None else None,
            })
        return answer

    with CACHE.open('a', encoding='utf-8') as file:
        for batch_n, group in enumerate(groups, 1):
            for row in infer_batch(group):
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
                file.flush()  # interrupt-safe: every finished sample is saved
                processed += 1
                count = len(row['qa']) if row['qa'] is not None else 'INVALID'
                print(f'[{processed}/{len(queued)}] batch={batch_n}/{len(groups)} '
                      f'DialogueID={row["DialogueID"]} QA={count} '
                      f'tokens={row["generated_tokens"]}', flush=True)
    print(f'Completed {processed}. Total expected cached rows: {len(done) + processed}', flush=True)


if __name__ == '__main__':
    main()
