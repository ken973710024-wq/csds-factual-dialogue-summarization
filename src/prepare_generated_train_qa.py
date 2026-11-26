#!/usr/bin/env python3
"""Generate ONLY Stage-1 predicted QA on CSDS TRAIN dialogues; checkpoint each row.
No gold summary or CSDS human QA is ever in the QA inference prompt.

python src/prepare_generated_train_qa.py --sample-size 512 --check
python src/prepare_generated_train_qa.py --sample-size 512 --limit 5
python src/prepare_generated_train_qa.py --sample-size 512
python src/prepare_generated_train_qa.py --sample-size 9101
"""
import argparse, json
from pathlib import Path
from qa_summary_study_common import (
    BASE_MODEL, QA_ADAPTER, CACHE, DECODE_MAX,
    load_train, select_rows, provenance, qa_input, sha_text,
    cache_rows, validate_cache_record,
)
from validate_qa_lora import canonical_items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample-size', type=int, default=512)
    ap.add_argument('--limit', type=int, default=0, help='only run N selected rows; 0 means all')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    rows = load_train()
    selected = select_rows(rows, args.sample_size)
    if args.limit < 0 or args.limit > len(selected):
        ap.error('--limit must be from 0 to sample-size')
    if args.limit:
        selected = selected[:args.limit]
    prov = provenance()
    done = cache_rows()
    by_source = {str(x['DialogueID']):x['source'] for x in rows}
    for key, item in done.items():
        if key not in by_source:
            raise ValueError(f'Unknown cache ID: {key}')
        validate_cache_record(item, by_source[key], prov)
    todo = [row for row in selected if str(row['DialogueID']) not in done]
    print(f'QA cache: existing={len(done)} selected={len(selected)}, pending={len(todo)}')
    print(f'Gold summary & gold QA never enter the generation prompt. Output: {CACHE}')
    if args.check:
        for path in (BASE_MODEL/'config.json', QA_ADAPTER/'adapter_config.json'):
            print(('OK: ' if path.is_file() else 'MISSING: ') + str(path))
            if not path.is_file():
                raise FileNotFoundError(path)
        return
    if not todo:
        return
    for path in (BASE_MODEL/'config.json', QA_ADAPTER/'adapter_config.json'):
        if not path.is_file(): raise FileNotFoundError(path)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL),quantization_config=quant,
                                                dtype=torch.bfloat16,device_map={'':0},local_files_only=True)
    model = PeftModel.from_pretrained(base,str(QA_ADAPTER),is_trainable=False)
    model.eval()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    with CACHE.open('a',encoding='utf-8') as output:
        for i, row in enumerate(todo, 1):
            prompt = qa_input(row['source'])
            messages = [{'role':'user','content':prompt}]
            inputs = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            tok = tokenizer([inputs],return_tensors='pt').to('cuda:0')
            if tok['input_ids'].shape[1]+DECODE_MAX > 8192:
                raise ValueError(f'QA prompt exceeds context guard for {row["DialogueID"]}')
            with torch.inference_mode():
                result = model.generate(**tok,max_new_tokens=DECODE_MAX,do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
            ids = result[0, tok['input_ids'].shape[1]:]
            raw = tokenizer.decode(ids,skip_special_tokens=True).strip()
            qa, error, _, _ = canonical_items(raw, prompt)
            data = {
                'DialogueID':row['DialogueID'], 'qa_prompt_sha256':sha_text(prompt),
                'provenance':prov, 'qa':qa, 'valid':qa is not None,
                'error':error,'generated_tokens':len(ids),
                'hit_max_new_tokens':len(ids)>=DECODE_MAX,
                'raw':raw if qa is None else None,
            }
            output.write(json.dumps(data, ensure_ascii=False)+'\n')
            output.flush()
            print(f'[{i}/{len(todo)}] DialogueID={row["DialogueID"]} '
                  f'QA={len(qa) if qa else "INVALID"} tokens={len(ids)}',flush=True)

if __name__=='__main__': main()
