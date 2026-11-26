#!/usr/bin/env python3
"""Build strictly paired direct vs QA-conditioned Stage-2 train sets.

Both groups have the SAME dialogue IDs, SAME targets, SAME length eligibility,
SAME frozen SUMMARY_RULES. Only the QA context is different. Any overlength
example is removed from BOTH, never silently truncated.

python src/prepare_summary_qa_training.py --sample-size 512 --max-seq-tokens 3072
python src/prepare_summary_qa_training.py --sample-size 9101 --max-seq-tokens 3072
"""
import argparse, json
from pathlib import Path
from qa_summary_study_common import (
    BASE_MODEL, CACHE, load_train, select_rows, label_for_size,
    provenance, stage2_input, cache_rows, validate_cache_record,
    atomic_json, sha_file, sha_text, SUMMARY_RULES,
)


def make_records(rows, qa_cache, prov):
    """CPU-only source join, useful for unit tests and output inspection."""
    for row in rows:
        key = str(row['DialogueID'])
        if key not in qa_cache:
            raise ValueError(f'Missing generated QA for train DialogueID={key}; run QA generation first')
        item = qa_cache[key]
        validate_cache_record(item, row['source'], prov)
        direct = stage2_input(row['source'])
        aware = stage2_input(row['source'], item['qa'])
        label = row['target']
        meta = {'DialogueID':row['DialogueID'],'qa_valid':item['qa'] is not None}
        yield ({**meta,'prompt':direct,'target':label},
               {**meta,'prompt':aware,'target':label})


def tokenize_pair(tokenizer, direct, aware):
    def length(record):
        messages=[{'role':'user','content':record['prompt']},{'role':'assistant','content':record['target']}]
        return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    return length(direct), length(aware)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample-size',type=int,default=512)
    ap.add_argument('--max-seq-tokens',type=int,default=3072)
    ap.add_argument('--check',action='store_true')
    args=ap.parse_args()
    if args.max_seq_tokens < 512:
        ap.error('--max-seq-tokens must be at least 512')
    rows=load_train()
    selected=select_rows(rows,args.sample_size)
    label=label_for_size(args.sample_size,len(rows))
    folder=Path('data/qa_stage2_research')/label
    direct_path=folder/'direct.jsonl'
    aware_path=folder/'aware.jsonl'
    manifest_path=folder/'manifest.json'
    qa_cache=cache_rows()
    prov=provenance()
    pairs=list(make_records(selected,qa_cache,prov))
    same_targets=all(a['target']==b['target'] and a['DialogueID']==b['DialogueID'] for a,b in pairs)
    if not same_targets:
        raise AssertionError('Direct and aware supervision is not paired')
    if args.check:
        print(f'CHECK OK: source={len(rows)} selected={len(selected)} cache={len(qa_cache)}; '
              f'paired labels and IDs identical. Tokenizer needed only for actual build.')
        print(f'Stage2 prompt SHA256={sha_text(SUMMARY_RULES)}')
        print(f'Direct preview:\n{pairs[0][0]["prompt"][:600]}\n')
        print(f'QA-aware preview:\n{pairs[0][1]["prompt"][:900]}\n')
        return
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(BASE_MODEL),local_files_only=True)
    included=[]; dropped=[]; invalid_count=0; lengths=[]
    for direct,aware in pairs:
        dl,al=tokenize_pair(tokenizer,direct,aware)
        lengths.append({'DialogueID':direct['DialogueID'],'direct':dl,'qa':al})
        if max(dl,al)>args.max_seq_tokens:
            dropped.append({'DialogueID':direct['DialogueID'],'direct_tokens':dl,'qa_tokens':al})
            continue
        included.append((direct,aware))
        invalid_count+=int(not aware['qa_valid'])
    if len(included)<2:
        raise ValueError('Too few paired examples after length filtering; increase --max-seq-tokens')
    folder.mkdir(parents=True,exist_ok=True)
    if direct_path.exists() or aware_path.exists() or manifest_path.exists():
        raise FileExistsError(f'{folder}: already exists. Do not overwrite a fixed experiment.')
    with direct_path.open('w',encoding='utf-8') as a,aware_path.open('w',encoding='utf-8') as b:
        for d,q in included:
            a.write(json.dumps(d,ensure_ascii=False)+'\n')
            b.write(json.dumps(q,ensure_ascii=False)+'\n')
    data={
        'version':'qa_stage2_paired_v1', 'tag':label,'requested_size':args.sample_size,
        'selected_ids_sha256':sha_text(json.dumps([str(x['DialogueID']) for x in selected])),
        'max_seq_tokens':args.max_seq_tokens,
        'selected_count':len(selected),'paired_retained':len(included),
        'paired_dropped':len(dropped), 'dropped_overlength':dropped,
        'predicted_qa_invalid_fallback_to_direct':invalid_count,
        'max_direct_tokens':max(x['direct'] for x in lengths),
        'max_qa_tokens':max(x['qa'] for x in lengths),
        'direct_path':str(direct_path),'aware_path':str(aware_path),
        'direct_sha256':sha_file(direct_path),'aware_sha256':sha_file(aware_path),
        'stage2_prompt_sha256':sha_text(SUMMARY_RULES),'qa_cache_sha256':sha_file(CACHE),
        'provenance':prov,
        'same_ids_and_targets':True,
        'warning':'QA predictions only in inputs, original CSDS reference summaries only in target; no human CSDS QA annotations in inputs.',
    }
    atomic_json(manifest_path,data)
    print(json.dumps({k:v for k,v in data.items() if k!='dropped_overlength'},ensure_ascii=False,indent=2))
    if dropped:
        print(f'WARNING: {len(dropped)} paired overlength examples dropped from BOTH arms. Check manifest.')

if __name__=='__main__': main()
