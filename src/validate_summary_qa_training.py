#!/usr/bin/env python3
"""Fair paired Stage-2 study evaluation on the exact saved 100 validation cases.

Compare old A/C and *new* G matched-direct / F matched-QA-aware adapters.
The new pair has the same base, training labels, seed, training data, epochs,
architecture, inference rules and decoding; only QA inclusion differs.

python src/validate_summary_qa_training.py --tag pilot512 --check
python src/validate_summary_qa_training.py --tag pilot512 --limit 5
python src/validate_summary_qa_training.py --tag pilot512 --limit 20
python src/validate_summary_qa_training.py --tag pilot512 --limit 100
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from qa_summary_study_common import (
    BASE_MODEL, SUMMARY_MAX, atomic_json, sha_file, sha_text,
    stage2_input, SUMMARY_RULES, score,
)
from validate_qa_aware_pipeline import read_cohort

ABC=Path('outputs/validate_qa_aware_pipeline_100.json')
VAL=Path('data/CSDS/val.json')


def load_cohort():
    ordered, dev20, base_meta=read_cohort()
    prior=json.loads(ABC.read_text(encoding='utf-8'))
    if prior.get('metadata',{}).get('version')!='qa_aware_pipeline_abc_v1':
        raise ValueError('Unexpected old A/B/C result version')
    by_id={str(x['DialogueID']):x for x in prior['results']}
    if len(by_id)!=100:
        raise ValueError('Expected 100 unique old A/B/C results')
    original_val=json.loads(VAL.read_text(encoding='utf-8'))
    pair=[]
    for old in ordered:
        key=str(old['DialogueID'])
        abc=by_id.get(key)
        if abc is None or abc.get('index')!=old['index']:
            raise ValueError(f'Old C prediction mismatches current cohort: {key}')
        for field in ('direct','lora_qa_guided','lora_qa'):
            if field not in abc: raise ValueError(f'Missing prior {field}: {key}')
        sample=original_val[old['index']]
        if str(sample['DialogueID'])!=key:
            raise ValueError('Official validation ID mismatch')
        pair.append((old,abc,sample))
    return pair,dev20


def baseline_scores(row):
    return {
        'A_old_direct':row['direct']['scores'],
        'C_old_qa_guided':row['lora_qa_guided']['scores'],
    }


def boot(differences,n=3000):
    rng=random.Random(20261122)
    k=len(differences)
    means=sorted(sum(differences[rng.randrange(k)] for _ in range(k))/k for _ in range(n))
    return [round(means[int(n*0.025)],4),round(means[min(n-1,int(n*0.975))],4)]


def report_group(rows,name):
    if not rows: return {'cohort':name,'completed':0}
    keys=('A_old_direct','C_old_qa_guided','G_new_matched_direct','F_new_qa_trained')
    def mean(k):
        return {field:round(sum(r[k]['scores'][field] for r in rows)/len(rows),4)
                for field in ('rouge_l_f1','rouge_l_precision','rouge_l_recall','chars')}
    def compare(left,right):
        delta=[r[right]['scores']['rouge_l_f1']-r[left]['scores']['rouge_l_f1'] for r in rows]
        return {'f1_delta':round(sum(delta)/len(delta),4),'paired_bootstrap_ci95':boot(delta),
                'right_wins':sum(d>1e-10 for d in delta),
                'left_wins':sum(d< -1e-10 for d in delta),
                'ties':sum(abs(d)<=1e-10 for d in delta)}
    return {'cohort':name,'completed':len(rows),
            **{k:mean(k) for k in keys},
            'F_minus_G_matched_QA_training_effect':compare('G_new_matched_direct','F_new_qa_trained'),
            'F_minus_C_new_vs_old_QA':compare('C_old_qa_guided','F_new_qa_trained'),
            'F_minus_A_new_vs_old_direct':compare('A_old_direct','F_new_qa_trained')}


def render_outputs(tag,meta,ordered,done):
    out=Path('outputs')/f'validate_summary_qa_training_{tag}_100.json'
    summary=Path('outputs')/f'validate_summary_qa_training_{tag}_100_summary.json'
    review=Path('outputs')/f'validate_summary_qa_training_{tag}_100_review.md'
    rows=[done[str(item[0]['DialogueID'])] for item in ordered if str(item[0]['DialogueID']) in done]
    complete=[r for r in rows if all(k in r for k in ('G_new_matched_direct','F_new_qa_trained'))]
    payload={'metadata':meta,'results':rows}
    atomic_json(out,payload)
    groups={
        'warning':'Prior 20 were used for prompt development; 80 only relatively held out, not pristine test. ROUGE-L is not factuality.',
        'all':report_group(complete,'all'),
        'dev20':report_group([r for r in complete if r['split']=='dev20'],'development20'),
        'holdout80':report_group([r for r in complete if r['split']=='holdout80'],'prior_holdout80'),
    }
    atomic_json(summary,groups)
    text=['# QA-aware Stage-2 matched training comparison',
          'G: retrained direct; F: retrained with predicted QA. A/C are original saved baselines.',
          'Gold reference for OFFLINE scoring only, never model inference.', '']
    for row in complete:
        text.extend([f'## DialogueID {row["DialogueID"]} [{row["split"]}]',
             '### Original dialogue', '```text',row['dialogue'],'```','',
             '### Stage-1 QA (same generated QA as old C)','```json',
             json.dumps(row['qa_used'],ensure_ascii=False,indent=2),'```',''])
        for title,field in [('A original direct','A_old_direct'),('C old QA-guided','C_old_qa_guided'),
                            ('G matched-direct trained','G_new_matched_direct'),('F matched-QA trained','F_new_qa_trained')]:
            text.extend([f'### {title}', '```text',row[field]['prediction'],'```',
                         f'ROUGE-L F1 = {row[field]["scores"]["rouge_l_f1"]:.4f}', ''])
        text.extend(['### Reference summary (offline)','```text',row['reference'],'```',
                     'Manual review: key time, status, object, omission, unsupported claim, repetition.', '', '---',''])
    review.write_text('\n'.join(text),encoding='utf-8')
    return groups


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag',default='pilot512')
    ap.add_argument('--limit',type=int,default=20)
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--evaluate-only',action='store_true')
    args=ap.parse_args()
    if not 1<=args.limit<=100: ap.error('--limit must be in 1..100')
    ordered,dev20=load_cohort()
    data_dir=Path('data/qa_stage2_research')/args.tag
    manifest=json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('tag')!=args.tag:
        raise ValueError('Wrong paired data manifest')
    direct=Path('outputs')/f'qlora_summary_matched_direct_{args.tag}'/'final_adapter'
    aware=Path('outputs')/f'qlora_summary_matched_aware_{args.tag}'/'final_adapter'
    meta={'version':'matched_stage2_train_eval_v1','tag':args.tag,
          'qa_stage1_source_sha256':sha_file(ABC),
          'training_manifest_sha256':sha_file(data_dir/'manifest.json'),
          'stage2_prompt_sha256':sha_text(SUMMARY_RULES),
          'base_model':str(BASE_MODEL), 'direct_adapter':str(direct),
          'aware_adapter':str(aware),'eval_decode':'greedy',
          'summary_max_new_tokens':SUMMARY_MAX,'count':100,
          'dev20_indices':dev20,'no_reference_in_prompt':True}
    abc=json.loads(ABC.read_text(encoding='utf-8'))['metadata']
    if abc['stage2_prompt_sha256']!=meta['stage2_prompt_sha256']:
        raise ValueError('SUMMARY_RULES have changed since original ABC!')
    to_check=[BASE_MODEL/'config.json',direct/'adapter_config.json',aware/'adapter_config.json']
    if args.check:
        for p in to_check:
            print(('OK' if p.is_file() else 'MISSING'),p)
        print('Source cohort=100, first 20 development; all old C QA predictions available.')
        print('Summary prompt hash:',meta['stage2_prompt_sha256'])
        if any(not x.is_file() for x in to_check):
            print('Some models are not yet trained; run the 2 matched training arms first.')
        return
    dest=Path('outputs')/f'validate_summary_qa_training_{args.tag}_100.json'
    existing=json.loads(dest.read_text(encoding='utf-8')) if dest.exists() else None
    if existing and existing.get('metadata')!=meta:
        raise ValueError('Existing evaluation checkpoint has different experiment configuration')
    done={str(row['DialogueID']):row for row in existing['results']} if existing else {}
    if len(done)!=(len(existing['results']) if existing else 0):
        raise ValueError('Duplicate IDs in evaluation checkpoint')
    ids={str(row[0]['DialogueID']) for row in ordered}
    if not set(done).issubset(ids): raise ValueError('Out-of-cohort cached results')
    if args.evaluate_only:
        print(json.dumps(render_outputs(args.tag,meta,ordered,done),ensure_ascii=False,indent=2))
        return
    pending=[]
    for old,abc_row,orig in ordered[:args.limit]:
        key=str(old['DialogueID'])
        if key not in done or not all(k in done[key] for k in ('G_new_matched_direct','F_new_qa_trained')):
            pending.append((old,abc_row,orig))
    if not pending:
        print(json.dumps(render_outputs(args.tag,meta,ordered,done),ensure_ascii=False,indent=2))
        return
    for p in to_check:
        if not p.is_file(): raise FileNotFoundError(p)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    tokenizer=AutoTokenizer.from_pretrained(str(BASE_MODEL),local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                             bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained(str(BASE_MODEL),quantization_config=quant,
                                             dtype=torch.bfloat16,device_map={'':0},local_files_only=True)
    model=PeftModel.from_pretrained(base,str(direct),adapter_name='matched_direct',is_trainable=False)
    model.load_adapter(str(aware),adapter_name='matched_aware',is_trainable=False)
    model.eval()
    torch.manual_seed(42)

    def generate(prompt):
        messages=[{'role':'user','content':prompt}]
        t=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        toks=tokenizer([t],return_tensors='pt').to('cuda:0')
        if toks['input_ids'].shape[1]+SUMMARY_MAX>8192:
            raise ValueError('Summary token length exceeds fixed guard: no silent truncation')
        with torch.inference_mode():
            result=model.generate(**toks,do_sample=False,max_new_tokens=SUMMARY_MAX,
                                  pad_token_id=tokenizer.eos_token_id)
        output=result[0,toks['input_ids'].shape[1]:]
        return tokenizer.decode(output,skip_special_tokens=True).strip(),len(output)

    for i,(old,abc_row,orig) in enumerate(pending,1):
        key=str(old['DialogueID'])
        row=done.get(key)
        if row is None:
            qa=abc_row['lora_qa']['parsed']
            row={
                'index':old['index'],'DialogueID':old['DialogueID'],
                'split':'dev20' if old['index'] in set(dev20) else 'holdout80',
                'dialogue':old['dialogue'],'reference':old['reference'],
                'qa_used':qa,'qa_valid':qa is not None,
                'A_old_direct':abc_row['direct'],
                'C_old_qa_guided':abc_row['lora_qa_guided'],
            }
        def save():
            done[key]=row
            atomic_json(dest,{'metadata':meta,'results':[done[str(x[0]['DialogueID'])] for x in ordered if str(x[0]['DialogueID']) in done]})
        if 'G_new_matched_direct' not in row:
            model.set_adapter('matched_direct')
            pred,n=generate(stage2_input(orig))
            row['G_new_matched_direct']={'prediction':pred,'scores':{
                **score(pred,row['reference']),'generated_tokens':n,'hit_max_new_tokens':n>=SUMMARY_MAX}}
            save()
        if 'F_new_qa_trained' not in row:
            model.set_adapter('matched_aware')
            pred,n=generate(stage2_input(orig,row['qa_used']))
            row['F_new_qa_trained']={'prediction':pred,'scores':{
                **score(pred,row['reference']),'generated_tokens':n,'hit_max_new_tokens':n>=SUMMARY_MAX}}
            save()
        print(f'[{i}/{len(pending)}] DialogueID={key}: '
              f'G={row["G_new_matched_direct"]["scores"]["rouge_l_f1"]:.4f} '
              f'F={row["F_new_qa_trained"]["scores"]["rouge_l_f1"]:.4f}',flush=True)
        render_outputs(args.tag,meta,ordered,done)
    print(json.dumps(render_outputs(args.tag,meta,ordered,done),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
