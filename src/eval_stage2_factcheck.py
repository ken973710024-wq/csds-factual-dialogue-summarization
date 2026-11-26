#!/usr/bin/env python3
"""Frozen-adapter QA fact verification ablation; run from csds-project root.

Smoke cases 2168/5853 are *not* part of the randomly chosen diagnostic30.
No references or gold QA appear in prompts. Existing baseline/model files never overwritten.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
from qa_summary_study_common import BASE_MODEL, SUMMARY_MAX, atomic_json, stage2_input, score, qa_input
from validate_qa_lora import numbered_speakers

ABC = Path('outputs/validate_qa_aware_pipeline_100.json')
VAL = Path('data/CSDS/val.json')
SEED = 20261123
SMOKE = {'2168', '5853'}
CHECK_RULES = '''\n【生成前的业务事实核查规则（优先遵循）】
候选 QA 只是定位业务问题的草稿，不是事实真值；所有写进摘要的处理结果都必须得到【原始对话】支持。
按订单、商品或服务事项分别核对，并沿对话顺序确认各事项最后一个有依据的处理状态；不同事项不得合并状态。
严格区分“建议申请”“已经提交”“等待审核”“处理成功”“退款成功”“尚未确认”，不得把未来承诺当成已经完成。
用户确认的事项应保留说话人归属，不得写成客服确认。
保留影响下一步行动的重要条件、金额、时间、最终状态以及客服承诺；重复追问合并。
原 QA 漏掉的关键进展可以从原始对话补入；原 QA 错误或与原文冲突时以原文为准。
有矛盾或最终结果未确认则明确说未确认；不要自行推断结果。
只输出简洁编号的客服摘要，不要输出分析过程或证据清单。
'''

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_cases():
    data=json.loads(ABC.read_text(encoding='utf-8'))
    val=json.loads(VAL.read_text(encoding='utf-8'))
    rows=[]
    for r in data['results']:
        idx=int(r['index']); src=val[idx]
        if str(src['DialogueID'])!=str(r['DialogueID']):
            raise ValueError('DialogueID/index mismatch')
        rows.append({'DialogueID':str(r['DialogueID']),'index':idx, 'split':r['split'],
                     'source':src, 'qa':r['lora_qa']['parsed'],
                     'reference':r['reference'], 'A_old':r['direct']})
    if len(rows)!=100 or len(set(r['DialogueID'] for r in rows))!=100:
        raise ValueError('100 unique validation rows required')
    return rows

def choose(rows, mode):
    if mode=='smoke':
        selected=[r for r in rows if r['DialogueID'] in SMOKE]
        if len(selected)!=2: raise ValueError('2168/5853 not found')
        return selected
    pool=[r for r in rows if r['split']!='dev20' and r['DialogueID'] not in SMOKE]
    if len(pool)<30: raise ValueError('insufficient eligible validation cases')
    indices=set(random.Random(SEED).sample(range(len(pool)),30))
    return [r for i,r in enumerate(pool) if i in indices]

def correction_path(mode):
    return Path('data/qa_stage2_research/factuality_study')/f'corrected_qa_{mode}.json'

def make_corrections(rows, mode):
    target=correction_path(mode)
    if target.exists():
        print('EXISTS (left untouched):',target)
        return target
    payload={'instructions':'Only read ORIGINAL dialogue to correct QA. Never consult reference summaries/gold QA. Set reviewed=true and describe changes after checking.',
             'mode':mode,'items':[
                {'DialogueID':r['DialogueID'],'reviewed':False,'change_note':'',
                 'qa':r['qa'] if r['qa'] is not None else []} for r in rows]}
    atomic_json(target,payload)
    print('CREATED:',target)
    return target

def load_corrections(rows,mode):
    obj=json.loads(correction_path(mode).read_text(encoding='utf-8'))
    mapping={str(x['DialogueID']):x for x in obj['items']}
    if set(mapping)!={r['DialogueID'] for r in rows} or len(mapping)!=len(obj['items']):
        raise ValueError('correction file sample set mismatch')
    out={}
    for row in rows:
        x=mapping[row['DialogueID']]
        if x.get('reviewed') is not True or not x.get('change_note','').strip():
            raise ValueError('Manual correction not reviewed: '+row['DialogueID'])
        qa=x['qa']
        if not isinstance(qa,list) or not qa:
            raise ValueError('Correction must be non-empty QA list: '+row['DialogueID'])
        _,agent_ids,_=numbered_speakers(qa_input(row['source']))
        user_ids,_,_=numbered_speakers(qa_input(row['source']))
        for item in qa:
            if not isinstance(item,dict) or set(item)!={'category','question','answer','question_ids','evidence_ids'}:
                raise ValueError('Wrong QA schema: '+row['DialogueID'])
            if not all(isinstance(item[k],str) and item[k].strip() for k in ('category','question','answer')):
                raise ValueError('Missing QA string: '+row['DialogueID'])
            if not isinstance(item['question_ids'],list) or not isinstance(item['evidence_ids'],list) or not item['evidence_ids']:
                raise ValueError('Bad ID lists: '+row['DialogueID'])
            if any(q not in user_ids for q in item['question_ids']) or any(q not in agent_ids for q in item['evidence_ids']):
                raise ValueError('Evidence / question reference not in dialogue: '+row['DialogueID'])
        out[row['DialogueID']]=qa
    return out

def render_results(path,meta,rows,done):
    ordered=[done[r['DialogueID']] for r in rows if r['DialogueID'] in done]
    atomic_json(path,{'metadata':meta,'results':ordered})
    def compare(a,b):
        pairs=[(r[a]['scores']['rouge_l_f1'],r[b]['scores']['rouge_l_f1']) for r in ordered if a in r and b in r]
        if not pairs: return {'n':0}
        delta=[y-x for x,y in pairs]
        rng=random.Random(271828)
        n=len(delta)
        bs=sorted(sum(delta[rng.randrange(n)] for _ in range(n))/n for _ in range(3000))
        return {'n':n,'mean_delta':round(sum(delta)/n,4),
                'ci95':[round(bs[74],4),round(bs[2924],4)],
                'wins':sum(x>0 for x in delta),'losses':sum(x<0 for x in delta),
                'ties':sum(x==0 for x in delta)}
    report={'sample':meta['mode'],'completed':len(ordered),
            'baseline_to_checked':compare('F_baseline','F_checked'),
            'checked_to_corrected':compare('F_checked','F_corrected'),
            'A_to_checked':compare('A_old','F_checked'),
            'warning':'Development/smoke data not independent. ROUGE is not factuality.'}
    summary=path.with_name(path.stem+'_summary.json')
    atomic_json(summary,report)
    review=path.with_name(path.stem+'_review.md')
    blocks=['# Frozen F model: final-state verification study','Source-only QA correction; reference displayed BELOW predictions for offline scoring only.','']
    for row in ordered:
        blocks+=['## DialogueID '+row['DialogueID'],'### Original dialogue','```text',row['dialogue'],'```',
                 '### Original QA','```json',json.dumps(row['qa_used'],ensure_ascii=False,indent=2),'```']
        for name,label in [('A_old','A original direct'),('F_baseline','F original QA'),
                            ('F_checked','F + state verification'),('F_corrected','F + verified prompt + human-corrected QA')]:
            if name in row:
                blocks+=['### '+label,'```text',row[name]['prediction'],'```',
                         f'ROUGE-L F1: {row[name]["scores"]["rouge_l_f1"]:.4f}']
        blocks+=['### Human gold summary (offline only)','```text',row['reference'],'```','','---','']
    review.write_text('\n'.join(blocks),encoding='utf-8')
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag',default='full',choices=['full','pilot512'])
    ap.add_argument('--mode',default='smoke',choices=['smoke','diagnostic30'])
    ap.add_argument('--arms',default='baseline,checked',help='Comma-separated: baseline,checked,corrected')
    ap.add_argument('--check',action='store_true')
    ap.add_argument('--prepare-corrections',action='store_true')
    ap.add_argument('--limit',type=int,default=0,help='For smoke/dev only; fixed sampling remains stable')
    args=ap.parse_args()
    arms=args.arms.split(',')
    if not arms or len(set(arms))!=len(arms) or not set(arms)<={'baseline','checked','corrected'}:
        ap.error('Invalid --arms')
    rows=choose(load_cases(),args.mode)
    if args.limit: rows=rows[:args.limit]
    if args.prepare_corrections:
        if args.limit: raise ValueError('Prepare corrections on full selected cohort only')
        make_corrections(rows,args.mode)
        return
    adapter=Path('outputs')/f'qlora_summary_matched_aware_{args.tag}'/'final_adapter'
    checks=[ABC,VAL,BASE_MODEL/'config.json',adapter/'adapter_config.json']
    if args.check:
        for f in checks: print('OK' if f.is_file() else 'MISSING',f)
        print('SELECTED',[(r['DialogueID'],r['split']) for r in rows])
        print('Corrections:',correction_path(args.mode))
        return
    for f in checks:
        if not f.is_file(): raise FileNotFoundError(f)
    corrected=load_corrections(rows,args.mode) if 'corrected' in arms else {}
    path=Path('outputs')/f'eval_factuality_F_{args.tag}_{args.mode}_{"_".join(arms)}.json'
    cfg={'version':'frozen_F_factcheck_v1','mode':args.mode,'tag':args.tag,'ids':[r['DialogueID'] for r in rows],
         'arms':arms,'source_abc_sha256':digest(ABC),'model_adapter_config_sha256':digest(adapter/'adapter_config.json'),
         'model_weights_sha256':digest(adapter/'adapter_model.safetensors'),
         'prompt_sha256':hashlib.sha256(CHECK_RULES.encode('utf-8')).hexdigest(),
         'corrections_sha256':digest(correction_path(args.mode)) if 'corrected' in arms else None,
         'decode':{'do_sample':False,'max_new_tokens':SUMMARY_MAX}}
    existing=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    if existing and existing.get('metadata')!=cfg:
        raise ValueError('Experiment configuration changed: use a new output filename; do not overwrite')
    done={str(x['DialogueID']):x for x in existing['results']} if existing else {}
    if len(done)!=(len(existing['results']) if existing else 0):raise ValueError('Duplicate IDs')
    pending=[]
    for r in rows:
        d=done.get(r['DialogueID'],{})
        if any('F_'+a not in d for a in arms):pending.append(r)
    if not pending:
        print(json.dumps(render_results(path,cfg,rows,done),ensure_ascii=False,indent=2));return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    tokenizer=AutoTokenizer.from_pretrained(str(BASE_MODEL),local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                             bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained(str(BASE_MODEL),quantization_config=quant,
                                              dtype=torch.bfloat16,device_map={'':0},local_files_only=True)
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
    model.eval()
    torch.manual_seed(42)
    def generate(prompt):
        chat=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
        batch=tokenizer([chat],return_tensors='pt').to('cuda:0')
        if batch['input_ids'].shape[1]+SUMMARY_MAX>8192:raise ValueError('Prompt exceeds 8192-token guard; not truncating')
        with torch.inference_mode():
            result=model.generate(**batch,max_new_tokens=SUMMARY_MAX,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        new_tokens=result[0,batch['input_ids'].shape[1]:]
        return tokenizer.decode(new_tokens,skip_special_tokens=True).strip(),len(new_tokens)
    for index,r in enumerate(pending,1):
        k=r['DialogueID']
        d=done.get(k) or {'DialogueID':k,'split':r['split'],'index':r['index'],
                           'dialogue':r['source'].get('Dialogue') and '\n'.join(str(z) for z in r['source']['Dialogue']) or '',
                           'qa_used':r['qa'],'reference':r['reference'],'A_old':r['A_old']}
        from validate_qa_aware_pipeline import dialogue_to_text
        d['dialogue']=dialogue_to_text(r['source'])
        for arm in arms:
            name='F_'+arm
            if name in d:continue
            qa=corrected[k] if arm=='corrected' else r['qa']
            prompt=stage2_input(r['source'],qa)
            if arm!='baseline':prompt+=CHECK_RULES
            pred,n=generate(prompt)
            d[name]={'prediction':pred,'scores':{**score(pred,r['reference']),
                     'generated_tokens':n,'hit_max_new_tokens':n>=SUMMARY_MAX}}
            done[k]=d
            render_results(path,cfg,rows,done)
            print(f'[{index}/{len(pending)}] ID={k} {arm}: ROUGE-L={d[name]["scores"]["rouge_l_f1"]:.4f}',flush=True)
    print(json.dumps(render_results(path,cfg,rows,done),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
