#!/usr/bin/env python3
"""Create *human* evidence-backed factuality audits and score completed judgments.

Stage1: outputs/validate_qa_lora_dev50.json, with QA Base vs QA-LoRA blinded.
Stage2: outputs/eval_factuality_F_<tag>_<mode>[_arms].json, optional multiple files.
Gold QA and gold summary are NEVER used as human fact-bank seeds.
"""
from __future__ import annotations
import argparse,csv,json,random,re
from collections import defaultdict
from pathlib import Path

FACT_COLS=['dialogue_id','fact_id','topic_id','fact_text','source_ids','key_stage1','key_summary','final_state','notes']
CLAIM_COLS=['dialogue_id','arm_blind','output_index','claim_id','claim_text','atomic_checked','supported',
            'object_correct','speaker_correct','state_correct','is_state_claim','matched_fact_ids','issue_type','notes']

def readcsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def writecsv(path,fields,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def protect_write(path,callback):
    if path.exists():
        print('PRESERVED existing:',path)
    else:callback();print('Created:',path)

def speaker_text(prompt):
    return prompt.split('【带编号的客服对话】',1)[-1].strip()

def prepare_stage1(outroot,source):
    rows=json.loads(source.read_text(encoding='utf-8'))['results']
    if len(rows)!=50:raise ValueError('Expected 50 stage1 dev dialogues')
    rng=random.Random(20261123)
    tasks=[];claims=[];blindmap={}
    for r in rows:
        k=str(r['DialogueID']);labels=['X','Y'];rng.shuffle(labels)
        mapping=dict(zip(('base','qa_lora'),labels));blindmap[k]=mapping
        tasks+=['## DialogueID '+k,'\n### Source dialogue with speaker IDs\n','```text',speaker_text(r['prompt']),'```','']
        for origin in ('base','qa_lora'):
            blind=mapping[origin];items=r[origin].get('parsed') or []
            tasks+=['### Anonymous QA '+blind, '```json',json.dumps(items,ensure_ascii=False,indent=2),'```','']
            if not items:
                claims.append(dict(zip(CLAIM_COLS,[k,blind,'-1',blind+'-invalid','<invalid QA output>','','','','','','','','invalid_schema',''])))
            for i,item in enumerate(items):
                claims.append({'dialogue_id':k,'arm_blind':blind,'output_index':i,
                     'claim_id':f'{blind}-{i}-1','claim_text':item['answer'],
                     'atomic_checked':'','supported':'','object_correct':'','speaker_correct':'',
                     'state_correct':'','is_state_claim':'','matched_fact_ids':'','issue_type':'','notes':''})
        tasks+=['Mark ORIGINAL dialogue facts before judging X/Y. Split multi-fact answers into atomic claim rows in the CSV.','---','']
    outroot.mkdir(parents=True,exist_ok=True)
    protect_write(outroot/'source_packets.md',lambda:(outroot/'source_packets.md').write_text('\n'.join(tasks),encoding='utf-8'))
    protect_write(outroot/'fact_bank.csv',lambda:writecsv(outroot/'fact_bank.csv',FACT_COLS,[]))
    protect_write(outroot/'claims_to_judge.csv',lambda:writecsv(outroot/'claims_to_judge.csv',CLAIM_COLS,claims))
    protect_write(outroot/'_unblind_after_review.json',lambda:(outroot/'_unblind_after_review.json').write_text(json.dumps(blindmap,ensure_ascii=False,indent=2),encoding='utf-8'))
    print('Stage1 cases:',len(rows),'Initial QA answer entries:',len(claims))

def prepare_stage2(outroot,sources):
    allrows={}
    for f in sources:
        for r in json.loads(f.read_text(encoding='utf-8'))['results']:
            k=str(r['DialogueID']);allrows.setdefault(k,{}).update(r)
    outroot.mkdir(parents=True,exist_ok=True)
    rng=random.Random(20261123)
    tasks=[];claims=[];blindmap={}
    for k,r in allrows.items():
        arms=[a for a in ('A_old','F_baseline','F_checked','F_corrected') if a in r]
        if not arms:continue
        labels=['W','X','Y','Z'][:len(arms)];rng.shuffle(labels)
        mapping=dict(zip(arms,labels));blindmap[k]=mapping
        tasks+=['## DialogueID '+k,'\n### Source dialogue (NO reference shown)\n','```text',r['dialogue'],'```','']
        for arm in arms:
            blind=mapping[arm];pred=r[arm]['prediction']
            tasks+=['### Anonymous summary '+blind,'```text',pred,'```','']
            fragments=[p.strip() for p in re.split(r'(?m)^\s*\d+[.、．]\s*',pred) if p.strip()]
            if not fragments:fragments=[pred]
            for i,p in enumerate(fragments):
                claims.append({'dialogue_id':k,'arm_blind':blind,'output_index':i,
                   'claim_id':f'{blind}-{i}-1','claim_text':p,'atomic_checked':'',
                   'supported':'','object_correct':'','speaker_correct':'','state_correct':'',
                   'is_state_claim':'','matched_fact_ids':'','issue_type':'','notes':''})
        tasks+=['Manually split each summary line into atomic claims if multiple facts.','---','']
    protect_write(outroot/'source_packets.md',lambda:(outroot/'source_packets.md').write_text('\n'.join(tasks),encoding='utf-8'))
    protect_write(outroot/'fact_bank.csv',lambda:writecsv(outroot/'fact_bank.csv',FACT_COLS,[]))
    protect_write(outroot/'claims_to_judge.csv',lambda:writecsv(outroot/'claims_to_judge.csv',CLAIM_COLS,claims))
    protect_write(outroot/'_unblind_after_review.json',lambda:(outroot/'_unblind_after_review.json').write_text(json.dumps(blindmap,ensure_ascii=False,indent=2),encoding='utf-8'))
    print('Stage2 cases:',len(allrows),'Initial summary lines:',len(claims))

def binary(v,why):
    if str(v).strip() not in ('0','1'):raise ValueError('Expected binary 0/1 for '+why)
    return int(v)

def score(outroot,kind,allow_partial):
    facts=readcsv(outroot/'fact_bank.csv');claims=readcsv(outroot/'claims_to_judge.csv')
    mapping=json.loads((outroot/'_unblind_after_review.json').read_text(encoding='utf-8'))
    if not facts: raise ValueError('fact_bank.csv empty; annotate dialogue facts first')
    factmap=defaultdict(dict)
    for f in facts:
        k=f['dialogue_id']; fid=f['fact_id'].strip()
        if not fid or fid in factmap[k]:raise ValueError('missing/duplicate fact_id: '+k)
        for col in ('key_stage1','key_summary','final_state'):binary(f[col],k+'/'+fid+'/'+col)
        if not f['source_ids'].strip():raise ValueError('Fact missing source evidence ids: '+k+'/'+fid)
        factmap[k][fid]=f
    per_arm=defaultdict(lambda:defaultdict(lambda:{'claims':0,'correct':0,'covered':set(),
                                                 'wrong_state':0,'status_claims':0}))
    pending=[]
    for c in claims:
        k=c['dialogue_id'];arm=c['arm_blind'];idx=c['claim_id']
        if k not in mapping or arm not in mapping[k].values():raise ValueError('unknown blinded arm: '+k+'/'+arm)
        if not factmap[k]:pending.append(k+':missing_facts');continue
        if not c['claim_text'].strip():pending.append(k+'/'+idx+':empty');continue
        keys=['atomic_checked','supported','object_correct','speaker_correct','state_correct','is_state_claim']
        if any(c[x].strip() not in ('0','1') for x in keys):
            pending.append(k+'/'+idx+':unreviewed_or_multifact');continue
        if binary(c['atomic_checked'],k+'/'+idx)!=1:
            pending.append(k+'/'+idx+':split_into_atomic_claims');continue
        scored=per_arm[k][arm];scored['claims']+=1
        good=all(binary(c[x],k+'/'+idx+'/'+x) for x in ('supported','object_correct','speaker_correct','state_correct'))
        if good:
            scored['correct']+=1
            match=[x.strip() for x in c['matched_fact_ids'].split(';') if x.strip()]
            unknown=set(match)-set(factmap[k])
            if unknown:raise ValueError('Unknown matched fact IDs: '+repr(unknown))
            scored['covered'].update(match)
        if binary(c['is_state_claim'],k+'/'+idx):
            scored['status_claims']+=1
            if not good or not binary(c['state_correct'],k+'/'+idx):scored['wrong_state']+=1
    if pending and not allow_partial:
        raise ValueError(f'{len(pending)} unreviewed claims. First examples: '+repr(pending[:8]))
    summary=defaultdict(lambda:{'correct':0,'claims':0,'covered':0,'keys':0,'wrong_state_dialogues':0,
                             'status_dialogues':0,'missing_state_dialogues':0,'dialogues':0})
    labels=set()
    for k,arms in mapping.items():
        if k not in factmap:
            if not allow_partial:raise ValueError('Unannotated case: '+k)
            continue
        desired={f['fact_id'] for f in factmap[k].values() if f['key_stage1' if kind=='stage1' else 'key_summary']=='1'}
        state_desired={f['fact_id'] for f in factmap[k].values() if f['final_state']=='1'}
        for true,anon in arms.items():
            key=true if kind=='stage2' else true
            scores=per_arm[k][anon]
            if not scores['claims']:
                if not allow_partial:raise ValueError('No reviewed claims: '+k+'/'+anon)
                continue
            agg=summary[key];labels.add(key)
            agg['correct']+=scores['correct'];agg['claims']+=scores['claims'];agg['keys']+=len(desired)
            agg['covered']+=len(desired & scores['covered']);agg['dialogues']+=1
            if state_desired:
                agg['status_dialogues']+=1
                agg['wrong_state_dialogues']+=int(scores['wrong_state']>0)
                agg['missing_state_dialogues']+=int(not state_desired.issubset(scores['covered']))
    report={'note':'Human source-grounded fact-level scoring; strict supported+entity+speaker+state.',
            'caveat':'If a correct claim is not in the fact bank, add its source-grounded fact before scoring.',
            'pending_claims':len(pending),'systems':{}}
    for name,a in summary.items():
        report['systems'][name]={**a,
           'fact_support_precision':round(a['correct']/a['claims'],4) if a['claims'] else None,
           'key_fact_recall':round(a['covered']/a['keys'],4) if a['keys'] else None,
           'final_status_wrong':f'{a["wrong_state_dialogues"]}/{a["status_dialogues"]}',
           'final_status_missing':f'{a["missing_state_dialogues"]}/{a["status_dialogues"]}'}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    (outroot/'human_fact_scores.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind',choices=['stage1','stage2'],required=True)
    ap.add_argument('--action',choices=['prepare','score'],required=True)
    ap.add_argument('--source',nargs='*',type=Path)
    ap.add_argument('--outdir',type=Path)
    ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args()
    out=args.outdir or Path('data/qa_stage2_research/factuality_study')/('stage1_dev50' if args.kind=='stage1' else 'stage2_review')
    if args.action=='prepare':
        if args.kind=='stage1':
            source=(args.source or [Path('outputs/validate_qa_lora_dev50.json')])[0]
            prepare_stage1(out,source)
        else:
            if not args.source:ap.error('stage2 prepare requires --source results.json')
            prepare_stage2(out,args.source)
    else:score(out,args.kind,args.allow_partial)

if __name__=='__main__':main()
