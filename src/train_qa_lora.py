#!/usr/bin/env python3
"""Train a NEW Stage-1 CSDS QA-extraction adapter with Qwen3-4B QLoRA.

Project-root usage (after checking token lengths):
  python src/train_qa_lora.py --max-tokens 2048 --dry-run
  python src/train_qa_lora.py --max-tokens 2048

Never silently truncates labels. By default, ANY overlength record aborts training
before loading the GPU model. Use explicit --overlength-policy drop only after
reviewing the token-length report; every dropped DialogueID is logged.

Inputs:  data/qa_sft_train.jsonl, data/qa_sft_dev.jsonl
Outputs: outputs/qlora_qa_stage1/final_adapter (does not touch Summary-LoRA).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'


def load_examples(path,tokenizer,max_tokens,policy):
    examples=[]; too_long=[]; seen=set(); total_qa=0
    with path.open(encoding='utf-8') as f:
        for line_num,line in enumerate(f,1):
            row=json.loads(line)
            d_id=str(row['DialogueID'])
            if d_id in seen: raise ValueError(f'{path}:{line_num}: duplicate DialogueID={d_id}')
            seen.add(d_id)
            target=json.loads(row['target'])
            if not isinstance(target,list) or not target:
                raise ValueError(f'{path}:{line_num}: QA target must be a nonempty JSON array')
            total_qa+=len(target)
            messages=[{'role':'user','content':row['prompt']}]
            prompt_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
            full_ids=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':row['target']}],tokenize=True,add_generation_prompt=False)
            if full_ids[:len(prompt_ids)]!=prompt_ids:
                raise ValueError(f'{path}:{line_num}: chat-template prefix mismatch')
            if len(full_ids)==len(prompt_ids):
                raise ValueError(f'{path}:{line_num}: empty target tokens')
            if len(full_ids)>max_tokens:
                too_long.append({'DialogueID':row['DialogueID'],'total_tokens':len(full_ids),'prompt_tokens':len(prompt_ids),'qa_count':len(target)})
                continue
            examples.append({'input_ids':full_ids,'attention_mask':[1]*len(full_ids),
                             'labels':[-100]*len(prompt_ids)+full_ids[len(prompt_ids):],
                             '_dialogue_id':row['DialogueID']})
    if too_long and policy=='error':
        biggest=sorted(too_long,key=lambda x:x['total_tokens'],reverse=True)[:5]
        raise ValueError(f'{path}: {len(too_long)} dialogues exceed {max_tokens} tokens. '
                         f'No truncation performed. Examples: {biggest}. '
                         f'Check token-length report; change --max-tokens after GPU memory assessment, '
                         f'or explicitly use --overlength-policy drop.')
    if not examples: raise ValueError(f'{path}: zero examples after length checks')
    # Metadata is kept separately, not passed through the model/collator.
    for x in examples: x.pop('_dialogue_id')
    return examples,too_long,total_qa


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',default='models/Qwen3-4B-Instruct-2507')
    ap.add_argument('--train',default='data/qa_sft_train.jsonl')
    ap.add_argument('--dev',default='data/qa_sft_dev.jsonl')
    ap.add_argument('--output',default='outputs/qlora_qa_stage1')
    ap.add_argument('--max-tokens',type=int,default=2048)
    ap.add_argument('--overlength-policy',choices=['error','drop'],default='error')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--epochs',type=float,default=1.0)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--grad-accum',type=int,default=8)
    args=ap.parse_args()
    if args.max_tokens<512: ap.error('--max-tokens must be >=512')
    if args.epochs<=0 or args.grad_accum<1: ap.error('invalid epochs or grad-accum')

    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    tokenizer.padding_side='right'
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    train,train_dropped,train_total_qa=load_examples(Path(args.train),tokenizer,args.max_tokens,args.overlength_policy)
    dev,dev_dropped,dev_total_qa=load_examples(Path(args.dev),tokenizer,args.max_tokens,args.overlength_policy)
    info={
        'model':args.model,'train_file':args.train,'dev_file':args.dev,
        'max_tokens':args.max_tokens,'overlength_policy':args.overlength_policy,
        'epochs':args.epochs,'learning_rate':args.lr,'gradient_accumulation':args.grad_accum,
        'train_dialogues_used':len(train),'dev_dialogues_used':len(dev),
        'train_total_qa_before_filter':train_total_qa,'dev_total_qa_before_filter':dev_total_qa,
        'train_dropped':train_dropped,'dev_dropped':dev_dropped,
        'schema':['category','question','answer','question_ids','evidence_ids'],
        'important':'Only existing CSDS QA labels supervised; object and status NOT supervised. Targets are never truncated.',
    }
    print(json.dumps({k:v for k,v in info.items() if k not in ('train_dropped','dev_dropped')},ensure_ascii=False,indent=2),flush=True)
    if args.dry_run:
        print('Dry run finished: NO GPU model loaded; NO training performed.',flush=True)
        return

    output=Path(args.output)
    adapter=output/'final_adapter'
    if adapter.exists():
        raise SystemExit(f'{adapter} already exists; refusing to overwrite completed adapter')
    output.mkdir(parents=True,exist_ok=True)
    manifest=output/'training_manifest.json'
    if manifest.is_file():
        prev=json.loads(manifest.read_text(encoding='utf-8'))
        comparable=['model','train_file','dev_file','max_tokens','overlength_policy','epochs','learning_rate','gradient_accumulation']
        if any(prev.get(k)!=info[k] for k in comparable):
            raise SystemExit('Existing checkpoint manifest uses different training settings: refusing unsafe resume')
    else:
        manifest.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')

    import torch
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM,BitsAndBytesConfig,TrainingArguments,Trainer
    from transformers.trainer_utils import get_last_checkpoint
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required. This script targets RTX 4090.')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA BF16 required for this configuration.')
    cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    model=AutoModelForCausalLM.from_pretrained(args.model,quantization_config=cfg,dtype=torch.bfloat16,device_map={'':0},local_files_only=True)
    model.config.use_cache=False
    model=prepare_model_for_kbit_training(model,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',
                              target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    model.print_trainable_parameters()

    def collate(batch):
        max_len=max(len(row['input_ids']) for row in batch)
        pad_id=tokenizer.pad_token_id
        return {
            'input_ids':torch.tensor([r['input_ids']+[pad_id]*(max_len-len(r['input_ids'])) for r in batch],dtype=torch.long),
            'attention_mask':torch.tensor([r['attention_mask']+[0]*(max_len-len(r['attention_mask'])) for r in batch],dtype=torch.long),
            'labels':torch.tensor([r['labels']+[-100]*(max_len-len(r['labels'])) for r in batch],dtype=torch.long),
        }
    ta=TrainingArguments(
        output_dir=str(output),num_train_epochs=args.epochs,per_device_train_batch_size=1,
        per_device_eval_batch_size=1,gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,lr_scheduler_type='cosine',warmup_ratio=0.03,
        bf16=True,optim='adamw_torch',max_grad_norm=1.0,
        logging_steps=20,logging_first_step=True,
        save_strategy='steps',save_steps=300,save_total_limit=2,
        eval_strategy='epoch',report_to='none',seed=20261122,
        remove_unused_columns=False,dataloader_num_workers=0,
    )
    trainer=Trainer(model=model,args=ta,train_dataset=train,eval_dataset=dev,data_collator=collate)
    last_ckpt=get_last_checkpoint(str(output))
    if last_ckpt: print(f'Resuming QA-LoRA from {last_ckpt}',flush=True)
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    print(f'Completed QA-LoRA training: {adapter}',flush=True)

if __name__=='__main__':
    main()
