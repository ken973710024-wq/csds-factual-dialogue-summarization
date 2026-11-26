#!/usr/bin/env python3
"""Matched 4-bit QLoRA SFT: --arm direct and --arm aware, independent adapters.

Same paired samples, same summary target, hyperparameters, seed, architecture.
Only difference: the aware prompt contains Stage-1 predicted QA.

python src/train_summary_qa_conditioned.py --tag pilot512 --arm direct --check
python src/train_summary_qa_conditioned.py --tag pilot512 --arm direct
python src/train_summary_qa_conditioned.py --tag pilot512 --arm aware
"""
import argparse,json,hashlib
from pathlib import Path
from qa_summary_study_common import BASE_MODEL, atomic_json, jsonl, sha_file


def prepare_tokenized(tokenizer, records, cap):
    examples=[]
    for i,row in enumerate(records):
        user=[{'role':'user','content':row['prompt']}]
        full=user+[{'role':'assistant','content':row['target']}]
        prefix=tokenizer.apply_chat_template(user,tokenize=True,add_generation_prompt=True)
        ids=tokenizer.apply_chat_template(full,tokenize=True,add_generation_prompt=False)
        if ids[:len(prefix)]!=prefix:
            raise ValueError(f'Chat template mismatch {row["DialogueID"]}')
        if len(ids)>cap:
            raise ValueError(f'Overlength after manifest filtering {row["DialogueID"]}: {len(ids)}>{cap}')
        if len(ids)<=len(prefix):
            raise ValueError('No trainable target tokens')
        examples.append({'input_ids':ids,'attention_mask':[1]*len(ids),
                         'labels':[-100]*len(prefix)+ids[len(prefix):]})
    return examples


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag',default='pilot512')
    ap.add_argument('--arm',choices=['direct','aware'],required=True)
    ap.add_argument('--lr',type=float,default=1e-4)
    ap.add_argument('--epochs',type=float,default=1.0)
    ap.add_argument('--grad-accum',type=int,default=8)
    ap.add_argument('--check',action='store_true')
    args=ap.parse_args()
    data_dir=Path('data/qa_stage2_research')/args.tag
    paired=json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    if paired.get('version')!='qa_stage2_paired_v1' or paired.get('tag')!=args.tag:
        raise ValueError('Wrong paired training manifest')
    direct_path=data_dir/'direct.jsonl'
    aware_path=data_dir/'aware.jsonl'
    for path,name in [(direct_path,'direct'),(aware_path,'aware')]:
        if sha_file(path)!=paired[name+'_sha256']:
            raise ValueError(f'Training source changed after preparation: {path}')
    # This catches accidental differences between arms other than prompts.
    direct=list(jsonl(direct_path))
    aware=list(jsonl(aware_path))
    if len(direct)!=paired['paired_retained'] or len(aware)!=len(direct):
        raise ValueError('Paired training count mismatch')
    for a,b in zip(direct,aware):
        if (str(a['DialogueID']),a['target'])!=(str(b['DialogueID']),b['target']):
            raise ValueError('The two experiments no longer use identical labels')
    print(f'Arm={args.arm}, tag={args.tag}, n={len(direct)}, same supervision confirmed.')
    path=direct_path if args.arm=='direct' else aware_path
    config={'version':'stage2_qa_matched_train_v1','tag':args.tag,'arm':args.arm,
            'dataset_manifest_sha256':sha_file(data_dir/'manifest.json'),
            'dataset_sha256':sha_file(path),'base_model':str(BASE_MODEL),
            'seed':42,'lr':args.lr,'epochs':args.epochs,'gradient_accumulation':args.grad_accum,
            'batch_size':1,'lora_r':16,'lora_alpha':32,'lora_dropout':0.05,
            'max_seq_tokens':paired['max_seq_tokens']}
    out=Path('outputs')/f'qlora_summary_matched_{args.arm}_{args.tag}'
    final=out/'final_adapter'
    if args.check:
        print(json.dumps(config,ensure_ascii=False,indent=2))
        for req in [BASE_MODEL/'config.json',BASE_MODEL/'tokenizer_config.json']:
            print('OK' if req.exists() else 'MISSING',req)
            if not req.exists(): raise FileNotFoundError(req)
        return
    if final.exists():
        raise FileExistsError(f'{final} already exists; refusing to overwrite completed model')
    out.mkdir(parents=True,exist_ok=True)
    train_manifest=out/'training_manifest.json'
    if train_manifest.exists():
        if json.loads(train_manifest.read_text(encoding='utf-8'))!=config:
            raise ValueError('Existing checkpoints have different training conditions')
    else:
        atomic_json(train_manifest,config)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,Trainer,TrainingArguments
    from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
    from transformers.trainer_utils import get_last_checkpoint
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 CUDA required')
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(str(BASE_MODEL),local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    records=direct if args.arm=='direct' else aware
    dataset=prepare_tokenized(tokenizer,records,paired['max_seq_tokens'])
    def collate(batch):
        m=max(len(x['input_ids']) for x in batch)
        pad=tokenizer.pad_token_id
        return {
            'input_ids':torch.tensor([x['input_ids']+[pad]*(m-len(x['input_ids'])) for x in batch],dtype=torch.long),
            'attention_mask':torch.tensor([x['attention_mask']+[0]*(m-len(x['attention_mask'])) for x in batch],dtype=torch.long),
            'labels':torch.tensor([x['labels']+[-100]*(m-len(x['labels'])) for x in batch],dtype=torch.long),
        }
    cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                           bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained(str(BASE_MODEL),quantization_config=cfg,
                                              dtype=torch.bfloat16,device_map={'':0},local_files_only=True)
    base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,gradient_checkpointing_kwargs={'use_reentrant':False})
    model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
        bias='none',task_type='CAUSAL_LM',target_modules=[
            'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    model.print_trainable_parameters()
    training_args=TrainingArguments(
        output_dir=str(out),num_train_epochs=args.epochs,per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,learning_rate=args.lr,
        lr_scheduler_type='cosine',warmup_ratio=0.03,bf16=True,
        optim='adamw_torch',max_grad_norm=1.0,logging_steps=20,
        logging_first_step=True,save_strategy='steps',save_steps=200,
        save_total_limit=2,eval_strategy='no',report_to='none',seed=42,data_seed=42,
        remove_unused_columns=False,dataloader_num_workers=0)
    trainer=Trainer(model=model,args=training_args,train_dataset=dataset,data_collator=collate)
    ckpt=get_last_checkpoint(str(out))
    if ckpt: print('Resuming from',ckpt,flush=True)
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f'Completed independent adapter: {final}',flush=True)

if __name__=='__main__': main()
