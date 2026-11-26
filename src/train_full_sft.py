"""CSDS AgentSumm | Qwen3-4B QLoRA baseline SFT (1 epoch).

Run from /root/workspace/csds-project after uploading this file to src/.
Uses data/train_sft.jsonl created earlier. Model files remain frozen.
"""
import json
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

MODEL_DIR = "models/Qwen3-4B-Instruct-2507"
TRAIN_FILE = Path("data/train_sft.jsonl")
OUTPUT_DIR = Path("outputs/qlora_sft")
MAX_TOKENS = 1536

# 1) Tokenize each dialogue + reference summary. Ignore the prompt in loss.
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
tokenizer.padding_side = "right"
examples = []
with TRAIN_FILE.open(encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)
        user_msg = [{"role": "user", "content": sample["prompt"]}]
        full_msg = user_msg + [{"role": "assistant", "content": sample["target"]}]
        prompt_ids = tokenizer.apply_chat_template(
            user_msg, tokenize=True, add_generation_prompt=True
        )
        full_ids = tokenizer.apply_chat_template(
            full_msg, tokenize=True, add_generation_prompt=False
        )
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Chat template prefix mismatch")
        if len(full_ids) > MAX_TOKENS:
            raise ValueError(
                f"DialogueID={sample['DialogueID']} has {len(full_ids)} tokens; "
                f"MAX_TOKENS={MAX_TOKENS}"
            )
        examples.append({
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
        })
print(f"Prepared {len(examples)} training examples; first length={len(examples[0]['input_ids'])}", flush=True)

# 2) Frozen 4-bit Qwen + trainable LoRA matrices.
quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    quantization_config=quantization,
    dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model, gradient_checkpointing_kwargs={"use_reentrant": False}
)
model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
))
model.print_trainable_parameters()

# 3) One pass through all 9,101 training samples. Save limited checkpoints.
args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    optim="adamw_torch",
    max_grad_norm=1.0,
    logging_steps=20,
    logging_first_step=True,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    eval_strategy="no",
    report_to="none",
    seed=42,
    remove_unused_columns=False,
    dataloader_num_workers=0,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=examples,
    data_collator=default_data_collator,
)
last_ckpt = get_last_checkpoint(str(OUTPUT_DIR)) if OUTPUT_DIR.is_dir() else None
if last_ckpt:
    print(f"Resuming from {last_ckpt}", flush=True)
trainer.train(resume_from_checkpoint=last_ckpt)

final_path = OUTPUT_DIR / "final_adapter"
trainer.save_model(str(final_path))
tokenizer.save_pretrained(str(final_path))
print(f"Done. LoRA adapter saved in {final_path}", flush=True)
