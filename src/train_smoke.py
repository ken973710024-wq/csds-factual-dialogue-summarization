import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

torch.manual_seed(42)

MODEL_PATH = "models/Qwen3-4B-Instruct-2507"
MAX_LENGTH = 1536
GRAD_ACCUM = 2

# 1. 读取4条训练数据
with open("data/train_sft.jsonl", encoding="utf-8") as f:
    samples = [json.loads(next(f)) for _ in range(4)]

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, local_files_only=True
)

# 2. 加载4-bit基础模型
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

# 3. 添加LoRA可训练矩阵
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.train()

model.print_trainable_parameters()

# 4. 定义优化器，只更新LoRA参数
optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-4
)

optimizer.zero_grad()
device = next(model.parameters()).device

# 5. 开始训练
for i, sample in enumerate(samples):

    user_message = [
        {"role": "user", "content": sample["prompt"]}
    ]

    full_messages = user_message + [
        {"role": "assistant", "content": sample["target"]}
    ]

    # 输入部分：只包含用户Prompt和assistant起始标记
    prompt_ids = tokenizer.apply_chat_template(
        user_message,
        tokenize=True,
        add_generation_prompt=True
    )

    # 完整训练序列：Prompt + 人工摘要
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False
    )

    assert full_ids[:len(prompt_ids)] == prompt_ids, (
        "Prompt与完整训练序列前缀不一致"
    )

    assert len(full_ids) <= MAX_LENGTH, (
        f"样本长度{len(full_ids)}超过{MAX_LENGTH}"
    )

    input_ids = torch.tensor(
        [full_ids], dtype=torch.long, device=device
    )

    # Prompt不参与Loss，只有人工摘要参与
    labels = torch.tensor(
        [[-100] * len(prompt_ids) +
         full_ids[len(prompt_ids):]],
        dtype=torch.long,
        device=device
    )

    assert (labels != -100).sum().item() > 0

    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels
    )

    loss = outputs.loss

    if not torch.isfinite(loss).item():
        raise RuntimeError("Loss出现NaN或Inf")

    (loss / GRAD_ACCUM).backward()

    print(
        f"Sample {i+1}/4 | "
        f"tokens={len(full_ids)} | "
        f"loss={loss.item():.4f}",
        flush=True
    )

    if (i + 1) % GRAD_ACCUM == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            max_norm=1.0
        )

        optimizer.step()
        optimizer.zero_grad()

        print(
            f"Optimizer step {(i+1)//GRAD_ACCUM} | "
            f"gradient norm={float(grad_norm):.4f}",
            flush=True
        )

print(
    "Peak GPU memory:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB"
)

print("Training smoke test successful!")
