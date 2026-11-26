import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

model_path = "models/Qwen3-4B-Instruct-2507"

# 1. 配置4-bit量化
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. 加载量化后的Qwen
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quant_config,
    dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# 3. 定义LoRA训练参数
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

# 4. 为Qwen添加可训练的LoRA矩阵
model = get_peft_model(model, lora_config)

# 5. 检查训练参数与显存
model.print_trainable_parameters()

print("4-bit loaded:", model.is_loaded_in_4bit)
print(
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)

print("QLoRA setup successful!")
