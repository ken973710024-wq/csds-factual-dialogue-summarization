"""Experiment 3 diagnostic: identify WHY QA extraction failed (NO training).

Upload to /root/workspace/csds-project/src/ and run from project root:
    python src/validate_qa_prompt_diagnostic.py --show-prompts
    python src/validate_qa_prompt_diagnostic.py

Uses the SAME first five dev dialogues from outputs/validate_qa_aware_dev.json.
Compares 2 model states x 3 prompt variants. Reuses the already-run
LoRA+original condition, so only five new generations per dialogue.
Never uses gold AgentSumm/QA annotations as model input.
"""
import argparse
import contextlib
import json
import re
from pathlib import Path

MODEL_DIR = Path("models/Qwen3-4B-Instruct-2507")
ADAPTER_DIR = Path("outputs/qlora_sft/final_adapter")
PRIOR_FILE = Path("outputs/validate_qa_aware_dev.json")
OUTPUT_FILE = Path("outputs/validate_qa_prompt_diagnostic.json")
SUMMARY_FILE = Path("outputs/validate_qa_prompt_diagnostic_summary.json")
PROMPTS_FILE = Path("outputs/validate_qa_prompt_diagnostic_prompts.txt")
MAX_NEW_TOKENS = 768
REQUIRED = ("category", "question", "answer", "status", "evidence")
STATUSES = {"信息答复", "已完成", "承诺后续处理", "处理中", "未确认"}

# Copied word-for-word from the existing validate_qa_aware.py EXTRACT_PROMPT.
ORIGINAL_PROMPT = """你是客服对话的结构化事实抽取器。暂时不要生成最终摘要。
逐项识别对话中所有重要的、互不重复的具体问题和客服主动给出的重要处理方案。
不能只按宽泛类别（如物流、售后）合并：同一类别中不同问题要分别记录。
按对话中的最终答复确认结果；存在前后矛盾时保留冲突，不要擅自解决。
每一项输出：
- category: 业务分类（如物流、发票、退款、售后、账号、商品、其他）
- question: 用户具体咨询的问题；如果是客服主动提供的方案，写“客服主动说明：...”
- answer: 客服明确提供的答复、关键条件和限制；没有明确答复写“未明确答复”
- status: 只能是“信息答复”“已完成”“承诺后续处理”“处理中”“未确认”之一
- evidence: 可追溯的客服原话摘录，不要编造
只根据对话事实填写；不要把用户要求或预计事项写成已完成。
仅返回 JSON 数组，每条记录为一个对象，不加 Markdown 或额外说明。

【原始对话】
"""

# Purely SYNTHETIC example, not a test/validation answer and not from CSDS.
DEMONSTRATION = """【示范对话（仅演示输出格式，不是待分析的数据）】
用户：我的订单已经取消了吗？
客服：还没取消，我明天下午帮您处理。
用户：运费是多少？
客服：运费是8元。
【示范的正确 QA 结构】
[
  {"category":"售后","question":"订单是否已经取消？","answer":"尚未取消，客服承诺明天下午处理。","status":"承诺后续处理","evidence":"还没取消，我明天下午帮您处理。"},
  {"category":"物流","question":"运费是多少？","answer":"运费为8元。","status":"信息答复","evidence":"运费是8元。"}
]
【以下是新的对话，只分析新的对话，不要复制示范内容】
"""

# Revised INSTRUCTIONS ONLY. No worked example in this condition.
CLEAR_PROMPT = """任务：抽取客服对话中的问答事实。当前只执行 QA 抽取，禁止生成最终摘要。

处理步骤：
1. 找出每一个不同的具体用户问题。即使同属物流类，不同问题也应分开。
2. 对每个问题找到客服实际给出的答复；必要时也记录客服主动给出的重要处理方案。
3. 区分信息答复、已完成、承诺后续处理、处理中、未确认。不能把「会处理」当作「处理完成」。
4. 为每条答复复制一句原始客服话语作为 evidence，必须是原始对话中的原文片段。
5. 只输出一个可以直接被 json.loads() 解析的 JSON 数组。

格式规则：每个数组元素必须是对象，包含五个字符串字段：
category：业务类别；question：具体问题；answer：对应的客服答复；
status：只能从「信息答复」「已完成」「承诺后续处理」「处理中」「未确认」中选择；
evidence：客服原话中的证据片段。
不得输出编号列表、Markdown 代码围栏、引言、注释、解释或最终摘要。
输出的第一个字符必须是 [，最后一个字符必须是 ]。
若未找到明确答复，answer 写「未明确答复」，status 写「未确认」。
不能使用测试对话外的信息。

【原始对话】
"""

CONDITIONS = (
    ("lora_original", "lora", "original"),  # reused from old 5-case run
    ("base_original", "base", "original"),
    ("base_example", "base", "example"),
    ("lora_example", "lora", "example"),
    ("base_clear", "base", "clear"),
    ("lora_clear", "lora", "clear"),
)


def prompt_for(variant, dialogue):
    if variant == "original":
        return ORIGINAL_PROMPT + dialogue
    if variant == "example":
        # Other instructions remain the same; we add exactly one demonstration.
        before, ending = ORIGINAL_PROMPT.rsplit("【原始对话】\n", 1)
        assert ending == ""
        return before + DEMONSTRATION + "\n【原始对话】\n" + dialogue
    if variant == "clear":
        return CLEAR_PROMPT + dialogue
    raise ValueError(variant)


def parse_json(raw):
    stripped = raw.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    candidates = [stripped]
    start, end = stripped.find("["), stripped.rfind("]")
    if start >= 0 and end > start:
        candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(value, list) and all(isinstance(x, dict) for x in value):
            return value
    return None


def evaluate(raw, dialogue, num_tokens):
    parsed = parse_json(raw)
    valid = parsed is not None and all(
        all(isinstance(item.get(k), str) and item[k].strip() for k in REQUIRED)
        and item["status"] in STATUSES
        for item in parsed
    ) and len(parsed) > 0
    evidence_ok = bool(valid) and all(item["evidence"] in dialogue for item in parsed)
    return {
        "raw": raw,
        "json_parsed": parsed is not None,
        "schema_valid": bool(valid),
        "evidence_exact_match": evidence_ok,
        "items": parsed,
        "qa_count": len(parsed) if parsed is not None else 0,
        "generated_tokens": num_tokens,
        "hit_max_new_tokens": num_tokens >= MAX_NEW_TOKENS,
    }


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def summarise(results):
    return {
        "evaluated_dialogues": len(results),
        "conditions": {
            name: {
                "evaluated": len([r for r in results if name in r["conditions"]]),
                "json_parsed": sum(r["conditions"][name]["json_parsed"] for r in results if name in r["conditions"]),
                "schema_valid": sum(r["conditions"][name]["schema_valid"] for r in results if name in r["conditions"]),
                "evidence_exact_match": sum(r["conditions"][name]["evidence_exact_match"] for r in results if name in r["conditions"]),
                "hit_max_new_tokens": sum(r["conditions"][name]["hit_max_new_tokens"] for r in results if name in r["conditions"]),
            }
            for name, _, _ in CONDITIONS
        },
        "warning": "JSON/schema pass is NOT proof that QA category, answers and statuses are semantically correct. Manually inspect items against dialogue.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Number of first saved dev examples (1..5)")
    parser.add_argument("--show-prompts", action="store_true", help="Print and save EXACT readable inputs; no GPU needed")
    args = parser.parse_args()
    if not 1 <= args.limit <= 5:
        parser.error("--limit must be 1..5")
    if not PRIOR_FILE.exists():
        raise FileNotFoundError(f"Missing {PRIOR_FILE}; upload the existing 5-case dev JSON into outputs/")
    old = json.loads(PRIOR_FILE.read_text(encoding="utf-8"))["results"]
    if len(old) < args.limit:
        raise ValueError("Prior dev output contains fewer examples than requested")
    old = old[:args.limit]
    if args.show_prompts:
        pieces = []
        for variant in ("original", "example", "clear"):
            pieces.append("=" * 18 + f" {variant.upper()} (same DialogueID={old[0]['DialogueID']}) " + "=" * 18)
            pieces.append(prompt_for(variant, old[0]["dialogue"]))
        PROMPTS_FILE.parent.mkdir(exist_ok=True, parents=True)
        PROMPTS_FILE.write_text("\n\n".join(pieces), encoding="utf-8")
        print("\n\n".join(pieces))
        print(f"\nSaved visible prompts: {PROMPTS_FILE}")
        print("Note: at inference the tokenizer also wraps this user content in the Qwen chat template.")
        return

    # Identical samples and identical Original Prompt to previously failed QA test.
    metadata = {
        "sample_indices": [x["index"] for x in old],
        "dialogue_ids": [x["DialogueID"] for x in old],
        "conditions": [name for name, _, _ in CONDITIONS],
        "max_new_tokens": MAX_NEW_TOKENS,
        "base_quantization": "4-bit NF4 double quant, bf16 compute",
        "original_prompt": "verbatim from validate_qa_aware.py",
        "example_source": "fully synthetic, NOT validation labels",
        "gold_used_in_prompt": False,
        "cached_condition": "lora_original reused from outputs/validate_qa_aware_dev.json",
    }
    if OUTPUT_FILE.exists():
        saved = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        if saved["metadata"] != metadata:
            raise ValueError("Existing results use different settings; refusing to overwrite")
        results = saved["results"]
    else:
        results = [
            {"index": x["index"], "DialogueID": x["DialogueID"], "dialogue": x["dialogue"], "conditions": {
                "lora_original": evaluate(x["stage1_qa_raw"], x["dialogue"], x["stage1_generated_tokens"])
            }}
            for x in old
        ]
        save(OUTPUT_FILE, {"metadata": metadata, "results": results})

    pending = [(r, name, mode, variant) for r in results for name, mode, variant in CONDITIONS if name not in r["conditions"]]
    if not pending:
        print("Already completed.\n" + json.dumps(summarise(results), ensure_ascii=False, indent=2))
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available. Execute on the RTX 4090 server.")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), quantization_config=quant, dtype=torch.bfloat16,
        device_map={"": 0}, local_files_only=True,
    )
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=False)
    model.eval()
    torch.manual_seed(42)

    for count, (r, name, mode, variant) in enumerate(pending, 1):
        prompt_content = prompt_for(variant, r["dialogue"])
        messages = [{"role": "user", "content": prompt_content}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda:0")
        context = model.disable_adapter() if mode == "base" else contextlib.nullcontext()
        with context:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
        output_ids = generated[0, inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        r["conditions"][name] = evaluate(raw, r["dialogue"], len(output_ids))
        save(OUTPUT_FILE, {"metadata": metadata, "results": results})
        save(SUMMARY_FILE, summarise(results))
        s = r["conditions"][name]
        print(f"[{count}/{len(pending)}] DialogueID={r['DialogueID']} {name}: "
              f"JSON={'OK' if s['json_parsed'] else 'FAIL'}, "
              f"schema={'OK' if s['schema_valid'] else 'FAIL'}, "
              f"evidence={'OK' if s['evidence_exact_match'] else 'FAIL'}", flush=True)

    print("\n" + json.dumps(summarise(results), ensure_ascii=False, indent=2))
    print(f"Details: {OUTPUT_FILE}\nSummary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
