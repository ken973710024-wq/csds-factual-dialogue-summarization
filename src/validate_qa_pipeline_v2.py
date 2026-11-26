"""Experiment 3: evidence-linked QA extraction -> LoRA summarisation (no training).

From /root/workspace/csds-project:
    python src/validate_qa_pipeline_v2.py --show-prompts
    python src/validate_qa_pipeline_v2.py --limit 5
    python src/validate_qa_pipeline_v2.py --limit 20

Input: outputs/validate_base_vs_lora_100.json (experiment 2's fixed 100 cases).
Same 20-case development split as validate_qa_aware.py; first five match the
previous QA diagnostic. Do NOT run on the 80 holdout cases during development.
No human reference or preannotated QA is passed to the model.
"""
import argparse
import json
import random
import re
from pathlib import Path

BASELINE_FILE = Path("outputs/validate_base_vs_lora_100.json")
MODEL_DIR = Path("models/Qwen3-4B-Instruct-2507")
ADAPTER_DIR = Path("outputs/qlora_sft/final_adapter")
OUTPUT_FILE = Path("outputs/validate_qa_pipeline_v2_dev.json")
SUMMARY_FILE = Path("outputs/validate_qa_pipeline_v2_dev_summary.json")
REVIEW_FILE = Path("outputs/validate_qa_pipeline_v2_dev_review.md")
SPLIT_SEED, DEV_SIZE = 20261122, 20
STAGE1_TOKENS, STAGE2_TOKENS = 1024, 384
STATUSES = {"信息答复", "已完成", "承诺后续处理", "处理中", "未确认"}
CATEGORIES = "物流、订单、退款、售后、发票、商品、优惠、账户、客服服务、其他"

EXTRACT_RULES = f"""你是客服对话的问答证据提取器。这一步只提取 QA，不生成最终摘要。
以下每行有唯一的发言编号：[U...] 是用户或商家，[C...] 是客服。

任务：
1. 先识别独立的具体业务问题；同一问题的重复追问应合并，不能逐句都当新问题。
2. 找到客服对于该问题的最终明确答复；客服主动提出的重要方案也需要记录。
3. 每条记录引用能够支持答复的客服原话编号 evidence_ids（仅限 [C...] 编号）。
4. 忽略寒暄、道谢、纯情绪表达及没有业务意义的重复内容。
5. 处理状态：
   - 已完成：客服明确确认操作已完成，例如“已为您提交申请”。
   - 承诺后续处理：客服表示之后会处理、回电或安排，但尚未确认完成。
   - 处理中：明确正在审核、等待处理或核查中。
   - 信息答复：告知政策、建议、条件、不支持等，并非执行操作。
   - 未确认：客服未明确回答该问题，或无法确认结果。
   “明天联系下”是建议，不是客服承诺明天回电；“无法承诺送达时间”不是承诺后续处理。
6. 不要为了凑数量添加没有答复的随口抱怨；不准凭空猜测类别或状态。

每条输出五个字段：
- category: 业务类目，优先从 {CATEGORIES} 选择
- question: 具体问题/重要主动方案的简述
- answer: 根据客服原话归纳的最终答复，保留关键条件与否定
- status: 只能是 信息答复 / 已完成 / 承诺后续处理 / 处理中 / 未确认
- evidence_ids: 对话中真实存在的客服发言编号数组，例如 ["C3", "C5"]

只返回严格合法的 JSON 数组，无 Markdown、无解释。evidence_ids 不得引用用户发言。
【带编号的客服对话】
"""

# Apart from the optional evidence list, CONTROL and QA-GUIDED conditions have
# exactly the same final summarisation rules and decoding/model settings.
SUMMARY_RULES = """你是一名客服摘要员。请根据原始对话生成简洁、忠实的客服答复摘要。
逐项覆盖重要的客服实际答复和主动提供的处理方案；合并同一问题的重复追问。
特别区分已完成、承诺将处理、处理中、信息答复、未确认。
不把用户想做的事情当成客服已完成，不凭空补充细节。
如果提供了【QA 核对清单】，它只是模型生成的辅助草稿，不是真实标签；
请逐项对照原始对话核实，必要时纠正或舍弃错误的 QA 项，并补充遗漏的关键答复。
最终只输出简洁编号列表，不输出 QA JSON、原文证据、注释和无关用户背景。
【原始对话】
"""


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def clean_chars(value):
    value = re.sub(r"(?m)^\s*\d+[.、．]\s*", "", value)
    return list(re.sub(r"\s+", "", value))


def score(prediction, reference):
    """Exactly experiment 2's character-level ROUGE-L LCS definition."""
    a, b = clean_chars(prediction), clean_chars(reference)
    if not a or not b:
        return {"rouge_l_precision": 0., "rouge_l_recall": 0., "rouge_l_f1": 0., "chars": len(a)}
    prev = [0] * (len(b) + 1)
    for char in a:
        curr = [0] * (len(b) + 1)
        for j, other in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if char == other else max(prev[j], curr[j - 1])
        prev = curr
    overlap = prev[-1]
    p, r = overlap / len(a), overlap / len(b)
    return {
        "rouge_l_precision": round(p, 6), "rouge_l_recall": round(r, 6),
        "rouge_l_f1": round(2*p*r/(p+r), 6) if p+r else 0., "chars": len(a),
    }


def numbered_dialogue(dialogue):
    """Identify all actual speaker turns; give assistant lines stable IDs."""
    lines, available = [], {}
    u = c = 0
    for raw_line in dialogue.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("客服："):
            ident = f"C{c}"; c += 1
            available[ident] = line[len("客服："):]
        elif line.startswith(("用户：", "商家：")):
            ident = f"U{u}"; u += 1
        else:
            # Continuations are appended to the previous turn, not new speakers.
            if lines:
                lines[-1] += " " + line
                if lines[-1].startswith("[C"):
                    prior = lines[-1].split("]", 1)[0][1:]
                    available[prior] += " " + line
            continue
        lines.append(f"[{ident}] {line}")
    return "\n".join(lines), available


def parse_qa(raw, available):
    """Reject malformed structure and ungrounded evidence IDs; no silent fallback."""
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:].strip()
    if raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, "not_valid_json"
    if not isinstance(parsed, list) or not parsed:
        return None, "not_nonempty_array"
    out = []
    for i, item in enumerate(parsed):
        required = ("category", "question", "answer", "status", "evidence_ids")
        if not isinstance(item, dict) or any(k not in item for k in required):
            return None, f"item_{i}_wrong_fields"
        if not all(isinstance(item[k], str) and item[k].strip() for k in required[:-1]):
            return None, f"item_{i}_invalid_string"
        if item["status"] not in STATUSES:
            return None, f"item_{i}_invalid_status"
        ids = item["evidence_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(v, str) for v in ids):
            return None, f"item_{i}_missing_evidence_ids"
        if any(v not in available for v in ids):
            return None, f"item_{i}_invalid_evidence_id"
        out.append({
            "category": item["category"].strip(),
            "question": item["question"].strip(),
            "answer": item["answer"].strip(),
            "status": item["status"],
            "evidence_ids": ids,
            "evidence_quotes": [available[v] for v in ids],
        })
    return out, None


def samples():
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(f"Missing {BASELINE_FILE} (from experiment 2)")
    source = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    rows = source["results"]
    if len(rows) != 100 or len({r["index"] for r in rows}) != 100:
        raise ValueError("Expected the 100 unique validation examples from experiment 2")
    dev_ids = set(random.Random(SPLIT_SEED).sample(sorted(r["index"] for r in rows), DEV_SIZE))
    selected = sorted((r for r in rows if r["index"] in dev_ids), key=lambda x: x["index"])
    assert len(selected) == DEV_SIZE
    return selected


def aggregate(results):
    completed = [r for r in results if r.get("stage1_valid")]
    report = {
        "processed": len(results),
        "stage1_schema_and_evidence_ids_valid": len(completed),
        "stage1_rejected": len(results) - len(completed),
        "stage1_failure_reasons": [
            {"DialogueID": r["DialogueID"], "reason": r.get("stage1_error")}
            for r in results if not r.get("stage1_valid")
        ],
        "warning": "Evidence IDs and ROUGE are mechanical checks, NOT semantic QA or factual correctness. Audit individual records.",
    }
    if not completed:
        return report
    for name, key in (("existing_sft", "existing_sft_scores"),
                      ("direct_control", "direct_scores"),
                      ("qa_guided", "qa_guided_scores")):
        report[name] = {
            metric: round(sum(r[key][metric] for r in completed) / len(completed), 4)
            for metric in ("rouge_l_f1", "rouge_l_precision", "rouge_l_recall", "chars")
        }
    report["qa_guided_f1_beats_direct"] = sum(
        r["qa_guided_scores"]["rouge_l_f1"] > r["direct_scores"]["rouge_l_f1"]
        for r in completed
    )
    report["qa_guided_recall_beats_direct"] = sum(
        r["qa_guided_scores"]["rouge_l_recall"] > r["direct_scores"]["rouge_l_recall"]
        for r in completed
    )
    return report


def save_review(results):
    """Human-readable comparison for checking QA quality, not just ROUGE."""
    lines = ["# Experiment 3 — QA-aware vs direct (Development)",
             "", "This report contains model predictions and references for human inspection.",
             "**Do not assume stage-1 QA facts are correct just because IDs exist.**", ""]
    for r in sorted(results, key=lambda x: x["index"]):
        lines += [f"## DialogueID {r['DialogueID']} (validation index {r['index']})", "",
                  "### Original dialogue", "", "```text", r["dialogue"], "```", "",
                  "### Stage 1: extracted QA (original Qwen)", ""]
        if not r.get("stage1_valid"):
            lines += [f"**Extraction failed:** {r.get('stage1_error')}", "",
                      "```text", r.get("stage1_raw", ""), "```", ""]
        else:
            for i, qa in enumerate(r["stage1_qa"], 1):
                cites = ", ".join(qa["evidence_ids"])
                quotations = " / ".join(qa["evidence_quotes"])
                lines += [f"{i}. **[{qa['category']}] {qa['question']}**",
                          f"   - Answer: {qa['answer']}",
                          f"   - Status: {qa['status']}",
                          f"   - Evidence: {cites} = {quotations}"]
            lines += [""]
        lines += ["### Experiment 2: saved standard SFT", "",
                  "```text", r["existing_sft_prediction"], "```", ""]
        if r.get("stage1_valid"):
            lines += ["### Direct control: same summary Prompt, NO QA list", "",
                      "```text", r["direct_prediction"], "```", "",
                      "### QA-guided: same summary Prompt, WITH QA list", "",
                      "```text", r["qa_guided_prediction"], "```", "",
                      f"Character ROUGE-L F1: SFT={r['existing_sft_scores']['rouge_l_f1']:.3f} | "
                      f"Direct={r['direct_scores']['rouge_l_f1']:.3f} | "
                      f"QA-guided={r['qa_guided_scores']['rouge_l_f1']:.3f}", ""]
        lines += ["### Reference (used ONLY for scoring/review)", "",
                  "```text", "\n".join(r["reference"]), "```", "",
                  "Manual checks: Did stage 1 match each question to the right answer? "
                  "Did it confuse planned actions with completed ones? Did QA-guided restore any missing points?", "", "---", ""]
    REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text("\n".join(lines), encoding="utf-8")


def generate(model, tokenizer, content, max_new_tokens):
    import torch
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(result, skip_special_tokens=True).strip(), len(result)


def main():
    ap = argparse.ArgumentParser(description="Experiment 3 QA-aware pipeline on fixed dev cases")
    ap.add_argument("--limit", type=int, default=5, help="Number of dev cases (1..20); resumes earlier cases")
    ap.add_argument("--show-prompts", action="store_true", help="Print first case prompts (no GPU)")
    args = ap.parse_args()
    if not 1 <= args.limit <= DEV_SIZE:
        ap.error("--limit must be 1..20")
    selected = samples()
    if args.show_prompts:
        tagged, _ = numbered_dialogue(selected[0]["dialogue"])
        print("\n===== STAGE 1: ORIGINAL QWEN (LoRA disabled) =====\n")
        print(EXTRACT_RULES + tagged)
        print("\n===== STAGE 2: SAME LoRA, DIRECT-CONTROL PROMPT =====\n")
        print(SUMMARY_RULES + selected[0]["dialogue"])
        print("\n===== STAGE 2: SAME LoRA, QA-GUIDED PROMPT TEMPLATE =====\n")
        print(SUMMARY_RULES + selected[0]["dialogue"] + "\n【QA 核对清单】\n" + "<first-stage QA appears here>")
        return

    meta = {
        "experiment": "Experiment 3 QA evidence-linked v2: base extract / same LoRA direct vs QA-guided",
        "dev_indices": [r["index"] for r in selected],
        "seed": SPLIT_SEED,
        "adapter": str(ADAPTER_DIR),
        "extract_prompt_version": "qa_ids_v2",
        "summary_prompt_version": "same_rules_with_optional_qa_v2",
        "model_precision": "4-bit NF4, bf16, double quant",
        "generation": "greedy", "stage1_max_tokens": STAGE1_TOKENS,
        "stage2_max_tokens": STAGE2_TOKENS,
        "gold_reference_or_gold_qa_in_prompt": False,
    }
    if OUTPUT_FILE.exists():
        previous = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        if previous["metadata"] != meta:
            raise ValueError(f"Existing {OUTPUT_FILE} uses a different setup; will not overwrite")
        results = previous["results"]
    else:
        results = []
    completed_ids = {r["index"] for r in results}
    if len(completed_ids) != len(results):
        raise ValueError("Duplicate saved result indices")
    pending = [r for r in selected[:args.limit] if r["index"] not in completed_ids]
    if not pending:
        print(json.dumps(aggregate(results), ensure_ascii=False, indent=2))
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("Requires CUDA; execute on your 4090 instance")
    for p in (MODEL_DIR / "config.json", ADAPTER_DIR / "adapter_model.safetensors"):
        if not p.is_file():
            raise FileNotFoundError(p)
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
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

    for i, row in enumerate(pending, 1):
        tagged, available = numbered_dialogue(row["dialogue"])
        with model.disable_adapter():
            stage1_raw, extracted_tokens = generate(model, tok, EXTRACT_RULES + tagged, STAGE1_TOKENS)
        qa, issue = parse_qa(stage1_raw, available)
        report = {
            "index": row["index"], "DialogueID": row["DialogueID"],
            "reference": row["reference"],  # stored for scoring/review, never in model input
            "dialogue": row["dialogue"],
            "numbered_dialogue": tagged,
            "existing_sft_prediction": row["lora_prediction"],
            "existing_sft_scores": row["lora"],
            "stage1_raw": stage1_raw, "stage1_tokens": extracted_tokens,
            "stage1_valid": issue is None,
            "stage1_error": issue,
            "stage1_qa": qa,
        }
        if qa is not None:
            # Control and guided summaries use exactly the same LoRA and instructions.
            direct, direct_tokens = generate(model, tok, SUMMARY_RULES + row["dialogue"], STAGE2_TOKENS)
            qa_context = json.dumps(qa, ensure_ascii=False, indent=2)
            guided_prompt = SUMMARY_RULES + row["dialogue"] + "\n【QA 核对清单】\n" + qa_context
            guided, guided_tokens = generate(model, tok, guided_prompt, STAGE2_TOKENS)
            reference = "\n".join(row["reference"])
            report.update({
                "direct_prediction": direct,
                "direct_tokens": direct_tokens,
                "direct_scores": score(direct, reference),
                "qa_guided_prediction": guided,
                "qa_guided_tokens": guided_tokens,
                "qa_guided_scores": score(guided, reference),
            })
        results.append(report)
        save_json(OUTPUT_FILE, {"metadata": meta, "results": results})
        save_json(SUMMARY_FILE, aggregate(results))
        save_review(results)
        print(f"[{i}/{len(pending)}] DialogueID={row['DialogueID']} "
              f"QA={'OK (' + str(len(qa)) + ')' if qa is not None else issue}"
              + (f" | direct={report['direct_scores']['rouge_l_f1']:.3f} "
                 f"guided={report['qa_guided_scores']['rouge_l_f1']:.3f}" if qa is not None else " | summary skipped"),
              flush=True)
    print("\n" + json.dumps(aggregate(results), ensure_ascii=False, indent=2))
    print(f"\nOutputs: {OUTPUT_FILE}\nSummary: {SUMMARY_FILE}\nReadable review: {REVIEW_FILE}")


if __name__ == "__main__":
    main()
