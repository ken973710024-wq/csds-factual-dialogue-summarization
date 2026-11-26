"""Compare QA-aware training against inference-only QA on the same 30 dialogues.

Put this file in the csds-project root and run it there with Python. It uses
the diagnostic30 cohort and the original Stage 2 prompt, plus the project's
saved matched G and F outputs. Only G with QA needs new inference.
"""
import argparse
import fcntl
import json
import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from eval_stage2_factcheck import ABC, choose, load_cases
from qa_summary_study_common import (
    BASE_MODEL, SUMMARY_MAX, SUMMARY_RULES, atomic_json, score, sha_file, sha_text, stage2_input,
)
from validate_qa_aware_pipeline import dialogue_to_text, generate
from validate_summary_qa_training import boot

DIRECT = "G_direct"
INFERENCE_ONLY = "G_inference_only_QA"
QA_TRAINED = "F_trained_with_QA"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(tag):
    outputs = Path("outputs")
    prior_path = outputs / f"validate_summary_qa_training_{tag}_100.json"
    if not prior_path.is_file():
        raise FileNotFoundError(f"{prior_path} missing. First run "
                                f"python src/validate_summary_qa_training.py --tag {tag} --limit 100")
    prior = read_json(prior_path)
    old_meta = prior["metadata"]
    direct = outputs / f"qlora_summary_matched_direct_{tag}" / "final_adapter"
    aware = outputs / f"qlora_summary_matched_aware_{tag}" / "final_adapter"
    required = {"tag": tag, "base_model": str(BASE_MODEL), "direct_adapter": str(direct),
                "aware_adapter": str(aware), "eval_decode": "greedy", "summary_max_new_tokens": SUMMARY_MAX,
                "qa_stage1_source_sha256": sha_file(ABC), "stage2_prompt_sha256": sha_text(SUMMARY_RULES)}
    for key, value in required.items():
        if old_meta.get(key) != value:
            raise ValueError(f"Saved evaluation differs at {key}. Check {prior_path}.")
    saved = {str(row["DialogueID"]): row for row in prior["results"]}
    if len(saved) != len(prior["results"]):
        raise ValueError("Duplicate DialogueID in the matched evaluation.")
    cases = choose(load_cases(), "diagnostic30")
    inputs = {}
    for case in cases:
        key = case["DialogueID"]
        row = saved.get(key, {})
        if "G_new_matched_direct" not in row or "F_new_qa_trained" not in row:
            raise ValueError(f"Missing matched predictions for {key}. Complete "
                             f"python src/validate_summary_qa_training.py --tag {tag} --limit 100")
        if (row["index"] != case["index"] or row["reference"] != case["reference"]
                or row["dialogue"] != dialogue_to_text(case["source"]) or row["qa_used"] != case["qa"]):
            raise ValueError(f"Saved inputs differ for DialogueID {key}.")
        inputs[key] = stage2_input(case["source"], case["qa"])
    weights = {}
    for name, adapter in (("G", direct), ("F", aware)):
        weights[name] = sha_file(adapter / "adapter_model.safetensors")
        weights[name + "_config"] = sha_file(adapter / "adapter_config.json")
    prompt_digest = sha_text(json.dumps(inputs, sort_keys=True, ensure_ascii=False))
    metadata = {"study_version": 1, "tag": tag, "ids": list(inputs), "matched_file_sha256": sha_file(prior_path),
                "adapter_sha256": weights, "input_prompts_sha256": prompt_digest, "decode": "greedy",
                "max_new_tokens": SUMMARY_MAX, "cohort": "diagnostic30"}
    return inputs, saved, direct, metadata


def load_direct(adapter):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Run inference on the project's CUDA server.")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL), quantization_config=quantization,
                                                dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True)
    model = PeftModel.from_pretrained(base, str(adapter), adapter_name="matched_direct", is_trainable=False)
    model.set_adapter("matched_direct")
    model.eval()
    torch.manual_seed(42)
    return model, tokenizer


def aggregate(rows):
    systems = (DIRECT, INFERENCE_ONLY, QA_TRAINED)
    scores = {system: [row[system]["scores"]["rouge_l_f1"] for row in rows] for system in systems}
    report = {"n": len(rows), "mean_rouge_l_f1": {s: round(mean(scores[s]), 4) for s in systems}}
    for left, right in ((QA_TRAINED, INFERENCE_ONLY), (INFERENCE_ONLY, DIRECT)):
        delta = [a - b for a, b in zip(scores[left], scores[right])]
        report[f"{left}_minus_{right}"] = {
            "mean_delta": round(mean(delta), 4), "paired_dialogue_bootstrap_ci95": boot(delta),
            "wins": sum(d > 0 for d in delta), "ties": delta.count(0), "losses": sum(d < 0 for d in delta),
        }
    return report


def write_blind_review(path, rows):
    if path.exists():
        return
    blocks = ["# QA 训练对照 · 事实盲审", "",
              "先从对话中记下应保留的对象、最终状态、时限或条件，再读 X 和 Y。分别记录写错与遗漏，允许平局。", ""]
    for row in rows:
        blocks += [f"## 对话 {row['DialogueID']}", "", "~~~text", row["dialogue"], "~~~", "",
                   "关键事实与原文依据：", ""]
        for label, system in row["blind_key"].items():
            blocks += [f"### 摘要 {label}", "", "~~~text", row[system]["prediction"], "~~~", "",
                       "有误的事实及依据：", "", "遗漏的关键事实：", ""]
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(blocks), encoding="utf-8")
    tmp.replace(path)


def run(tag, check):
    prompts, saved, adapter, metadata = prepare(tag)
    print(f"{len(prompts)} paired cases. Same dialogue, QA, prompt and decoding. Adapter: {adapter}", flush=True)
    if check:
        return
    dest = Path("outputs") / f"validate_qa_training_effect_{tag}_diagnostic30.json"
    with dest.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An evaluation is already writing this result.")
        state = read_json(dest) if dest.exists() else {"metadata": metadata, "results": []}
        if state["metadata"] != metadata:
            raise ValueError(f"Inputs or weights changed. Archive the prior run before reusing {dest}.")
        done = {str(r["DialogueID"]) for r in state["results"]}
        if len(done) != len(state["results"]) or not done.issubset(prompts):
            raise ValueError("Cached DialogueIDs do not match this cohort.")
        pending = [key for key in prompts if key not in done]
        if pending:
            model, tokenizer = load_direct(adapter)
            for key in pending:
                old = saved[key]
                prediction, count = generate(model, tokenizer, prompts[key], SUMMARY_MAX, key)
                order = [INFERENCE_ONLY, QA_TRAINED]
                random.Random(f"diagnostic30/qa-training/{key}").shuffle(order)
                row = {"DialogueID": key, "dialogue": old["dialogue"], "qa": old["qa_used"],
                       "reference": old["reference"], DIRECT: old["G_new_matched_direct"],
                       QA_TRAINED: old["F_new_qa_trained"], "blind_key": dict(zip(("X", "Y"), order)),
                       INFERENCE_ONLY: {"prediction": prediction, "scores": score(prediction, old["reference"]),
                                        "generated_tokens": count, "hit_max_new_tokens": count >= SUMMARY_MAX}}
                state["results"].append(row)
                atomic_json(dest, state)
                print(f"[{len(state['results'])}/{len(prompts)}] DialogueID={key}", flush=True)
        state["summary"] = aggregate(state["results"])
        atomic_json(dest, state)
        review = dest.with_name(dest.stem + "_blind.md")
        write_blind_review(review, state["results"])
        print(json.dumps(state["summary"], ensure_ascii=False, indent=2))
        print(f"Review X/Y in {review} before opening scores and identities in {dest}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", choices=("pilot512", "full"), default="pilot512")
    parser.add_argument("--check", action="store_true", help="Check inputs and saved adapters on CPU.")
    args = parser.parse_args()
    run(args.tag, args.check)
