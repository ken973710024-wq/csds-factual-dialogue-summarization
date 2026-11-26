"""Test five source-grounded state updates on four previously reviewed CSDS dialogues.

Save this file in the csds-project root. Run
    python validate_summary_state_evidence.py --check
to inspect the changes, or omit --check to generate the four summaries.
The comparator is the saved F_checked prediction on each identical dialogue.
"""
import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from eval_stage2_factcheck import ABC, CHECK_RULES, choose, load_cases
from qa_summary_study_common import BASE_MODEL, SUMMARY_MAX, atomic_json, qa_input, sha_file, sha_text, stage2_input
from validate_qa_aware_pipeline import dialogue_to_text, generate
from validate_qa_lora import numbered_speakers


PRIOR = Path("outputs/eval_factuality_F_pilot512_diagnostic30_baseline_checked.json")
ADAPTER = Path("outputs/qlora_summary_matched_aware_pilot512/final_adapter")
OUTPUT = Path("outputs/validate_summary_state_evidence_pilot512_4.json")
# Each tuple identifies the existing QA item (zero-based) and an original agent
# utterance. The literal is used only to verify its identity before adding its ID.
ADDITIONS = {
    "7365": [(0, "C14", "我帮您修改好了")],
    "9089": [(2, "C18", "这边已经帮您联系配送配送稍后给您[数字]回电")],
    "246": [(0, "C12", "已经给您提交申请")],
    "8618": [(0, "C5", "已经备注站点的哦"), (1, "C9", "好的，小妹通知站点了哦")],
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    previous = read(PRIOR)
    meta = previous["metadata"]
    rows = {str(r["DialogueID"]): r for r in previous["results"]}
    cases = choose(load_cases(), "diagnostic30")
    ids = [str(case["DialogueID"]) for case in cases]
    if len(ids) != 30 or len(rows) != 30 or len(previous["results"]) != 30 or set(ids) != set(rows) or ids != meta["ids"]:
        raise ValueError("The saved result and diagnostic30 cohort do not match.")
    expected = {
        "version": "frozen_F_factcheck_v1", "tag": "pilot512", "mode": "diagnostic30",
        "prompt_sha256": sha_text(CHECK_RULES), "source_abc_sha256": sha_file(ABC),
        "decode": {"do_sample": False, "max_new_tokens": SUMMARY_MAX},
    }
    for field, value in expected.items():
        if meta.get(field) != value:
            raise ValueError(f"Earlier checked evaluation has a different {field}.")

    prompts, records = {}, {}
    for case in cases:
        key = str(case["DialogueID"])
        if key not in ADDITIONS:
            continue
        old, source = rows[key], case["source"]
        if (case["qa"] != old["qa_used"] or dialogue_to_text(source) != old["dialogue"]
                or case["index"] != old["index"] or "F_checked" not in old):
            raise ValueError(f"Different dialogue, QA or comparator for {key}.")
        _, agent_ids, quotes = numbered_speakers(qa_input(source))
        qa = deepcopy(case["qa"])
        all_original_refs = {ref for item in qa for ref in item["evidence_ids"]}
        additional = []
        for i, ref, expected_quote in ADDITIONS[key]:
            if ref not in agent_ids or ref in all_original_refs or "".join(quotes[ref].split()) != expected_quote:
                raise ValueError(f"New evidence was already present or differs at {key}/{ref}.")
            qa[i]["evidence_ids"].append(ref)
            additional.append({"qa_index": i, "id": ref, "quote": quotes[ref]})
        prompts[key] = stage2_input(source, qa) + CHECK_RULES
        records[key] = {
            "DialogueID": key, "dialogue": old["dialogue"], "qa_before": case["qa"], "qa_after": qa,
            "added_evidence": additional, "checked_summary": old["F_checked"]["prediction"],
        }
    if set(prompts) != set(ADDITIONS):
        raise ValueError("One of the four targeted dialogues is absent.")
    metadata = {
        "version": 1, "tag": "pilot512", "comparator_file_sha256": sha_file(PRIOR),
        "weights_sha256": meta["model_weights_sha256"], "config_sha256": meta["model_adapter_config_sha256"],
        "decode": meta["decode"], "new_prompt_sha256": {k:sha_text(p) for k,p in prompts.items()},
    }
    return metadata, prompts, records


def load_model(metadata):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Run inference in the csds-project CUDA environment.")
    if (sha_file(ADAPTER / "adapter_model.safetensors") != metadata["weights_sha256"]
            or sha_file(ADAPTER / "adapter_config.json") != metadata["config_sha256"]):
        raise ValueError("F adapter weights or configuration differ from the checked run.")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL), quantization_config=config,
                                               dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True)
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False)
    model.eval()
    torch.manual_seed(42)
    return model, tokenizer


def run(check_only):
    meta, prompts, records = prepare()
    for key, record in records.items():
        print(key, " / ".join(e["id"] + " " + e["quote"] for e in record["added_evidence"]))
    if check_only:
        print("Verified 4 inputs, 5 added quotes. The generation run also checks the adapter weights.")
        return
    with OUTPUT.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This comparison is already running.")
        report = read(OUTPUT) if OUTPUT.exists() else {"metadata": meta, "results": []}
        if report["metadata"] != meta:
            raise ValueError("Saved experiment has different inputs. Use a separate output path.")
        done = {r["DialogueID"] for r in report["results"]}
        if len(done) != len(report["results"]) or not done.issubset(prompts):
            raise ValueError("Duplicate or unexpected cached dialogue.")
        pending = [k for k in prompts if k not in done]
        if pending:
            model, tokenizer = load_model(meta)
            for k in pending:
                new, tokens = generate(model, tokenizer, prompts[k], SUMMARY_MAX, k)
                result = dict(records[k], new_summary=new, generated_tokens=tokens, hit_max_new_tokens=tokens >= SUMMARY_MAX)
                report["results"].append(result)
                atomic_json(OUTPUT, report)
                print(f"[{len(report['results'])}/4] DialogueID={k}", flush=True)
        print(f"Paired before/after summaries and source quotes: {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify source and prompt inputs on CPU.")
    run(parser.parse_args().check)
