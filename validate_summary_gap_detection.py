"""Check detection of factual errors and missing key facts on 12 existing CSDS development pairs.

Unpack into csds-project, beside validate_summary_evidence_revision.py.
  python validate_summary_gap_detection.py run    # frozen GPU verifier, then score
  python validate_summary_gap_detection.py score  # score a completed run on CPU

The model receives one unlabelled original dialogue and one existing summary per call.
The two drafts of each pair are inspected in separate calls, in shuffled order.
The scoring key is opened only after all predictions have been saved.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re

import validate_summary_evidence_revision as revision

study = revision.previous
ROOT = Path(__file__).resolve().parent
PACK = Path("csds_gap_check")
OUT = Path("outputs/summary_gap_detection_pairs")
AXES = ("error", "omission")
LIMIT = 384
PROMPT = """请独立评估一份客服交接摘要。完整原对话是唯一事实依据。
分别判断两件事，两者可以同时成立，也可以同时不成立。
error：摘要是否写入至少一条与原文冲突或原文不足以支持的业务陈述？检查主体、条件、状态及时间。
omission：是否遗漏了至少一项接手客服需要知道的关键信息？看重要诉求、实际答复、进展与下一步行动。
同义改写视为已覆盖；寒暄、重复和不影响交接的细节允许省略。区分客户所述和客服已确认。
回答一个 JSON 对象，顶层字段为 error 和 omission。每个字段都是一个对象，含 present、reason、turns。
present 用 JSON 布尔值 true 或 false；reason 用一句具体理由；turns 是有关原文行号的字符串数组。
error 为真时另填 summary_quote，逐字摘录有问题的摘要短句。若缺少证实且无可引用的原文，可以留空 turns。
omission 为真时，reason 写明一个遗漏的事实，turns 至少引用一个支持它的原文行号。
判为 false 时也给简要判断依据。只评价这一份摘要，保持其原文。"""
TURN = re.compile(r"^\[([CU]\d+)\] ")


def unique_rows(items):
    index = {r["id"]:r for r in items}
    if len(index) != len(items) or not index:
        raise ValueError("Empty input or duplicate summary codes.")
    return index


def prompt_for(row):
    return PROMPT + "\n【原对话】\n" + row["dialogue"] + "\n【待评摘要】\n" + row["summary"]


def parse_response(raw, row):
    body = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.I | re.S)
    errors, votes = {}, {}
    try:
        reply = json.loads(fence[1] if fence else body)
    except (ValueError, RecursionError):
        return {}, {"response":"invalid_json"}
    if not isinstance(reply, dict):
        return {}, {"response":"expected_object"}
    allowed = {m[1] for line in row["dialogue"].splitlines() if (m := TURN.match(line))}
    for axis in AXES:
        obj = reply.get(axis)
        if not isinstance(obj, dict) or type(obj.get("present")) is not bool:
            errors[axis] = "missing_explicit_decision"
            continue
        reason, turns = obj.get("reason"), obj.get("turns")
        if (not isinstance(reason,str) or not reason.strip() or not isinstance(turns,list)
                or not all(isinstance(t,str) and t in allowed for t in turns)):
            errors[axis] = "invalid_reason_or_source_ids"
            continue
        if obj["present"] and axis == "omission" and not turns:
            errors[axis] = "missing_source_for_omission"
            continue
        if obj["present"] and axis == "error":
            quote = obj.get("summary_quote")
            if not isinstance(quote,str) or not quote.strip() or quote not in row["summary"]:
                errors[axis] = "unmatched_summary_quote"
                continue
        votes[axis] = obj
    return votes, errors


def read_inputs():
    items = study.read(PACK / "inputs.json")
    unique_rows(items)
    for row in items:
        if set(row) != {"id","dialogue","summary"} or any(not isinstance(v,str) or not v.strip() for v in row.values()):
            raise ValueError("Blind input must contain only the code, source dialogue and raw summary.")
    return items


def predict(items):
    path = OUT / "predictions.json"
    profile = {"inputs":study.digest(items), "program":study.file_hash(Path(__file__)),
               "original_code":revision.code_version(), "model":revision.weight_profile(),
               "prompt":study.digest(PROMPT), "limit":LIMIT, "greedy":True, "adapter":"disabled"}
    state = study.read(path) if path.exists() else {"profile":profile,"results":[]}
    if state["profile"] != profile:
        raise ValueError("This output directory contains a different frozen run.")
    done = {r["id"] for r in state["results"]}
    if len(done) != len(state["results"]) or not done <= unique_rows(items).keys():
        raise ValueError("Unexpected or duplicate cached prediction.")
    pending = [r for r in items if r["id"] not in done]
    if pending:
        model, tokenizer = revision.load_model()
        with model.disable_adapter():
            for row in pending:
                raw, info = study.completion(model, tokenizer, prompt_for(row), LIMIT, row["id"] + "/detect")
                if info["context_limit"] or info["hit_token_limit"]:
                    votes, errors = {}, {"response":"context_or_output_limit"}
                else:
                    votes, errors = parse_response(raw,row)
                state["results"].append({"id":row["id"], "votes":votes, "invalid":errors, "raw":raw, "generation":info})
                study.atomic_json(path,state)
                print(f"[{len(state['results'])}/{len(items)}] {row['id']}",flush=True)
    return state


def score(items, state):
    # The label file and past reviewers' paraphrases cannot enter model prompts.
    key = study.read(PACK / "scoring_key.json")
    if not (key["input_sha256"] == study.digest(items) == state["profile"]["inputs"]):
        raise ValueError("Input packet, predictions and scoring key do not match.")
    if (state["profile"]["program"] != study.file_hash(Path(__file__))
            or state["profile"]["original_code"] != revision.code_version()):
        raise ValueError("Evaluation code changed after prediction.")
    cases = unique_rows(items)
    preds = unique_rows(state["results"])
    if preds.keys() != cases.keys():
        raise ValueError("Scoring needs all saved predictions. Resume with run.")
    pair_ids = [pair[role]["id"] for pair in key["pairs"] for role in ("issue","control")]
    if len(set(pair_ids)) != len(pair_ids) or set(pair_ids) != cases.keys():
        raise ValueError("Unmatched or repeated scoring-key rows.")
    metrics = {axis:{"pairs":0,"issue_detected":0,"control_correctly_cleared":0,"pair_both_correct":0,
                     "issue_missed":0,"control_false_alarm":0,"issue_unscored":0,"control_unscored":0} for axis in AXES}
    examples = []
    for pair in key["pairs"]:
        axis = pair["axis"]
        m = metrics[axis]
        m["pairs"] += 1
        both = True
        if cases[pair["issue"]["id"]]["dialogue"] != cases[pair["control"]["id"]]["dialogue"]:
            raise ValueError("Paired summaries have different sources.")
        for role in ("issue","control"):
            expected = role == "issue"
            ref = pair[role]
            if bool(ref["errors" if axis == "error" else "omissions"]) != expected:
                raise ValueError("Invalid pair construction.")
            result = preds[ref["id"]]
            answer = result["votes"].get(axis)
            decision = None if answer is None else answer["present"]
            if decision is None:
                m[role + "_unscored"] += 1
            elif decision == expected:
                m["issue_detected" if expected else "control_correctly_cleared"] += 1
            else:
                m["issue_missed" if expected else "control_false_alarm"] += 1
            both = both and decision is expected
            examples.append({"axis":axis,"original_dialogue_id":pair["dialogue_id"],"case_id":ref["id"],
                             "expected_issue":expected,"model":answer,"invalid":result["invalid"],
                             "raw_summary":cases[ref["id"]]["summary"],"dialogue":cases[ref["id"]]["dialogue"],
                             "locked_reference_for_review":ref["errors" if axis=="error" else "omissions"]})
        m["pair_both_correct"] += int(both)
    report = {"scope":"enriched previously reviewed development pairs; detection only",
              "n_summaries":len(items),"n_dialogues":len(key["pairs"]),
              "reference":"one previous AI annotation, not independent human judgement",
              "scoring":"binary verdicts; a matching verdict still needs a semantically correct rationale",
              "reference_key_sha256":study.digest(key), "axes":metrics}
    study.atomic_json(OUT / "diagnostic_scores.json",report)
    study.atomic_json(OUT / "case_review.json",examples)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase",choices=["run","score"])
    args = ap.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another process is using this run.")
        items = read_inputs()
        state = predict(items) if args.phase=="run" else study.read(OUT / "predictions.json")
        report = score(items,state)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        print("逐条核对依据：",OUT / "case_review.json")


if __name__=="__main__":
    main()
