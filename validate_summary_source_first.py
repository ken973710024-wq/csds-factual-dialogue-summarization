"""Test evidence selection before editing the six cached CSDS training drafts.

Place beside validate_summary_evidence_revision.py and validate_unseen_summaries.py
in the CSDS project root. Run: python validate_summary_source_first.py smoke
Reads the completed original smoke run. Writes outputs/summary_source_first_smoke/.
Outputs are a diagnostic on previously inspected training examples.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re

import validate_summary_evidence_revision as revision

study = revision.previous
OUT = Path("outputs/summary_source_first_smoke")
SELECT_MAX = 384
SELECT = """你在为接手这次会话的客服查找原始证据。只读下面按时间排列的对话。
选中能说明重要诉求、处理进展或下一步行动的原话。对于同一事项结合上下文保留必要的主体、条件、
期限以及尚待确认的结论。留意后续发言是否更新了早先的说法。省去寒暄和没有新增信息的重复发言。
用一个 JSON 对象作答。字段 turn_ids 是字符串数组，每项为原文方括号内的行号。只选有用的行；
如果没有实质业务信息，数组可以为空。"""
GUIDE = """以下是从同一对话中选出的原话。逐项与草稿核对：检查漏掉会改变理解或下一步行动的信息，
也检查是否已有等义表达。仍按完整原对话判断，沿用上面的修改格式。"""


def load_cached_pairs():
    directory = revision.SMOKE
    manifest = study.read(directory / "cohort.json")
    cache = study.read(directory / "private_predictions.json")
    profile = cache["profile"]
    ids = manifest["ids"]
    rows = study.by_id(cache["results"])
    if (manifest["split"] != "training_smoke" or manifest["protocol"] != "csds_direct_evidence_revision_v1"
            or profile["study_id"] != study.digest(manifest) or len(ids) != revision.N_SMOKE
            or len(set(ids)) != len(ids) or set(ids) != rows.keys()):
        raise ValueError("A complete paired training smoke run is required. Check its cohort and predictions.")
    if manifest["inputs"]["train_sha256"] != study.file_hash(revision.TRAIN):
        raise ValueError("The training dialogue file differs from the one used for the cached drafts.")
    if profile["code_sha256"] != revision.code_version():
        raise ValueError("The original evaluation code has changed since these drafts were produced.")
    wanted = set(ids)
    train = study.by_id([revision.source_only(row) for row in study.read(revision.TRAIN)
                         if str(row["DialogueID"]) in wanted])
    if train.keys() != wanted:
        raise ValueError("The cached dialogue IDs are missing from the original training split.")
    return ids, train, rows, profile


def decode_ids(raw, numbered):
    """Keep actual source wording, speaker and chronology. Validate every requested pointer."""
    body = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.S | re.I)
    obj = json.loads(fence[1] if fence else body)
    if not isinstance(obj, dict) or set(obj) != {"turn_ids"} or not isinstance(obj["turn_ids"], list):
        raise ValueError("expected_turn_ids_array")
    by_id = {}
    for line in numbered.splitlines():
        match = revision.TURN.fullmatch(line)
        if not match or match[1] in by_id:
            raise ValueError("invalid_numbered_source")
        by_id[match[1]] = line
    requested = set()
    for idx in obj["turn_ids"]:
        if not isinstance(idx, str) or idx not in by_id:
            raise ValueError("unknown_turn_id")
        requested.add(idx)
    return [line for idx, line in by_id.items() if idx in requested]


def incomplete(info):
    return info["context_limit"] or info["hit_token_limit"]


def check_pair(model, tokenizer, source, cached):
    key = str(source["DialogueID"])
    draft = cached["systems"]["A"]
    previous_text = cached["systems"]["revised"]
    result = {"DialogueID": key, "draft": draft, "previous_revision": previous_text, "summary": draft,
              "selected_lines": [], "selector": None, "editor": None, "edits": [], "reason": ""}
    if not draft.strip() or incomplete(cached["generation"]["A"]):
        return {**result, "status": "fallback_incomplete_draft"}
    numbered, _ = study.review_source(source)
    # A fresh model call sees only the source. The preceding draft and human judgements are never in this prompt.
    with model.disable_adapter():
        raw, info = study.completion(model, tokenizer, SELECT + "\n【原对话】\n" + numbered,
                                     SELECT_MAX, key + "/select")
        result["selector"] = {**info, "raw": raw}
        if incomplete(info):
            return {**result, "status": "fallback_selector_limit"}
        try:
            lines = decode_ids(raw, numbered)
        except (ValueError, RecursionError) as exc:
            return {**result, "status": "fallback_invalid_selection", "reason": str(exc)}
        result["selected_lines"] = lines
        if not lines:
            # With no highlighted lines the editor input is identical to the cached control.
            return {**result, "summary": previous_text, "status": "empty_selection_reused_previous"}
        prompt, evidence = revision.edit_prompt(source, draft)
        prompt += "\n【原文复核】\n" + GUIDE + "\n" + "\n".join(lines)
        raw, info = study.completion(model, tokenizer, prompt, revision.EDIT_MAX, key + "/edit")
    result["editor"] = {**info, "raw": raw}
    if incomplete(info):
        return {**result, "status": "fallback_editor_limit"}
    try:
        revised, status, edits = revision.apply_edits(raw, draft, evidence)
    except (ValueError, RecursionError) as exc:
        return {**result, "status": "fallback_invalid_edits", "reason": str(exc)}
    return {**result, "status": status, "summary": revised, "edits": edits}


def write_review(out, results, sources):
    counts = Counter(r["status"] for r in results)
    calls = {step: [r[step] for r in results if r[step] is not None] for step in ("selector", "editor")}
    stats = {"n_training": len(results), "selected_any_evidence": sum(bool(r["selected_lines"]) for r in results),
             "changed_vs_A": sum(r["summary"] != r["draft"] for r in results),
             "changed_vs_previous_checker": sum(r["summary"] != r["previous_revision"] for r in results),
             "status_counts": dict(counts), "call_attempts": {k: len(v) for k,v in calls.items()},
             "generated_tokens": {k: sum(s["tokens"] for s in v) for k,v in calls.items()}}
    study.atomic_json(out / "run_summary.json", stats)
    parts = ["# 先选原文，再核对草稿", "训练集诊断。先核查选中的行能否说明关键事实，再核查最终摘要是否正确使用了它。"]
    for r in results:
        k = r["DialogueID"]
        parts += ["## 对话 " + k, "### 原对话", study.review_source(sources[k])[0],
                  "### A 的原草稿", r["draft"] or "（空）", "### 上一轮直接核查的摘要", r["previous_revision"] or "（空）",
                  "### 仅看原文选出的证据", "\n".join(r["selected_lines"]) or "（无）",
                  "### 新核查器提出的修改", r["editor"]["raw"] if r["editor"] else "（没有调用编辑步骤）",
                  "### 最终摘要", r["summary"] or "（空）", "处理记录　" + r["status"]]
        if r["reason"]:
            parts.append("原因　" + r["reason"])
    path = out / "smoke_review.md"
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    tmp.replace(path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("核对原文、选中证据和最终摘要：", path)


def run(out):
    ids, sources, cached, parent = load_cached_pairs()
    path = out / "paired_predictions.json"
    profile = {"method": "source_turn_selection_then_original_editor_v1", "parent_study_id": parent["study_id"],
               "parent_profile_sha256": study.digest(parent), "cached_pairs_sha256": study.digest([cached[k] for k in ids]),
               "script_sha256": study.file_hash(Path(__file__)), "token_limits": [SELECT_MAX, revision.EDIT_MAX],
               "original_code_sha256": revision.code_version()}
    state = study.read(path) if path.exists() else {"profile": profile, "results": []}
    if state["profile"] != profile:
        raise ValueError("Saved source-first predictions belong to different code, inputs or drafts.")
    done = study.by_id(state["results"])
    if not done.keys() <= set(ids):
        raise ValueError("Unexpected dialogue in the saved source-first run.")
    pending = [k for k in ids if k not in done]
    if pending:
        if parent["inputs"] != revision.weight_profile():
            raise ValueError("Model or generation helper files differ from the cached comparison.")
        model, tokenizer = revision.load_model()
        for k in pending:
            record = check_pair(model, tokenizer, sources[k], cached[k])
            state["results"].append(record)
            study.atomic_json(path, state)
            print(f"[{len(state['results'])}/{len(ids)}] DialogueID={k}  {record['status']}", flush=True)
    ordered = study.by_id(state["results"])
    write_review(out, [ordered[k] for k in ids], sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["smoke"], nargs="?", default="smoke")
    parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another source-first run is using this directory.")
        run(OUT)


if __name__ == "__main__":
    main()
