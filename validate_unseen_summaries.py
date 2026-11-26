"""Prepare, run and score the four frozen CSDS systems on fresh development dialogues.

Place this file in the csds-project root. Commands:
  python validate_unseen_summaries.py prepare
  python validate_unseen_summaries.py predict
  python validate_unseen_summaries.py score
Prepare takes an optional --seen-ids file with one additional previously examined ID per line.
Complete source_facts.json before looking at blind_summaries.json. Then fill the latter
without opening private_predictions.json. Score requires the two completed review files.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import random
import secrets
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from prepare_qa_sft_data import numbered_lines
from qa_summary_study_common import BASE_MODEL, QA_ADAPTER, SUMMARY_MAX, atomic_json, qa_input, stage2_input
from validate_qa_aware_pipeline import QA_MAX_NEW_TOKENS, SUMMARY_ADAPTER, dialogue_to_text, generate, qa_result

DATA = Path("data/CSDS")
HISTORY = Path("outputs/validate_qa_aware_pipeline_100.json")
F_ADAPTER = Path("outputs/qlora_summary_matched_aware_pilot512/final_adapter")
OUT = Path("outputs/summary_generalization_dev60")
SIZE, SEED = 60, 20261124
ARMS = ("base", "A", "QA_list", "F")
VERDICTS = ("有据", "冲突", "无据")
PRIMARY_COMPARISON = {
    "candidate": "F",
    "comparator": "A",
    "primary_metric": "supported_claim_fraction",
    "guardrail": "key_fact_coverage",
    "primary_delta_ci_lower_must_exceed": 0.0,
    "coverage_delta_ci_lower_must_exceed": -0.02,
    "interval": "paired two-sided 95% bootstrap interval by dialogue, pooling counts in each resample",
    "scope": "development estimate only. Confirm the frozen comparison on untouched test data.",
}


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def digest(obj):
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def by_id(rows):
    index = {str(r["DialogueID"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate DialogueID within one split or review file.")
    return index


def signature(row):
    # Ignore name of customer role, Unicode presentation forms and whitespace.
    turns = []
    for t in row["Dialogue"]:
        u = t["utterance"]
        text = " ".join(map(str, u)) if isinstance(u, list) else str(u)
        turns.append((t["speaker"] == "A", "".join(unicodedata.normalize("NFKC", text).split())))
    return digest(turns)


def select(val, train, old, extra):
    rows = by_id(val)
    past = by_id(old)
    if len(past) != 100 or not past.keys() <= rows.keys():
        raise ValueError("Expected the project's previous 100 validation dialogues.")
    for k,r in past.items():
        if dialogue_to_text(rows[k]) != r["dialogue"]:
            raise ValueError(f"Earlier dialogue differs from validation data at {k}.")
    used_ids = set(by_id(train)) | set(past) | set(extra)
    used_text = {signature(r) for r in train}
    used_text.update(signature(rows[k]) for k in (used_ids & rows.keys()))
    pool = []
    for k in sorted(rows):
        row = rows[k]
        if k not in used_ids:
            sig = signature(row)
            if sig not in used_text:
                pool.append(row)
                used_text.add(sig)
    if len(pool) < SIZE:
        raise ValueError(f"Only {len(pool)} eligible unique dialogues for the fixed cohort of {SIZE}.")
    lengths = sorted(len(row["Dialogue"]) for row in pool)
    cut1, cut2 = [lengths[(len(lengths)-1)*q//3] for q in (1,2)]
    layers = defaultdict(list)
    for row in pool:
        n = len(row["Dialogue"])
        intent = next((q["intent"] for q in row.get("QA", []) if isinstance(q, dict) and q.get("intent")), "未标注")
        layers[(int(n > cut1) + int(n > cut2), intent)].append(row)
    rng = random.Random(SEED)
    ordered = []
    for layer in sorted(layers):
        rng.shuffle(layers[layer])
        ordered.extend(layers[layer])
    # A random start and equally spaced positions sample roughly in proportion
    # to each layer. All pool members have the same inclusion probability.
    start = rng.randrange(len(pool))
    chosen = [ordered[(start + i * len(pool)) // SIZE] for i in range(SIZE)]
    rng.shuffle(chosen)
    return chosen, len(pool), [cut1, cut2]


def ensure_template(path, template):
    if not path.exists():
        atomic_json(path, template)
        return
    existing = read(path)
    for field in ("study_id", "prediction_id"):
        if existing.get(field) != template.get(field):
            raise ValueError(f"Review belongs to a different cohort or generation run: {path}")


def review_source(row):
    numbered, ids, _ = numbered_lines(row["Dialogue"], row["DialogueID"])
    # Stage 1 numbers non-agent lines under 用户. Restore the actual customer
    # role in the review copy, e.g. 商家, without changing model inputs or IDs.
    text = "\n".join(line.replace("] 用户：", f"] {row['QRole']}：", 1) if line.startswith("[U") else line
                     for line in numbered.splitlines())
    return text, set(ids.values())


def prepare(extra_ids_path):
    manifest_path = OUT / "cohort.json"
    saved = read(manifest_path) if manifest_path.exists() else {}
    extra = saved.get("extra_seen_ids", [])
    if extra_ids_path is not None:
        extra = sorted(set(extra_ids_path.read_text(encoding="utf-8").split()))
    val_path, train_path = DATA / "val.json", DATA / "train.json"
    chosen, n, cutoffs = select(read(val_path), read(train_path), read(HISTORY)["results"], extra)
    # Only raw dialogue, its ordering and role labels are allowed downstream.
    chosen = [{key:row[key] for key in ("DialogueID","Dialogue","QRole")} for row in chosen]
    meta = {"protocol":"csds_fresh_dev_v1","primary_comparison":PRIMARY_COMPARISON,
            "seed":SEED,"eligible":n,"length_cutoffs":cutoffs,
            "ids":[str(r["DialogueID"]) for r in chosen],"extra_seen_ids":extra,
            "train_sha256":file_hash(train_path),"val_sha256":file_hash(val_path),"old100_sha256":file_hash(HISTORY)}
    if saved and saved != meta:
        raise ValueError("Inputs or cohort changed since preparation. Keep the existing comparison intact.")
    if not saved:
        atomic_json(manifest_path, meta)
    study_id = digest(meta)
    sources = []
    for row in chosen:
        text, _ = review_source(row)
        sources.append({"DialogueID":str(row["DialogueID"]),"dialogue":text,"reviewed":False,"key_facts":[]})
    ensure_template(OUT / "source_facts.json", {"study_id":study_id,
        "instructions":"先独立读原对话。每个关键事实填一项，给出唯一编号、具体表述和原文行号。若某个业务结果没有得到确认，为它建一个事实并标 unresolved=true。同一事项只建一个这样的待定结果。完成该对话后设 reviewed=true。无实质业务事实可保留空列表。",
        "fact_format":{"id":"K1","text":"<一项有业务意义的事实，含必要条件>","evidence_ids":["C0"],"unresolved":False},
        "dialogues":sources})
    return study_id, chosen


def model_files():
    weights = sorted(BASE_MODEL.glob("*.safetensors")) or sorted(BASE_MODEL.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"No base model weights under {BASE_MODEL}")
    files = set(weights) | {BASE_MODEL/"config.json"}
    for pat in ("*index.json","generation_config.json","tokenizer*","*chat_template*","vocab.json","merges.txt","special_tokens_map.json"):
        files.update(p for p in BASE_MODEL.glob(pat) if p.is_file())
    for path in (QA_ADAPTER, SUMMARY_ADAPTER, F_ADAPTER):
        files.update([path/"adapter_model.safetensors",path/"adapter_config.json"])
    files.update(ROOT/"src"/name for name in ("prepare_qa_sft_data.py","qa_summary_study_common.py",
                 "validate_qa_aware_pipeline.py","validate_qa_lora.py","validate_qa_pipeline_dev20.py"))
    return {p.name if p.parent==ROOT/"src" else str(p):file_hash(p) for p in sorted(files)}


def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("The predict command needs the project's CUDA server.")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL),local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(str(BASE_MODEL),quantization_config=config,dtype=torch.bfloat16,
                                               device_map={"":0},local_files_only=True)
    model = PeftModel.from_pretrained(base,str(QA_ADAPTER),adapter_name="qa",is_trainable=False)
    model.load_adapter(str(SUMMARY_ADAPTER),adapter_name="direct",is_trainable=False)
    model.load_adapter(str(F_ADAPTER),adapter_name="aware",is_trainable=False)
    model.eval()
    torch.manual_seed(42)
    return model, tokenizer


def completion(model, tokenizer, prompt, limit, label):
    try:
        text, tokens = generate(model, tokenizer, prompt, limit, label)
    except ValueError as error:
        if "exceeds safety guard" not in str(error):
            raise
        return "", {"context_limit": True,"tokens":0,"hit_token_limit":False}
    return text, {"context_limit":False,"tokens":tokens,"hit_token_limit":tokens>=limit}


def run_one(model, tokenizer, source):
    key = str(source["DialogueID"])
    texts, logs = {}, {}
    direct = stage2_input(source)
    model.set_adapter("direct")
    with model.disable_adapter():
        texts["base"], logs["base"] = completion(model,tokenizer,direct,SUMMARY_MAX,key+"/base")
    texts["A"], logs["A"] = completion(model,tokenizer,direct,SUMMARY_MAX,key+"/A")
    model.set_adapter("qa")
    prompt = qa_input(source)
    raw, logs["stage1"] = completion(model,tokenizer,prompt,QA_MAX_NEW_TOKENS,key+"/QA")
    qa = qa_result(raw,prompt,logs["stage1"]["tokens"],QA_MAX_NEW_TOKENS)
    parsed = qa["parsed"]
    logs["QA_fallback"] = parsed is None
    if parsed is None:
        texts["QA_list"] = texts["F"] = texts["A"]
    else:
        answers = dict.fromkeys(q["answer"].strip() for q in parsed)
        texts["QA_list"] = "\n".join(f"{i}. {a}" for i,a in enumerate(answers,1))
        model.set_adapter("aware")
        texts["F"], logs["F"] = completion(model,tokenizer,stage2_input(source,parsed),SUMMARY_MAX,key+"/F")
        if logs["F"]["context_limit"]:
            texts["F"] = texts["A"]
    return {"DialogueID":key,"systems":texts,"generation":logs,"predicted_qa":qa}


def blind_views(state):
    template, keys = [], {}
    for row in state["results"]:
        k, texts = row["DialogueID"], row["systems"]
        order = list(ARMS)
        random.Random(digest([state["blinding_salt"],k])).shuffle(order)
        unique, mask = {}, {}
        for arm in order:
            txt = texts[arm]
            if txt not in unique:
                unique[txt] = f"S{len(unique)+1}"
            mask[arm] = unique[txt]
        template.append({"DialogueID":k,"summaries":[{"label":code,"text":txt,"reviewed":False,
                         "claims":[],"covered_fact_ids":[],"false_completion_ids":[]} for txt,code in unique.items()]})
        keys[k]=mask
    return template, keys


def predict(study_id, chosen):
    path = OUT/"private_predictions.json"
    profile = {"study_id":study_id,"inputs":model_files(),"max_tokens":[QA_MAX_NEW_TOKENS,SUMMARY_MAX],
               "decoding":"greedy","qa_failure":"A for both QA systems","F_context_overflow":"A",
               "generation_protocol":"original QA prompt and original summary prompt, version 1"}
    state = read(path) if path.exists() else {
        "profile":profile,"blinding_salt":secrets.token_hex(20),"results":[]}
    if state["profile"] != profile:
        raise ValueError("Saved predictions have different weights, prompts, helper code or cohort.")
    done = by_id(state["results"])
    if not done.keys() <= set(str(r["DialogueID"]) for r in chosen):
        raise ValueError("Saved predictions contain an unexpected dialogue.")
    pending = [r for r in chosen if str(r["DialogueID"]) not in done]
    if pending:
        model,tok = load_model()
        for row in pending:
            result=run_one(model,tok,row)
            state["results"].append(result)
            atomic_json(path,state)
            print(f"[{len(state['results'])}/{SIZE}] DialogueID={result['DialogueID']}",flush=True)
    blind,_=blind_views(state)
    ensure_template(OUT/"blind_summaries.json",{"study_id":study_id,"prediction_id":digest(blind),"instructions":
        "先固定原文事实，再看匿名摘要。将实质陈述按语义拆开、去重，每项标有据/冲突/无据。"
        "有据和冲突给出原文行号。covered_fact_ids 填语义完整保留的关键事实。false_completion_ids 填被当成办妥的待定事项编号。"
        "允许多处遗漏或相同输出。无法独立判断的空话不计陈述。每份核对后设 reviewed=true。",
        "claim_format":{"text":"<摘要陈述>","verdict":"有据","evidence_ids":["C0"]},"dialogues":blind})


def unique_ids(values, allowed):
    if not isinstance(values,list) or len(set(values))!=len(values) or not set(values)<=allowed:
        raise ValueError(f"Repeated or unknown IDs: {values}")
    return set(values)


def collect(study_id, chosen):
    source,blind,state=(read(OUT/n) for n in ("source_facts.json","blind_summaries.json","private_predictions.json"))
    if not (source["study_id"]==blind["study_id"]==state["profile"]["study_id"]==study_id):
        raise ValueError("Source, reviews and predictions use different cohorts.")
    canonical,keys=blind_views(state)
    sources,reviews,expected=map(by_id,(source["dialogues"],blind["dialogues"],canonical))
    if blind.get("prediction_id") != digest(canonical):
        raise ValueError("Blind summaries no longer match the saved predictions.")
    if not (sources.keys()==reviews.keys()==keys.keys()==by_id(chosen).keys()):
        raise ValueError("Missing or unexpected dialogues. Complete every paired review.")
    records=[]
    for row in chosen:
        k=str(row["DialogueID"])
        src=sources[k]
        numbered,allowed=review_source(row)
        if src["dialogue"]!=numbered or src["reviewed"] is not True:
            raise ValueError(f"Incomplete or changed source review for {k}.")
        facts=src["key_facts"]
        fids=[f["id"] for f in facts]
        if len(set(fids))!=len(fids) or not all(isinstance(fid,str) and fid.strip() for fid in fids):
            raise ValueError(f"Invalid fact IDs for {k}")
        for fact in facts:
            if not fact["text"].strip() or not unique_ids(fact["evidence_ids"],allowed) or type(fact["unresolved"])!=bool:
                raise ValueError(f"Fact lacks source evidence or an unresolved flag at {k}.")
        open_ids={f["id"] for f in facts if f["unresolved"]}
        official={s["label"]:s["text"] for s in expected[k]["summaries"]}
        summaries=reviews[k]["summaries"]
        if len(official)!=len(summaries) or {s["label"]:s["text"] for s in summaries}!=official:
            raise ValueError(f"Blind labels or prediction text changed for {k}.")
        entries={}
        for s in summaries:
            if s["reviewed"] is not True:
                raise ValueError(f"Unfinished summary at {k}/{s['label']}")
            supported=0
            for c in s["claims"]:
                refs=unique_ids(c["evidence_ids"],allowed)
                if not c["text"].strip() or c["verdict"] not in VERDICTS or (not refs and c["verdict"]!="无据"):
                    raise ValueError(f"Unclassified or ungrounded judgement at {k}/{s['label']}.")
                supported+=int(c["verdict"]=="有据")
            hit=unique_ids(s["covered_fact_ids"],set(fids))
            wrong=unique_ids(s["false_completion_ids"],open_ids)
            if hit & wrong or (not s["text"].strip() and (s["claims"] or hit or wrong)):
                raise ValueError(f"Inconsistent coverage or claims on an empty summary at {k}/{s['label']}.")
            if hit and not supported:
                raise ValueError(f"Covered facts have no supported claim at {k}/{s['label']}.")
            if wrong and all(c['verdict']=='有据' for c in s['claims']):
                raise ValueError(f"A false completion must have a conflicting or unsupported claim at {k}/{s['label']}.")
            entries[s["label"]]=[supported,len(s["claims"]),len(hit),len(facts),len(wrong),len(open_ids)]
        records.append({arm:entries[keys[k][arm]] for arm in ARMS})
    return records,state


def score_reviews(study_id,chosen):
    import numpy as np
    records,state=collect(study_id,chosen)
    data=np.asarray([[r[a] for a in ARMS] for r in records],dtype=int)
    metrics=("supported_claim_fraction","key_fact_coverage","false_completion_fraction")
    def ratios(x):
        numer=x[...,::2]
        denom=x[...,1::2]
        return np.divide(numer,denom,out=np.full(numer.shape,np.nan),where=denom>0)
    def number(x):
        return float(x) if np.isfinite(x) else None
    total=data.sum(axis=0)
    point=ratios(total)
    ix=np.random.default_rng(SEED).integers(0,len(records),(10000,len(records)))
    draws=ratios(data[ix].sum(axis=1))
    contrasts={}
    for left,right in (("F","A"),("F","QA_list"),("A","base"),("QA_list","A")):
        a,b=ARMS.index(left),ARMS.index(right)
        ci={}
        for col,m in enumerate(metrics):
            delta=draws[:,a,col]-draws[:,b,col]
            valid=delta[np.isfinite(delta)]
            ci[m]={"delta":number(point[a,col]-point[b,col]),"ci95":
                list(map(float,np.quantile(valid,[.025,.975]))) if len(valid)==len(draws) else None}
        contrasts[left+"_minus_"+right]=ci
    raw=state["results"]
    report={"cohort":"fresh development only","primary_comparison":PRIMARY_COMPARISON,"n":len(records),
        "source_review_sha256":file_hash(OUT/"source_facts.json"),"blind_review_sha256":file_hash(OUT/"blind_summaries.json"),
        "bootstrap_replicates":len(draws),"bootstrap_seed":SEED,
        "count_order":["supported_claims","all_judgeable_claims","covered_facts","all_key_facts","false_completions","unresolved_items"],
        "counts":{a:total[i].tolist() for i,a in enumerate(ARMS)},
        "metrics":{a:{m:number(point[i,j]) for j,m in enumerate(metrics)} for i,a in enumerate(ARMS)},
        "empty_outputs":{a:sum(not r['systems'][a].strip() for r in raw) for a in ARMS},
        "no_judgeable_claims":{a:int((data[:,i,1]==0).sum()) for i,a in enumerate(ARMS)},
        "invalid_qa_fallbacks":sum(r['generation']['QA_fallback'] for r in raw),
        "F_context_fallbacks":sum(r['generation'].get('F',{}).get('context_limit',False) for r in raw),
        "paired_dialogue_bootstrap":contrasts}
    atomic_json(OUT/"review_scores.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["prepare","predict","score"])
    parser.add_argument("--seen-ids",type=Path,help="Additional previously seen validation IDs, whitespace separated")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another run is using this cohort.")
        study_id,chosen=prepare(args.seen_ids)
        if args.phase=="predict":
            predict(study_id,chosen)
        elif args.phase=="score":
            score_reviews(study_id,chosen)
        print("Output directory:",OUT)
        if args.phase=="prepare":
            print("Read source_facts.json first. Reserved official test data were not opened.")


if __name__=="__main__":
    main()
