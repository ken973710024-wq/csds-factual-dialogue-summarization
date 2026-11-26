"""Shared data and model setup for the matched handoff training experiment."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import sys

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import validate_unseen_summaries as review
from qa_summary_study_common import BASE_MODEL, load_train
from validate_qa_aware_pipeline import SUMMARY_ADAPTER

OUT = Path("outputs/summary_handoff_pilot128")
ARMS = ("original", "handoff")
SEED = 42
MAX_SEQUENCE = 3072
DECODE_MAX = 384
INSTRUCTION = (
    "请给下一位客服写一份简洁的中文交接摘要。保留本次实际诉求、重要答复、"
    "最后确认的处理状态，以及会影响下一步行动的条件、时限或责任人。"
    "分清用户或商家的说法与客服核实的结果。原文仍未确定的事项按原有确定程度表达，"
    "同一事项有更新时写清最新进展。只写有对话依据且值得交接的信息，保留脱敏标记。"
    "按要点编号输出，无需引用行号。"
)


def make_prompt(dialogue):
    return INSTRUCTION + "\n\n对话\n" + dialogue


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"另一进程正在使用 {directory}") from exc
        yield


def freeze(path, value):
    if path.exists():
        if review.read(path) != value:
            raise ValueError(f"已有记录与本次输入不同，已保留原文件 {path}")
    else:
        review.atomic_json(path, value)


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_candidates():
    path = PKG / "handoff_targets_128.jsonl"
    info = review.read(PKG / "data_manifest.json")
    if info["version"] != "csds_handoff_pilot128_v1" or review.file_hash(path) != info["targets_sha256"]:
        raise ValueError("训练候选答案或文件版本与清单不符")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ids = review.by_id(rows)
    if len(ids) != 128 or len({r["dialogue"] for r in rows}) != 128:
        raise ValueError("预期有 128 个不同的训练 ID 和对话文本")
    if ids.keys() & set(info["previously_seen_dev_ids"]):
        raise ValueError("训练名单与历史开发名单相交")
    return info, rows


def matched_pairs(candidates, train, info):
    indexed = review.by_id(train)
    pairs = []
    for row in candidates:
        k = str(row["DialogueID"])
        if k not in indexed:
            raise KeyError(f"{k} 不在本地原始训练集中")
        dialogue, turns = review.review_source(indexed[k]["source"])
        if row["dialogue"] != dialogue:
            raise ValueError(f"{k} 的训练对话与抽样时不同")
        target = indexed[k]["target"]
        prior = info["seed_original_target_sha256"].get(k)
        if prior is not None and text_sha(target) != prior:
            raise ValueError(f"{k} 的原训练答案与上一批 30 条不一致")
        lines = row["handoff_target"].splitlines()
        evidence = row["evidence"]
        if not lines or len(lines) != len(evidence):
            raise ValueError(f"{k} 的训练目标与证据行数不符")
        for n, (line, item) in enumerate(zip(lines, evidence, strict=True), 1):
            if (not line.startswith(f"{n}. ") or not line[len(f"{n}. "):].strip()
                    or item["target_line"] != f"T{n}"
                    or not review.unique_ids(item["source_turns"], turns)):
                raise ValueError(f"{k}/T{n} 缺少有效的原文依据")
        pairs.append({"DialogueID": k, "dialogue": dialogue, "original": target, "handoff": row["handoff_target"]})
    return pairs


def prepare_pairs():
    info, candidates = read_candidates()
    pairs = matched_pairs(candidates, load_train(), info)
    with locked(OUT / "data"):
        freeze(OUT / "data/paired_targets.json", {"split": "train", "rows": pairs})
    return pairs


def weight_file(adapter):
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        path = adapter / name
        if path.is_file() and path.stat().st_size:
            return path
    raise FileNotFoundError(f"缺少 adapter 权重 {adapter}")


def base_identity():
    weights = sorted(BASE_MODEL.glob("*.safetensors")) or sorted(BASE_MODEL.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"缺少基础模型权重 {BASE_MODEL}")
    paths = set(weights) | {BASE_MODEL / "config.json", BASE_MODEL / "tokenizer_config.json"}
    for pat in ("*index.json", "generation_config.json", "tokenizer*", "*chat_template*", "vocab.*",
                "merges.txt", "added_tokens.json", "special_tokens_map.json"):
        paths.update(p for p in BASE_MODEL.glob(pat) if p.is_file())
    return {str(p): review.file_hash(p) for p in sorted(paths)}


def adapter_identity(adapter):
    names = (adapter / "adapter_config.json", weight_file(adapter))
    return {p.name: review.file_hash(p) for p in names}


def get_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base():
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("请在支持 BF16 的 CUDA 训练服务器运行")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL), quantization_config=quant, dtype=torch.bfloat16,
        device_map={"": 0}, local_files_only=True,
    )

