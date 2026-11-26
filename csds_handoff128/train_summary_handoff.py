"""Train two matched copies of the existing A adapter on the same 128 train dialogues."""
import argparse
from functools import partial
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import handoff_common as h
from train_summary_qa_conditioned import prepare_tokenized

EPOCHS = 2
ACCUM = 8
LR = 5e-5
EXPECTED_STEPS = 128 * EPOCHS // ACCUM


def encode_pairs(pairs, tokenizer):
    # Request token IDs explicitly on Transformers versions that also return BatchEncoding.
    chat = SimpleNamespace(apply_chat_template=partial(tokenizer.apply_chat_template, return_dict=False))
    encoded, stats = {}, {}
    for arm in h.ARMS:
        records = [{"DialogueID": row["DialogueID"], "prompt": h.make_prompt(row["dialogue"]), "target": row[arm]}
                   for row in pairs]
        examples = prepare_tokenized(chat, records, h.MAX_SEQUENCE)
        counts = [sum(v != -100 for v in e["labels"]) for e in examples]
        stats[arm] = {"n": len(examples), "supervised_tokens_per_epoch": sum(counts),
                      "max_sequence_tokens": max(len(e["input_ids"]) for e in examples)}
        encoded[arm] = examples
    for row, left, right in zip(pairs, encoded["original"], encoded["handoff"], strict=True):
        n = left["labels"].count(-100)
        if (n != right["labels"].count(-100) or left["input_ids"][:n] != right["input_ids"][:n]
                or not n or min(len(left["labels"]), len(right["labels"])) <= n):
            raise ValueError(f"{row['DialogueID']} 两组的 prompt 或监督位置不匹配")
    return encoded, stats


def conditions(pairs, stats):
    from train_summary_qa_conditioned import __file__ as tokenizer_helper
    return {
        "protocol": "csds_handoff_same_input_different_target_v1",
        "pair_data_sha256": h.review.digest(pairs), "prompt_sha256": h.text_sha(h.INSTRUCTION),
        "base": str(h.BASE_MODEL), "base_files": h.base_identity(),
        "initial_adapter": str(h.SUMMARY_ADAPTER), "initial_adapter_files": h.adapter_identity(h.SUMMARY_ADAPTER),
        "lora_config": "inherited from initial adapter", "seed": h.SEED, "data_seed": h.SEED,
        "batch_size": 1, "gradient_accumulation": ACCUM, "epochs": EPOCHS, "lr": LR,
        "scheduler": "cosine", "warmup_steps": 1, "optimizer": "adamw_torch", "gradient_clip": 1.0,
        "precision": "NF4 / double quantization / bf16", "max_sequence": h.MAX_SEQUENCE,
        "optimizer_steps_per_arm": EXPECTED_STEPS, "token_counts": stats,
        "code": {Path(p).name: h.review.file_hash(Path(p)) for p in (__file__, h.__file__, tokenizer_helper)},
        "versions": {pkg: version(pkg) for pkg in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")},
    }


def completed(final, config):
    if not final.exists():
        return False
    saved = h.review.read(final / "completion.json")
    if (saved["training_config_sha256"] != h.review.digest(config)
            or saved["optimizer_steps"] != EXPECTED_STEPS
            or saved["adapter_files"] != h.adapter_identity(final)):
        raise ValueError(f"已有 adapter 与本次实验不符，已保留 {final}")
    return True


def checkpoint_to_resume(directory):
    # A power cut can leave an incomplete last checkpoint.
    checkpoints = (p for p in directory.glob("checkpoint-*") if p.name[11:].isdigit())
    for path in sorted(checkpoints, key=lambda p: int(p.name[11:]), reverse=True):
        needed = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json")
        if all((path / name).is_file() and (path / name).stat().st_size for name in needed):
            try:
                state = h.review.read(path / "trainer_state.json")
                h.weight_file(path)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if state.get("global_step") == int(path.name[11:]):
                return str(path)
    return None


def train(arm, examples, config):
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, default_data_collator, set_seed

    directory = h.OUT / arm
    final = directory / "final_adapter"
    with h.locked(directory):
        config = {**config, "target_arm": arm}
        h.freeze(directory / "training_manifest.json", config)
        if completed(final, config):
            print(f"已完成，复用 {final}", flush=True)
            return
        set_seed(h.SEED)
        base = h.load_base()
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(base, gradient_checkpointing_kwargs={"use_reentrant": False})
        model = PeftModel.from_pretrained(base, str(h.SUMMARY_ADAPTER), is_trainable=True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not trainable:
            raise RuntimeError("起始 adapter 没有可训练参数")
        args = TrainingArguments(
            output_dir=str(directory), num_train_epochs=EPOCHS, per_device_train_batch_size=1,
            gradient_accumulation_steps=ACCUM, learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=1,
            max_grad_norm=1.0, optim="adamw_torch", bf16=True, seed=h.SEED, data_seed=h.SEED,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="no", save_strategy="epoch", save_total_limit=2, report_to="none",
            logging_steps=4, logging_first_step=True, remove_unused_columns=False, dataloader_num_workers=0,
        )
        # Per-device batch size is one, so the standard collator needs no padding across examples.
        trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=default_data_collator)
        checkpoint = checkpoint_to_resume(directory)
        if checkpoint:
            print(f"从 {checkpoint} 继续", flush=True)
        trainer.train(resume_from_checkpoint=checkpoint)
        steps = int(trainer.state.global_step)
        if steps != EXPECTED_STEPS:
            raise RuntimeError(f"预期 {EXPECTED_STEPS} 个优化步，实际 {steps}")
        with tempfile.TemporaryDirectory(prefix=".export-", dir=directory) as tmp:
            staging = Path(tmp) / "final_adapter"
            trainer.save_model(str(staging))
            h.review.atomic_json(staging / "completion.json", {
                "arm": arm, "optimizer_steps": steps, "trainable_parameters": trainable,
                "training_config_sha256": h.review.digest(config), "adapter_files": h.adapter_identity(staging),
            })
            staging.rename(final)
        print(f"完成 {arm}，{steps} 个优化步，已保存 {final}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check", "both", *h.ARMS))
    args = parser.parse_args()
    os.chdir(h.ROOT)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("此实验固定为单卡、单进程训练")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0]
    if args.phase == "both":
        for arm in h.ARMS:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), arm], check=True)
        return
    pairs = h.prepare_pairs()
    encoded, stats = encode_pairs(pairs, h.get_tokenizer())
    config = conditions(pairs, stats)
    with h.locked(h.OUT / "data"):
        h.freeze(h.OUT / "data/comparison_manifest.json", config)
    print(json.dumps({"n": len(pairs), "identical_prompts": True, "optimizer_steps_per_arm": EXPECTED_STEPS,
                      "initial_adapter": str(h.SUMMARY_ADAPTER), "token_counts": stats},
                     indent=2, ensure_ascii=False), flush=True)
    if args.phase != "check":
        train(args.phase, encoded[args.phase], config)


if __name__ == "__main__":
    main()

