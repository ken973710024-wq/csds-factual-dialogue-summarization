# Reproduction and local artifacts

## Scope and environment

Run all commands from the repository root. The historical project was run on a Linux CUDA server. Python 3.12 is recommended; original exact package versions are recorded in the author's private `backup_info.json` and comparison manifests, not supplied with this release. `requirements.txt` lists dependencies inferred from the source and is not a tested lockfile. Use the archived versions and a matching PyTorch/CUDA build when available. Some older scripts depend on the Transformers chat-template return format; do not silently change it for historical comparisons. The final training script explicitly requests token IDs. CPU tests require no model downloads.

## Required local artifacts (not committed)

| Artifact | Expected path |
|---|---|
| CSDS original data | `data/CSDS/train.json`, `data/CSDS/val.json` |
| Exact Qwen base | `models/Qwen3-4B-Instruct-2507/` |
| Direct SFT adapter A | `outputs/qlora_sft/final_adapter/` |
| QA Stage 1 adapter | `outputs/qlora_qa_stage1/final_adapter/` |
| Stage 2 study adapters/data | `outputs/qlora_summary_matched_*_pilot512/`, `data/qa_stage2_research/` |
| Fixed final training candidates and manifest | `csds_handoff128/handoff_targets_128.jsonl`, `csds_handoff128/data_manifest.json` |
| Final matched adapters and training manifest | `outputs/summary_handoff_pilot128/` |
| Final evaluation inputs and predictions | `outputs/summary_handoff_pilot128_dev60/` |

The original CSDS data should be obtained from its upstream distribution, subject to its terms. The exact handoff candidates, previously inspected IDs, trained adapters and annotated scores are private experiment artifacts; there is no public download for them in this release. Keep them outside Git or restore locally under ignored paths. A new set of candidates is a new experiment, not exact historical reproduction.

## Direct model and early studies

For a fresh authorized local workspace with the base and train/val data:

```bash
python src/prepare_sft.py
python src/train_full_sft.py
python src/prepare_qa_sft_data.py
python src/train_qa_lora.py
```

These are original training entry points and may require the archived package versions. Inspect their paths and options before launching. They write to `data/` and `outputs/`; never run training in the only copy of the private backup. Earlier baseline scripts can generate outputs needed by the historical validators; consult script docstrings for specific prerequisites.

With the generated SFT data and trained QA adapter, the paired Stage 2 pilot entry points are:

```bash
python src/prepare_generated_train_qa_batched.py --sample-size 512 --batch-size 4
python src/prepare_summary_qa_training.py --sample-size 512 --max-seq-tokens 3072
python src/train_summary_qa_conditioned.py --tag pilot512 --arm direct
python src/train_summary_qa_conditioned.py --tag pilot512 --arm aware
python src/validate_summary_qa_training.py --tag pilot512 --check
```

Validation needs the earlier evaluation artifacts specified by the validator. These commands are source-derived, not newly executed GPU runs.

## Main matched handoff experiment (requires private candidates)

After restoring the required private candidates, data, exact base, and A adapter:

```bash
python csds_handoff128/train_summary_handoff.py check
python csds_handoff128/train_summary_handoff.py both
python csds_handoff128/validate_summary_handoff.py prepare
```

At this point, finish and lock source facts using only the selected source dialogues. Run `predict` only after that is complete. Complete the blinded claim-level review without revealing model identities, then run `score`:

```bash
python csds_handoff128/validate_summary_handoff.py predict
python csds_handoff128/validate_summary_handoff.py score
```

Use `--seen-ids` at the prepare stage for additional previously examined development IDs when needed. Existing manifests and hashes guard against inconsistent reuse. Do not overwrite a fixed experiment.

To re-score the historical experiment in a private copy, use the original backup and restore the fixed scored JSON files from `reviewed/handoff128_dev60/` into `outputs/summary_handoff_pilot128_dev60/` before invoking `score`. This requires the exact original Python source, candidate manifest and data. The public repository preserves original Python bytes to reduce fingerprint drift. The complete reviewed artifacts were not supplied for this packaging pass; historical re-scoring was not executed here.

## CPU-only validation

```bash
python -m unittest discover -s tests -v
PYTHONPATH=src python tests/test_study_contract.py
python scripts/demo_summary.py --prompt-only
python scripts/show_results.py
```

The first two are the existing project's tests. The last two are release utilities; they do not perform GPU inference or recompute the blinded assessment.
