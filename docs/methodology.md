# Methodology and research findings

## 1. Direct summarization and QA extraction

The direct summarizer uses Qwen3-4B-Instruct-2507 with parameter-efficient 4-bit QLoRA. `src/prepare_sft.py` constructs a prompt and a numbered target from the CSDS training records. `src/train_full_sft.py` masks prompt tokens from supervised loss. QA preparation and training use `src/prepare_qa_sft_data.py` and `src/train_qa_lora.py`. The QA format includes category, question, answer, question IDs and evidence IDs. Generated QA rather than reference QA is used as Stage 2 context.

The controlled QA-aware pipeline compares a common direct summary prompt against the same summarizer using QA extracted by the base model or the QA-LoRA model. The Stage 2 paired training scripts construct examples with identical dialogue IDs and summary targets; the QA context differs. Length filtering removes a pair from both arms if either is too long. The CPU contract tests check the separation of reference labels from inputs, paired targets, and prompt masking. The pilot512 adapters are present in the private archive; a full matched pair is not verified.

## 2. Source-grounded diagnosis

The root `validate_*.py` modules and fact-repair scripts investigate errors in QA transmission, final-state tracking, evidence revision, missing facts, and source-first summarization. These modules are earlier research stages, not claims of performance improvement. A useful distinction is lexical overlap (character-level ROUGE-L in earlier experiments) versus factual correctness measured against source dialogue.

## 3. Matched target intervention (main experiment)

`csds_handoff128/train_summary_handoff.py` continues two copies of the same A adapter on the same 128 training dialogues, with the same handoff prompt and optimization settings. Original uses the original summary targets; Handoff uses AI-assisted handoff-style targets. Source-turn evidence and input hashes are checked. The script records the supervised token counts separately, checks prompt equality, and verifies the final adapter and optimizer step count. The reported configuration is 2 epochs, batch size 1, gradient accumulation 8, 32 optimizer updates per arm, learning rate 5e-5, seed 42, and maximum sequence length 3072.

The A baseline uses its own original direct prompt at evaluation; Original and Handoff share the handoff prompt. Accordingly, the primary comparison is Handoff vs Original, not Handoff vs A. All three use greedy decoding with a cap of 384 new tokens.

## 4. Evaluation protocol

`csds_handoff128/validate_summary_handoff.py` selects 60 development dialogues after excluding known previously inspected development IDs and matching training dialogue IDs/text. Reference summaries and QA annotations are stripped from the source before preparation. Key facts are locked before predictions are reviewed. Predictions are blinded; identical outputs for one dialogue are assessed once. Claims are assessed as supported, contradicted or unsupported against source turns, and full matches to key facts determine coverage. The actual reported review was performed by a single AI reviewer, not independent human reviewers.

- Key-fact coverage = fully covered pre-established key facts / all pre-established key facts (239 pooled across dialogues).
- Source-supported claim fraction = supported judged claims / all judged claims written by the corresponding system. The denominator varies by system.
- The scoring implementation uses 10,000 paired bootstrap replicates, resampling whole dialogues and computing pooled ratios; two-sided 95% intervals are available in the private scoring report. This release does not invent or transcribe unavailable interval endpoints.

## 5. Interpretation

Handoff targets improved coverage relative to matched Original continuation but also increased unsupported or contradicted content as a share of judged claims. The project retained A rather than adopting Handoff for a reliability-sensitive application. The study suggests a coverage–faithfulness trade-off under this intervention, not a general causal claim for all models or datasets.

### Illustrative fictional case (not an evaluation record)

A customer is told that a courier has been asked to call back. A summary saying that the courier has already called back changes a pending action into a completed action. This can cause the next agent to skip follow-up. The synthetic example in `examples/` demonstrates this distinction without publishing a real dialogue.

## 6. Limitations and next steps

The reported cohort is a development set used for model selection. The evaluation relies on a single AI judge and AI-assisted target drafting; independent human review is needed to quantify judge bias. The matched training study is small and target lengths vary. No new-model official test result, runtime or memory benchmark is asserted. Next steps are an independent human audit, evaluation on an untouched cohort, and same-GPU latency/peak-memory measurement under matched input and decoding settings.
