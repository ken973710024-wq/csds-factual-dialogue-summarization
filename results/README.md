# Reported aggregate results

`handoff_dev60_counts.csv` transcribes the integer counts in the supplied internship project card. `python scripts/show_results.py` calculates pooled ratios and matched point differences from these counts. It does not reproduce the AI judgments or bootstrap intervals: the original per-dialogue scores, blinded annotations and raw predictions were not supplied for the public release. The private backup's `reviewed/handoff128_dev60/review_scores.json` is the original scoring report.

Evaluation scope: 60 development dialogues, 239 source-grounded key facts, 180 outputs from three systems, 169 distinct summaries after same-dialogue deduplication, one blinded AI reviewer. Do not present these as official test or independently human-verified results.
