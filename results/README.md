# Result Artifacts

- `leaderboard.csv`: exact aggregate values for the five selected runs.
- `metrics/*.metrics.json`: complete split and per-source-dataset reports.
- `summaries/*.summary.json`: parameters, runtime, memory, counts, and failures.
- `predictions/*.jsonl.gz`: complete raw outputs and parsed predictions.
- `SHA256SUMS`: checksums of the files published in this directory.

The selected LFM2.5 result is `lfm2_5_vl_450m_coco_prompt`. Its Task A/B rows
come from the full LFM2.5 run and its Task C rows come from the shorter selected
COCO-style caption run, as recorded by the summary JSON. The unselected verbose
caption run is also retained for prompt-comparison auditability.

Aborted LFM2-VL and SmolVLM2 runs, parser-bug backups, and incomplete generation
experiments are intentionally excluded because they are not valid benchmark
results.
