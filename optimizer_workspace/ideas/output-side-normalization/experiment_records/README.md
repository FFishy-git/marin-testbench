# Complete Experiment Records Archive (`experiment_records/`)

Self-contained archive of **every run** referenced in the paper and confirmation campaign (`152` total runs: `133` complete training runs + `19` matched 8-GPU timing benchmarks), excluding multi-GB model weight checkpoints.

## Contents
- `index.json` — Master index mapping every run to its original resolved config (`run.json`), lossless gzipped per-update training/evaluation log (`metrics_from_*.jsonl.gz`), summary evaluation metrics, 16-domain Paloma JSON (where applicable), and verified SwanLab URL.
- `runs/` & `campaigns/` — Per-run original `run.json` (contains the exact resolved `config`, hardware, git SHA, and completion metadata) and lossless `metrics_from_*.jsonl.gz` (all per-update train losses, held-out selection losses, C4 losses, and runtime counters).
- `paloma_eval/` — All `22` post-hoc 16-domain Paloma evaluation JSONs across Phases 2, 3A, 3B, and 4.
- `swanlab_receipts/` — All `18` verified SwanLab upload receipts covering `133 / 133` complete training runs across private projects `siyuc3141/output-side-normalization-marin` and `siyuc3141/output-side-normalization-confirmation`.
- `dataset_experiment_receipt.json` — Verified tokenized dataset provenance receipt.

## Totals
- Archived runs: `152` (`133` complete training runs, `133` verified on SwanLab)
- Compressed `metrics_from_*.jsonl.gz` size: `42.21 MB`
- `run.json` configs size: `1.61 MB`
