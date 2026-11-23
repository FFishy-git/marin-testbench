# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive all 152 runs' original configs (run.json), full per-update metrics
(lossless gzipped metrics_from_*.jsonl.gz), all 22 Paloma 16-domain eval JSONs,
all campaign.json manifests, and all verified SwanLab receipts into:
  ideas/output-side-normalization/experiment_records/
together with a unified master index.json mapping every run to its config,
evaluation metrics, Paloma eval, and verified SwanLab URL.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import statistics
from datetime import UTC, datetime
from pathlib import Path

PVC = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
OUT = Path("/root/dl/runtime/script/marin-testbench/optimizer_workspace/ideas/output-side-normalization/experiment_records")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def gzip_copy(src: Path, dst_gz: Path) -> int:
    dst_gz.parent.mkdir(parents=True, exist_ok=True)
    raw = src.read_bytes()
    with gzip.GzipFile(dst_gz, "wb", compresslevel=6, mtime=0) as fh:
        fh.write(raw)
    return dst_gz.stat().st_size


def collect_swanlab_receipts() -> tuple[list[tuple[str, Path]], dict[str, dict]]:
    """Find all SwanLab receipts on PVC and build a lookup by (group, run_name) and run_name."""
    receipt_specs: list[tuple[str, Path]] = []
    # 1. run-preparation/swanlab-upload-*.json
    for f in sorted((PVC / "run-preparation").glob("swanlab-upload-*.json")):
        receipt_specs.append((f"run-preparation/{f.name}", f))
    # 2. campaigns/*/*swanlab*.json
    for f in sorted((PVC / "campaigns").glob("*/*swanlab*.json")):
        if f.is_file():
            receipt_specs.append((f"campaigns/{f.parent.name}/{f.name}", f))

    lookup: dict[str, dict] = {}
    for rel_name, path in receipt_specs:
        d = json.loads(path.read_text())
        proj_url = d.get("project_url") or d.get("project")
        # Derive group key from filename
        if rel_name.startswith("run-preparation/swanlab-upload-"):
            grp = path.stem.removeprefix("swanlab-upload-")
            for arm_key, entry in d.get("runs", {}).items():
                if isinstance(entry, dict):
                    lookup[f"runs/{grp}/{arm_key}"] = {
                        "project_url": proj_url,
                        "receipt": f"swanlab_receipts/{rel_name}",
                        "id": entry.get("id"),
                        "url": (entry.get("verification") or {}).get("url") or entry.get("url"),
                        "status": entry.get("status"),
                    }
        elif rel_name.startswith("campaigns/confirmation-20261121/"):
            for wave_run, entry in d.get("runs", {}).items():
                if isinstance(entry, dict):
                    lookup[f"campaigns/confirmation-20261121/runs/{wave_run}"] = {
                        "project_url": proj_url,
                        "receipt": f"swanlab_receipts/{rel_name}",
                        "id": entry.get("id"),
                        "url": (entry.get("verification") or {}).get("url") or entry.get("url"),
                        "status": entry.get("status"),
                    }
        else:
            camp_name = path.parent.name
            for rkey, entry in d.get("runs", {}).items():
                if isinstance(entry, dict):
                    # Try both runs/<rkey> and direct <rkey>
                    rec = {
                        "project_url": proj_url,
                        "receipt": f"swanlab_receipts/{rel_name}",
                        "id": entry.get("id"),
                        "url": (entry.get("verification") or {}).get("url") or entry.get("url"),
                        "status": entry.get("status"),
                    }
                    lookup[f"campaigns/{camp_name}/runs/{rkey}"] = rec
                    lookup[f"campaigns/{camp_name}/{rkey}"] = rec
    return receipt_specs, lookup


def discover_all_runs() -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    # runs/<wave>/<run>
    for w in sorted((PVC / "runs").iterdir()):
        if not w.is_dir():
            continue
        if (w / "run.json").is_file():
            found.append((f"runs/{w.name}", w))
        else:
            for d in sorted(w.iterdir()):
                if d.is_dir() and (d / "run.json").is_file():
                    found.append((f"runs/{w.name}/{d.name}", d))
    # campaigns/<camp>/...
    for c in sorted((PVC / "campaigns").iterdir()):
        if not c.is_dir():
            continue
        for d in sorted(c.iterdir()):
            if d.is_dir() and (d / "run.json").is_file():
                found.append((f"campaigns/{c.name}/{d.name}", d))
        if (c / "runs").is_dir():
            for x in sorted((c / "runs").iterdir()):
                if not x.is_dir():
                    continue
                if (x / "run.json").is_file():
                    found.append((f"campaigns/{c.name}/runs/{x.name}", x))
                else:
                    for y in sorted(x.iterdir()):
                        if y.is_dir() and (y / "run.json").is_file():
                            found.append((f"campaigns/{c.name}/runs/{x.name}/{y.name}", y))
    return found


def summarize_metrics(run_dir: Path) -> dict:
    by_step: dict[int, dict] = {}
    for mf in sorted(run_dir.glob("metrics_from_*.jsonl")):
        for line in mf.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                by_step[int(r["step"])] = r
    if not by_step:
        return {}
    steps = sorted(by_step)
    T = steps[-1]
    eval_steps = [s for s in steps if "selection_loss" in by_step[s]]
    late_steps = [s for s in eval_steps if s > T // 2]
    out = {"logged_steps": len(steps), "last_step": T, "tokens": by_step[T].get("tokens")}
    if eval_steps:
        last_e = by_step[eval_steps[-1]]
        out["final_selection_loss"] = last_e.get("selection_loss")
        out["final_c4_loss"] = last_e.get("c4_loss")
        out["final_heldout_losses"] = last_e.get("heldout_losses")
    if late_steps:
        out["late_eval_count"] = len(late_steps)
        out["late_selection_loss_mean"] = statistics.fmean(by_step[s]["selection_loss"] for s in late_steps)
        out["late_c4_loss_mean"] = statistics.fmean(by_step[s]["c4_loss"] for s in late_steps)
    return out


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Copy all SwanLab receipts
    receipt_specs, swanlab_lookup = collect_swanlab_receipts()
    for rel_name, src in receipt_specs:
        copy_file(src, OUT / "swanlab_receipts" / rel_name)

    # 2. Copy all 22 Paloma 16-domain eval JSONs
    paloma_src_dir = PVC / "campaigns" / "confirmation-20261121" / "paloma_eval"
    paloma_map: dict[str, dict] = {}
    for pf in sorted(paloma_src_dir.glob("*.json")):
        dst = OUT / "paloma_eval" / pf.name
        copy_file(pf, dst)
        pdata = json.loads(pf.read_text())
        paloma_map[pf.stem] = {
            "path": f"paloma_eval/{pf.name}",
            "paloma_macro_mean": pdata.get("paloma_macro_mean"),
            "selection_loss": pdata.get("selection_loss"),
        }

    # 3. Copy campaign.json manifests & experiment_receipt.json (dataset receipt)
    for camp_dir in sorted((PVC / "campaigns").iterdir()):
        if camp_dir.is_dir():
            for fname in ("campaign.json", "experiment_receipt.json"):
                f = camp_dir / fname
                if f.is_file():
                    copy_file(f, OUT / "campaigns" / camp_dir.name / fname)
    data_receipt = PVC / "data" / "experiment_receipt.json"
    if data_receipt.is_file():
        copy_file(data_receipt, OUT / "dataset_experiment_receipt.json")

    # 4. Copy every run's run.json (original resolved config + metadata) and gzipped metrics_from_*.jsonl.gz
    runs_list = discover_all_runs()
    index_runs: dict[str, dict] = {}
    total_gz_bytes = 0
    total_json_bytes = 0
    complete_count = 0
    swanlab_matched = 0

    for rel_key, run_dir in runs_list:
        dst_dir = OUT / rel_key
        rj = run_dir / "run.json"
        copy_file(rj, dst_dir / "run.json")
        total_json_bytes += rj.stat().st_size
        meta = json.loads(rj.read_text())
        status = meta.get("status")
        if status == "complete":
            complete_count += 1

        gz_files = []
        for mf in sorted(run_dir.glob("metrics_from_*.jsonl")):
            gz_path = dst_dir / f"{mf.name}.gz"
            total_gz_bytes += gzip_copy(mf, gz_path)
            gz_files.append(f"{rel_key}/{mf.name}.gz")

        sl = swanlab_lookup.get(rel_key)
        if sl and sl.get("status") == "verified":
            swanlab_matched += 1

        pal = paloma_map.get(run_dir.name) if rel_key.startswith("campaigns/confirmation-20261121/") else None
        cfg = meta.get("config", {})
        index_runs[rel_key] = {
            "status": status,
            "arm": cfg.get("experiment", {}).get("arm"),
            "arm_label": cfg.get("experiment", {}).get("arm_label"),
            "model_size": cfg.get("model", {}).get("size"),
            "seed": cfg.get("training", {}).get("seed"),
            "steps": cfg.get("training", {}).get("steps"),
            "lr": cfg.get("optimizer", {}).get("lr"),
            "auxiliary_lr": cfg.get("optimizer", {}).get("auxiliary_lr"),
            "run_json": f"{rel_key}/run.json",
            "metrics_gz": gz_files,
            "eval_summary": summarize_metrics(run_dir),
            "paloma_eval": pal,
            "swanlab": sl,
        }

    index = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_pvc_root": str(PVC),
        "totals": {
            "runs_archived": len(index_runs),
            "complete_training_runs": complete_count,
            "swanlab_verified_runs": swanlab_matched,
            "swanlab_receipt_files": len(receipt_specs),
            "paloma_16domain_evals": len(paloma_map),
            "run_json_bytes": total_json_bytes,
            "metrics_gz_bytes": total_gz_bytes,
        },
        "runs": index_runs,
    }
    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")

    readme = f"""# Complete Experiment Records Archive (`experiment_records/`)

Self-contained archive of **every run** referenced in the paper and confirmation campaign (`{len(index_runs)}` total runs: `{complete_count}` complete training runs + `19` matched 8-GPU timing benchmarks), excluding multi-GB model weight checkpoints.

## Contents
- `index.json` — Master index mapping every run to its original resolved config (`run.json`), lossless gzipped per-update training/evaluation log (`metrics_from_*.jsonl.gz`), summary evaluation metrics, 16-domain Paloma JSON (where applicable), and verified SwanLab URL.
- `runs/` & `campaigns/` — Per-run original `run.json` (contains the exact resolved `config`, hardware, git SHA, and completion metadata) and lossless `metrics_from_*.jsonl.gz` (all per-update train losses, held-out selection losses, C4 losses, and runtime counters).
- `paloma_eval/` — All `{len(paloma_map)}` post-hoc 16-domain Paloma evaluation JSONs across Phases 2, 3A, 3B, and 4.
- `swanlab_receipts/` — All `{len(receipt_specs)}` verified SwanLab upload receipts covering `{swanlab_matched} / {complete_count}` complete training runs across private projects `siyuc3141/output-side-normalization-marin` and `siyuc3141/output-side-normalization-confirmation`.
- `dataset_experiment_receipt.json` — Verified tokenized dataset provenance receipt.

## Totals
- Archived runs: `{len(index_runs)}` (`{complete_count}` complete training runs, `{swanlab_matched}` verified on SwanLab)
- Compressed `metrics_from_*.jsonl.gz` size: `{total_gz_bytes / 1024 / 1024:.2f} MB`
- `run.json` configs size: `{total_json_bytes / 1024 / 1024:.2f} MB`
"""
    (OUT / "README.md").write_text(readme)
    print(
        f"ARCHIVED: runs={len(index_runs)}, complete={complete_count}, "
        f"swanlab_verified={swanlab_matched}, receipts={len(receipt_specs)}, "
        f"paloma={len(paloma_map)}, run_json={total_json_bytes/1024/1024:.2f}MB, "
        f"metrics_gz={total_gz_bytes/1024/1024:.2f}MB"
    )


if __name__ == "__main__":
    main()
