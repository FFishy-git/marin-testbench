# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export and plot the exact same-wave WT-MLP-Lag and WT-Sym-Lag histories."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parents[1]
RUNS = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/runs")
WAVE = "wave_sym_fastlag"
ARMS = ("inner_outer_mlpdiag_lag", "inner_outer_sym_mlpdiag_lag")
FIELDS = ("step", "tokens", "wt_mlp_lag", "wt_sym_lag", "sym_minus_mlp")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_comparison(runs_root):
    """Require completed, workload-matched runs and every ordered update."""
    manifests, histories, sources = [], [], []
    for arm in ARMS:
        directory = runs_root / WAVE / arm
        manifest_path = directory / "run.json"
        metrics_path = directory / "metrics_from_000000.jsonl"
        manifest = json.loads(manifest_path.read_text())
        config = manifest["config"]
        steps = config["training"]["steps"]
        if (
            manifest["status"] != "complete"
            or manifest["completed_steps"] != steps
            or manifest["execution_check"]
            or config["experiment"]["arm"] != arm
        ):
            raise ValueError(f"{arm}: expected a completed quality run")
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        if len(rows) != steps:
            raise ValueError(f"{arm}: incomplete update history")
        for step, row in enumerate(rows, 1):
            if row["step"] != step or row["tokens"] != step * config["training"]["batch_tokens"]:
                raise ValueError(f"{arm}: update or token sequence differs at row {step}")
            if not math.isfinite(row["train_loss"]) or row["train_loss"] < 0:
                raise ValueError(f"{arm}: invalid training loss at update {step}")
        manifests.append(manifest)
        histories.append(rows)
        sources.append(
            {
                "run": f"{WAVE}/{arm}",
                "manifest": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "metrics": str(metrics_path),
                "metrics_sha256": sha256(metrics_path),
                "command": manifest["command"],
                "source_revision": manifest["source_revision"],
                "source_sha256": manifest["source_sha256"],
                "data": manifest["data"],
                "data_receipt_sha256": manifest["data_receipt_sha256"],
                "gpu": manifest["gpu"],
                "world_size": manifest["world_size"],
            }
        )
    configs = []
    for manifest in manifests:
        config = manifest["config"]
        configs.append(
            {
                **config,
                "experiment": {key: value for key, value in config["experiment"].items() if key != "arm"},
            }
        )
    if configs[0] != configs[1]:
        raise ValueError("Comparison configurations differ beyond the arm")
    for field in ("source_revision", "source_sha256", "data", "data_receipt_sha256", "gpu", "world_size"):
        if manifests[0][field] != manifests[1][field]:
            raise ValueError(f"Comparison runs differ in {field}")
    paired = [
        {
            "step": mlp["step"],
            "tokens": mlp["tokens"],
            "wt_mlp_lag": mlp["train_loss"],
            "wt_sym_lag": sym["train_loss"],
            "sym_minus_mlp": sym["train_loss"] - mlp["train_loss"],
        }
        for mlp, sym in zip(*histories, strict=True)
    ]
    return paired, {"wave": WAVE, "sources": sources, "common_config_except_arm": configs[0]}


def export_comparison(rows, receipt, data_path, receipt_path):
    """Preserve round-trip float values and source identities for offline rebuilds."""
    with data_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter=" ", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    window = receipt["common_config_except_arm"]["schedule"]["stable_steps"]
    receipt.update(
        rows=len(rows),
        update_range=[rows[0]["step"], rows[-1]["step"]],
        training_loss="Logged global-batch mean over all accumulation passes and DDP ranks, before the update.",
        smoothing="none",
        downsampling="none",
        path_simplification=False,
        data_sha256=sha256(data_path),
        paired_window_means=[
            {
                "steps": [start + 1, min(start + window, len(rows))],
                "sym_minus_mlp": statistics.fmean(row["sym_minus_mlp"] for row in rows[start : start + window]),
            }
            for start in range(0, len(rows), window)
        ],
    )
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


def read_export(data_path, receipt_path):
    """Validate an exported series before plotting without access to the run share."""
    receipt = json.loads(receipt_path.read_text())
    if sha256(data_path) != receipt["data_sha256"]:
        raise ValueError("Exported training history differs from its receipt")
    with data_path.open(newline="") as handle:
        rows = [
            {key: int(value) if key in ("step", "tokens") else float(value) for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter=" ")
        ]
    if len(rows) != receipt["rows"]:
        raise ValueError("Exported training history has the wrong row count")
    return rows, receipt


def plot_comparison(rows, receipt, output):
    """Draw each recorded point, with a separate zoom for the early loss gap."""
    steps = [row["step"] for row in rows]
    mlp = [row["wt_mlp_lag"] for row in rows]
    sym = [row["wt_sym_lag"] for row in rows]
    differences = [row["sym_minus_mlp"] for row in rows]
    stable = receipt["common_config_except_arm"]["schedule"]["stable_steps"]
    early_start = 129
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "path.simplify": False,
        }
    )
    figure = Figure(figsize=(9.0, 6.6), layout="constrained")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.05, 1))
    full = figure.add_subplot(grid[0, :])
    early = figure.add_subplot(grid[1, 0])
    paired = figure.add_subplot(grid[1, 1])
    for axis in (full, early):
        axis.plot(steps, mlp, color="#1B4F9C", linewidth=0.65, label="WT-MLP-Lag")
        axis.plot(steps, sym, color="#C0392B", linewidth=0.65, label="WT-Sym-Lag")
        axis.set_ylabel("Training cross-entropy (nats)")
    full.set_title("(a) Exact full-budget training losses", loc="left", weight="bold")
    full.set_xlim(1, steps[-1])
    full.set_xlabel("Completed training updates")
    full.legend(loc="upper right", frameon=False, ncol=2)
    full.axvline(stable, color="#5F6B73", linewidth=0.8, linestyle="--")
    full.annotate(
        "Linear LR decay begins",
        xy=(stable, 0.88),
        xycoords=("data", "axes fraction"),
        xytext=(8, 0),
        textcoords="offset points",
        color="#5F6B73",
        fontsize=8,
    )
    early.set_title("(b) Constant-LR zoom: updates 129-1,024", loc="left", weight="bold")
    early.set_xlim(early_start, stable)
    zoom = [row[key] for row in rows[early_start - 1 : stable] for key in ("wt_mlp_lag", "wt_sym_lag")]
    early.set_ylim(min(zoom) - 0.06, max(zoom) + 0.06)
    early.set_xlabel("Completed training updates")
    paired.plot(steps, differences, color="#0E7470", linewidth=0.55)
    paired.axhline(0, color="#5F6B73", linewidth=0.8)
    paired.axvline(stable, color="#5F6B73", linewidth=0.8, linestyle="--")
    paired.set_title("(c) Exact paired training-loss difference", loc="left", weight="bold")
    paired.set_ylabel("WT-Sym-Lag minus WT-MLP-Lag (nats)")
    paired.set_xlabel("Completed training updates")
    paired.set_xlim(1, steps[-1])
    for axis in (full, early, paired):
        axis.grid(color="#D2D7DB", linewidth=0.4, alpha=0.7)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.tick_params(labelsize=8)
    figure.savefig(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=RUNS)
    parser.add_argument("--from-data", action="store_true", help="Replot the checked export without reading run logs.")
    parser.add_argument("--data", type=Path, default=HERE / "figs/wt_lag_training.dat")
    parser.add_argument("--receipt", type=Path, default=HERE / "results/wt_lag_training.json")
    parser.add_argument("--figure", type=Path, default=HERE / "figs/wt_lag_training.pdf")
    args = parser.parse_args()
    if args.from_data:
        rows, receipt = read_export(args.data, args.receipt)
    else:
        rows, receipt = load_comparison(args.runs_root)
        export_comparison(rows, receipt, args.data, args.receipt)
    plot_comparison(rows, receipt, args.figure)
    print(f"Plotted {len(rows)} exact updates per arm: {args.figure}")


if __name__ == "__main__":
    main()
