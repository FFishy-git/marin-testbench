# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit measured eight-A100 training timings without launching or modifying runs."""

import argparse
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path

from compare_runs import PVC, RunHistory, history, pairing
from freeze_wave5 import sha256

TASK = Path(__file__).resolve().parents[1]
METHODS = {
    "muont": "MuonT",
    "muon": "Muon",
    "inner_outer_mlpdiag": "WT-MLP",
    "inner_outer_mlpdiag_lag": "WT-MLP-Lag",
    "inner_outer_sym_mlpdiag": "WT-Sym",
    "inner_outer_sym_mlpdiag_lag": "WT-Sym-Lag",
    "inner_outer_leftdiag": "OD-All",
    "outer_leftdiag_upstream_fused": "OD-Outer",
    "inner_outer_all": "DT-All",
    "inner_outer_all_lag": "DT-All-Lag",
    "null_mlpdiag": "Null-WT",
    "null_all": "Null-Dense",
    "null_leftdiag_upstream_fused": "Null-OD",
    "outer_left_all": "Abl-OuterDense",
    "inner_outer_hidden": "Abl-No2048",
    "outer_left_down": "Abl-DownOnly",
    "down_hybrid": "Abl-DownSwap",
    "left_all": "Abl-OutOnly",
    "inner_outer_all_diag": "Abl-BothDiag",
}
BENCH_WINDOW = (37, 96)
FULL_WINDOW = (2049, 4096)
REPORT_START = "<!-- measured-8gpu-audit:start -->"
REPORT_END = "<!-- measured-8gpu-audit:end -->"


def timing_window(rows: list[dict], first: int, last: int) -> dict:
    """Summarize every consecutive training update, retaining all timing outliers."""
    selected = [row for row in rows if first <= row["step"] <= last]
    if [row["step"] for row in selected] != list(range(first, last + 1)):
        raise ValueError(f"Incomplete timing window {first}..{last}")
    times = [row["step_seconds"] for row in selected]
    if any(value <= 0 for value in times):
        raise ValueError("Training step times must be positive")
    mean = statistics.fmean(times)
    midpoint = len(times) // 2
    halves = [statistics.fmean(times[:midpoint]), statistics.fmean(times[midpoint:])]
    return {
        "window": [first, last],
        "measured_updates": len(times),
        "mean_step_seconds": mean,
        "median_step_seconds": statistics.median(times),
        "std_step_seconds": statistics.stdev(times),
        "min_step_seconds": min(times),
        "max_step_seconds": max(times),
        "total_step_seconds": sum(times),
        "half_window_means": halves,
        "half_window_change_pct": 100 * (halves[1] / halves[0] - 1),
        "coefficient_of_variation_pct": 100 * statistics.stdev(times) / mean,
        "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in rows) / 2**30,
        "input_fallbacks_in_window": (
            selected[-1].get("input_fallbacks", 0) - rows[first - 2].get("input_fallbacks", 0)
        ),
        "output_fallbacks_in_window": (
            selected[-1].get("output_fallbacks", 0) - rows[first - 2].get("output_fallbacks", 0)
        ),
    }


def measured_run(directory: Path, arm: str, first: int, last: int) -> tuple[dict, RunHistory]:
    run = history(directory)
    manifest = run.manifest
    config = manifest["config"]
    if manifest["world_size"] != 8 or manifest["gpu"] != "NVIDIA A100-SXM4-40GB":
        raise ValueError(f"{directory}: expected measured eight-A100-SXM4-40GB training")
    if config["experiment"]["arm"] != arm or config["training"]["world_size"] != 8:
        raise ValueError(f"{directory}: method or configured world size mismatch")
    if config != json.loads((directory / "config.json").read_text()):
        raise ValueError(f"{directory}: persisted configuration differs from run manifest")
    if manifest["status"] not in ("complete", "execution_check_complete"):
        raise ValueError(f"{directory}: run did not finish successfully")
    if manifest["completed_steps"] != len(run.rows):
        raise ValueError(f"{directory}: completed-step count differs from metrics")
    if config["runtime"]["compile"] == "eager" or ("root" in config and not config["root"]["graphs"]):
        raise ValueError(f"{directory}: compilation or root graphs were disabled")
    result = timing_window(run.rows, first, last)
    evidence_files = [directory / "run.json", directory / "config.json", *sorted(directory.glob("metrics_*.jsonl"))]
    result.update(
        {
            "run_directory": str(directory),
            "status": manifest["status"],
            "completed_steps": manifest["completed_steps"],
            "start_time": manifest["start_time"],
            "end_time": manifest["end_time"],
            "world_size": manifest["world_size"],
            "gpu": manifest["gpu"],
            "torch": manifest["torch"],
            "cuda": manifest["cuda"],
            "driver": manifest["driver"],
            "source_sha256": manifest["source_sha256"],
            "data_receipt_sha256": manifest["data_receipt_sha256"],
            "command": manifest["command"],
            "config": config,
            "files_sha256": {path.name: sha256(path) for path in evidence_files},
            "init_model_from": manifest.get("init_model_from"),
            "data_step_offset": manifest.get("data_step_offset", 0),
            "optimizer_state": "fresh" if manifest.get("bench_midtrain") else "training_trajectory",
        }
    )
    return result, run


def audit(root: Path) -> dict:
    """Verify shared benchmark provenance and inventory full-budget records."""
    bench = root / "runs" / "bench_8gpu_opt"
    checkpoint = root / "runs" / "wave_sym_fastlag" / "muont" / "checkpoints" / "step_002500"
    receipt = json.loads((checkpoint / "complete.json").read_text())
    if receipt["step"] != 2500 or receipt["world"] != 8:
        raise ValueError("Mid-training initialization checkpoint has a mismatched identity")
    baseline, base_run = measured_run(bench / "muont", "muont", *BENCH_WINDOW)
    source = bench / "_code"
    changed = [
        name
        for name, expected in baseline["source_sha256"].items()
        if sha256(source / name) != expected
    ]
    if changed:
        raise ValueError(f"Benchmark source no longer matches its manifest: {changed}")
    data_digest = sha256(root / "data" / "experiment_receipt.json")
    missing = []
    methods = {}
    for arm, tax_id in METHODS.items():
        directory = bench / arm
        if not (directory / "run.json").is_file():
            missing.append(arm)
            methods[arm] = {"tax_id": tax_id, "measured_benchmark": None}
            continue
        result, run = measured_run(directory, arm, *BENCH_WINDOW)
        manifest = run.manifest
        if (
            not manifest.get("bench_midtrain")
            or manifest.get("data_step_offset") != 2500
            or manifest.get("init_model_from") != str(checkpoint)
            or manifest["data_receipt_sha256"] != data_digest
            or manifest["source_sha256"] != baseline["source_sha256"]
        ):
            raise ValueError(f"{directory}: benchmark provenance differs")
        pairing(base_run, run)
        for key in ("training", "runtime", "schedule"):
            if run.manifest["config"][key] != base_run.manifest["config"][key]:
                raise ValueError(f"{directory}: shared {key} settings differ from MuonT")
        optimizer = {key: value for key, value in run.manifest["config"]["optimizer"].items() if key != "name"}
        base_optimizer = {key: value for key, value in base_run.manifest["config"]["optimizer"].items() if key != "name"}
        if arm != "muon" and optimizer != base_optimizer:
            raise ValueError(f"{directory}: shared optimizer settings differ from MuonT")
        if arm != "muon":
            for key in ("root", "statistics"):
                if run.manifest["config"][key] != base_run.manifest["config"][key]:
                    raise ValueError(f"{directory}: shared {key} settings differ from MuonT")
        if any("eval_seconds" in row or "checkpoint_seconds" in row for row in run.rows):
            raise ValueError(f"{directory}: benchmark includes evaluation or checkpoint IO")
        result["mean_overhead_vs_muont_pct"] = 100 * (
            result["mean_step_seconds"] / baseline["mean_step_seconds"] - 1
        )
        result["median_overhead_vs_muont_pct"] = 100 * (
            result["median_step_seconds"] / baseline["median_step_seconds"] - 1
        )
        methods[arm] = {"tax_id": tax_id, "measured_benchmark": result, "full_budget_records": []}

    for manifest_path in sorted((root / "runs").glob("*/*/run.json")):
        manifest = json.loads(manifest_path.read_text())
        config = manifest["config"]
        arm = config["experiment"]["arm"]
        training = config["training"]
        if (
            arm not in methods
            or manifest["status"] != "complete"
            or manifest.get("execution_check")
            or training["batch_tokens"] != 524288
            or training["seed"] != 0
            or manifest.get("completed_steps") != 5120
        ):
            continue
        result, _ = measured_run(manifest_path.parent, arm, *FULL_WINDOW)
        result["runner"] = (
            "optimized_wave_sym_fastlag" if manifest_path.parent.parent.name == "wave_sym_fastlag" else "historical"
        )
        methods[arm].setdefault("full_budget_records", []).append(result)

    for item in methods.values():
        records = item.get("full_budget_records", [])
        at_recipe = [
            record
            for record in records
            if record["config"]["optimizer"]["lr"] == baseline["config"]["optimizer"]["lr"]
            or item["tax_id"] == "Muon"
        ]
        preferred = sorted(
            at_recipe,
            key=lambda record: (
                record["runner"] != "optimized_wave_sym_fastlag",
                record["run_directory"],
            ),
        )
        item["representative_full_budget_record"] = preferred[0] if preferred else None

    return {
        "audit_time": datetime.now(UTC).isoformat(),
        "scope": "19 paper taxonomic methods, 0.5M-token batch, seed 0",
        "protocol": {
            "gpu": "NVIDIA A100-SXM4-40GB",
            "world_size": 8,
            "global_batch_sequences": 128,
            "sequence_length": 4096,
            "micro_batch_size": 4,
            "gradient_accumulation": 4,
            "benchmark_window": list(BENCH_WINDOW),
            "warmup_updates": 36,
            "measured_updates": 60,
            "full_budget_window": list(FULL_WINDOW),
            "estimator": "arithmetic mean of all step_seconds in the stated window; no outlier removal",
            "timing_scope": (
                "Rank-0 real-data training step, including data transfer and NCCL DDP; main-stream synchronized"
            ),
            "initialization": "MuonT step-2500 model weights; fresh optimizer/statistics; data and LR offset 2500",
            "excludes": "compilation/startup warmup, evaluation and checkpoint IO",
            "limitations": [
                "One short benchmark per method; 60 correlated updates do not establish precise percent-level rankings.",
                "Fresh optimizer/statistics differ from each method's own converged mid-training state.",
                "Rank-0 timing is recorded; per-rank maxima and historical GPU occupancy were not logged.",
                "Current workspace training implementation differs from the frozen optimized benchmark implementation.",
            ],
        },
        "covered_methods": len(METHODS) - len(missing),
        "total_methods": len(METHODS),
        "missing_methods": missing,
        "methods": methods,
    }


def report_section(result: dict) -> str:
    lines = [
        REPORT_START,
        "## Measured Eight-A100 Timing Audit",
        "",
        f"Measured real-data training records cover {result['covered_methods']}/{result['total_methods']} "
        "paper methods. No stage estimates are used below. These records completed on November 20, 2026.",
        "Each frozen optimized-runner benchmark starts from MuonT model weights at update 2,500, "
        "with fresh optimizer/statistics and data/LR offset 2,500. It executes 96 training updates "
        "on eight A100-SXM4-40GB GPUs; the first 36 are discarded and all updates 37-96 are averaged.",
        "The model uses 32 layers, width 512, MLP width 2048, sequence length 4096, "
        "global batch 128 sequences, microbatch 4 and four accumulation passes. "
        "Eight GPUs share one node. BF16 model execution, FP32 moments, TF32 disabled, "
        "compiled model/update paths, cached-root CUDA graphs, and NCCL DDP are retained. "
        "Exact roots run for the initial 20 optimizer updates; the configured host refresh interval is four updates. "
        "Evaluation and checkpoint IO are disabled during these short timing runs.",
        "Timing is Rank-0 host elapsed time for data loading/transfers, all four forward/backward "
        "accumulation passes, clipping, optimizer/statistics work and DDP communication, ending "
        "after main-stream synchronization. It is not the maximum elapsed time across ranks; "
        "background work may overlap the next update.",
        "",
        "| Method | Short Mean (s) | Short Median (s) | Step SD (s) | Trajectory Mean (s) | Trajectory Runner |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in result["methods"].values():
        measured = item["measured_benchmark"]
        if measured is None:
            lines.append(f"| {item['tax_id']} | missing | | | | |")
            continue
        full = item["representative_full_budget_record"]
        full_mean = f"{full['mean_step_seconds']:.4f}" if full else "missing"
        full_runner = (
            "optimized" if full and full["runner"] == "optimized_wave_sym_fastlag" else "historical" if full else ""
        )
        lines.append(
            f"| {item['tax_id']} | {measured['mean_step_seconds']:.4f} | "
            f"{measured['median_step_seconds']:.4f} | {measured['std_step_seconds']:.4f} | "
            f"{full_mean} | {full_runner} |"
        )
    lines += [
        "",
        "Means retain all slow steps and timing outliers, whose causes are not separated in these logs. "
        "Standard deviations describe step variability; they are not confidence intervals. "
        "A single 60-update window with fresh optimizer state is insufficient for precise "
        "percent-level speed rankings or a full-budget wall-clock claim. "
        "The 36-update exclusion exceeds the configured 20-update exact-root startup, "
        "but mature statistics and long-run fallback frequencies have not been established. "
        "Historical occupancy and independent repeat measurements were not recorded in this audit.",
        "",
        "Trajectory means use updates 2049-4096 (2048 measured updates) of completed 5120-update training "
        "trajectories at matrix LR 0.02262741696 (Muon: 0.016). "
        "The selected record is the optimized wave when available, otherwise the first eligible "
        "historical directory in lexicographic order. "
        "Other full-budget records use historical implementations and cannot be compared "
        "directly with the optimized short means. Cross-method trajectory-mean comparisons also "
        "mix runner versions. The JSON audit retains every eligible "
        "full-budget record, including rate variants, without pooling runner versions.",
        "",
        "Regenerate the audit and report from existing records, without launching training:",
        "",
        "```bash",
        "uv run --no-project --python /opt/conda/envs/torch-base/bin/python \\",
        "  python tracking/audit_8gpu_timings.py \\",
        "  --out paper/results/measured_8gpu_timings.json \\",
        "  --report paper/REPORT.md",
        "```",
        "",
        "Raw benchmarks: "
        "`/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/runs/bench_8gpu_opt`.",
        "The [JSON audit](results/measured_8gpu_timings.json) records exact commands, source/configuration "
        "identities, metric file hashes, measurement windows, half-window means, and full-budget record paths. "
        "The current workspace source differs from this frozen optimized runner; these results "
        "do not measure later implementation changes or newly prepared related-method adapters.",
        REPORT_END,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PVC)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    result["audit_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    if args.report:
        text = args.report.read_text()
        section = report_section(result)
        if REPORT_START in text:
            before, remainder = text.split(REPORT_START, 1)
            _, after = remainder.split(REPORT_END, 1)
            text = before + section + after
        else:
            text = text.rstrip() + "\n\n---\n\n" + section + "\n"
        args.report.write_text(text)
    print(f"Coverage: {result['covered_methods']}/{result['total_methods']}; missing: {result['missing_methods']}")
    print(f"{'Method':<17} {'Mean':>9} {'Median':>9} {'SD':>9} {'Mean vs MuonT':>14} {'Full Runs':>9}")
    for item in result["methods"].values():
        measured = item["measured_benchmark"]
        if measured:
            print(
                f"{item['tax_id']:<17} {measured['mean_step_seconds']:9.4f} "
                f"{measured['median_step_seconds']:9.4f} {measured['std_step_seconds']:9.4f} "
                f"{measured['mean_overhead_vs_muont_pct']:+13.1f}% {len(item['full_budget_records']):9d}"
            )
    print(args.out)


if __name__ == "__main__":
    main()
