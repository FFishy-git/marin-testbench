# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare token-aligned loss and runtime histories against a baseline.

Only identical manifest seed/data identities permit paired-update labels.
Different batches use token-weighted training windows, not paired updates.
Sparse evaluations are never interpolated. --descriptive permits explicitly
unpaired comparisons; --json [PATH] emits JSON (stdout when PATH is omitted).
"""

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

PVC = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
WINDOW = 536870912
WARMUP_TOKENS = 100 * 524288
EVALUATION_KEYS = ("c4_loss", "selection_loss")
RUNTIME_KEYS = (
    "step_seconds",
    "elapsed_seconds",
    "end_to_end_seconds",
    "eval_seconds",
    "checkpoint_seconds",
    "peak_allocated_bytes",
)


@dataclass
class RunHistory:
    manifest: dict
    rows: list[dict]
    segments: list[list[dict]]


def positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name}: expected a positive integer")
    return value


def validate_rows(rows: list[dict], batch_tokens: int | None = None) -> None:
    """Require a consecutive prefix from update one with an exact token axis."""
    if not rows:
        raise ValueError("Missing training history")
    if batch_tokens is None:
        batch_tokens = positive_integer(rows[0].get("tokens"), "tokens")
    tokens_seen = set()
    steps_seen = set()
    for expected_step, row in enumerate(rows, start=1):
        step = positive_integer(row.get("step"), "step")
        tokens = positive_integer(row.get("tokens"), "tokens")
        if step in steps_seen or tokens in tokens_seen:
            raise ValueError(f"Duplicate step or token history at update {step}, tokens {tokens}")
        if step != expected_step:
            raise ValueError(f"Missing or out-of-order steps: expected {expected_step}, found {step}")
        if tokens != step * batch_tokens:
            raise ValueError(f"Missing or inconsistent token history at update {step}")
        steps_seen.add(step)
        tokens_seen.add(tokens)
        for key in ("train_loss", *EVALUATION_KEYS, *RUNTIME_KEYS):
            if key != "train_loss" and key not in row:
                continue
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"Invalid {key} at update {step}")


def history(path: Path) -> RunHistory:
    """Load a manifest and all metric segments, rejecting gaps and overlap."""
    manifest = json.loads((path / "run.json").read_text())
    training = manifest["config"]["training"]
    steps = positive_integer(training["steps"], "training.steps")
    batch_tokens = positive_integer(training["batch_tokens"], "training.batch_tokens")
    budget = positive_integer(training["tokens"], "training.tokens")
    if budget != steps * batch_tokens:
        raise ValueError(f"{path}: manifest step/token budget mismatch")
    segments = []
    for segment in sorted(path.glob("metrics_from_*.jsonl")):
        with segment.open() as stream:
            rows = [json.loads(line) for line in stream]
        if not rows:
            raise ValueError(f"{segment}: missing training history")
        segments.append(rows)
    rows = [row for segment in segments for row in segment]
    validate_rows(rows, batch_tokens)
    if len(rows) > steps:
        raise ValueError(f"{path}: history exceeds the manifest budget")
    return RunHistory(manifest, rows, segments)


def window_means(rows: list[dict], window: int = WINDOW) -> dict[int, float]:
    """Return token-weighted train means keyed by complete window endpoints.

    An update crossing a boundary contributes its batch mean in proportion to
    the overlapping tokens. Default windows align exactly with 0.5M/1M batches.
    """
    positive_integer(window, "window tokens")
    validate_rows(rows)
    means = {}
    index = 0
    for end in range(window, rows[-1]["tokens"] + 1, window):
        start = end - window
        weighted = []
        while index < len(rows):
            row = rows[index]
            previous = rows[index - 1]["tokens"] if index else 0
            overlap = min(end, row["tokens"]) - max(start, previous)
            if overlap > 0:
                weighted.append(overlap * row["train_loss"])
            if row["tokens"] >= end:
                break
            index += 1
        means[end] = math.fsum(weighted) / window
    return means


def evaluations(rows: list[dict], key: str) -> dict[int, float]:
    """Return sparse evaluations keyed by tokens, with no filling or deduping."""
    validate_rows(rows)
    return {row["tokens"]: row[key] for row in rows if key in row}


def completion(run: RunHistory) -> dict:
    """Expose manifest completion separately from observed history coverage."""
    training = run.manifest["config"]["training"]
    reasons = []
    if run.manifest.get("status") != "complete":
        reasons.append("manifest_not_complete")
    if len(run.rows) != training["steps"]:
        reasons.append("history_truncated")
    if run.manifest.get("completed_steps") != training["steps"]:
        reasons.append("manifest_step_count_mismatch")
    if run.manifest.get("execution_check"):
        reasons.append("execution_check")
    return {
        "complete": not reasons,
        "manifest_status": run.manifest.get("status"),
        "manifest_completed_steps": run.manifest.get("completed_steps"),
        "observed_steps": len(run.rows),
        "expected_steps": training["steps"],
        "observed_tokens": run.rows[-1]["tokens"],
        "expected_tokens": training["tokens"],
        "reasons": reasons,
    }


def pairing(base: RunHistory, run: RunHistory, descriptive: bool = False) -> dict:
    """Verify provenance before distinguishing paired updates from token matches."""
    base_config = base.manifest["config"]
    config = run.manifest["config"]
    mismatches = []
    for key in ("seed", "heldout_windows"):
        expected = base_config["training"].get(key)
        actual = config["training"].get(key)
        if type(expected) is not int or type(actual) is not int or expected != actual:
            mismatches.append(f"config.training.{key}")
    for key in ("data", "model"):
        expected = base_config.get(key)
        actual = config.get(key)
        if not expected or not actual or expected != actual:
            mismatches.append(f"config.{key}")
    expected_receipt = base.manifest.get("data_receipt_sha256")
    actual_receipt = run.manifest.get("data_receipt_sha256")
    if not expected_receipt or not actual_receipt or expected_receipt != actual_receipt:
        mismatches.append("data_receipt_sha256")
    base_eval_limit = base_config.get("runtime", {}).get("eval_max_windows", 0)
    eval_limit = config.get("runtime", {}).get("eval_max_windows", 0)
    if base_eval_limit != eval_limit:
        mismatches.append("config.runtime.eval_max_windows")
    if mismatches and not descriptive:
        raise ValueError(
            f"Unverified pairing ({', '.join(mismatches)}); refusing paired-update comparison. "
            "Use --descriptive for an explicitly unpaired comparison."
        )
    same_batch = config["training"]["batch_tokens"] == base_config["training"]["batch_tokens"]
    kind = "descriptive_unpaired" if descriptive else "paired_updates" if same_batch else "matched_tokens"
    return {
        "kind": kind,
        "baseline_seed": base_config["training"].get("seed"),
        "arm_seed": config["training"].get("seed"),
        "baseline_batch_tokens": base_config["training"]["batch_tokens"],
        "arm_batch_tokens": config["training"]["batch_tokens"],
        "per_update_paired": not descriptive and not mismatches and same_batch,
        "seed_match": "config.training.seed" not in mismatches,
        "data_identity_match": not any(
            key in mismatches for key in ("config.data", "data_receipt_sha256", "config.training.heldout_windows")
        ),
        "mismatches": mismatches,
    }


def evaluation_comparison(base: RunHistory, run: RunHistory, key: str, common_tokens: int, final: bool) -> dict:
    """Compare exact-token evaluations, reserving final fields for the full budget."""
    base_values = evaluations(base.rows, key)
    values = evaluations(run.rows, key)
    shared = sorted(base_values.keys() & values.keys())
    late = [tokens for tokens in shared if common_tokens < 2 * tokens]
    last = shared[-1] if shared else None
    final_token = common_tokens if final and common_tokens in shared else None
    return {
        "shared_tokens": shared,
        "late_start_tokens_exclusive": common_tokens / 2,
        "late_end_tokens": common_tokens,
        "late_tokens": late,
        "mean_late_delta": statistics.fmean(values[token] - base_values[token] for token in late) if late else None,
        "last_shared_tokens": last,
        "last_shared_value": values[last] if last is not None else None,
        "last_shared_baseline_value": base_values[last] if last is not None else None,
        "last_shared_delta": values[last] - base_values[last] if last is not None else None,
        "final_tokens": final_token,
        "final_value": values[final_token] if final_token is not None else None,
        "final_baseline_value": base_values[final_token] if final_token is not None else None,
        "final_delta": values[final_token] - base_values[final_token] if final_token is not None else None,
    }


def runtime(run: RunHistory, common_tokens: int) -> dict:
    """Separate startup, training steps and observed wall time across segments."""
    rows = run.rows
    shared = [row for row in rows if row["tokens"] <= common_tokens]
    exact_endpoint = bool(shared) and shared[-1]["tokens"] == common_tokens
    batch_tokens = run.manifest["config"]["training"]["batch_tokens"]
    steady = [row for row in shared if row["tokens"] - batch_tokens >= WARMUP_TOKENS]
    step_seconds = math.fsum(row["step_seconds"] for row in rows) if all("step_seconds" in r for r in rows) else None
    shared_seconds = (
        math.fsum(row["step_seconds"] for row in shared)
        if exact_endpoint and all("step_seconds" in r for r in shared)
        else None
    )
    steady_times = [row["step_seconds"] for row in steady] if all("step_seconds" in r for r in steady) else []
    startup = []
    totals = []
    for segment in run.segments:
        first, last = segment[0], segment[-1]
        if "elapsed_seconds" in first and "step_seconds" in first:
            startup.append(first["elapsed_seconds"] - first["step_seconds"])
        if "end_to_end_seconds" in last:
            totals.append(last["end_to_end_seconds"])
    complete = completion(run)["complete"]
    # The manifest wall clock is process-local too; replace only the final segment.
    manifest_total = run.manifest.get("elapsed_seconds") if complete else None
    if manifest_total is not None:
        if isinstance(manifest_total, bool) or not isinstance(manifest_total, (int, float)):
            raise ValueError("Invalid manifest elapsed_seconds")
        if not math.isfinite(manifest_total):
            raise ValueError("Invalid manifest elapsed_seconds")
        if len(totals) == len(run.segments):
            totals[-1] = manifest_total
        elif len(run.segments) == 1:
            totals = [manifest_total]
    total = math.fsum(totals) if len(totals) == len(run.segments) else None
    peaks = [row["peak_allocated_bytes"] for row in rows if "peak_allocated_bytes" in row]
    return {
        "common_tokens": common_tokens,
        "startup_seconds": math.fsum(startup) if len(startup) == len(run.segments) else None,
        "training_step_seconds": step_seconds,
        "shared_step_seconds": shared_seconds,
        "shared_step_seconds_per_million_tokens": (
            shared_seconds * 1e6 / common_tokens if shared_seconds is not None else None
        ),
        "warmup_tokens": WARMUP_TOKENS,
        "median_step_seconds": statistics.median(steady_times) if steady_times else None,
        "steady_step_seconds_per_million_tokens": (
            math.fsum(steady_times) * 1e6 / (len(steady_times) * batch_tokens) if steady_times else None
        ),
        "eval_seconds": (
            math.fsum(row.get("eval_seconds", 0) for row in rows) if any("eval_seconds" in row for row in rows) else None
        ),
        "checkpoint_seconds": (
            math.fsum(row.get("checkpoint_seconds", 0) for row in rows)
            if any("checkpoint_seconds" in row for row in rows)
            else None
        ),
        "total_seconds": total,
        "total_is_final": complete and total is not None,
        "segments": len(run.segments),
        "peak_allocated_gib": max(peaks) / 2**30 if peaks else None,
    }


def compare(runs: dict[str, RunHistory], baseline: str, window: int = WINDOW, *, descriptive: bool = False) -> dict:
    """Build a JSON-ready comparison with explicit completion and pairing scope."""
    positive_integer(window, "window tokens")
    base = runs[baseline]
    base_completion = completion(base)
    base_windows = window_means(base.rows, window)
    comparisons = {}
    for arm, run in runs.items():
        identity = pairing(base, run, descriptive)
        progress = completion(run)
        common_tokens = min(base_completion["observed_tokens"], progress["observed_tokens"])
        final = (
            base_completion["complete"]
            and progress["complete"]
            and base_completion["expected_tokens"] == progress["expected_tokens"]
        )
        means = window_means(run.rows, window)
        training = [
            {
                "start_tokens": end - window,
                "end_tokens": end,
                "baseline_mean": base_windows[end],
                "mean": means[end],
                "delta": means[end] - base_windows[end],
            }
            for end in sorted(base_windows.keys() & means.keys())
        ]
        base_runtime = runtime(base, common_tokens)
        arm_runtime = runtime(run, common_tokens)
        deltas = {}
        for key in (
            "startup_seconds",
            "shared_step_seconds",
            "shared_step_seconds_per_million_tokens",
            "median_step_seconds",
            "steady_step_seconds_per_million_tokens",
            "total_seconds",
        ):
            actual, expected = arm_runtime[key], base_runtime[key]
            deltas[key] = actual - expected if actual is not None and expected is not None else None
        deltas["final_total_seconds"] = deltas["total_seconds"] if final else None
        comparisons[arm] = {
            "pairing": identity,
            "completion": progress,
            "common_tokens": common_tokens,
            "final_comparison_eligible": final,
            "training_windows": training,
            "evaluations": {key: evaluation_comparison(base, run, key, common_tokens, final) for key in EVALUATION_KEYS},
            "runtime": {
                "baseline": base_runtime,
                "arm": arm_runtime,
                "delta": deltas,
                "total_scope": "full_budget" if final else "observed_progress",
            },
        }
    return {
        "schema_version": 1,
        "baseline": baseline,
        "window_tokens": window,
        "delta_direction": "arm_minus_baseline",
        "evaluation_scope": "final_half_of_pairwise_common_observed_tokens",
        "runtime_scope": "sum_of_observed_process_segments_excluding_resume_downtime",
        "interpretation": "exploratory_descriptive_deltas_not_replicated_statistical_inference",
        "comparisons": comparisons,
    }


def format_number(value: float | None, spec: str = ".3f") -> str:
    return format(value, spec) if value is not None else "n/a"


def format_report(result: dict) -> str:
    """Render progress, token-window deltas and runtime without premature finals."""
    comparisons = result["comparisons"]
    lines = [
        f"Baseline {result['baseline']}; positive loss deltas are worse.",
        "Exploratory comparisons, not replicated or independent-update statistical inference.",
        "Train windows are token weighted; evaluations match exact tokens only.",
        "",
        "| Arm | Steps | Observed / Budget Tokens | Complete | Comparison |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for arm, item in comparisons.items():
        progress = item["completion"]
        label = item["pairing"]["kind"]
        if item["pairing"]["mismatches"]:
            label += f" ({', '.join(item['pairing']['mismatches'])})"
        lines.append(
            f"| {arm} | {progress['observed_steps']} / {progress['expected_steps']} | "
            f"{progress['observed_tokens']} / {progress['expected_tokens']} | "
            f"{progress['complete']} ({progress['manifest_status']}) | {label} |"
        )
    ends = sorted({window["end_tokens"] for item in comparisons.values() for window in item["training_windows"]})
    if ends:
        lines += [
            "",
            "| Arm | " + " | ".join(f"Train Tokens ({end - result['window_tokens']}, {end}]" for end in ends) + " |",
            "|" + " --- |" * (1 + len(ends)),
        ]
        for arm, item in comparisons.items():
            windows = {window["end_tokens"]: window["delta"] for window in item["training_windows"]}
            lines.append(f"| {arm} | " + " | ".join(format_number(windows.get(end), "+.5f") for end in ends) + " |")
    for key in EVALUATION_KEYS:
        lines += [
            "",
            f"{key}: late mean over exact shared evaluations in (common_tokens / 2, common_tokens].",
            "",
            "| Arm | Common Tokens | Late Points | Mean Late Delta | Last Shared Tokens | Last Shared Value | "
            "Last Shared Delta | Final Value | Final Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm, item in comparisons.items():
            evaluation = item["evaluations"][key]
            lines.append(
                f"| {arm} | {item['common_tokens']} | {len(evaluation['late_tokens'])} | "
                f"{format_number(evaluation['mean_late_delta'], '+.6f')} | "
                f"{evaluation['last_shared_tokens'] or 'n/a'} | "
                f"{format_number(evaluation['last_shared_value'], '.6f')} | "
                f"{format_number(evaluation['last_shared_delta'], '+.6f')} | "
                f"{format_number(evaluation['final_value'], '.6f')} | "
                f"{format_number(evaluation['final_delta'], '+.6f')} |"
            )
    lines += [
        "",
        "Final loss fields require both complete runs, equal budgets, and an evaluation at the budget endpoint.",
        f"Steady step statistics exclude the first {WARMUP_TOKENS} tokens; total time includes eval/checkpoints.",
        "Observed total deltas may cover different progress. Shared step time requires an exact common endpoint.",
        "Segment totals exclude unlogged tails and resume downtime; startup excludes first-step compilation.",
        "",
        "| Arm | Startup Seconds | Startup Delta | Median Step Seconds | Steady Seconds / MTok | "
        "Shared Step Delta | Observed Total Seconds | Observed Total Delta | Final Total Delta | Peak GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm, item in comparisons.items():
        timing = item["runtime"]["arm"]
        delta = item["runtime"]["delta"]
        lines.append(
            f"| {arm} | {format_number(timing['startup_seconds'])} | "
            f"{format_number(delta['startup_seconds'], '+.3f')} | "
            f"{format_number(timing['median_step_seconds'])} | "
            f"{format_number(timing['steady_step_seconds_per_million_tokens'])} | "
            f"{format_number(delta['shared_step_seconds'], '+.3f')} | "
            f"{format_number(timing['total_seconds'], '.1f')} | "
            f"{format_number(delta['total_seconds'], '+.1f')} | "
            f"{format_number(delta['final_total_seconds'], '+.1f')} | "
            f"{format_number(timing['peak_allocated_gib'])} |"
        )
    return "\n".join(lines)


def report(runs: dict[str, RunHistory], baseline: str, window: int = WINDOW, *, descriptive: bool = False) -> str:
    """Compare loaded histories and return the human-readable report."""
    return format_report(compare(runs, baseline, window, descriptive=descriptive))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="WAVE/ARM directories under PVC/runs")
    parser.add_argument("--root", type=Path, default=PVC)
    parser.add_argument("--baseline", default="wave0/muont")
    parser.add_argument("--window", "--window-tokens", type=int, default=WINDOW, help="Complete window size in tokens")
    parser.add_argument("--descriptive", action="store_true", help="Allow and label unpaired seed/data comparisons")
    parser.add_argument("--json", nargs="?", const=Path("-"), type=Path, help="Write JSON to PATH, or stdout if omitted")
    args = parser.parse_args()
    names = list(dict.fromkeys([args.baseline, *args.runs]))
    try:
        runs = {name: history(args.root / "runs" / name) for name in names}
        smallest_batch = min(run.manifest["config"]["training"]["batch_tokens"] for run in runs.values())
        if args.window < smallest_batch:
            raise ValueError(
                f"--window is measured in tokens and must be at least {smallest_batch}; "
                "use --window-tokens 536870912 for the 1024-update 0.5M comparison"
            )
        result = compare(runs, args.baseline, args.window, descriptive=args.descriptive)
        if args.json is None:
            print(format_report(result))
            return
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.json == Path("-"):
            print(payload, end="")
            return
        temporary = args.json.with_suffix(args.json.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(args.json)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
