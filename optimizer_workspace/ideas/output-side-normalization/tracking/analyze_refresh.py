# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize recorded fixed-age refresh events over complete shared token windows."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from analyze_tuning import executed_sources, normalized_sources
from compare_runs import WINDOW, completion, history, validate_rows
from prepare_fixed_age_tuning import require_source
from prepare_tuning import sha256
from refresh_history import EVENT_FIELDS, scalar_history_row
from wave7_release import ARMS

COUNTERS = ("input_fallbacks", "input_extra_roots")
SAMPLED_FIELDS = ("input_max_basis_age", "input_pending_refreshes")


def event_summary(rows: list[dict], age: int) -> dict:
    """Validate actual installation records and count each rank-zero event once."""
    groups = {}
    for row in rows:
        scalar_history_row(row)
        for field in EVENT_FIELDS:
            seen = set()
            for event in row.get(field, []):
                identity = event["group"], event["index"]
                if identity in seen:
                    raise ValueError("Duplicate refresh event for one engine/group/index at an update")
                seen.add(identity)
                actual_age = event["installation_step"] - event["source_step"]
                if actual_age != age:
                    raise ValueError("Observed host refresh violates the fixed installation age")
                group = groups.setdefault(
                    f"{field}/{event['group']}",
                    {"events": 0, "installed": 0, "discarded": 0, "installation_ages": {}},
                )
                group["events"] += 1
                group["installed" if event["installed"] else "discarded"] += 1
                group["installation_ages"][actual_age] = group["installation_ages"].get(actual_age, 0) + 1
    return groups


def summarize(rows: list[dict], previous: dict | None, age: int) -> dict:
    """Report event coverage and recorded counters without estimating unlogged host waits."""
    groups = event_summary(rows, age)
    result = {
        "updates": len(rows),
        "events": sum(group["events"] for group in groups.values()),
        "installed": sum(group["installed"] for group in groups.values()),
        "discarded": sum(group["discarded"] for group in groups.values()),
        "groups": groups,
        "event_field_rows": {field: sum(field in row for row in rows) for field in EVENT_FIELDS},
        "counter_deltas": {},
        "recorded_samples": {},
    }
    for key in COUNTERS:
        if not all(key in row for row in rows) or (previous is not None and key not in previous):
            continue
        start = 0 if previous is None else previous[key]
        values = [start, *(row[key] for row in rows)]
        if any(type(value) is not int or value < 0 for value in values) or any(
            after < before for before, after in pairwise(values)
        ):
            raise ValueError("Recorded cumulative refresh counter is invalid or decreases")
        result["counter_deltas"][key] = values[-1] - start
    for key in SAMPLED_FIELDS:
        values = [row[key] for row in rows if key in row]
        if values:
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError("Recorded refresh diagnostic is invalid")
            result["recorded_samples"][key] = {
                "rows": len(values),
                "mean": statistics.mean(values),
                "max": max(values),
            }
    timing = [row["step_seconds"] for row in rows if "step_seconds" in row]
    if timing:
        if any(value <= 0 for value in timing):
            raise ValueError("Recorded step timing must be positive")
        result["step_timing"] = {
            "rows": len(timing),
            "sum": math.fsum(timing),
            "median": statistics.median(timing),
            "p95": sorted(timing)[math.ceil(0.95 * len(timing)) - 1],
        }
    return result


def refresh_windows(rows: list[dict], common_tokens: int, window_tokens: int, age: int) -> list[dict]:
    """Use complete batch-aligned token windows, retaining sparse telemetry as sparse."""
    validate_rows(rows)
    batch_tokens = rows[0]["tokens"]
    if window_tokens <= 0 or window_tokens % batch_tokens:
        raise ValueError("Refresh event windows must align with optimizer updates")
    result = []
    for endpoint in range(window_tokens, min(common_tokens, rows[-1]["tokens"]) + 1, window_tokens):
        start = endpoint - window_tokens
        first = start // batch_tokens
        last = endpoint // batch_tokens
        selected = rows[first:last]
        result.append(
            {
                "start_tokens": start,
                "end_tokens": endpoint,
                **summarize(selected, rows[first - 1] if first else None, age),
            }
        )
    return result


def sha256_rows(rows: list[dict]) -> str:
    """Hash the exact parsed observation so later appends cannot change its identity."""
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()


def analyze(root: Path) -> dict:
    """Bind read-only diagnostics to the four executed Wave-7 source/config/data identities."""
    receipt = root / "run-preparation/wave7-release-binding.json"
    frozen = require_source(root, receipt)
    candidate = Path(frozen["candidate"])
    expected_sources = executed_sources(candidate, frozen)
    runs = {}
    provenance = {}
    for arm in ARMS:
        directory = root / "runs/wave7" / arm
        run = history(directory)
        config = json.loads((candidate / f"experiment/configs/{arm}_0p5m_seed0.json").read_text())
        if (
            run.manifest["config"] != config
            or run.manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
            or normalized_sources(candidate, run.manifest["source_sha256"]) != expected_sources
            or not all("input_refresh_installations" in row for row in run.rows)
        ):
            raise ValueError("Refresh diagnostic run identity differs from the validated Wave-7 source")
        runs[arm] = run
        provenance[arm] = {
            "manifest_sha256": sha256(directory / "run.json"),
            "observed_updates": len(run.rows),
            "observed_rows_sha256": sha256_rows(run.rows),
        }
    common_tokens = min(run.rows[-1]["tokens"] for run in runs.values())
    return {
        "time": datetime.now(UTC).isoformat(),
        "status": "recorded_refresh_diagnostic",
        "wave": "wave7",
        "source_release_sha256": sha256(receipt),
        "window_tokens": WINDOW,
        "common_observed_tokens": common_tokens,
        "refresh_install_age": 1,
        "provenance": provenance,
        "runs": {
            arm: {
                "completion": completion(run),
                "all_observed": summarize(run.rows, None, 1),
                "complete_shared_windows": refresh_windows(run.rows, common_tokens, WINDOW, 1),
            }
            for arm, run in runs.items()
        },
        "limitation": (
            "Rank-zero recorded events/counters only. Installation age is snapshot age at installation, "
            "not current basis age across all ranks. Missing counters remain missing. Step timings include "
            "startup and do not isolate host joins, CUDA stages, evaluation or checkpoint costs. "
            "No causal or full-budget efficiency claim follows from this diagnostic."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "common_observed_tokens": result["common_observed_tokens"],
                "complete_shared_windows": len(next(iter(result["runs"].values()))["complete_shared_windows"]),
                "events": {arm: item["all_observed"]["events"] for arm, item in result["runs"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
