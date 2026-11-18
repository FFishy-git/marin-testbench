# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report uncensored evaluation crossings without interpolating sparse losses."""

import argparse
import json
import math
from pathlib import Path

from compare_runs import completion, history


def crossings(run, target: float) -> dict:
    if not math.isfinite(target):
        raise ValueError("Target must be finite")
    previous = None
    for row in run.rows:
        if "c4_loss" not in row:
            continue
        if row["c4_loss"] <= target:
            return {
                "target": target,
                "censored": False,
                "previous_evaluation_tokens": None if previous is None else previous["tokens"],
                "crossing_tokens": row["tokens"],
                "crossing_c4": row["c4_loss"],
                "recorded_main_seconds": row.get("end_to_end_seconds"),
                "previous_recorded_main_seconds": None if previous is None else previous.get("end_to_end_seconds"),
                "interpretation": "First completed evaluation at or below target; no loss/time interpolation.",
            }
        previous = row
    return {
        "target": target,
        "censored": True,
        "observed_tokens": run.rows[-1]["tokens"],
        "last_evaluation_tokens": None if previous is None else previous["tokens"],
        "last_evaluation_c4": None if previous is None else previous["c4_loss"],
        "recorded_main_seconds": run.rows[-1].get("end_to_end_seconds"),
        "run_complete": completion(run)["complete"],
    }


def report(root: Path, names: list[str], targets: list[float]) -> dict:
    runs = {name: history(root / "runs" / name) for name in names}
    if any(len(run.segments) != 1 for run in runs.values()):
        raise ValueError("Crossing timing currently requires one uninterrupted process segment")
    return {
        "targets": targets,
        "timing_scope": (
            "Recorded main-function timer; includes setup/evaluation/checkpointing, "
            "excludes Python imports and parent launch overhead."
        ),
        "interpretation": (
            "Exploratory elapsed-time diagnostic, not launch-based confirmatory "
            "time-to-target or statistical replication."
        ),
        "runs": {name: [crossings(run, target) for target in targets] for name, run in runs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target", type=float, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    result = report(args.root, args.runs, args.target)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
