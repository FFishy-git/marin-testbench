# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Save five broad-token-window comparisons and uncensored elapsed-time diagnostics for Wave 8."""

import argparse
import json
from pathlib import Path

from compare_runs import WINDOW, compare, history
from launch_wave5 import sha256
from time_to_target import report
from wave8_history import ARMS, verify_histories


def analyze(root: Path) -> dict:
    verify_histories(root)
    pairs = (
        ("muont", ARMS[1:]),
        ("outer_leftdiag", ("outer_leftdiag_fused",)),
        ("outer_leftdiag_fused", ("outer_leftdiag_combined_fused",)),
    )
    analyses = []
    for baseline, names in pairs:
        runs = {f"wave8/{arm}": history(root / "runs/wave8" / arm) for arm in (baseline, *names)}
        analyses.append(compare(runs, f"wave8/{baseline}", WINDOW))
    return {
        "wave": "wave8",
        "window_tokens": WINDOW,
        "analyses": analyses,
        "unpaired_runs": {},
        "source_release_sha256": sha256(root / "run-preparation/wave8-launch-binding.json"),
        "elapsed_time_diagnostic": report(root, [f"wave8/{arm}" for arm in ARMS], [3.59, 3.52, 3.46]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    path = args.root / "run-preparation/analysis-wave8.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
