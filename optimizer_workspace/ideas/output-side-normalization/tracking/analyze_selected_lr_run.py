# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare selected-LR full-budget arms using complete large token windows."""

import argparse
from pathlib import Path

from compare_runs import WINDOW, compare, history
from launch_wave5 import sha256
from selected_lr_run_history import ARMS, WAVE, paths, verify_histories
from time_to_target import report
from watch_wave5 import write


def analyze(root: Path) -> dict:
    verify_histories(root)
    runs = {f"{WAVE}/{arm}": history(root / f"runs/{WAVE}" / arm) for arm in ARMS}
    return {
        "wave": WAVE,
        "window_tokens": WINDOW,
        "analyses": [compare(runs, f"{WAVE}/muont", WINDOW)],
        "unpaired_runs": {},
        "source_release_sha256": sha256(paths(root)[0]),
        "elapsed_time_diagnostic": report(root, list(runs), [3.59, 3.52, 3.46]),
        "limitation": (
            "Exploratory seed-0 transfer of short-study LR choices to the full schedule; "
            "full-budget LR optima and fresh-seed superiority remain unestablished. "
            "Time-to-target thresholds are retrospective and recorded timing excludes "
            "Python import/parent launch overhead."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    path = args.root / f"run-preparation/analysis-{WAVE}.json"
    write(path, analyze(args.root))
    print(path)


if __name__ == "__main__":
    main()
