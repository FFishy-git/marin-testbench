# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze full-budget attribution waves with matched-token and matched-null comparisons."""

import argparse
from pathlib import Path

from compare_runs import WINDOW, compare, history
from fused_attribution_production_contract import FAMILIES, WAVES, paths, wave_spec
from fused_attribution_run_history import verify_histories
from launch_wave5 import sha256
from time_to_target import report
from watch_wave5 import write


def analyze(root: Path, wave: str) -> dict:
    verify_histories(root, wave)
    runs = {f"{wave}/{family}": history(root / f"runs/{wave}/{family}") for family in FAMILIES}
    return {
        "wave": wave,
        "window_tokens": WINDOW,
        "analyses": [
            compare(runs, f"{wave}/muont", WINDOW),
            compare(runs, f"{wave}/null_leftdiag_fused", WINDOW),
        ],
        "unpaired_runs": {},
        "source_release_sha256": sha256(paths(root, wave)[0]),
        "matrix_lr": wave_spec(wave)["matrix_lr"],
        "elapsed_time_diagnostic": report(root, list(runs), [3.59, 3.52, 3.46]),
        "limitation": (
            "Exploratory seed-0 full-schedule LR and matched-null attribution. Family LR winners "
            "require all nine full-budget runs and eleven exact final-half selection evaluations. "
            "Endpoint winners require equal-family log-LR expansion. Nearby evaluations are not "
            "replications; fresh-seed superiority and measured efficiency remain unestablished."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wave", choices=WAVES, required=True)
    args = parser.parse_args()
    destination = args.root / f"run-preparation/analysis-{args.wave}.json"
    write(destination, analyze(args.root, args.wave))
    print(destination)


if __name__ == "__main__":
    main()
