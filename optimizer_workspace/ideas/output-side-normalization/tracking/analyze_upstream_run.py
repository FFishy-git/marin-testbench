# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare exact-token Wave-9 histories without attributing short or unreplicated gains."""

import argparse
import json
from pathlib import Path

from compare_runs import WINDOW, compare, history
from launch_wave5 import sha256
from time_to_target import report
from upstream_run_history import ARMS, paths, verify_histories


def analyze(root: Path) -> dict:
    verify_histories(root)
    pairs = (
        ("muont", ARMS[1:]),
        ("outer_leftdiag_fused", ("outer_leftdiag_upstream_fused",)),
        ("null_leftdiag_upstream_fused", ("outer_leftdiag_upstream_fused",)),
        ("muont", ("null_leftdiag_upstream_fused",)),
    )
    analyses = []
    for baseline, names in pairs:
        runs = {f"wave9/{arm}": history(root / "runs/wave9" / arm) for arm in (baseline, *names)}
        analyses.append(compare(runs, f"wave9/{baseline}", WINDOW))
    return {
        "wave": "wave9",
        "window_tokens": WINDOW,
        "analyses": analyses,
        "unpaired_runs": {},
        "source_release_sha256": sha256(paths(root)[0]),
        "elapsed_time_diagnostic": report(root, [f"wave9/{arm}" for arm in ARMS], [3.59, 3.52, 3.46]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    path = args.root / "run-preparation/analysis-wave9.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
