# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare Wave-10 broad token windows and the unchanged Wave-9 systems baseline."""

import argparse
import json
from pathlib import Path

from compare_runs import WINDOW, compare, history
from launch_wave5 import sha256
from owned_output_run_history import ARMS, WAVE, paths, verify_histories
from time_to_target import report


def analyze(root: Path) -> dict:
    verify_histories(root)
    analyses = []
    for baseline, names in (
        ("muont", ARMS[1:]),
        ("outer_leftdiag_fused", ("outer_leftdiag_upstream_fused",)),
        ("null_leftdiag_upstream_fused", ("outer_leftdiag_upstream_fused",)),
    ):
        runs = {f"{WAVE}/{arm}": history(root / f"runs/{WAVE}" / arm) for arm in (baseline, *names)}
        analyses.append(compare(runs, f"{WAVE}/{baseline}", WINDOW))
    release = json.loads(paths(root)[0].read_text())
    predecessor = json.loads((root / "run-preparation/wave9-launch-binding.json").read_text())
    if release["predecessor_release_sha256"] != sha256(root / "run-preparation/wave9-launch-binding.json"):
        raise ValueError("Wave-10 cross-wave systems reference changed")
    source = Path(release["candidate"])
    reference = Path(predecessor["candidate"])
    for arm in ARMS:
        config = f"experiment/configs/{arm}_0p5m_seed0.json"
        if (source / config).read_bytes() != (reference / config).read_bytes():
            raise ValueError("Cross-wave ownership comparison requires identical configurations")
        runs = {f"{wave}/{arm}": history(root / "runs" / wave / arm) for wave in ("wave9", WAVE)}
        analyses.append(compare(runs, f"wave9/{arm}", WINDOW))
    return {
        "wave": WAVE,
        "window_tokens": WINDOW,
        "analyses": analyses,
        "unpaired_runs": {},
        "source_release_sha256": sha256(paths(root)[0]),
        "elapsed_time_diagnostic": report(root, [f"{WAVE}/{arm}" for arm in ARMS], [3.59, 3.52, 3.46]),
        "limitation": (
            "Seed-0 base-LR systems experiment; cross-wave runtime is observational. "
            "Independent tuning and fresh seeds remain required."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    path = args.root / f"run-preparation/analysis-{WAVE}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(path)


if __name__ == "__main__":
    main()
