# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist broad-window comparisons selected before interpreting a wave."""

import argparse
import json
from pathlib import Path

from compare_runs import WINDOW, compare, completion, evaluations, format_report, history, runtime, window_means

COMPARISONS = {
    "wave0": (("wave0/muont", ("wave0/muon", "wave0/outer_left_down", "wave0/down_hybrid")),),
    "wave1": (
        ("wave0/muont", ("wave1/inner_outer_all", "wave1/left_all", "wave1/inner_outer_all_diag", "wave1/null_all")),
    ),
    "wave2": (
        (
            "wave0/muont",
            (
                "wave2/inner_outer_hidden",
                "wave2/inner_outer_mlpdiag",
                "wave2/inner_outer_leftdiag",
                "wave2/outer_left_all",
            ),
        ),
    ),
    "wave3": (
        ("wave3/muont_seed1", ("wave3/inner_outer_mlpdiag_seed1",)),
        ("wave0/muont", ("wave3/null_mlpdiag", "wave3/inner_outer_all_lag")),
        ("wave3/null_mlpdiag", ("wave2/inner_outer_mlpdiag",)),
        ("wave1/inner_outer_all", ("wave3/inner_outer_all_lag",)),
    ),
    "wave4": (
        ("wave2/inner_outer_mlpdiag", ("wave4/inner_outer_mlpdiag_lag",)),
        ("wave4/muont_1m", ("wave4/inner_outer_mlpdiag_1m",)),
    ),
    "wave4-baseline": (("wave4-baseline/muont_seed2", ("wave4/inner_outer_mlpdiag_seed2",)),),
    "wave5": (
        (
            "wave5/muont",
            (
                "wave5/inner_outer_mlpdiag",
                "wave5/inner_outer_mlpdiag_fused",
                "wave5/inner_outer_leftdiag_fused",
            ),
        ),
        ("wave5/inner_outer_mlpdiag", ("wave5/inner_outer_mlpdiag_fused",)),
        ("wave2/inner_outer_leftdiag", ("wave5/inner_outer_leftdiag_fused",)),
    ),
}
UNPAIRED = {"wave4": ("wave4/inner_outer_mlpdiag_seed2",)}


def analyze(root: Path, wave: str) -> dict:
    comparisons = []
    for baseline, names in COMPARISONS[wave]:
        runs = {name: history(root / "runs" / name) for name in (baseline, *names)}
        comparisons.append(compare(runs, baseline, WINDOW))
    unpaired = {}
    for name in UNPAIRED.get(wave, ()):
        run = history(root / "runs" / name)
        unpaired[name] = {
            "interpretation": "No same-seed baseline yet; raw values are not an optimizer improvement claim.",
            "completion": completion(run),
            "training_windows": window_means(run.rows, WINDOW),
            "evaluations": {key: evaluations(run.rows, key) for key in ("c4_loss", "selection_loss")},
            "runtime": runtime(run, run.rows[-1]["tokens"]),
        }
    return {"wave": wave, "window_tokens": WINDOW, "analyses": comparisons, "unpaired_runs": unpaired}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wave", choices=COMPARISONS, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.wave)
    directory = args.root / "run-preparation"
    directory.mkdir(parents=True, exist_ok=True)
    for suffix, payload in (
        ("json", json.dumps(result, indent=2, allow_nan=False) + "\n"),
        (
            "txt",
            "\n\n".join(format_report(item) for item in result["analyses"])
            + "\n\nUnpaired runs (no same-seed baseline):\n"
            + json.dumps(result["unpaired_runs"], indent=2)
            + "\n",
        ),
    ):
        path = directory / f"analysis-{args.wave}.{suffix}"
        temporary = path.with_suffix(f".{suffix}.tmp")
        temporary.write_text(payload)
        temporary.replace(path)
    print(directory / f"analysis-{args.wave}.json")


if __name__ == "__main__":
    main()
