# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare immutable strict captures of live Wave-7 histories with the existing token-window logic."""

import argparse
import json
from pathlib import Path

from analyze_tuning import executed_sources, normalized_sources
from compare_runs import WINDOW, compare, history
from prepare_fixed_age_tuning import require_source
from run_fixed_age_validation import COMPARISONS
from watch_wave5 import write
from wave7_release import ARMS, read, sha256


def capture(root: Path, output: Path) -> dict:
    """Preserve one complete byte snapshot per file; reject malformed or unfinished rows."""
    frozen = require_source(root, root / "run-preparation/wave7-release-binding.json")
    candidate = Path(frozen["candidate"])
    expected_sources = executed_sources(candidate, frozen)
    if output.resolve().is_relative_to(candidate.resolve()):
        raise ValueError("Live observation output must stay outside the frozen candidate")
    if output.exists():
        raise FileExistsError("Observation already exists; preserve its exact captured evidence")
    payloads = {}
    identities = {}
    for arm in ARMS:
        directory = root / "runs/wave7" / arm
        manifest_bytes = (directory / "run.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        config = read(candidate / f"experiment/configs/{arm}_0p5m_seed0.json")
        if (
            manifest["config"] != config
            or manifest["status"] != "running"
            or manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
            or normalized_sources(candidate, manifest["source_sha256"]) != expected_sources
        ):
            raise ValueError(
                "Captured live Wave-7 manifest is not running or differs from its source/config/data binding"
            )
        payloads[Path(arm) / "run.json"] = manifest_bytes
        segments = sorted(directory.glob("metrics_from_*.jsonl"))
        if not segments:
            raise ValueError("Live Wave-7 observation has no scalar history")
        for segment in segments:
            raw = segment.read_bytes()
            if not raw or not raw.endswith(b"\n"):
                raise ValueError("Live Wave-7 observation has an unfinished JSONL row; retry the read")
            for row in raw.splitlines():
                json.loads(row)
            payloads[Path(arm) / segment.name] = raw
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        destination = output / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(raw)
    for arm in ARMS:
        run = history(output / arm)
        identities[arm] = {
            "observed_updates": len(run.rows),
            "manifest_sha256": sha256(output / arm / "run.json"),
            "history_sha256": {path.name: sha256(path) for path in sorted((output / arm).glob("metrics_from_*.jsonl"))},
        }
    result = {
        "status": "captured_strict_wave7_observation",
        "source_release_sha256": sha256(root / "run-preparation/wave7-release-binding.json"),
        "observation": str(output.resolve()),
        "runs": identities,
        "scope": "Immutable parsed byte observations only; no omitted malformed rows or inferred telemetry.",
    }
    write(output / "observation.json", result)
    return result


def analyze_capture(observation: Path) -> dict:
    """Reuse the comparator on one observation per arm, preventing different-progress rereads."""
    record = read(observation / "observation.json")
    runs = {}
    for arm in ARMS:
        identity = record["runs"][arm]
        directory = observation / arm
        if identity["manifest_sha256"] != sha256(directory / "run.json") or identity["history_sha256"] != {
            path.name: sha256(path) for path in sorted(directory.glob("metrics_from_*.jsonl"))
        }:
            raise ValueError("Immutable Wave-7 observation changed after capture")
        run = history(directory)
        if run.manifest["status"] != "running":
            raise ValueError("Captured Wave-7 comparison requires running manifests; no completed-wave outcome")
        if len(run.rows) != identity["observed_updates"]:
            raise ValueError("Captured Wave-7 history coverage changed")
        runs[f"wave7/{arm}"] = run
    return {
        "wave": "wave7",
        "window_tokens": WINDOW,
        "analyses": [
            compare({name: runs[name] for name in (baseline, *names)}, baseline, WINDOW)
            for baseline, names in COMPARISONS
        ],
        "unpaired_runs": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    args = parser.parse_args()
    observed = capture(args.root, args.observation)
    result = analyze_capture(args.observation)
    path = args.observation / "analysis.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"observation": observed["observation"], "analysis": str(path)}, indent=2))


if __name__ == "__main__":
    main()
