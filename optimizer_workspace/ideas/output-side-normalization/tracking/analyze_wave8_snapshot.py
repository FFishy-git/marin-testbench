# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare immutable, fully parsed observations of running Wave-8 histories."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from analyze_tuning import executed_sources, normalized_sources
from boundary_handoff import ARMS, release
from compare_runs import WINDOW, RunHistory, compare, history, validate_rows
from launch_wave5 import PYTHON, read, sha256
from watch_wave5 import write

PAIRS = (("muont", ARMS[1:]), ("outer_leftdiag", (ARMS[2],)), (ARMS[2], (ARMS[3],)))


def capture(root: Path, output: Path) -> dict:
    """Read each history once as bytes; never skip, trim or repair incomplete scalar rows."""
    frozen = release(root)
    script = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from wave8_history import binding
print(json.dumps(binding(Path(sys.argv[2]))))
"""
    subprocess.run(
        [PYTHON, "-c", script, frozen["helpers"], str(root)],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    candidate = Path(frozen["candidate"])
    numerical = read(root / "run-preparation/wave8-candidate.json")
    expected_sources = executed_sources(candidate, numerical)
    if output.exists():
        raise FileExistsError("Wave-8 observation already exists; preserve it without overwrite")
    if output.resolve().is_relative_to(candidate.resolve()):
        raise ValueError("Observation output must remain outside the frozen candidate")
    payloads, identities = {}, {}
    for arm in ARMS:
        directory = root / "runs/wave8" / arm
        manifest_bytes = (directory / "run.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        config = read(candidate / f"experiment/configs/{arm}_0p5m_seed0.json")
        if (
            manifest["config"] != config
            or manifest["status"] != "running"
            or manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
            or normalized_sources(candidate, manifest["source_sha256"]) != expected_sources
        ):
            raise ValueError("Live Wave-8 manifest differs from running source/config/data identity")
        segments = sorted(directory.glob("metrics_from_*.jsonl"))
        if [path.name for path in segments] != ["metrics_from_000000.jsonl"]:
            raise ValueError("Wave-8 observation requires its single original non-resumed history")
        raw = segments[0].read_bytes()
        if not raw or not raw.endswith(b"\n"):
            raise ValueError("Live Wave-8 history ends with an unfinished row; retry the observation")
        rows = [json.loads(line) for line in raw.splitlines()]
        validate_rows(rows, config["training"]["batch_tokens"])
        if len(rows) > config["training"]["steps"]:
            raise ValueError("Live Wave-8 history exceeds its budget")
        payloads[Path(arm) / "run.json"] = manifest_bytes
        payloads[Path(arm) / segments[0].name] = raw
        identities[arm] = {
            "observed_updates": len(rows),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "history_sha256": {segments[0].name: hashlib.sha256(raw).hexdigest()},
        }
        compare({arm: RunHistory(manifest, rows, [rows])}, arm, WINDOW)
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        destination = output / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(raw)
    result = {
        "status": "captured_strict_wave8_observation",
        "source_release_sha256": sha256(root / "run-preparation/wave8-launch-binding.json"),
        "observation": str(output.resolve()),
        "runs": identities,
        "scope": "Fully parsed immutable byte observations; no omitted rows, interpolation or completed-wave claim.",
    }
    write(output / "observation.json", result)
    return result


def analyze_capture(observation: Path) -> dict:
    record = read(observation / "observation.json")
    if record["status"] != "captured_strict_wave8_observation" or set(record["runs"]) != set(ARMS):
        raise ValueError("Wave-8 observation has a different protocol or arm inventory")
    runs = {}
    for arm in ARMS:
        directory, identity = observation / arm, record["runs"][arm]
        if identity["manifest_sha256"] != sha256(directory / "run.json") or identity["history_sha256"] != {
            path.name: sha256(path) for path in sorted(directory.glob("metrics_from_*.jsonl"))
        }:
            raise ValueError("Immutable Wave-8 observation changed after capture")
        run = history(directory)
        if run.manifest["status"] != "running" or len(run.rows) != identity["observed_updates"]:
            raise ValueError("Captured Wave-8 running identity or coverage changed")
        runs[f"wave8/{arm}"] = run
    return {
        "wave": "wave8",
        "window_tokens": WINDOW,
        "source_release_sha256": record["source_release_sha256"],
        "analyses": [
            compare({f"wave8/{arm}": runs[f"wave8/{arm}"] for arm in (baseline, *names)}, f"wave8/{baseline}", WINDOW)
            for baseline, names in PAIRS
        ],
        "unpaired_runs": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    args = parser.parse_args()
    observed = capture(args.root, args.observation)
    analysis = analyze_capture(args.observation)
    (args.observation / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"observation": observed["observation"], "status": "strict_comparison_saved"}))


if __name__ == "__main__":
    main()
