# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare immutable strict byte captures of running production histories."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from analyze_tuning import executed_sources, normalized_sources
from boundary_handoff import release
from compare_runs import EVALUATION_KEYS, WINDOW, RunHistory, compare, evaluations, history, validate_rows
from launch_wave5 import PYTHON, read, sha256
from selected_lr_validation import ARMS as SELECTED_LR_ARMS
from upstream_validation import ARMS
from watch_wave5 import write

PAIRS = (
    ("muont", ARMS[1:]),
    ("outer_leftdiag_fused", ("outer_leftdiag_upstream_fused",)),
    ("null_leftdiag_upstream_fused", ("outer_leftdiag_upstream_fused",)),
    ("muont", ("null_leftdiag_upstream_fused",)),
)
PROTOCOLS = {
    "wave9": ("upstream_run_history", ARMS, PAIRS),
    "wave10": ("owned_output_run_history", ARMS, PAIRS),
    "wave12": (
        "selected_lr_run_history",
        SELECTED_LR_ARMS,
        (("muont", SELECTED_LR_ARMS[1:]), ("outer_leftdiag", ("outer_leftdiag_fused",))),
    ),
}


def verify_production_binding(root: Path, wave: str) -> tuple[dict, str]:
    """Check frozen helper bytes before importing their independent saved-evidence verifier."""
    module = "selected_lr_run_history" if wave == "wave11" else PROTOCOLS[wave][0]
    binding_path = root / f"run-preparation/{wave}-launch-binding.json"
    raw = binding_path.read_bytes()
    frozen = json.loads(raw)
    if wave == "wave9" and release(root, wave=wave) != frozen:
        raise ValueError("Wave9 predecessor binding changed before verification")
    binding_digest = hashlib.sha256(raw).hexdigest()
    helpers = root / f"controllers/{wave}-launch-20261117/tracking"
    inventory = frozen["helpers_sha256"]
    if (
        frozen["helpers"] != str(helpers)
        or f"{module}.py" not in inventory
        or any(Path(name).name != name or sha256(helpers / name) != digest for name, digest in inventory.items())
        or sha256(Path(frozen["archive"])) != frozen["archive_sha256"]
    ):
        raise ValueError(f"{wave} frozen helper path, bytes or archive differs")
    script = (
        "import json,sys\nfrom pathlib import Path\n"
        f"from {module} import binding\n"
        "print(json.dumps(binding(Path(sys.argv[1]))))"
    )
    checked = subprocess.run(
        [PYTHON, "-P", "-c", script, str(root)],
        cwd=frozen["helpers"],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": frozen["helpers"]},
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if (
        json.loads(checked.stdout) != frozen
        or binding_path.read_bytes() != raw
        or any(sha256(helpers / name) != digest for name, digest in inventory.items())
    ):
        raise ValueError(f"{wave} captured observation differs from its frozen production binding")
    return frozen, binding_digest


def capture(root: Path, output: Path, *, wave: str = "wave9") -> dict:
    """Read every row once as bytes and reject incomplete data before publishing a capture."""
    _, arms, _ = PROTOCOLS[wave]
    frozen, binding_digest = verify_production_binding(root, wave)
    if tuple(frozen["arms"]) != arms:
        raise ValueError(f"{wave} captured arm inventory differs from its frozen production binding")
    candidate = Path(frozen["candidate"])
    if output.exists():
        raise FileExistsError(f"{wave} observation already exists; preserve its exact identity")
    if output.resolve().is_relative_to(candidate.resolve()):
        raise ValueError(f"{wave} observation must stay outside the frozen candidate")
    expected_sources = executed_sources(candidate, frozen)
    payloads, identities = {}, {}
    for arm in arms:
        directory = root / "runs" / wave / arm
        manifest_bytes = (directory / "run.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        config = read(candidate / f"experiment/configs/{arm}_0p5m_seed0.json")
        training = config["training"]
        if (
            manifest["config"] != config
            or manifest["status"] != "running"
            or manifest.get("execution_check")
            or manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]
            or normalized_sources(candidate, manifest["source_sha256"]) != expected_sources
            or config["experiment"]["arm"] != arm
            or training["steps"] != 5120
            or training["tokens"] != 2684354560
            or training["batch_tokens"] != 524288
            or training["world_size"] != 8
            or training["seed"] != 0
        ):
            raise ValueError(f"{wave} live manifest differs from its running full-budget source/config/data")
        segments = sorted(directory.glob("metrics_from_*.jsonl"))
        if [path.name for path in segments] != ["metrics_from_000000.jsonl"]:
            raise ValueError(f"{wave} capture requires the single original non-resumed history")
        raw = segments[0].read_bytes()
        if not raw or not raw.endswith(b"\n"):
            raise ValueError(f"{wave} capture ends with an unfinished row; retry the observation")
        rows = [json.loads(line) for line in raw.splitlines()]
        validate_rows(rows, training["batch_tokens"])
        if len(rows) > training["steps"]:
            raise ValueError(f"{wave} capture exceeds the full training budget")
        compare({arm: RunHistory(manifest, rows, [rows])}, arm, WINDOW)
        payloads[Path(arm) / "run.json"] = manifest_bytes
        payloads[Path(arm) / segments[0].name] = raw
        identities[arm] = {
            "observed_updates": len(rows),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "history_sha256": {segments[0].name: hashlib.sha256(raw).hexdigest()},
        }
    if sha256(root / f"run-preparation/{wave}-launch-binding.json") != binding_digest:
        raise ValueError(f"{wave} verified release bytes changed during history capture")
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        destination = output / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(raw)
    result = {
        "status": f"captured_strict_{wave}_observation",
        "source_release_sha256": binding_digest,
        "observation": str(output.resolve()),
        "runs": identities,
        "scope": "Immutable fully parsed running prefixes; no omitted rows, final result or cloud import.",
    }
    write(output / "observation.json", result)
    return result


def analyze_capture(observation: Path) -> dict:
    """Rederive all matched pairs from the same immutable per-arm observation."""
    record = read(observation / "observation.json")
    wave = {f"captured_strict_{name}_observation": name for name in PROTOCOLS}.get(record["status"])
    if wave is None:
        raise ValueError("Production observation protocol differs")
    _, arms, pairs = PROTOCOLS[wave]
    if set(record["runs"]) != set(arms):
        raise ValueError("Production observation arm inventory differs")
    runs = {}
    for arm in arms:
        directory, identity = observation / arm, record["runs"][arm]
        if identity["manifest_sha256"] != sha256(directory / "run.json") or identity["history_sha256"] != {
            path.name: sha256(path) for path in sorted(directory.glob("metrics_from_*.jsonl"))
        }:
            raise ValueError("Immutable production observation changed after capture")
        run = history(directory)
        if run.manifest["status"] != "running" or len(run.rows) != identity["observed_updates"]:
            raise ValueError("Production captured running identity or update coverage changed")
        runs[f"{wave}/{arm}"] = run
    return {
        "wave": wave,
        "window_tokens": WINDOW,
        "source_release_sha256": record["source_release_sha256"],
        "analyses": [
            compare({f"{wave}/{arm}": runs[f"{wave}/{arm}"] for arm in (baseline, *names)}, f"{wave}/{baseline}", WINDOW)
            for baseline, names in pairs
        ],
        "unpaired_runs": {},
    }


def captured_cohort(observation: Path) -> dict:
    """Compare every captured arm on one token prefix and the same sparse evaluations."""
    analysis = analyze_capture(observation)
    wave = analysis["wave"]
    observed = {arm: history(observation / arm) for arm in PROTOCOLS[wave][1]}
    common_tokens = min(run.rows[-1]["tokens"] for run in observed.values())
    prefixes = {arm: [row for row in run.rows if row["tokens"] <= common_tokens] for arm, run in observed.items()}
    common_evaluations = {}
    for key in EVALUATION_KEYS:
        shared = sorted(set.intersection(*(set(evaluations(rows, key)) for rows in prefixes.values())))
        late = [token for token in shared if common_tokens < 2 * token]
        common_evaluations[key] = {"shared_tokens": shared, "late_tokens": late}
    runs = {}
    for arm, rows in prefixes.items():
        selected = [
            {
                key: value
                for key, value in row.items()
                if key not in EVALUATION_KEYS or row["tokens"] in common_evaluations[key]["shared_tokens"]
            }
            for row in rows
        ]
        runs[f"{wave}/{arm}"] = RunHistory(observed[arm].manifest, selected, [selected])
    result = compare(runs, f"{wave}/muont", WINDOW)
    result["captured_observed_updates"] = {arm: len(run.rows) for arm, run in observed.items()}
    result["common_tokens"] = common_tokens
    result["common_evaluations"] = common_evaluations
    return result


def refresh_running_capture(root: Path, now: datetime, *, wave: str = "wave9") -> dict:
    """Capture all-running histories for the existing document monitor, preserving any refusal."""
    release_path = root / f"run-preparation/{wave}-launch-binding.json"
    manifests = [root / "runs" / wave / arm / "run.json" for arm in PROTOCOLS[wave][1]]
    if not release_path.is_file() or not all(path.is_file() for path in manifests):
        return {"status": f"{wave}_capture_not_initialized"}
    if any(read(path).get("status") != "running" for path in manifests):
        return {"status": f"{wave}_capture_not_all_running"}
    stamp = now.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / f"run-preparation/{wave}-observations" / f"capture-{stamp}"
    observed = capture(root, output, wave=wave)
    analysis = analyze_capture(output)
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    return {"status": "strict_comparison_saved", "observation": observed["observation"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--wave", choices=tuple(PROTOCOLS), default="wave9")
    args = parser.parse_args()
    observed = capture(args.root, args.observation, wave=args.wave)
    analysis = analyze_capture(args.observation)
    (args.observation / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"observation": observed["observation"], "status": "strict_comparison_saved"}))


if __name__ == "__main__":
    main()
