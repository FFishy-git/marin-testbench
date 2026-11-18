# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Save run manifests, latest metrics, and fresh process evidence without importing Torch."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def snapshot(root: Path, wave: str) -> dict:
    launch = json.loads((root / "run-preparation" / f"{wave}.json").read_text())
    runs = []
    for entry in launch["launches"]:
        directory = Path(entry["out"])
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    f"root@{entry['node']}",
                    f"ps -p {int(entry['pid'])} -o pid=,stat=,etime=,args=",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            result = subprocess.CompletedProcess([], 124, "", "SSH observation timed out; liveness unknown")
        manifest_path = directory / "run.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        latest = None
        for segment in sorted(directory.glob("metrics_from_*.jsonl")):
            with segment.open("rb") as stream:
                stream.seek(max(0, segment.stat().st_size - 32768))
                lines = stream.read().splitlines()
            for line in reversed(lines):
                try:
                    latest = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        process = result.stdout.strip()
        fields = process.split()
        alive = (
            result.returncode == 0
            and bool(process)
            and "-m torch.distributed.run" in process
            and str(directory) in process
        )
        missing = result.returncode == 1 and not process
        zombie = result.returncode == 0 and len(fields) > 1 and fields[1].startswith("Z")
        status = "unknown"
        if zombie or missing:
            status = "complete" if manifest and manifest["status"] == "complete" else "stopped_incomplete"
        elif alive:
            status = "running"
        runs.append(
            {
                "name": directory.name,
                "arm": entry["arm"],
                "node": entry["node"],
                "pid": entry["pid"],
                "observation_exit_code": result.returncode,
                "process": process,
                "observation_error": result.stderr.strip(),
                "status": status,
                "latest_metrics": latest,
                "manifest_status": None if manifest is None else manifest["status"],
                "completed_steps": None if manifest is None else manifest.get("completed_steps"),
                "planned_steps": None if manifest is None else manifest["config"]["training"]["steps"],
                "seed": None if manifest is None else manifest["config"]["training"]["seed"],
                "batch_tokens": None if manifest is None else manifest["config"]["training"]["batch_tokens"],
            }
        )
    state = "complete" if runs and all(run["status"] == "complete" for run in runs) else "active"
    if any(run["status"] == "stopped_incomplete" for run in runs):
        state = "needs_attention"
    if len(runs) != launch.get("expected_launches", len(runs)):
        state = "partial"
    return {"time": datetime.now(UTC).isoformat(), "wave": wave, "status": state, "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wave", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = snapshot(args.root, args.wave)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(record["status"])


if __name__ == "__main__":
    main()
