# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gate and launch the missing seed-2 MuonT companion after wave-4 MuonT."""

import argparse
import contextlib
import io
import json
import os
import subprocess
from pathlib import Path

from experiment.data_access import require_receipt
from experiment.launch import ENVIRONMENT, PYTHON, config_path, start_local
from experiment.records import atomic_json, sha256, utc_now
from experiment.reference import TASK
from extend_validation import extend

NODE = "28.216.17.144"
WAVE = "wave4-baseline"
KEEPER = Path("/root/dl/runtime/script/hy_infuse_moe/gpu_resource_manager.sh")


def launch(root: Path) -> None:
    preparation = root / "run-preparation"
    queued = json.loads((preparation / "seed2-companion-queue.json").read_text())
    state_path = preparation / "seed2-companion-state.json"
    report_path = preparation / f"{WAVE}.json"
    validation = preparation / f"validation-{WAVE}.json"
    original = preparation / "validation-wave4.json"
    data, data_receipt = root / "data", root / "data/experiment_receipt.json"
    config = config_path("muont", 2, "0p5m")
    out = root / "runs" / WAVE / "muont_seed2"
    state = {
        "time": utc_now(), "status": "preflight", "node": NODE, "config": str(config),
        "config_sha256": sha256(config), "original_validation_sha256": sha256(original),
    }
    atomic_json(state_path, state)
    if queued["node"] != NODE or queued["config_sha256"] != sha256(config):
        raise ValueError("Queued companion node/configuration changed")
    if queued["validation_sha256"] != sha256(original):
        raise ValueError("Queued validation receipt changed")
    if any(sha256(TASK / name) != expected for name, expected in queued["helpers_sha256"].items()):
        raise ValueError("Queued helper source changed")
    if out.exists() or report_path.exists() or validation.exists():
        raise FileExistsError("Follow-up has existing launch/validation evidence; inspect it before retrying")
    launch_record = json.loads((preparation / "wave4.json").read_text())
    previous = next(entry for entry in launch_record["launches"] if entry["node"] == NODE)
    if previous["arm"] != "muont" or previous["batch"] != "1m" or previous["seed"] != 0:
        raise ValueError("Unexpected predecessor on the companion node")
    process = subprocess.run(
        ["ps", "-p", str(previous["pid"]), "-o", "stat=,args="], capture_output=True, text=True, check=False,
    )
    if process.returncode not in (0, 1):
        raise RuntimeError("Could not establish predecessor process liveness")
    if process.returncode == 0 and not process.stdout.strip().startswith("Z"):
        raise RuntimeError("Predecessor is still running or its PID was reused")
    manifest = json.loads((Path(previous["out"]) / "run.json").read_text())
    if manifest["status"] != "complete" or manifest["completed_steps"] != manifest["config"]["training"]["steps"]:
        raise ValueError("Predecessor did not complete its budget")
    restored = root / "runs/wave4" / f"keeper-restore-{NODE}.log"
    if "keeper start exit 0" not in restored.read_text():
        raise RuntimeError("Wait for predecessor keeper watcher to finish before transferring this node")
    prior = json.loads(original.read_text())
    expected = {str(path.relative_to(TASK)): sha256(path) for path in sorted((TASK / "experiment").rglob("*.py"))}
    if prior["source_sha256"] != expected or prior["environment"] != ENVIRONMENT or not all(prior["checks"].values()):
        raise ValueError("Source/environment/validation changed; companion cannot reuse these gates")
    require_receipt(data, data_receipt)
    if sha256(data_receipt) != prior["data_receipt_sha256"]:
        raise ValueError("Prepared data receipt changed")
    if config.resolve() != (TASK / "experiment/configs/muont_0p5m_seed2.json").resolve():
        raise ValueError("Unexpected companion configuration")
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    ).splitlines()
    if len(devices) != 8 or any("A100-SXM4-40GB" not in device for device in devices):
        raise ValueError("Companion requires eight A100 40GB GPUs")
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
    ).split()
    if occupied:
        keeper_pid = int(Path("/root/dl/runtime/gpu_matmul.pid").read_text())
        arguments = (Path("/proc") / str(keeper_pid) / "cmdline").read_bytes().split(b"\0")
        if b"/root/dl/runtime/script/hy_infuse_moe/gpu_matmul.py" not in arguments:
            raise RuntimeError("GPU occupancy does not belong to the known keeper")
        for pid in occupied:
            fields = (Path("/proc") / pid / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) != keeper_pid:
                raise RuntimeError("Unrelated CUDA process present; companion refused")
        state["released_keeper_pid"] = keeper_pid
        subprocess.run(
            ["bash", str(KEEPER), "stop"], env={**os.environ, "GPU_RESOURCE_MANAGER_LOCAL_ONLY": "1"}, check=True,
        )
    state.update(status="validating", time=utc_now())
    atomic_json(state_path, state)
    training_started = False
    try:
        extend(
            original, validation, root / "validation/wave4_baseline_seed2",
            data, data_receipt, {"muont_seed2": config},
        )
        receipt = json.loads(validation.read_text())
        if not all(receipt["checks"].values()) or receipt["config_sha256"] != {"muont_seed2": sha256(config)}:
            raise ValueError("Companion gates incomplete")
        state.update(status="benchmarking", time=utc_now())
        atomic_json(state_path, state)
        benchmark = preparation / "benchmark-diagonal-20261116.json"
        with (preparation / "benchmark-diagonal-20261116.log").open("x") as log:
            probe = subprocess.run(
                [PYTHON, str(TASK / "tracking/benchmark_diagonal.py"), "--out", str(benchmark)],
                cwd=TASK, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **ENVIRONMENT}, check=False,
            )
        state["benchmark_exit_code"] = probe.returncode
        state["benchmark_receipt"] = str(benchmark) if probe.returncode == 0 else None
        current = {str(path.relative_to(TASK)): sha256(path) for path in sorted((TASK / "experiment").rglob("*.py"))}
        if receipt["source_sha256"] != current or receipt["config_sha256"] != {"muont_seed2": sha256(config)}:
            raise ValueError("Validated source/configuration changed before companion launch")
        occupied = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
        ).strip()
        if occupied:
            raise RuntimeError("CUDA occupancy remains after benchmark; companion launch refused")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            start_local(config, data, data_receipt, out)
        started = json.loads(output.getvalue())
        training_started = True
        report = {
            "time": utc_now(), "environment": ENVIRONMENT, "blockers": [],
            "validation_receipt": str(validation), "validation_sha256": sha256(validation),
            "launches": [{"node": NODE, "arm": "muont", "seed": 2, "batch": "0p5m", **started}],
        }
        atomic_json(report_path, report)
        with (out.parent / f"keeper-restore-{NODE}.log").open("x") as stream:
            watcher = subprocess.Popen(
                ["bash", str(TASK / "tracking/watch_keeper.sh"), str(started["pid"]), str(out)],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
        state.update(
            status="launched", time=utc_now(), training_pid=started["pid"],
            keeper_watcher_pid=watcher.pid, launch_receipt=str(report_path),
        )
        atomic_json(state_path, state)
        print(json.dumps(state, indent=2))
    except Exception:
        state.update(status="needs_attention", time=utc_now(), training_started=training_started)
        atomic_json(state_path, state)
        if not training_started:
            occupied = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
            ).strip()
            if not occupied:
                subprocess.run(
                    ["bash", str(KEEPER), "start"],
                    env={**os.environ, "GPU_RESOURCE_MANAGER_LOCAL_ONLY": "1"}, check=True,
                )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    addresses = subprocess.check_output(["hostname", "-I"], text=True).split()
    if NODE not in addresses:
        raise RuntimeError("Run this gate on the predecessor's node")
    launch(args.root)


if __name__ == "__main__":
    main()
