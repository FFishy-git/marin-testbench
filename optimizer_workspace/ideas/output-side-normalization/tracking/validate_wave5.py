# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the frozen wave-5 candidate on the completed wave-4 lagged node."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

NODE = "28.216.17.9"
PYTHON = "/opt/conda/envs/torch-base/bin/python"
KEEPER = "/root/dl/runtime/script/hy_infuse_moe/gpu_resource_manager.sh"
ENVIRONMENT = {
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONFAULTHANDLER": "1",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_state(path: Path, state: dict) -> None:
    state["time"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate(root: Path, candidate: Path) -> None:
    preparation = root / "run-preparation"
    state_path = preparation / "wave5-validation-state.json"
    queued = json.loads((preparation / "wave5-candidate.json").read_text())
    state = {"status": "preflight", "candidate": str(candidate), "node": NODE}
    save_state(state_path, state)
    if NODE not in subprocess.check_output(["hostname", "-I"], text=True).split():
        raise RuntimeError("Candidate validation must run on its designated predecessor node")
    if candidate.resolve() != Path(queued["candidate"]).resolve():
        raise ValueError("Unexpected candidate directory")
    if any(sha256(candidate / name) != expected for name, expected in queued["files_sha256"].items()):
        raise ValueError("Frozen candidate source/configuration changed")
    if sha256(root / "data/experiment_receipt.json") != queued["data_receipt_sha256"]:
        raise ValueError("Prepared data identity changed")
    if sha256(Path(__file__)) != queued["validator_sha256"]:
        raise ValueError("Queued validation controller changed")
    predecessor = next(
        entry for entry in json.loads((preparation / "wave4.json").read_text())["launches"] if entry["node"] == NODE
    )
    if predecessor["arm"] != "inner_outer_mlpdiag_lag" or predecessor["pid"] != queued["predecessor_pid"]:
        raise ValueError("Unexpected predecessor identity")
    process = subprocess.run(
        ["ps", "-p", str(predecessor["pid"]), "-o", "stat=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode not in (0, 1):
        raise RuntimeError("Cannot establish predecessor liveness")
    if process.returncode == 0 and not process.stdout.strip().startswith("Z"):
        raise RuntimeError("Predecessor is still live or its PID was reused")
    manifest = json.loads((Path(predecessor["out"]) / "run.json").read_text())
    if manifest["status"] != "complete" or manifest["completed_steps"] != manifest["config"]["training"]["steps"]:
        raise RuntimeError("Predecessor did not finish its token budget")
    if "keeper start exit 0" not in (root / "runs/wave4" / f"keeper-restore-{NODE}.log").read_text():
        raise RuntimeError("Predecessor keeper restoration has not finished")
    receipt = preparation / "validation-wave5-gpu.json"
    work = root / "validation/wave5_fused_diagonal"
    if receipt.exists() or work.exists():
        raise FileExistsError("Candidate validation already has evidence; inspect it instead of retrying")
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).split()
    if occupied:
        keeper_pid = int(Path("/root/dl/runtime/gpu_matmul.pid").read_text())
        arguments = (Path("/proc") / str(keeper_pid) / "cmdline").read_bytes().split(b"\0")
        if b"/root/dl/runtime/script/hy_infuse_moe/gpu_matmul.py" not in arguments:
            raise RuntimeError("CUDA occupancy is not the known task keeper")
        for pid in occupied:
            fields = (Path("/proc") / pid / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) != keeper_pid:
                raise RuntimeError("Unrelated CUDA process prevents validation")
        subprocess.run(
            ["bash", KEEPER, "stop"],
            env={**os.environ, "GPU_RESOURCE_MANAGER_LOCAL_ONLY": "1"},
            check=True,
        )
        state["released_keeper_pid"] = keeper_pid
    state["status"] = "validating"
    save_state(state_path, state)
    try:
        command = [
            PYTHON,
            "-m",
            "experiment.validate",
            "--gpu",
            "--data",
            str(root / "data"),
            "--data-receipt",
            str(root / "data/experiment_receipt.json"),
            "--work",
            str(work),
            "--receipt",
            str(receipt),
        ]
        with (preparation / "validation-wave5-gpu.log").open("x") as log:
            execution = subprocess.run(
                command,
                cwd=candidate,
                env={**os.environ, **ENVIRONMENT},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        state["exit_code"] = execution.returncode
        if execution.returncode:
            raise RuntimeError("Wave-5 candidate validation failed; inspect its preserved log and receipt")
        result = json.loads(receipt.read_text())
        if not all(result["checks"].values()):
            raise ValueError("Candidate validation returned without all required passing gates")
        current = {
            str(path.relative_to(candidate)): sha256(path) for path in sorted((candidate / "experiment").rglob("*.py"))
        }
        if result["source_sha256"] != current:
            raise ValueError("Validated candidate source identity changed")
        state.update(status="passed", validation_receipt=str(receipt), validation_sha256=sha256(receipt))
        save_state(state_path, state)
    except Exception:
        state["status"] = "needs_attention"
        save_state(state_path, state)
        raise
    finally:
        occupied = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if occupied:
            state["keeper_restore"] = "refused_cuda_occupancy"
        else:
            restoration = subprocess.run(
                ["bash", KEEPER, "start"],
                env={**os.environ, "GPU_RESOURCE_MANAGER_LOCAL_ONLY": "1"},
                check=False,
            )
            state["keeper_restore"] = restoration.returncode
        save_state(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    validate(args.root, args.candidate)


if __name__ == "__main__":
    main()
