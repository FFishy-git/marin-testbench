# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transfer completed allocations to the independently gated wave-5 candidate."""

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
PYTHON = "/opt/conda/envs/torch-base/bin/python"
KEEPER = "/root/dl/runtime/script/hy_infuse_moe/gpu_resource_manager.sh"
WORKLOAD = b"/root/dl/runtime/script/hy_infuse_moe/gpu_matmul.py"
NODES = ("28.216.17.9", "28.216.17.80", "28.216.17.144", "28.216.17.139")
REQUIRED_GATES = (
    "cpu_numerics",
    "cpu_distributed_statistics",
    "model_parameter_count",
    "gpu_collector",
    "distributed_statistics",
    "fa2",
    "compiled_model",
    "real_data_four_arm_smoke",
    "checkpoint_continuation",
    "gpu_down_updates",
    "cached_root_continuation",
    "microbatch_partition",
    "real_data_cached_root_continuation",
    "real_data_fused_diagonal_capture",
    "gpu_output_rows",
    "gpu_block_updates",
    "block_distributed_statistics",
    "block_compiled_body",
    "fused_diagonal_pipeline",
)
HELPERS = (
    "tracking/launch_wave5.py",
    "tracking/queue_wave5_launch.sh",
    "tracking/monitor.sh",
    "tracking/snapshot.py",
    "tracking/analyze_wave.py",
    "tracking/compare_runs.py",
    "tracking/swanlab_upload.py",
    "tracking/watch_keeper.sh",
    "tracking/watch_wave5.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, record: dict) -> None:
    record["time"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bind_queue(root: Path) -> None:
    preparation = root / "run-preparation"
    path = preparation / "wave5-launch-queue.json"
    archive = preparation / "source-wave5-launch-controller-20261116.tar"
    if path.exists() or archive.exists():
        raise FileExistsError("Wave-5 launch queue binding already exists")
    with tarfile.open(archive, "x") as stream:
        for name in HELPERS:
            stream.add(TASK / name, arcname=name, recursive=False)
    write(
        path,
        {
            "status": "queued_pending_validation_and_predecessors",
            "candidate_receipt_sha256": sha256(preparation / "wave5-candidate.json"),
            "helpers_sha256": {name: sha256(TASK / name) for name in HELPERS},
            "archive": str(archive),
            "archive_sha256": sha256(archive),
            "launch_condition": (
                "Candidate GPU gates/restoration pass; wave4 and seed2 baseline complete, "
                "analyzed and cloudverified; verified allocation transfer"
            ),
        },
    )


def binding(root: Path) -> tuple[Path, dict]:
    preparation = root / "run-preparation"
    queued = read(preparation / "wave5-launch-queue.json")
    if any(sha256(TASK / name) != expected for name, expected in queued["helpers_sha256"].items()):
        raise ValueError("Wave-5 launch helpers changed after queue binding")
    if sha256(preparation / "wave5-candidate.json") != queued["candidate_receipt_sha256"]:
        raise ValueError("Wave-5 candidate binding changed")
    frozen = read(preparation / "wave5-candidate.json")
    candidate = Path(frozen["candidate"])
    if any(sha256(candidate / name) != expected for name, expected in frozen["files_sha256"].items()):
        raise ValueError("Frozen candidate source/configuration/pinned sources changed")
    if sha256(root / "data/experiment_receipt.json") != frozen["data_receipt_sha256"]:
        raise ValueError("Prepared data identity changed")
    return candidate, frozen


def require_release(root: Path) -> tuple[Path, Path]:
    candidate, frozen = binding(root)
    preparation = root / "run-preparation"
    state = read(preparation / "wave5-validation-state.json")
    receipt_path = preparation / "validation-wave5-gpu.json"
    if state["status"] != "passed" or type(state["keeper_restore"]) is not int or state["keeper_restore"] != 0:
        raise ValueError("Candidate GPU validation or keeper restoration has not passed")
    if state["validation_sha256"] != sha256(receipt_path):
        raise ValueError("Candidate validation receipt changed")
    validation = read(receipt_path)
    if not all(validation["checks"][name] is True for name in REQUIRED_GATES):
        raise ValueError("Required wave-5 execution gates are incomplete")
    sources = {
        name: value
        for name, value in frozen["files_sha256"].items()
        if name.startswith("experiment/") and name.endswith(".py")
    }
    if validation["source_sha256"] != sources or validation["data_receipt_sha256"] != frozen["data_receipt_sha256"]:
        raise ValueError("Candidate validation source/data identity differs")
    cluster = read(candidate / "experiment/cluster.json")
    if tuple(cluster["nodes"]) != NODES or cluster["wave_name"] != "wave5":
        raise ValueError("Wave-5 allocation changed")
    expected = {
        entry["arm"]: sha256(
            candidate / "experiment/configs" / f"{entry['arm']}_{entry['batch']}_seed{entry['seed']}.json"
        )
        for entry in cluster["wave"]
    }
    if validation["config_sha256"] != expected or validation["environment"] != cluster["environment"]:
        raise ValueError("Full-model gates do not cover the executed wave settings")
    if not validation["evidence"]["gpu_commands"]:
        raise ValueError("Candidate gates have no execution evidence")
    for execution in validation["evidence"]["gpu_commands"]:
        if execution["exit_code"] != 0 or sha256(Path(execution["log"])) != execution["log_sha256"]:
            raise ValueError("Candidate gate execution failed or its log changed")
    for wave in ("wave4", "wave4-baseline"):
        launch = read(preparation / f"{wave}.json")
        cloud = read(preparation / f"swanlab-upload-{wave}.json")
        names = {Path(entry["out"]).name for entry in launch["launches"]}
        if cloud["status"] != "verified" or names != set(cloud["runs"]):
            raise ValueError("Predecessor wave has no complete verified cloud receipt")
        for name, imported in cloud["runs"].items():
            run = root / "runs" / wave / name
            if imported["status"] != "verified" or imported["manifest_sha256"] != sha256(run / "run.json"):
                raise ValueError("Predecessor cloud verification is stale")
            if imported["metrics_sha256"] != sha256(run / "metrics_from_000000.jsonl"):
                raise ValueError("Predecessor scalar history changed after verification")
    return candidate, receipt_path


def predecessor(root: Path, node: str) -> tuple[str, dict]:
    wave = "wave4-baseline" if node == "28.216.17.144" else "wave4"
    entry = next(entry for entry in read(root / "run-preparation" / f"{wave}.json")["launches"] if entry["node"] == node)
    if wave == "wave4-baseline" and (entry["arm"], entry["seed"], entry["batch"]) != ("muont", 2, "0p5m"):
        raise ValueError("Missing seed-2 baseline predecessor")
    return wave, entry


def queue_status(root: Path, directory: Path) -> str:
    preparation = root / "run-preparation"
    validation_path = preparation / "wave5-validation-state.json"
    validation = read(validation_path) if validation_path.exists() else {}
    if validation.get("status") == "needs_attention":
        return "needs_attention"
    restoration = validation.get("keeper_restore")
    if restoration is not None and (type(restoration) is not int or restoration != 0):
        return "needs_attention"
    for wave, filename in (("wave4", "latest-wave4.json"), ("wave4-baseline", "latest-baseline.json")):
        path = directory / filename
        if not path.exists():
            return "waiting"
        snapshot = read(path)
        if snapshot["status"] == "needs_attention":
            return "needs_attention"
        if snapshot["status"] != "complete":
            return "waiting"
        cloud = preparation / f"swanlab-upload-{wave}.json"
        if not cloud.exists() or read(cloud).get("status") != "verified":
            return "waiting"
        for entry in read(preparation / f"{wave}.json")["launches"]:
            log = root / "runs" / wave / f"keeper-restore-{entry['node']}.log"
            if not log.exists() or "keeper start exit 0" not in log.read_text():
                return "waiting"
    return (
        "ready" if validation.get("status") == "passed" and type(restoration) is int and restoration == 0 else "waiting"
    )


def node_action(root: Path, action: str) -> None:
    addresses = subprocess.check_output(["hostname", "-I"], text=True).split()
    node = next(node for node in NODES if node in addresses)
    if action == "restore":
        processes = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
        if any("-m torch.distributed.run" in row or "-m experiment.train" in row for row in processes):
            raise RuntimeError("Live training/torchrun prevents keeper restoration, including unacknowledged launches")
        occupied = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        )
        if occupied.strip():
            raise RuntimeError("Keeper restoration refused while CUDA workloads remain")
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                try:
                    if WORKLOAD in (path / "cmdline").read_bytes().split(b"\0"):
                        raise RuntimeError("Existing keeper-path process prevents broad manager startup cleanup")
                except FileNotFoundError:
                    continue
        Path("/root/dl/runtime/gpu_matmul.pid").unlink(missing_ok=True)
        subprocess.run(["bash", KEEPER, "start"], env={**os.environ, "GPU_RESOURCE_MANAGER_LOCAL_ONLY": "1"}, check=True)
        return
    candidate, _ = require_release(root)
    wave, previous = predecessor(root, node)
    process = subprocess.run(["ps", "-p", str(previous["pid"]), "-o", "stat=,args="], capture_output=True, text=True)
    if process.returncode not in (0, 1):
        raise RuntimeError("Cannot establish predecessor liveness")
    if process.returncode == 0 and not process.stdout.strip().startswith("Z"):
        raise RuntimeError("Predecessor is live or its PID was reused")
    out = Path(previous["out"])
    manifest = read(out / "run.json")
    training = manifest["config"]["training"]
    if manifest["status"] != "complete" or manifest["completed_steps"] != training["steps"]:
        raise ValueError("Predecessor did not finish its token budget")
    complete = read(out / "checkpoints" / f"step_{training['steps']:06d}/complete.json")
    if complete["step"] != training["steps"] or complete["world"] != 8:
        raise ValueError("Predecessor final checkpoint receipt differs")
    watcher_log = root / "runs" / wave / f"keeper-restore-{node}.log"
    if "keeper start exit 0" not in watcher_log.read_text():
        raise RuntimeError("Predecessor keeper watcher has not restored the allocation")
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(devices) != 8 or any("A100-SXM4-40GB" not in device for device in devices):
        raise ValueError("Wave-5 requires eight A100 40GB GPUs on every node")
    subprocess.run([PYTHON, "-m", "experiment.bootstrap", "--check-only"], cwd=candidate, check=True)
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).split()
    if occupied:
        keeper_pid = int(Path("/root/dl/runtime/gpu_matmul.pid").read_text())
        if WORKLOAD not in (Path("/proc") / str(keeper_pid) / "cmdline").read_bytes().split(b"\0"):
            raise RuntimeError("Allocation is not held by the identified keeper")
        for pid in occupied:
            fields = (Path("/proc") / pid / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[1]) != keeper_pid:
                raise RuntimeError("Unrelated CUDA workload prevents allocation transfer")
        parent_fields = (Path("/proc") / str(keeper_pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if int(parent_fields[2]) != keeper_pid or int(parent_fields[3]) != keeper_pid:
            raise RuntimeError("Keeper parent does not own its expected session/process group")
        generation = []
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            except FileNotFoundError:
                continue
            if int(fields[2]) == keeper_pid:
                if int(fields[3]) != keeper_pid:
                    raise RuntimeError("Keeper process group contains a foreign session")
                generation.append((int(path.name), fields[19]))
        if action == "release":
            current = (Path("/proc") / str(keeper_pid) / "stat").read_text().rsplit(")", 1)[1].split()
            if current[19] != parent_fields[19]:
                raise RuntimeError("Keeper parent PID reused before release")
            os.killpg(keeper_pid, signal.SIGTERM)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                alive = False
                for pid, started in generation:
                    try:
                        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
                    except FileNotFoundError:
                        continue
                    if fields[19] == started and fields[0] != "Z":
                        alive = True
                if not alive:
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Verified keeper generation did not stop; no broad process cleanup attempted")
            if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip():
                raise RuntimeError("CUDA occupancy remains after verified keeper release")
            pid_file = Path("/root/dl/runtime/gpu_matmul.pid")
            if pid_file.read_text().strip() == str(keeper_pid):
                pid_file.unlink()
    print(json.dumps({"node": node, "action": action, "keeper_occupied": bool(occupied)}))


def remote(root: Path, node: str, action: str) -> None:
    command = [PYTHON, str(Path(__file__).resolve()), "--root", str(root), "--node-action", action]
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"root@{node}", shlex.join(command)],
        check=True,
        timeout=180,
    )


def install_watchers(root: Path, report: dict) -> dict:
    errors = {}
    for entry in report["launches"]:
        if "keeper_watcher_pid" in entry:
            continue
        command = [
            PYTHON,
            str(TASK / "tracking/watch_wave5.py"),
            "--pid",
            str(entry["pid"]),
            "--out",
            entry["out"],
        ]
        try:
            result = subprocess.check_output(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    f"root@{entry['node']}",
                    shlex.join(command),
                ],
                text=True,
                timeout=45,
            )
            watched = json.loads(result)
            if watched["status"] not in ("watching", "restored"):
                raise RuntimeError("Watcher did not acknowledge a valid watching/restored state")
            entry["keeper_watcher_pid"] = watched["pid"]
            entry["keeper_watcher_status"] = watched["status"]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as error:
            errors[entry["node"]] = str(error)
        write(root / "run-preparation/wave5.json", report)
    return errors


def install_monitor(root: Path, state: dict) -> None:
    if "monitor_pid" in state:
        return
    directory = root / "run-preparation/monitor-wave5"
    directory.mkdir(exist_ok=True)
    with (directory / "monitor.log").open("a") as log:
        monitor = subprocess.Popen(
            ["bash", str(TASK / "tracking/monitor.sh"), str(root), "wave5"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state["monitor_pid"] = monitor.pid


def reconcile(root: Path, state: dict, report: dict) -> None:
    preparation = root / "run-preparation"
    report["expected_launches"] = 4
    state["status"] = "needs_attention"
    write(preparation / "wave5-launch-state.json", state)
    try:
        if report["launches"]:
            write(preparation / "wave5.json", report)
            state["watcher_errors"] = install_watchers(root, report)
            install_monitor(root, state)
    except Exception as error:
        state["monitoring_recovery_error"] = str(error)
    finally:
        active_nodes = {entry["node"] for entry in report["launches"]}
        restore_errors = {}
        for node in state["released_nodes"]:
            if node not in active_nodes:
                try:
                    remote(root, node, "restore")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    restore_errors[node] = str(error)
        state["restore_errors"] = restore_errors
        write(preparation / "wave5-launch-state.json", state)


def launch(root: Path) -> None:
    candidate, validation = require_release(root)
    preparation = root / "run-preparation"
    state_path = preparation / "wave5-launch-state.json"
    report_path = preparation / "wave5.json"
    if state_path.exists() or report_path.exists() or (root / "runs/wave5").exists():
        raise FileExistsError("Wave-5 already has execution evidence; inspect it instead of retrying")
    for wave in ("wave4", "wave4-baseline"):
        subprocess.run([PYTHON, str(TASK / "tracking/analyze_wave.py"), "--root", str(root), "--wave", wave], check=True)
        analysis = read(preparation / f"analysis-{wave}.json")
        for pair in analysis["analyses"]:
            if any(
                not comparison["final_comparison_eligible"] or len(comparison["training_windows"]) != 5
                for comparison in pair["comparisons"].values()
            ):
                raise ValueError("Predecessor broad-window analysis is not complete")
    for node in NODES:
        remote(root, node, "inspect")
    state = {"status": "transferring_allocation", "candidate": str(candidate), "released_nodes": []}
    write(state_path, state)
    report = {"launches": []}
    execution = None
    try:
        for node in NODES:
            state["released_nodes"].append(node)
            write(state_path, state)
            remote(root, node, "release")
        state["status"] = "launching"
        write(state_path, state)
        with (preparation / "launch-wave5.log").open("x") as log:
            execution = subprocess.Popen(
                [
                    PYTHON,
                    "-m",
                    "experiment.launch",
                    "--launch",
                    "--data",
                    str(root / "data"),
                    "--data-receipt",
                    str(root / "data/experiment_receipt.json"),
                    "--validation-receipt",
                    str(validation),
                    "--report",
                    str(report_path),
                    "--out",
                    str(root / "runs/wave5"),
                ],
                cwd=candidate,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            exit_code = execution.wait()
        report = read(report_path)
        report["expected_launches"] = 4
        state["watcher_errors"] = install_watchers(root, report)
        write(report_path, report)
        if report["launches"]:
            install_monitor(root, state)
        if exit_code or len(report["launches"]) != 4 or state["watcher_errors"]:
            raise RuntimeError("Wave-5 launch failed or only partially launched; inspect preserved evidence")
        state.update(status="launched", launch_receipt=str(report_path))
        write(state_path, state)
    except BaseException:
        if execution is not None and execution.poll() is None:
            try:
                execution.wait(timeout=180)
            except subprocess.TimeoutExpired:
                state["owned_launcher_join"] = "timed_out; preserved; restoration refuses live training"
        if report_path.exists() and not report["launches"]:
            report = read(report_path)
        reconcile(root, state, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node-action", choices=("inspect", "release", "restore"))
    parser.add_argument("--bind-queue", action="store_true")
    parser.add_argument("--queue-status", type=Path)
    args = parser.parse_args()
    if args.bind_queue:
        bind_queue(args.root)
    elif args.queue_status:
        print(queue_status(args.root, args.queue_status))
    elif args.node_action:
        node_action(args.root, args.node_action)
    else:
        interrupted_signals = []

        def interrupted(signum, frame):
            interrupted_signals.append(signum)

        # Allocation transfer is a bounded critical section. Defer termination
        # until the launcher joins and every started job has monitoring.
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        launch(args.root)
        if interrupted_signals:
            raise SystemExit(
                "Signal received during allocation transfer; controller joined and monitoring finalized: "
                f"{interrupted_signals}"
            )


if __name__ == "__main__":
    main()
