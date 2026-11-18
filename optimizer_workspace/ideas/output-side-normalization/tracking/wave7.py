# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze, validate and launch a fixed-age outer-diagonal wave without changing failed bindings."""

import argparse
import fcntl
import hashlib
import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import analyze_wave
from launch_wave5 import NODES, PYTHON, read, sha256
from launch_wave5 import node_action as keeper_action
from run_diagonal_diagnostic import PROCESS_MARKER, finish_child, release_keeper, tagged_processes
from validate_wave5 import ENVIRONMENT
from watch_wave5 import process, write
from watch_wave7 import ensure

TASK = Path(__file__).resolve().parents[1]
ARMS = ("muont", "inner_outer_leftdiag", "outer_leftdiag", "outer_leftdiag_fused")
PREDECESSORS = ("wave4", "wave4-baseline")
HELPERS = (
    "wave7.py",
    "analyze_wave7.py",
    "swanlab_wave7.py",
    "monitor_wave7.sh",
    "snapshot_wave7.py",
    "snapshot.py",
    "analyze_wave.py",
    "compare_runs.py",
    "swanlab_upload.py",
    "run_diagonal_diagnostic.py",
    "launch_wave5.py",
    "validate_wave5.py",
    "watch_wave5.py",
    "watch_wave7.py",
)
GATES = (
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
    "real_data_outer_diagonal_update_parity",
    "gpu_output_rows",
    "gpu_block_updates",
    "block_distributed_statistics",
    "block_compiled_body",
    "outer_diagonal_pipeline",
)


def sources(candidate: Path) -> dict:
    return {str(path.relative_to(candidate)): sha256(path) for path in sorted((candidate / "experiment").rglob("*.py"))}


def candidate_inventory(candidate: Path) -> dict:
    """Inventory executed code/configs and self-contained pinned files, rejecting aliases."""
    materialized = candidate / "artifacts/pinned-sources-materialized"
    alias = candidate / "artifacts/external-sources"
    if (
        not alias.is_symlink()
        or alias.readlink() != Path("pinned-sources-materialized")
        or materialized.is_symlink()
        or not materialized.is_dir()
        or alias.resolve() != candidate.resolve() / "artifacts/pinned-sources-materialized"
    ):
        raise ValueError("Candidate external-sources must be the exact candidate-local relative symlink")
    files = []
    for directory in (candidate / "experiment", materialized):
        if directory.is_symlink():
            raise ValueError("Candidate source directories must not be symlinks")
        for path in directory.rglob("*"):
            if "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                raise ValueError(f"Candidate source inventory contains a symlink: {path}")
            if path.is_file() and (directory == materialized or path.suffix in (".py", ".json")):
                files.append(path)
    return {str(path.relative_to(candidate)): sha256(path) for path in sorted(files)}


def verify_cpu_receipt(root: Path, candidate: Path, cpu: dict) -> None:
    """Bind CPU execution to the exact source, configurations, data and allocation."""
    configs = {arm: candidate / "experiment/configs" / f"{arm}_0p5m_seed0.json" for arm in ARMS}
    if (
        cpu["source_sha256"] != sources(candidate)
        or type(cpu["evidence"]["cpu_test_exit_code"]) is not int
        or cpu["evidence"]["cpu_test_exit_code"] != 0
        or not all(cpu["checks"][name] is True for name in GATES[:3])
        or type(cpu["evidence"]["model_parameter_count"]) is not int
        or cpu["evidence"]["model_parameter_count"] != 265585152
        or cpu["environment"] != ENVIRONMENT
        or Path(cpu["candidate_path"]).resolve() != candidate.resolve()
        or cpu["cluster_sha256"] != sha256(candidate / "experiment/cluster.json")
        or cpu["upstream_manifest_sha256"] != sha256(candidate / "experiment/upstream_manifest.json")
        or cpu["config_sha256"] != {arm: sha256(path) for arm, path in configs.items()}
        or cpu["config_paths"] != {arm: str(path.resolve()) for arm, path in configs.items()}
        or Path(cpu["data_receipt"]).resolve() != (root / "data/experiment_receipt.json").resolve()
        or cpu["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
    ):
        raise ValueError("Wave-7 CPU evidence is failed or stale for source/config/cluster/data/environment")


def freeze(root: Path, candidate: Path) -> dict:
    """Bind a new candidate and immutable controller copy without changing earlier bindings."""
    preparation = root / "run-preparation"
    path = preparation / "wave7-candidate.json"
    archive = preparation / "source-wave7-fixed-age-outer-diagonal-20261116.tar"
    controller = root / "controllers/wave7-20261116/tracking"
    if path.exists() or archive.exists() or controller.exists():
        raise FileExistsError("Wave-7 binding already has evidence; inspect it instead of overwriting")
    original = read(preparation / "wave5-candidate.json")
    failed_wave6 = read(preparation / "wave6-candidate.json")
    if candidate.resolve() == Path(failed_wave6["candidate"]).resolve():
        raise ValueError("Wave-7 must not reuse the GPU-failed Wave-6 candidate")
    cpu_path = preparation / "validation-wave7-cpu.json"
    cpu = read(cpu_path)
    verify_cpu_receipt(root, candidate, cpu)
    cluster = read(candidate / "experiment/cluster.json")
    planned = tuple((entry["arm"], entry["seed"], entry["batch"]) for entry in cluster["wave"])
    if planned != tuple((arm, 0, "0p5m") for arm in ARMS):
        raise ValueError("Wave-7 arm, seed or batch allocation differs from the reviewed wave")
    if tuple(cluster["nodes"]) != NODES or cluster["wave_name"] != "wave7" or cluster["environment"] != ENVIRONMENT:
        raise ValueError("Wave-7 cluster identity differs")
    for arm in ARMS:
        config = read(candidate / "experiment/configs" / f"{arm}_0p5m_seed0.json")
        previous = read(Path(failed_wave6["candidate"]) / "experiment/configs" / f"{arm}_0p5m_seed0.json")
        policy = config["experiment"].pop("refresh_install_age")
        if type(policy) is not int or policy != 1 or config != previous:
            raise ValueError("Wave-7 must retain the exact Wave-6 configuration apart from fixed installation age")
    pinned = {name: digest for name, digest in original["files_sha256"].items() if name.startswith("artifacts/")}
    files = candidate_inventory(candidate)
    if not pinned or {name: digest for name, digest in files.items() if name.startswith("artifacts/")} != pinned:
        raise ValueError("Wave-7 pinned source inventory differs from the frozen original")
    documents = {name: (TASK / name).read_bytes() for name in ("ENDEAVOR.md", "experiment/README.md")}
    controller.mkdir(parents=True)
    for name in HELPERS:
        shutil.copy2(TASK / "tracking" / name, controller / name)
    with tarfile.open(archive, "x") as stream:
        for name in sorted(files):
            stream.add(candidate / name, arcname=name, recursive=False)
        stream.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
        for name in HELPERS:
            stream.add(controller / name, arcname=f"tracking/{name}", recursive=False)
        for name, content in documents.items():
            item = tarfile.TarInfo(name)
            item.size = len(content)
            stream.addfile(item, io.BytesIO(content))
    record = {
        "status": "frozen_pending_gpu_validation",
        "candidate": str(candidate.resolve()),
        "files_sha256": files,
        "source_sha256": sources(candidate),
        "helpers": str(controller),
        "helpers_sha256": {name: sha256(controller / name) for name in HELPERS},
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "cpu_receipt_sha256": sha256(cpu_path),
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "required_gates": GATES,
        "arms": ARMS,
        "predecessors": PREDECESSORS,
        "documents_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in documents.items()},
        "policy": "Fixed-age host-refresh control, age 1 on every Toner engine; no approval of failed Waves 5/6.",
        "refresh_install_age": 1,
        "failed_wave6_binding_sha256": sha256(preparation / "wave6-candidate.json"),
        "failed_wave6_gpu_receipt_sha256": sha256(preparation / "validation-wave6-gpu.json"),
    }
    write(path, record)
    return record


def verify_helper_only(root: Path) -> dict:
    """Verify restoration code and archive without depending on candidate or data contents."""
    record = read(root / "run-preparation/wave7-candidate.json")
    helper_directory = Path(record["helpers"])
    if set(record["helpers_sha256"]) != set(HELPERS):
        raise ValueError("Frozen wave-7 helper inventory differs")
    if any(sha256(helper_directory / name) != digest for name, digest in record["helpers_sha256"].items()):
        raise ValueError("Frozen wave-7 controller changed")
    if Path(__file__).resolve() != (helper_directory / "wave7.py").resolve():
        raise ValueError("Execute the bound controller copy, not the mutable checkout")
    if sha256(Path(record["archive"])) != record["archive_sha256"]:
        raise ValueError("Wave-7 source archive changed")
    return record


def binding(root: Path) -> tuple[Path, dict]:
    preparation = root / "run-preparation"
    record = verify_helper_only(root)
    candidate = Path(record["candidate"])
    if tuple(record["arms"]) != ARMS or tuple(record["required_gates"]) != GATES:
        raise ValueError("Wave-7 release contract changed")
    if tuple(record["predecessors"]) != PREDECESSORS:
        raise ValueError("Wave-7 predecessor contract changed")
    if (
        record["refresh_install_age"] != 1
        or record["failed_wave6_binding_sha256"] != sha256(preparation / "wave6-candidate.json")
        or record["failed_wave6_gpu_receipt_sha256"] != sha256(preparation / "validation-wave6-gpu.json")
    ):
        raise ValueError("Wave-7 scheduling policy or preserved Wave-6 failure changed")
    if candidate_inventory(candidate) != record["files_sha256"]:
        raise ValueError("Frozen wave-7 candidate inventory changed")
    if sha256(preparation / "validation-wave7-cpu.json") != record["cpu_receipt_sha256"]:
        raise ValueError("Wave-7 CPU evidence changed")
    if sha256(root / "data/experiment_receipt.json") != record["data_receipt_sha256"]:
        raise ValueError("Prepared data identity changed")
    return candidate, record


@contextmanager
def defer_signals():
    """Finish bounded cleanup before restoring the caller's signal handlers."""
    signals = []

    def interrupted(signum, frame):
        signals.append(signum)

    handlers = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield signals
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


@contextmanager
def node_guard(root: Path, node: str, *, lock_timeout: float = 0):
    """Serialize mutations on one node without acquiring the parent controller's shared lock."""
    with defer_signals() as signals:
        with (root / "run-preparation" / f".wave7-node-{node}.lock").open("a") as lock:
            deadline = time.monotonic() + lock_timeout
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            yield signals


def restore_keeper(root: Path) -> None:
    verify_helper_only(root)
    keeper_action(root, "restore")


def local_node() -> str:
    addresses = subprocess.check_output(["hostname", "-I"], text=True).split()
    return next(node for node in NODES if node in addresses)


def predecessor(root: Path, node: str) -> tuple[str, dict]:
    wave = "wave4-baseline" if node == "28.216.17.144" else "wave4"
    entry = next(entry for entry in read(root / "run-preparation" / f"{wave}.json")["launches"] if entry["node"] == node)
    observed = process(entry["pid"])
    if observed is not None and observed[0] != "Z":
        raise RuntimeError("Predecessor is still live or its PID was reused")
    run = Path(entry["out"])
    manifest = read(run / "run.json")
    steps = manifest["config"]["training"]["steps"]
    if manifest["status"] != "complete" or manifest["completed_steps"] != steps:
        raise RuntimeError("Predecessor has not completed")
    complete = read(run / f"checkpoints/step_{steps:06d}/complete.json")
    if complete["step"] != steps or complete["world"] != 8:
        raise ValueError("Predecessor checkpoint receipt differs")
    if "keeper start exit 0" not in (run.parent / f"keeper-restore-{node}.log").read_text():
        raise RuntimeError("Predecessor keeper restoration is incomplete")
    return wave, entry


def idle_training() -> None:
    commands = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
    if any("-m torch.distributed.run" in command or "-m experiment.train" in command for command in commands):
        raise RuntimeError("Existing training or unacknowledged torchrun prevents allocation transfer")


def bounded_execution(command: list, cwd: Path, log: Path, state: dict, state_path: Path, timeout: float) -> int:
    """Join a bounded launcher and all marker-bound children, including independent rank sessions."""
    marker = uuid4().hex
    generation = {}
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env={**os.environ, **ENVIRONMENT, PROCESS_MARKER: marker},
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            state.update(command=command, child_pid=child.pid, child_marker=marker, log=str(log))
            write(state_path, state)
            deadline = time.monotonic() + timeout
            while True:
                generation.update(tagged_processes(marker))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    return child.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            try:
                finish_child(child, marker, generation)
                state["cleanup_exit_code"] = child.returncode
            except Exception as error:
                state["cleanup_error"] = str(error)
                raise
            finally:
                state["child_generation"] = generation
                write(state_path, state)


def validate(root: Path) -> dict:
    with node_guard(root, local_node()) as signals:
        state = validate_on_node(root)
        if signals:
            state["deferred_signals"] = signals
            write(root / "run-preparation/wave7-validation-state.json", state)
        return state


def validate_on_node(root: Path) -> dict:
    candidate, frozen = binding(root)
    if local_node() != "28.216.17.9":
        raise RuntimeError("Wave-7 validation must run on .9")
    preparation = root / "run-preparation"
    state_path = preparation / "wave7-validation-state.json"
    receipt = preparation / "validation-wave7-gpu.json"
    work = root / "validation/wave7_fixed_age_outer_diagonal"
    if state_path.exists() or receipt.exists() or work.exists():
        raise FileExistsError("Wave-7 validation already has evidence; inspect it instead of retrying")
    predecessor(root, "28.216.17.9")
    idle_training()
    state = {"status": "preflight", "candidate": str(candidate), "node": local_node()}
    write(state_path, state)
    try:
        state["keeper"] = release_keeper(state, state_path)
        state["status"] = "validating"
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
        code = bounded_execution(
            command, candidate, preparation / "validation-wave7-gpu.log", state, state_path, timeout=7200
        )
        state["exit_code"] = code
        if code:
            raise RuntimeError("Wave-7 validation failed; inspect preserved logs")
        checked = read(receipt)
        if checked["source_sha256"] != frozen["source_sha256"] or not all(
            checked["checks"][name] is True for name in GATES
        ):
            raise ValueError("Wave-7 required actual execution gates have not passed")
        state.update(status="passed", validation_sha256=sha256(receipt))
    except Exception as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if state.get("keeper_release_attempted"):
            try:
                if "cleanup_error" in state:
                    raise RuntimeError("Live or unverified child cleanup prevents keeper restoration")
                restore_keeper(root)
                state["keeper_restore"] = 0
            except Exception as error:
                state.update(status="needs_attention", keeper_restore=f"refused: {error}")
        write(state_path, state)
    if type(state.get("keeper_restore")) is not int or state["keeper_restore"] != 0:
        raise RuntimeError("Wave-7 keeper restoration failed")
    return state


def require_release(root: Path) -> tuple[Path, Path, dict]:
    candidate, frozen = binding(root)
    preparation = root / "run-preparation"
    state = read(preparation / "wave7-validation-state.json")
    receipt = preparation / "validation-wave7-gpu.json"
    if state["status"] != "passed" or type(state.get("keeper_restore")) is not int or state["keeper_restore"] != 0:
        raise ValueError("Wave-7 validation/restoration has not passed")
    if state["validation_sha256"] != sha256(receipt):
        raise ValueError("Wave-7 GPU evidence changed")
    checked = read(receipt)
    if checked["source_sha256"] != frozen["source_sha256"] or not all(checked["checks"][name] is True for name in GATES):
        raise ValueError("Wave-7 source or required gates differ")
    if checked["data_receipt_sha256"] != frozen["data_receipt_sha256"] or checked["environment"] != ENVIRONMENT:
        raise ValueError("Wave-7 data/environment validation differs")
    configs = {arm: sha256(candidate / "experiment/configs" / f"{arm}_0p5m_seed0.json") for arm in ARMS}
    if checked["config_sha256"] != configs:
        raise ValueError("Exact wave-7 configurations were not validated")
    commands = checked["evidence"]["gpu_commands"]
    if not commands or any(
        item["exit_code"] != 0 or sha256(Path(item["log"])) != item["log_sha256"] for item in commands
    ):
        raise ValueError("Wave-7 validation commands failed or their logs changed")
    for wave in PREDECESSORS:
        cloud = read(preparation / f"swanlab-upload-{wave}.json")
        entries = read(preparation / f"{wave}.json")["launches"]
        if cloud["status"] != "verified" or set(cloud["runs"]) != {Path(entry["out"]).name for entry in entries}:
            raise ValueError("Predecessor wave is not fully cloud-verified")
        for name, imported in cloud["runs"].items():
            run = root / "runs" / wave / name
            if imported["status"] != "verified" or imported["manifest_sha256"] != sha256(run / "run.json"):
                raise ValueError("Predecessor manifest verification is stale")
            if imported["metrics_sha256"] != sha256(run / "metrics_from_000000.jsonl"):
                raise ValueError("Predecessor scalar verification is stale")
        analysis = read(preparation / f"analysis-{wave}.json")
        fresh = json.loads(json.dumps(analyze_wave.analyze(root, wave), allow_nan=False))
        expected = analyze_wave.COMPARISONS[wave]
        if (
            not expected
            or fresh["window_tokens"] != analyze_wave.WINDOW
            or len(fresh["analyses"]) != len(expected)
            or any(
                pair["baseline"] != baseline
                or set(pair["comparisons"]) != {baseline, *names}
                or any(
                    comparison["final_comparison_eligible"] is not True or len(comparison["training_windows"]) != 5
                    for comparison in pair["comparisons"].values()
                )
                for pair, (baseline, names) in zip(fresh["analyses"], expected, strict=True)
            )
            or analysis != fresh
        ):
            raise ValueError("Predecessor analysis is incomplete, unrelated or stale for verified histories")
    return candidate, receipt, frozen


def node_operation(root: Path, phase: str, *, lock_timeout: float = 0) -> dict:
    node = local_node()
    if phase in ("release", "restore"):
        with node_guard(root, node, lock_timeout=lock_timeout) as signals:
            result = operate_node(root, node, phase)
            if signals:
                result["deferred_signals"] = signals
            return result
    return operate_node(root, node, phase)


def operate_node(root: Path, node: str, phase: str) -> dict:
    if phase == "restore":
        restore_keeper(root)
        return {"node": node, "keeper_restore": 0}
    candidate, _, _ = require_release(root)
    predecessor(root, node)
    idle_training()
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(devices) != 8 or any("A100-SXM4-40GB" not in device for device in devices):
        raise ValueError("Wave-7 requires eight A100 40GB GPUs per node")
    subprocess.run([PYTHON, "-m", "experiment.bootstrap", "--check-only"], cwd=candidate, check=True)
    if phase == "inspect":
        return {"node": node, "status": "preflight_passed"}
    path = root / "run-preparation" / f"wave7-release-{node}.json"
    if path.exists():
        raise FileExistsError("Allocation already has release evidence; reconcile before another attempt")
    state = {"status": "releasing", "node": node}
    write(path, state)
    try:
        state["keeper"] = release_keeper(state, path)
        state["status"] = "released"
        write(path, state)
        return state
    except BaseException as error:
        state.update(status="needs_attention", error=str(error))
        if state.get("keeper_release_attempted"):
            try:
                restore_keeper(root)
                state["keeper_restore"] = 0
            except Exception as restore_error:
                state["keeper_restore"] = str(restore_error)
        write(path, state)
        raise


def remote(root: Path, frozen: dict, node: str, phase: str) -> dict:
    command = [PYTHON, str(Path(frozen["helpers"]) / "wave7.py"), "--root", str(root), "--phase", phase]
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"root@{node}", shlex.join(command)],
        capture_output=True,
        text=True,
        timeout=240,
    )
    if result.returncode:
        raise RuntimeError(f"{node} {phase}: {result.stderr[-4000:]}")
    return json.loads(result.stdout.splitlines()[-1])


def discover(root: Path) -> dict:
    """Reconcile a lost launch acknowledgment only from exact wave/config/process identity."""
    node = local_node()
    with node_guard(root, node):
        return discover_on_node(root, node)


def discover_on_node(root: Path, node: str) -> dict:
    frozen = verify_helper_only(root)
    candidate = Path(frozen["candidate"])
    arm = ARMS[NODES.index(node)]
    out = root / "runs/wave7" / arm
    config = candidate / "experiment/configs" / f"{arm}_0p5m_seed0.json"
    inventory_path = root / "run-preparation" / f"wave7-observed-{node}.json"
    found = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            arguments = (directory / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if b"torch.distributed.run" not in arguments or str(out).encode() not in arguments:
            continue
        if str(config).encode() not in arguments or b"experiment.train" not in arguments:
            raise RuntimeError("Wave-7 output is occupied by an unexpected training identity")
        observed = process(int(directory.name))
        if observed is None or observed[0] == "Z":
            continue
        found.append(
            {
                "node": node,
                "arm": arm,
                "seed": 0,
                "batch": "0p5m",
                "pid": int(directory.name),
                "start_ticks": observed[1],
                "out": str(out),
                "command": [argument.decode() for argument in arguments if argument],
                "source_binding_sha256": sha256(root / "run-preparation/wave7-candidate.json"),
            }
        )
    if len(found) > 1:
        raise RuntimeError("Multiple wave-7 parents have the same output identity")
    if not found:
        known = []
        if inventory_path.exists():
            known = [read(inventory_path)]
        elif (root / "run-preparation/wave7.json").exists():
            try:
                known = [
                    entry for entry in read(root / "run-preparation/wave7.json")["launches"] if entry["node"] == node
                ]
            except (OSError, ValueError, KeyError, TypeError) as error:
                # An unreadable report is not idle evidence; check live training and CUDA separately.
                idle_training()
                if subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
                ).strip():
                    raise RuntimeError("Unidentified CUDA occupancy prevents an idle observation") from error
                return {"node": node, "status": "idle", "launch": None}
        if len(known) > 1:
            raise RuntimeError("Multiple acknowledged launches belong to this node")
        if not known:
            idle_training()
            if subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
            ).strip():
                raise RuntimeError("Unidentified CUDA occupancy prevents an idle observation")
            return {"node": node, "status": "idle", "launch": None}
        entry = known[0]
        if (
            entry["node"] != node
            or entry["out"] != str(out)
            or entry["arm"] != arm
            or entry.get("seed") != 0
            or entry.get("batch") != "0p5m"
            or str(config) not in entry["command"]
            or "torch.distributed.run" not in entry["command"]
            or "experiment.train" not in entry["command"]
            or type(entry["pid"]) is not int
            or entry["pid"] <= 0
        ):
            raise RuntimeError("Acknowledged launch does not match the bound node/config identity")
        observed = process(entry["pid"])
        if observed is not None and observed[0] != "Z":
            raise RuntimeError("Acknowledged training PID is live with an unexpected identity")
        status = "exited"
    else:
        entry = found[0]
        status = "active"
    result = {"node": node, "status": status, "launch": entry}
    # Keep the process identity even if its watcher or local persistence fails.
    try:
        write(inventory_path, entry)
    except Exception as error:
        result["inventory_error"] = str(error)
    try:
        watcher = ensure(entry["pid"], out, root, entry.get("start_ticks"))
        if watcher["status"] not in ("watching", "restoring", "restored"):
            raise RuntimeError("Watcher did not acknowledge watching, restoring or restored")
        entry.update(keeper_watcher_pid=watcher["pid"], keeper_watcher_status=watcher["status"])
    except Exception as error:
        entry.pop("keeper_watcher_pid", None)
        entry.pop("keeper_watcher_status", None)
        result["watcher_error"] = str(error)
    try:
        write(inventory_path, entry)
    except Exception as error:
        result["inventory_error"] = str(error)
    return result


def merge_launch(root: Path, report: dict, entry: dict) -> None:
    node = entry["node"]
    if node not in NODES:
        raise ValueError("Launch inventory contains an unexpected node")
    arm = ARMS[NODES.index(node)]
    if (
        entry["arm"] != arm
        or entry["out"] != str(root / "runs/wave7" / arm)
        or entry.get("seed") != 0
        or entry.get("batch") != "0p5m"
        or type(entry.get("pid")) is not int
        or entry["pid"] <= 0
    ):
        raise ValueError("Launch inventory contains an unexpected arm/output identity")
    report["launches"] = [known for known in report["launches"] if known["node"] != node]
    report["launches"].append(entry)


def load_launch_report(root: Path, report: dict) -> None:
    path = root / "run-preparation/wave7.json"
    if path.exists():
        saved = read(path)
        if not isinstance(saved["launches"], list):
            raise ValueError("Invalid launch inventory")
        nodes = [entry["node"] for entry in saved["launches"]]
        if len(nodes) != len(set(nodes)):
            raise ValueError("Launch inventory contains duplicate nodes")
        for entry in saved["launches"]:
            merge_launch(root, report, entry)
        report.update({name: value for name, value in saved.items() if name != "launches"})
    report["expected_launches"] = len(ARMS)


def observe_nodes(root: Path, frozen: dict, nodes: list | tuple, state: dict, report: dict) -> None:
    """Preserve every observed identity and attempt all watchers despite independent failures."""
    observations = state.setdefault("node_observations", {})
    errors = state.setdefault("discovery_errors", {})
    watcher_errors = state.setdefault("watcher_errors", {})
    report_errors = state.setdefault("report_errors", {})
    for node in nodes:
        observations[node] = "unknown"
        try:
            observed = remote(root, frozen, node, "discover")
            if observed["node"] != node or observed["status"] not in ("active", "exited", "idle"):
                raise ValueError("Invalid node observation")
            entry = observed["launch"]
            if entry is not None:
                merge_launch(root, report, entry)
                if "watcher_error" in observed or entry.get("keeper_watcher_status") not in (
                    "watching",
                    "restoring",
                    "restored",
                ):
                    watcher_errors[node] = observed.get("watcher_error", "Missing watcher acknowledgment")
                else:
                    watcher_errors.pop(node, None)
            elif observed["status"] != "idle" or any(entry["node"] == node for entry in report["launches"]):
                raise ValueError("Missing acknowledged launch identity; node is not proven idle")
            observations[node] = observed["status"]
            errors.pop(node, None)
            if "inventory_error" in observed:
                report_errors[node] = observed["inventory_error"]
        except Exception as error:
            errors[node] = str(error)
        try:
            write(root / "run-preparation/wave7.json", report)
        except Exception as error:
            report_errors[node] = str(error)


def reconcile(root: Path, frozen: dict, state: dict, report: dict) -> None:
    """Recover monitoring and positively idle nodes independently of report I/O failures."""
    state["status"] = "needs_attention"
    try:
        try:
            load_launch_report(root, report)
        except Exception as error:
            state["report_recovery_error"] = str(error)
        observe_nodes(root, frozen, state["released_nodes"], state, report)
        if report["launches"]:
            try:
                state["monitor_pid"] = install_monitor(root, frozen)
            except Exception as error:
                state["monitor_error"] = str(error)
    finally:
        restore_errors = {}
        restored = []
        unknown = []
        for node in state["released_nodes"]:
            status = state.get("node_observations", {}).get(node, "unknown")
            if status == "unknown":
                unknown.append(node)
            if status != "idle" or any(entry["node"] == node for entry in report["launches"]):
                continue
            try:
                remote(root, frozen, node, "restore")
                restored.append(node)
            except Exception as error:
                restore_errors[node] = str(error)
        state.update(restored_nodes=restored, unknown_nodes=unknown, restore_errors=restore_errors)
        try:
            write(root / "run-preparation/wave7-launch-state.json", state)
        except Exception as error:
            state["state_persistence_error"] = str(error)


def install_monitor(root: Path, frozen: dict) -> int:
    directory = root / "run-preparation/monitor-wave7"
    directory.mkdir(exist_ok=True)
    if (directory / "pid").exists():
        pid = int((directory / "pid").read_text())
        observed = process(pid)
        if observed is not None and observed[0] != "Z" and "monitor_wave7.sh" in observed[2]:
            return pid
        raise RuntimeError("Existing wave-7 monitor is stopped; inspect before retrying")
    with (root / "run-preparation/monitor-wave7.log").open("a") as log:
        child = subprocess.Popen(
            ["bash", str(Path(frozen["helpers"]) / "monitor_wave7.sh"), str(root)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if (directory / "pid").exists():
            pid = int((directory / "pid").read_text())
            observed = process(pid)
            if observed is not None and observed[0] != "Z" and pid == child.pid:
                return pid
        if child.poll() is not None:
            raise RuntimeError("Wave-7 monitor exited before startup acknowledgment")
        time.sleep(0.1)
    raise TimeoutError("Wave-7 monitor has not acknowledged startup")


def launch(root: Path) -> dict:
    candidate, validation, frozen = require_release(root)
    preparation = root / "run-preparation"
    state_path = preparation / "wave7-launch-state.json"
    report_path = preparation / "wave7.json"
    if state_path.exists() or report_path.exists() or (root / "runs/wave7").exists():
        raise FileExistsError("Wave-7 already has launch evidence; reconcile before another attempt")
    for node in NODES:
        remote(root, frozen, node, "inspect")
    state = {"status": "transferring_allocation", "released_nodes": [], "candidate": str(candidate)}
    write(state_path, state)
    report = {"expected_launches": 4, "launches": []}
    try:
        for node in NODES:
            state["released_nodes"].append(node)
            write(state_path, state)
            remote(root, frozen, node, "release")
        command = [
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
            str(root / "runs/wave7"),
        ]
        state["status"] = "launching"
        code = bounded_execution(command, candidate, preparation / "launch-wave7.log", state, state_path, timeout=1200)
        load_launch_report(root, report)
        observe_nodes(root, frozen, NODES, state, report)
        if report["launches"]:
            state["monitor_pid"] = install_monitor(root, frozen)
        if (
            code
            or state["discovery_errors"]
            or state["watcher_errors"]
            or state["report_errors"]
            or len(report["launches"]) != len(ARMS)
            or any(
                entry.get("keeper_watcher_status") not in ("watching", "restoring", "restored")
                for entry in report["launches"]
            )
        ):
            raise RuntimeError("Wave-7 launch incomplete; preserve and reconcile")
        state["status"] = "launched"
        write(state_path, state)
        return state
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        reconcile(root, frozen, state, report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--phase", choices=("freeze", "validate", "launch", "inspect", "release", "restore", "discover"))
    parser.add_argument("--node-lock-timeout", type=float, default=0)
    args = parser.parse_args()
    if args.node_lock_timeout and (args.phase != "restore" or not 0 < args.node_lock_timeout <= 120):
        parser.error("A bounded node-lock timeout is permitted only for restore (at most 120 seconds)")
    if args.phase == "freeze":
        if args.candidate is None:
            parser.error("--phase freeze requires --candidate")
        result = freeze(args.root, args.candidate)
    elif args.phase in ("validate", "launch"):
        with defer_signals() as signals:
            with (args.root / "run-preparation/.wave7-controller.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = validate(args.root) if args.phase == "validate" else launch(args.root)
        if signals:
            result["deferred_signals"] = signals
    elif args.phase == "discover":
        result = discover(args.root)
    else:
        result = node_operation(args.root, args.phase, lock_timeout=args.node_lock_timeout)
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
