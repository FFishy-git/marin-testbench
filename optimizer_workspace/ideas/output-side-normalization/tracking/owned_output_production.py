# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and release Wave 10 only after the original owned-output validator passes."""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

import owned_output_run_history as run_history
from launch_wave5 import NODES, PYTHON, node_action, read, sha256
from owned_output_handoff import keeper_observation, observed, same_generation
from owned_output_run_history import ARMS, CHANGES, EXTRA_HELPERS, WAVE, paths
from owned_output_validation import VALIDATION_ENVIRONMENT
from prepare_upstream_outer import replace_once
from run_diagonal_diagnostic import PROCESS_MARKER, finish_child, tagged_processes
from watch_wave5 import write
from wave7_release import candidate_inventory, defer_signals, idle_training, local_node, node_guard, release_keeper

WORKER_MARKER = "OUTPUT_MARIN_WAVE10_WORKER_ID"
MASK = "0,1,2,3,4,5,6,7"
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
OWNER_IMPORT = "from .owned_output_validation import VALIDATION_ENVIRONMENT, validation_commands"
OWNER_DENIAL = (
    "    if args.launch_release is not None:\n"
    '        raise RuntimeError("owned output full-budget production release has not been issued")'
)
OWNER_ACCEPTANCE = (
    "    if args.launch_release is not None:\n" "        require_production_owner(args, config)\n" "        return"
)


def original(root: Path) -> dict:
    frozen = read(root / "run-preparation/owned-output-trace-binding.json")
    subprocess.run(
        [PYTHON, str(Path(frozen["helpers"]) / "run_owned_output_trace.py"), "--root", str(root), "--phase", "check"],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""},
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    return frozen


def production_owner_text(previous: Path) -> str:
    text = replace_once(
        (previous / "experiment/execution_owner.py").read_text(),
        OWNER_IMPORT,
        OWNER_IMPORT + "\nfrom .owned_production_owner import require_production_owner",
    )
    return replace_once(text, OWNER_DENIAL, OWNER_ACCEPTANCE)


def prepare(root: Path, candidate: Path) -> dict:
    frozen = original(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.resolve().is_relative_to(previous.resolve()):
        raise FileExistsError("Wave-10 preparation needs a distinct unused candidate")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    helpers = Path(__file__).parent
    shutil.copy2(helpers / "owned_output_production_owner.py", candidate / "experiment/owned_production_owner.py")
    shutil.copy2(
        helpers / "test_owned_output_production_owner.py", candidate / "experiment/tests/test_owned_output_release.py"
    )
    (candidate / "experiment/execution_owner.py").write_text(production_owner_text(previous))
    cluster_path = candidate / "experiment/cluster.json"
    cluster = read(cluster_path)
    cluster["wave_name"] = WAVE
    write(cluster_path, cluster)
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError(f"Wave-10 preparation exceeds ownership-only scope: {sorted(changed ^ CHANGES)}")
    result = {
        "status": "prepared_owner_only_no_production_release",
        "candidate": str(candidate.resolve()),
        "validation_binding_sha256": sha256(root / "run-preparation/owned-output-trace-binding.json"),
        "files_sha256": files,
        "changed_files": sorted(CHANGES),
        "owner_sha256": sha256(helpers / "owned_output_production_owner.py"),
        "test_sha256": sha256(helpers / "test_owned_output_production_owner.py"),
        "policy": (
            "Numerical/config/training bytes unchanged; production requires all fourteen GPU commands and restoration."
        ),
    }
    write(candidate / "production-preparation.json", result)
    return result


def require_preparation(root: Path, candidate: Path) -> dict:
    frozen = original(root)
    previous = Path(frozen["candidate"])
    record = read(candidate / "production-preparation.json")
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    cluster, previous_cluster = read(candidate / "experiment/cluster.json"), read(previous / "experiment/cluster.json")
    if (
        record["status"] != "prepared_owner_only_no_production_release"
        or record["candidate"] != str(candidate.resolve())
        or record["validation_binding_sha256"] != sha256(root / "run-preparation/owned-output-trace-binding.json")
        or record["files_sha256"] != files
        or set(record["changed_files"]) != CHANGES
        or changed != CHANGES
        or (candidate / "experiment/execution_owner.py").read_text() != production_owner_text(previous)
        or record["owner_sha256"] != sha256(candidate / "experiment/owned_production_owner.py")
        or record["test_sha256"] != sha256(candidate / "experiment/tests/test_owned_output_release.py")
        or cluster["wave_name"] != WAVE
        or {key: value for key, value in cluster.items() if key not in ("wave_name", "time")}
        != {key: value for key, value in previous_cluster.items() if key not in ("wave_name", "time")}
        or any(
            (candidate / f"experiment/configs/{arm}_0p5m_seed0.json").read_bytes()
            != (previous / f"experiment/configs/{arm}_0p5m_seed0.json").read_bytes()
            for arm in ARMS
        )
    ):
        raise ValueError("Wave-10 preparation or unchanged numerical/configuration scope differs")
    return record


def cpu_command() -> list[str]:
    return [
        PYTHON,
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "-p",
        "no:cacheprovider",
        "-q",
        "-m",
        SAFE_MARKERS,
        "experiment/tests",
    ]


def require_archive(path: Path, files: dict, preparation_sha256: str) -> None:
    expected = files | {"production-preparation.json": preparation_sha256}
    with tarfile.open(path) as bundle:
        members = bundle.getmembers()
        regular = [member for member in members if member.isfile()]
        aliases = [member for member in members if not member.isfile()]
        if (
            len(regular) != len(expected)
            or {member.name for member in regular} != set(expected)
            or len(aliases) != 1
            or aliases[0].name != "artifacts/external-sources"
            or not aliases[0].issym()
            or aliases[0].linkname != "pinned-sources-materialized"
        ):
            raise ValueError("Wave-10 CPU source archive inventory or pinned alias differs")
        for member in regular:
            with bundle.extractfile(member) as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected[member.name]:
                    raise ValueError("Wave-10 archived bytes differ from CPU-tested source")


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    preparation = require_preparation(root, candidate)
    if out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Wave-10 CPU evidence needs an unused external directory")
    out.mkdir(parents=True, exist_ok=False)
    log, archive = out / "candidate.log", out / "source.tar"
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(),
                cwd=candidate,
                env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "CPU command exceeded 600 seconds; exit 124 is the timeout sentinel"
    preparation_digest = sha256(candidate / "production-preparation.json")
    with tarfile.open(archive, "x") as bundle:
        for name in preparation["files_sha256"]:
            bundle.add(candidate / name, arcname=name, recursive=False)
        bundle.add(candidate / "production-preparation.json", arcname="production-preparation.json", recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    try:
        if require_preparation(root, candidate) != preparation:
            raise ValueError("Wave-10 source changed during CPU verification")
        require_archive(archive, preparation["files_sha256"], preparation_digest)
    except ValueError as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": (
            "production_cpu_verified_gpu_release_pending"
            if type(code) is int and code == 0 and error is None
            else "production_cpu_failed"
        ),
        "candidate": str(candidate.resolve()),
        "preparation_sha256": preparation_digest,
        "files_sha256": preparation["files_sha256"],
        "command": cpu_command(),
        "cwd": str(candidate.resolve()),
        "exit_code": code,
        "log": str(log),
        "log_sha256": sha256(log),
        "cuda_visible_devices": "",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != "production_cpu_verified_gpu_release_pending":
        raise RuntimeError(f"Wave-10 CPU failure preserved at {out}")
    return result


def require_cpu(root: Path, path: Path) -> dict:
    record = read(path)
    candidate = Path(record["candidate"])
    preparation = require_preparation(root, candidate)
    if (
        record["status"] != "production_cpu_verified_gpu_release_pending"
        or record["preparation_sha256"] != sha256(candidate / "production-preparation.json")
        or record["files_sha256"] != preparation["files_sha256"]
        or record["command"] != cpu_command()
        or record["cwd"] != str(candidate.resolve())
        or type(record["exit_code"]) is not int
        or record["exit_code"] != 0
        or record["cuda_visible_devices"] != ""
        or "error" in record
        or record["log_sha256"] != sha256(Path(record["log"]))
        or record["archive_sha256"] != sha256(Path(record["archive"]))
    ):
        raise ValueError("Wave-10 CPU command, source, archive or outcome differs")
    require_archive(Path(record["archive"]), record["files_sha256"], record["preparation_sha256"])
    return record


def require_completed_validation(root: Path, *, accepted: dict | None = None) -> dict:
    """Rederive all original ranks/checkpoints and inspect the restored validator allocation."""
    frozen = original(root)
    state = read(root / "run-preparation/owned-output-trace-state.json")
    if state["status"] != "passed" or any(
        type(state.get(key)) is not int or state[key] != 0
        for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
    ):
        raise RuntimeError("Original full-trace validator has not passed, cleaned up and restored")
    script = """
import json,sys
from pathlib import Path
from run_owned_output_trace import binding,require_validation,CONTROLLER_MARKER
from wave7_release import node_guard
import owned_output_handoff as h
root=Path(sys.argv[1]); frozen=binding(root); previous=json.loads(sys.argv[2])
binding_path=root/"run-preparation/owned-output-trace-binding.json"
state_path=root/"run-preparation/owned-output-trace-state.json"
receipt_path=root/"run-preparation/validation-owned-output-trace-gpu.json"
intent_path=Path(frozen["helpers"]).parent/"controller.json"
def identities():
    return {key:h.sha256(path) for key,path in
        (("binding_sha256",binding_path),("state_sha256",state_path),
         ("receipt_sha256",receipt_path),("intent_sha256",intent_path))}
def probe():
    state=h.read(state_path); intent=h.read(intent_path)
    expected=[sys.executable,str(Path(frozen["helpers"])/"run_owned_output_trace.py"),
        "--root",str(root),"--phase","validate"]
    generation=intent.get("generation",{})
    if (state["status"]!="passed" or "error" in state or "cleanup_error" in state
        or any(type(state.get(k)) is not int or state[k]!=0
            for k in ("exit_code","cleanup_exit_code","keeper_restore"))
        or intent.get("status")!="acknowledged" or intent.get("command")!=expected
        or intent.get("binding_sha256")!=h.sha256(binding_path)
        or state.get("binding_sha256")!=intent["binding_sha256"]
        or intent.get("pid")!=state["controller_pid"]
        or intent.get("start_ticks")!=state["controller_start_ticks"]
        or generation.get("pid")!=intent["pid"] or generation.get("start_ticks")!=intent["start_ticks"]
        or generation.get("command")!=expected or generation.get("status") in ("Z","X","x")
        or intent.get("candidate")!=frozen["candidate"]
        or intent.get("cuda_visible_devices")!="0,1,2,3,4,5,6,7"
        or not intent.get("marker") or not state.get("child_marker")
        or state["validation_sha256"]!=h.sha256(receipt_path)):
        raise RuntimeError("Original full-trace validator acknowledgment or terminal evidence differs")
    for pid in (state["controller_pid"],state["child_pid"]):
        process=h.observed(pid)
        if process is not None and process["status"] not in ("Z","X","x"):
            raise RuntimeError("Original validator handle remains live")
    if h.marker_processes(h.CHILD_MARKER,state["child_marker"]):
        raise RuntimeError("Original diagnostic descendants remain live")
    keeper=h.keeper_observation()
    marked=h.marker_processes(CONTROLLER_MARKER,intent["marker"])
    if any(pid not in keeper["generation"] or not h.same_generation(keeper["generation"][pid],process)
        for pid,process in marked.items()):
        raise RuntimeError("Inherited controller marker exists outside the exact restored keeper")
    return state,keeper
with node_guard(root,frozen["node"]):
    accepted=identities(); state,keeper=probe()
    if previous is not None:
        if any(accepted[key]!=previous[key] for key in accepted) or set(keeper["generation"])!=set(
            map(int,previous["keeper"]["generation"])) or any(
            not h.same_generation(process,previous["keeper"]["generation"][str(pid)])
            for pid,process in keeper["generation"].items()):
            raise RuntimeError("Accepted validation allocation or source changed before publication")
        receipt=h.read(receipt_path)
    else:
        receipt=require_validation(root,frozen)
    binding(root)
    final_state,final_keeper=probe()
    if accepted!=identities() or state!=final_state or set(keeper["generation"])!=set(final_keeper["generation"]) or any(
        not h.same_generation(process,final_keeper["generation"][pid])
        for pid,process in keeper["generation"].items()):
        raise RuntimeError("Original validation or restored keeper changed during reconstruction")
print(json.dumps({"status":"owned_validation_rederived_absent_exact_keeper_verified",
    **accepted,
    "controller_pid":state["controller_pid"],"controller_start_ticks":state["controller_start_ticks"],
    "commands":len(receipt["evidence"]["gpu_commands"]),"checks":len(frozen["required_gates"]),
    "keeper":final_keeper},allow_nan=False))
"""
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "root@28.216.17.139",
            shlex.join(
                [
                    "env",
                    "CUDA_VISIBLE_DEVICES=",
                    "PYTHONPATH=" + frozen["helpers"],
                    PYTHON,
                    "-c",
                    script,
                    str(root),
                    json.dumps(accepted),
                ]
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=2400 if accepted is None else 300,
    )
    evidence = json.loads(result.stdout.splitlines()[-1])
    if evidence["status"] != "owned_validation_rederived_absent_exact_keeper_verified" or evidence["commands"] != 14:
        raise ValueError("Original owned-output validation reconstruction is incomplete")
    return evidence


def require_accepted_validation(root: Path, evidence: dict) -> None:
    """Reject changes to the independently accepted original evidence before publication."""
    files = {
        "binding_sha256": root / "run-preparation/owned-output-trace-binding.json",
        "state_sha256": root / "run-preparation/owned-output-trace-state.json",
        "receipt_sha256": root / "run-preparation/validation-owned-output-trace-gpu.json",
        "intent_sha256": root / "controllers/owned-output-trace-validation-20261117/controller.json",
    }
    if any(sha256(path) != evidence[key] for key, path in files.items()):
        raise ValueError("Accepted original validation evidence changed before production publication")


def freeze(root: Path, receipt_path: Path) -> dict:
    receipt = require_cpu(root, receipt_path)
    numerical = original(root)
    names = (*numerical["helpers_sha256"], *EXTRA_HELPERS)
    if len(names) != len(set(names)):
        raise ValueError("Wave-10 helper inventory contains duplicate identities")
    sources = {
        name: Path(numerical["helpers"]) / name if name in numerical["helpers_sha256"] else Path(__file__).parent / name
        for name in names
    }
    hashes = {name: sha256(source) for name, source in sources.items()}
    prepared = read(Path(receipt["candidate"]) / "production-preparation.json")
    if (
        hashes["owned_output_production_owner.py"] != prepared["owner_sha256"]
        or hashes["test_owned_output_production_owner.py"] != prepared["test_sha256"]
    ):
        raise ValueError("Wave-10 owner helpers differ from the tested production candidate")
    evidence = require_completed_validation(root)
    require_accepted_validation(root, evidence)
    if (
        require_cpu(root, receipt_path) != receipt
        or {name: sha256(source) for name, source in sources.items()} != hashes
    ):
        raise ValueError("Wave-10 source or helpers changed during validation reconstruction")
    release_path, helpers = paths(root)
    archive = root / "run-preparation/source-wave10-launch-20261117.tar"
    if any(path.exists() for path in (release_path, helpers, archive)):
        raise FileExistsError("Wave-10 release already has evidence; preserve without reissuing")
    helpers.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "x") as bundle:
        for name in names:
            shutil.copy2(sources[name], helpers / name)
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        for name in receipt["files_sha256"]:
            bundle.add(Path(receipt["candidate"]) / name, arcname=name, recursive=False)
        bundle.add(
            Path(receipt["candidate"]) / "production-preparation.json", arcname="preparation.json", recursive=False
        )
        bundle.add(
            Path(receipt["candidate"]) / "artifacts/external-sources",
            arcname="artifacts/external-sources",
            recursive=False,
        )
    if (
        require_cpu(root, receipt_path) != receipt
        or hashes != {name: sha256(helpers / name) for name in names}
        or hashes != {name: sha256(source) for name, source in sources.items()}
    ):
        raise ValueError("Wave-10 archive/source changed; preserve partial freeze without launch")
    evidence = require_completed_validation(root, accepted=evidence)
    require_accepted_validation(root, evidence)
    release = {
        "status": "frozen_validated_launch_release",
        "root": str(root),
        "candidate": receipt["candidate"],
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "validation_binding_sha256": evidence["binding_sha256"],
        "validation_sha256": evidence["receipt_sha256"],
        "state_sha256": evidence["state_sha256"],
        "validation_reconstruction": evidence,
        "preparation": str(Path(receipt["candidate"]) / "production-preparation.json"),
        "preparation_sha256": sha256(Path(receipt["candidate"]) / "production-preparation.json"),
        "cpu_receipt": str(receipt_path),
        "cpu_receipt_sha256": sha256(receipt_path),
        "files_sha256": receipt["files_sha256"],
        "source_sha256": {
            name: digest
            for name, digest in receipt["files_sha256"].items()
            if name.startswith("experiment/") and name.endswith(".py")
        },
        "data_receipt_sha256": numerical["data_receipt_sha256"],
        "predecessor_release_sha256": sha256(root / "run-preparation/wave9-launch-binding.json"),
        "environment": VALIDATION_ENVIRONMENT,
        "arms": ARMS,
        "nodes": NODES,
        "commands": run_history.commands(root, Path(receipt["candidate"])),
        "monitor_interval_seconds": 900,
        "training_timeout_seconds": 21600,
        "policy": (
            "Ownership-only migration; original base LRs/full budget; all owned GPU gates rederived; no resume/retry."
        ),
    }
    require_accepted_validation(root, evidence)
    write(release_path, release)
    return release


def binding(root: Path, *, observation: bool = False) -> dict:
    release = run_history.helper_binding(root) if observation else run_history.binding(root)
    if Path(__file__).resolve() != (Path(release["helpers"]) / "owned_output_production.py").resolve():
        raise ValueError("Wave 10 requires its exact frozen controller")
    if not observation:
        require_cpu(root, Path(release["cpu_receipt"]))
    return release


def node_for(arm: str) -> str:
    return NODES[ARMS.index(arm)]


def worker_path(root: Path, arm: str) -> Path:
    return root / "run-preparation" / f"wave10-worker-{node_for(arm)}.json"


def worker_command(root: Path, release: dict, arm: str) -> list[str]:
    return [
        PYTHON,
        str(Path(release["helpers"]) / "owned_output_production.py"),
        "--root",
        str(root),
        "--phase",
        "worker",
        "--arm",
        arm,
    ]


def allocation(root: Path, release: dict, arm: str) -> dict:
    if local_node() != node_for(arm) or os.environ.get("CUDA_VISIBLE_DEVICES") != MASK:
        raise RuntimeError("Wave-10 allocation requires its original node and explicit eight-device mask")
    idle_training()
    subprocess.run(
        [PYTHON, "-m", "experiment.bootstrap", "--check-only"],
        cwd=release["candidate"],
        env=os.environ | VALIDATION_ENVIRONMENT,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return keeper_observation()


def run_child(root: Path, release: dict, state: dict, path: Path) -> int:
    command = release["commands"][state["node"]]
    marker, generation = uuid4().hex, {}
    log = root / f"runs/{WAVE}" / f"{state['arm']}.torchrun.log"
    state.update(command=command, child_marker=marker, log=str(log))
    write(path, state)
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=release["candidate"],
            env=os.environ | VALIDATION_ENVIRONMENT | {PROCESS_MARKER: marker},
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            own = observed(child.pid)
            if own is None or own["status"] == "Z" or own["command"] != command:
                raise RuntimeError("Wave-10 torchrun lacks its matching generation")
            state.update(status="running", child_pid=child.pid, child_start_ticks=own["start_ticks"])
            write(path, state)
            deadline = time.monotonic() + release["training_timeout_seconds"]
            while True:
                generation.update(tagged_processes(marker))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, release["training_timeout_seconds"])
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
                state.update(child_generation=generation, log_sha256=sha256(log))
                write(path, state)


def run_arm(root: Path, arm: str) -> dict:
    release = binding(root)
    path = worker_path(root, arm)
    intent_path = path.with_suffix(".start.json")
    intent = read(intent_path)
    deadline = time.monotonic() + 30
    while ("generation" not in intent) and time.monotonic() < deadline:
        time.sleep(0.05)
        intent = read(intent_path)
    launch = read(root / "run-preparation/wave10-launch-state.json")
    own = observed(os.getpid())
    expected = intent.get("generation")
    if (
        own is None
        or own["status"] in ("Z", "X", "x")
        or expected is None
        or any(own[key] != expected[key] for key in ("pid", "start_ticks", "group", "session", "command"))
        or own["command"] != worker_command(root, release, arm)
        or intent["release_binding_sha256"] != sha256(paths(root)[0])
        or intent["worker_marker"] != launch["worker_markers"][arm]
        or os.environ.get(WORKER_MARKER) != intent["worker_marker"]
        or launch["binding_sha256"] != sha256(paths(root)[0])
        or launch["status"] not in ("launching", "launched", "needs_attention")
        or path.exists()
    ):
        raise RuntimeError("Wave-10 worker lacks its original one-shot generation acknowledgment")
    with node_guard(root, node_for(arm)) as signals:
        keeper = allocation(root, release, arm)
        state = {
            "status": "starting",
            "arm": arm,
            "node": node_for(arm),
            "worker_pid": os.getpid(),
            "worker_start_ticks": own["start_ticks"],
            "worker_command": own["command"],
            "worker_marker": intent["worker_marker"],
            "release_binding_sha256": sha256(paths(root)[0]),
            "out": str(root / f"runs/{WAVE}" / arm),
        }
        write(path, state)
        try:
            if Path(state["out"]).exists():
                raise FileExistsError("Preserve previous Wave-10 output without retry")
            current = keeper_observation()
            if set(current["generation"]) != set(keeper["generation"]) or any(
                not same_generation(value, current["generation"][pid]) for pid, value in keeper["generation"].items()
            ):
                raise RuntimeError("Wave-10 keeper changed after allocation preflight")
            state["keeper"] = release_keeper(state, path)
            code = run_child(root, release, state, path)
            state["exit_code"] = code
            if type(code) is not int or code != 0:
                raise RuntimeError("Wave-10 full-budget command failed")
            state["completed_run"] = run_history.completed_run(root, arm, release)
            state["status"] = "complete"
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if state.get("keeper_release_attempted"):
                try:
                    if "cleanup_error" in state or tagged_processes(state.get("child_marker", "")):
                        raise RuntimeError("Unverified training descendants prevent keeper restoration")
                    node_action(root, "restore")
                    state["restored_keeper"] = keeper_observation()
                    state["keeper_restore"] = 0
                except Exception as error:
                    state.update(status="needs_attention", keeper_restore=f"refused: {error}")
            if signals:
                state["deferred_signals"] = signals
            write(path, state)
    return state


def start_arm(root: Path, arm: str) -> dict:
    release = binding(root)
    if local_node() != node_for(arm):
        raise RuntimeError("Wave-10 worker startup requires its original node")
    path = worker_path(root, arm)
    intent_path = path.with_suffix(".start.json")
    with path.with_suffix(".start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() or intent_path.exists():
            raise FileExistsError("Wave-10 worker has intent or evidence; no retry")
        launch = read(root / "run-preparation/wave10-launch-state.json")
        if launch["status"] != "launching" or launch["binding_sha256"] != sha256(paths(root)[0]):
            raise RuntimeError("Wave-10 startup differs from its exact release")
        command = worker_command(root, release, arm)
        intent = {
            "status": "starting",
            "arm": arm,
            "node": node_for(arm),
            "command": command,
            "release_binding_sha256": sha256(paths(root)[0]),
            "worker_marker": launch["worker_markers"][arm],
            "starter": observed(os.getpid()),
        }
        write(intent_path, intent)
        try:
            with path.with_suffix(".log").open("x") as stream:
                child = subprocess.Popen(
                    command,
                    cwd=release["candidate"],
                    env=os.environ | VALIDATION_ENVIRONMENT | {WORKER_MARKER: intent["worker_marker"]},
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            own = observed(child.pid)
            if own is None or own["status"] == "Z" or own["command"] != command:
                raise RuntimeError("Wave-10 worker exited before generation acknowledgment")
            intent.update(pid=child.pid, start_ticks=own["start_ticks"], generation=own, status="acknowledged")
            write(intent_path, intent)
        except BaseException as error:
            intent.update(status="unacknowledged", error=f"{type(error).__name__}: {error}")
            write(intent_path, intent)
            raise
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if path.exists():
                state = read(path)
                if state["worker_pid"] != child.pid or state["worker_start_ticks"] != own["start_ticks"]:
                    raise RuntimeError("Wave-10 worker acknowledgment changed generation")
                return state
            if child.poll() is not None:
                raise RuntimeError("Wave-10 worker exited before state acknowledgment; no retry")
            time.sleep(0.1)
        raise TimeoutError("Wave-10 worker startup unknown; preserve the original process")


def remote(root: Path, release: dict, arm: str, phase: str) -> dict:
    command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={MASK}",
        PYTHON,
        str(Path(release["helpers"]) / "owned_output_production.py"),
        "--root",
        str(root),
        "--phase",
        phase,
        "--arm",
        arm,
    ]
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"root@{node_for(arm)}", shlex.join(command)],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return json.loads(result.stdout.splitlines()[-1])


def probe_arm(root: Path, arm: str) -> dict:
    release = binding(root, observation=True)
    if local_node() != node_for(arm):
        raise RuntimeError("Wave-10 observation requires its original node")
    path = worker_path(root, arm)
    if not path.exists():
        intent_path = path.with_suffix(".start.json")
        if not intent_path.exists():
            return {"arm": arm, "status": "not_launched"}
        intent = read(intent_path)
        starter = intent.get("starter")
        if starter and same_generation(starter, observed(starter["pid"])):
            return {"arm": arm, "status": "unknown", "receipt": intent}
        child = observed(intent["pid"]) if intent.get("pid") is not None else None
        if child is not None and child["status"] != "Z":
            return {"arm": arm, "status": "unknown", "receipt": intent}
        token = f"{WORKER_MARKER}={intent['worker_marker']}".encode()
        for directory in Path("/proc").iterdir():
            if not directory.name.isdigit():
                continue
            try:
                if token in (directory / "environ").read_bytes().split(b"\0"):
                    process = observed(int(directory.name))
                    if process is not None and process["status"] != "Z":
                        return {"arm": arm, "status": "live_orphaned_training", "receipt": intent}
            except (FileNotFoundError, ProcessLookupError):
                continue
        return {"arm": arm, "status": "needs_attention", "receipt": intent, "startup_absence_verified": True}
    state = read(path)
    if (
        state["arm"] != arm
        or state["node"] != node_for(arm)
        or state["release_binding_sha256"] != sha256(paths(root)[0])
    ):
        raise ValueError("Wave-10 saved worker identity differs")
    own = observed(state["worker_pid"])
    children = tagged_processes(state["child_marker"]) if state.get("child_marker") else {}
    if own is not None and own["status"] != "Z":
        if own["start_ticks"] != state["worker_start_ticks"] or own["command"] != worker_command(root, release, arm):
            return {"arm": arm, "status": "worker_generation_changed", "receipt": state, "descendants": children}
        return {"arm": arm, "status": "running", "receipt": state}
    if children:
        return {"arm": arm, "status": "live_orphaned_training", "receipt": state, "descendants": children}
    complete = (
        state["status"] == "complete"
        and "cleanup_error" not in state
        and "error" not in state
        and all(
            type(state.get(key)) is int and state[key] == 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
    )
    if complete and (
        state["completed_run"] != run_history.completed_run(root, arm, release)
        or state["log_sha256"] != sha256(Path(state["log"]))
    ):
        raise ValueError("Wave-10 completed worker history or log changed")
    return {"arm": arm, "status": "complete" if complete else "needs_attention", "receipt": state}


def snapshot(root: Path) -> dict:
    release = binding(root, observation=True)
    workers, runs, launches = [], [], []
    launch = read(root / "run-preparation/wave10-launch-state.json")
    for arm in ARMS:
        try:
            worker = remote(root, release, arm, "probe")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            worker = {"arm": arm, "status": "unknown", "error": str(error)}
        if worker["status"] == "not_launched" and launch["status"] == "needs_attention":
            worker["status"] = "not_launched_after_launch_failure"
        workers.append(worker)
        directory = root / f"runs/{WAVE}" / arm
        run = {
            "name": arm,
            "arm": arm,
            "node": node_for(arm),
            "status": worker["status"],
            "planned_steps": 5120,
            "observed_tokens": None,
            "latest_metrics": None,
        }
        if (directory / "run.json").exists():
            try:
                measured = run_history.measured_run(root, arm, release)
                run.update(
                    observed_tokens=measured["observed_tokens"], latest_metrics={"step": measured["observed_updates"]}
                )
            except (ValueError, OSError, KeyError) as error:
                run["accounting_error"] = str(error)
        elif worker["status"] in ("not_launched", "not_launched_after_launch_failure"):
            run["observed_tokens"] = 0
        state = worker.get("receipt", {})
        if state.get("child_pid"):
            launches.append(
                {
                    "arm": arm,
                    "node": node_for(arm),
                    "seed": 0,
                    "batch": "0p5m",
                    "out": str(directory),
                    "pid": state["child_pid"],
                    "start_ticks": state["child_start_ticks"],
                    "command": state["command"],
                }
            )
        runs.append(run)
    statuses = {worker["status"] for worker in workers}
    terminal = statuses <= {"complete", "needs_attention", "not_launched_after_launch_failure"}
    write(root / "run-preparation/wave10.json", {"expected_launches": 4, "launches": launches})
    return {
        "status": "complete" if statuses == {"complete"} else "needs_attention" if terminal else "active",
        "binding_sha256": sha256(paths(root)[0]),
        "workers": workers,
        "runs": runs,
        "all_workers_terminal": terminal,
    }


def launch(root: Path) -> dict:
    release = binding(root)
    require_completed_validation(root)
    path = root / "run-preparation/wave10-launch-state.json"
    directory = root / "run-preparation/monitor-wave10"
    if any(item.exists() for item in (path, directory, root / f"runs/{WAVE}")):
        raise FileExistsError("Wave-10 launch already has evidence; no retry")
    for arm in ARMS:
        if worker_path(root, arm).exists() or worker_path(root, arm).with_suffix(".start.json").exists():
            raise FileExistsError("Wave-10 worker already has launch evidence")
        remote(root, release, arm, "inspect")
    (root / f"runs/{WAVE}").mkdir(parents=True)
    directory.mkdir()
    state = {
        "status": "launching",
        "binding_sha256": sha256(paths(root)[0]),
        "workers": {},
        "worker_markers": {arm: uuid4().hex for arm in ARMS},
    }
    write(path, state)
    try:
        command = ["bash", str(Path(release["helpers"]) / "monitor_owned_output_run.sh"), str(root)]
        with (directory / "monitor.log").open("x") as log:
            monitor = subprocess.Popen(
                command,
                env=os.environ | VALIDATION_ENVIRONMENT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (directory / "pid").exists():
                own = observed(monitor.pid)
                if (
                    int((directory / "pid").read_text()) != monitor.pid
                    or own is None
                    or own["status"] == "Z"
                    or own["command"] != command
                ):
                    raise RuntimeError("Wave-10 monitor generation acknowledgment differs")
                state.update(monitor_pid=monitor.pid, monitor_start_ticks=own["start_ticks"])
                write(path, state)
                break
            if monitor.poll() is not None:
                raise RuntimeError("Wave-10 monitor exited before acknowledgment")
            time.sleep(0.1)
        else:
            raise TimeoutError("Wave-10 monitor startup unknown; preserve original process")
        for arm in ARMS:
            ack = remote(root, release, arm, "start")
            state["workers"][arm] = ack
            write(path, state)
            if (
                ack["arm"] != arm
                or ack["node"] != node_for(arm)
                or ack["worker_command"] != worker_command(root, release, arm)
                or ack["worker_marker"] != state["worker_markers"][arm]
                or ack["release_binding_sha256"] != state["binding_sha256"]
                or ack["status"] not in ("starting", "running", "complete")
            ):
                raise RuntimeError("Wave-10 worker acknowledgment differs; preserve partial launch")
        state["status"] = "launched"
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("prepare", "cpu", "freeze", "check", "launch", "inspect", "start", "worker", "probe", "snapshot"),
        required=True,
    )
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.phase in ("prepare", "cpu") and args.candidate is None:
        parser.error("Preparation/CPU requires --candidate")
    if args.phase in ("cpu", "snapshot") and args.out is None:
        parser.error("CPU/snapshot requires --out")
    if args.phase == "freeze" and args.cpu_receipt is None:
        parser.error("Freeze requires --cpu-receipt")
    if args.phase in ("inspect", "start", "worker", "probe") and args.arm is None:
        parser.error("Node operations require --arm")
    root = args.root.resolve()
    if args.phase == "prepare":
        result = prepare(root, args.candidate)
    elif args.phase == "cpu":
        result = cpu(root, args.candidate, args.out)
    elif args.phase == "freeze":
        with (root / "run-preparation/.wave10-freeze.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = freeze(root, args.cpu_receipt)
    elif args.phase == "check":
        result = binding(root)
    elif args.phase == "launch":
        with defer_signals():
            with (root / "run-preparation/.wave10-launch-controller.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = launch(root)
    elif args.phase == "inspect":
        release = binding(root)
        with node_guard(root, node_for(args.arm)):
            allocation(root, release, args.arm)
        result = {"status": "allocation_preflight_passed", "arm": args.arm, "node": node_for(args.arm)}
    elif args.phase == "start":
        result = start_arm(root, args.arm)
    elif args.phase == "worker":
        result = run_arm(root, args.arm)
    elif args.phase == "probe":
        result = probe_arm(root, args.arm)
    else:
        result = snapshot(root)
        write(args.out, result)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
