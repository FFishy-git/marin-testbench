# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and launch an ownership-only production successor of the corrected upstream validator."""

import argparse
import csv
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from uuid import uuid4

import upstream_run_history
from launch_wave5 import NODES, PYTHON, node_action, read, sha256
from prepare_upstream_outer import replace_once
from run_diagonal_diagnostic import PROCESS_MARKER, finish_child, keeper_generation, tagged_processes
from validate_wave5 import ENVIRONMENT
from watch_wave5 import write
from watch_wave7_release import observed_process
from wave7_release import candidate_inventory, defer_signals, idle_training, local_node, node_guard, release_keeper

ARMS = ("muont", "outer_leftdiag_fused", "outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused")
WORKER_MARKER = "OUTPUT_MARIN_WAVE9_WORKER_ID"
CHANGES = {
    "experiment/execution_owner.py",
    "experiment/production_owner.py",
    "experiment/cluster.json",
    "experiment/tests/test_upstream_release.py",
}
EXTRA_HELPERS = (
    "upstream_production.py",
    "upstream_production_owner.py",
    "test_upstream_production_owner.py",
    "test_upstream_production.py",
    "upstream_run_history.py",
    "analyze_upstream_run.py",
    "swanlab_upstream_run.py",
    "monitor_upstream_run.sh",
    "time_to_target.py",
)


def paths(root: Path) -> tuple[Path, Path]:
    return root / "run-preparation/wave9-launch-binding.json", root / "controllers/wave9-launch-20261117/tracking"


def original(root: Path) -> dict:
    path = root / "run-preparation/upstream-full-model-corrected-binding.json"
    frozen = read(path)
    subprocess.run(
        [PYTHON, str(Path(frozen["helpers"]) / "run_upstream_full_model.py"), "--root", str(root), "--phase", "check"],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    return frozen


def prepare(root: Path, candidate: Path) -> dict:
    frozen = original(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.resolve().is_relative_to(previous.resolve()):
        raise FileExistsError("Production preparation requires a distinct unused candidate")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    helpers = Path(__file__).parent
    shutil.copy2(helpers / "upstream_production_owner.py", candidate / "experiment/production_owner.py")
    shutil.copy2(helpers / "test_upstream_production_owner.py", candidate / "experiment/tests/test_upstream_release.py")
    path = candidate / "experiment/execution_owner.py"
    text = replace_once(
        path.read_text(),
        "from .reference import TASK",
        "from .reference import TASK\nfrom .production_owner import require_production_owner",
    )
    path.write_text(
        replace_once(
            text,
            "    if args.launch_release is not None:\n"
            '        raise RuntimeError("Upstream full-budget production release has not been issued")',
            "    if args.launch_release is not None:\n"
            "        require_production_owner(args, config)\n"
            "        return",
        )
    )
    cluster_path = candidate / "experiment/cluster.json"
    cluster = read(cluster_path)
    cluster["wave_name"] = "wave9"
    write(cluster_path, cluster)
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError(f"Production source migration exceeds ownership-only scope: {sorted(changed ^ CHANGES)}")
    result = {
        "status": "prepared_owner_only_no_production_release",
        "candidate": str(candidate.resolve()),
        "validation_binding_sha256": sha256(root / "run-preparation/upstream-full-model-corrected-binding.json"),
        "files_sha256": files,
        "changed_files": sorted(CHANGES),
        "owner_sha256": sha256(helpers / "upstream_production_owner.py"),
        "test_sha256": sha256(helpers / "test_upstream_production_owner.py"),
        "policy": (
            "Original numerical files/configs unchanged; full-budget release waits for all GPU gates and restoration."
        ),
    }
    write(candidate / "production-preparation.json", result)
    return result


def require_preparation(root: Path, candidate: Path) -> dict:
    frozen = original(root)
    preparation = read(candidate / "production-preparation.json")
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    previous = Path(frozen["candidate"])
    owner = replace_once(
        (previous / "experiment/execution_owner.py").read_text(),
        "from .reference import TASK",
        "from .reference import TASK\nfrom .production_owner import require_production_owner",
    )
    owner = replace_once(
        owner,
        "    if args.launch_release is not None:\n"
        '        raise RuntimeError("Upstream full-budget production release has not been issued")',
        "    if args.launch_release is not None:\n" "        require_production_owner(args, config)\n" "        return",
    )
    cluster, previous_cluster = read(candidate / "experiment/cluster.json"), read(previous / "experiment/cluster.json")
    if (
        preparation["status"] != "prepared_owner_only_no_production_release"
        or preparation["candidate"] != str(candidate.resolve())
        or preparation["validation_binding_sha256"]
        != sha256(root / "run-preparation/upstream-full-model-corrected-binding.json")
        or preparation["files_sha256"] != files
        or changed != CHANGES
        or (candidate / "experiment/execution_owner.py").read_text() != owner
        or sha256(candidate / "experiment/production_owner.py") != preparation["owner_sha256"]
        or sha256(candidate / "experiment/tests/test_upstream_release.py") != preparation["test_sha256"]
        or read(candidate / "experiment/cluster.json")["wave_name"] != "wave9"
        or {key: value for key, value in cluster.items() if key not in ("wave_name", "time")}
        != {key: value for key, value in previous_cluster.items() if key not in ("wave_name", "time")}
        or any(
            read(candidate / f"experiment/configs/{arm}_0p5m_seed0.json")
            != read(Path(frozen["candidate"]) / f"experiment/configs/{arm}_0p5m_seed0.json")
            for arm in ARMS
        )
    ):
        raise ValueError("Production preparation, unchanged numerical scope or configurations differ")
    return preparation


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
        "not slow and not integration and not data_integration and not cluster "
        "and not requires_cluster and not docker and not manual",
        "experiment/tests",
    ]


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    preparation = require_preparation(root, candidate)
    if out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Production CPU evidence requires an unused separate directory")
    out.mkdir(parents=True)
    log = out / "candidate.log"
    with log.open("x") as stream:
        code = subprocess.run(
            cpu_command(),
            cwd=candidate,
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=600,
        ).returncode
    if code:
        raise RuntimeError(f"Production CPU tests failed; preserve {log}")
    if require_preparation(root, candidate) != preparation:
        raise ValueError("Production source changed during CPU verification")
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in preparation["files_sha256"]:
            bundle.add(candidate / name, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    receipt = {
        "status": "production_cpu_verified_gpu_release_pending",
        "candidate": str(candidate.resolve()),
        "preparation_sha256": sha256(candidate / "production-preparation.json"),
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
    write(out / "receipt.json", receipt)
    return receipt


def require_cpu(root: Path, path: Path) -> dict:
    receipt = read(path)
    candidate = Path(receipt["candidate"])
    preparation = require_preparation(root, candidate)
    if (
        receipt["status"] != "production_cpu_verified_gpu_release_pending"
        or receipt["preparation_sha256"] != sha256(candidate / "production-preparation.json")
        or receipt["files_sha256"] != preparation["files_sha256"]
        or receipt["command"] != cpu_command()
        or receipt["cwd"] != str(candidate.resolve())
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["cuda_visible_devices"] != ""
        or sha256(Path(receipt["log"])) != receipt["log_sha256"]
        or sha256(Path(receipt["archive"])) != receipt["archive_sha256"]
    ):
        raise ValueError("Production CPU command, source or log evidence differs")
    return receipt


def require_completed_validation(root: Path) -> dict:
    """Use the exact original frozen queue to reconstruct all gates and its healthy restored keeper."""
    directory = root / "run-preparation/upstream-full-model-corrected-queue"
    script = """
import json,sys
from pathlib import Path
root=Path(sys.argv[2]); directory=root/"run-preparation/upstream-full-model-corrected-queue"
sys.path.insert(0,sys.argv[1])
from queue_upstream_full_model import binding,completed,observe,QUEUE_MARKER,CONTROLLER_MARKER
frozen=binding(root); installed=json.loads((directory/"installation.json").read_text())
state=json.loads((directory/"state.json").read_text()); intent=json.loads((directory/"controller.json").read_text())
if (state["status"]!="validated" or not (directory/"stopped_at").exists()
    or observe(installed,QUEUE_MARKER)!="exited" or observe(intent,CONTROLLER_MARKER)!="exited"):
    raise RuntimeError("Original full-model queue/controller has not completed")
print(json.dumps(completed(root,frozen,intent)))
"""
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "root@28.216.17.139",
            shlex.join([PYTHON, "-c", script, str(directory / "helpers"), str(root)]),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    return json.loads(result.stdout)


def freeze(root: Path, receipt_path: Path) -> dict:
    receipt = require_cpu(root, receipt_path)
    numerical = original(root)
    names = (*numerical["helpers_sha256"], *EXTRA_HELPERS)
    if len(names) != len(set(names)):
        raise ValueError("Wave-9 helper inventory has duplicate names")
    sources = {
        name: Path(numerical["helpers"]) / name if name in numerical["helpers_sha256"] else Path(__file__).parent / name
        for name in names
    }
    hashes = {name: sha256(source) for name, source in sources.items()}
    preparation = Path(receipt["candidate"]) / "production-preparation.json"
    prepared = read(preparation)
    if (
        hashes["upstream_production_owner.py"] != prepared["owner_sha256"]
        or hashes["test_upstream_production_owner.py"] != prepared["test_sha256"]
    ):
        raise ValueError("Production owner helpers differ from the executed CPU candidate")
    receipt_digest = sha256(receipt_path)
    evidence = require_completed_validation(root)
    if (
        sha256(receipt_path) != receipt_digest
        or require_cpu(root, receipt_path) != receipt
        or {name: sha256(source) for name, source in sources.items()} != hashes
    ):
        raise ValueError("Production candidate, CPU evidence or helpers changed during reconstruction")
    release_path, helpers = paths(root)
    archive = root / "run-preparation/source-wave9-launch-20261117.tar"
    publication_path = root / "run-preparation/wave9-freeze-publication.json"
    if publication_path.exists():
        publication = read(publication_path)
        release = publication["release"]
        if (
            release["files_sha256"] != receipt["files_sha256"]
            or release["helpers_sha256"] != hashes
            or release["cpu_receipt_sha256"] != receipt_digest
            or release["validation_sha256"] != evidence["receipt_sha256"]
            or release["state_sha256"] != evidence["state_sha256"]
        ):
            raise ValueError("Interrupted freeze differs from the original prepared release")
        publish_freeze(root, publication)
        return release
    if any(path.exists() for path in (release_path, helpers, archive)):
        raise FileExistsError("Wave-9 release already has evidence; preserve without refreezing")
    staging = root / "run-preparation" / f"wave9-freeze-staging-{uuid4().hex}"
    staged_helpers, staged_archive = staging / "tracking", staging / "source.tar"
    staged_helpers.mkdir(parents=True)
    with tarfile.open(staged_archive, "x") as bundle:
        for name in names:
            shutil.copy2(sources[name], staged_helpers / name)
            bundle.add(staged_helpers / name, arcname=f"tracking/{name}", recursive=False)
        for name in receipt["files_sha256"]:
            bundle.add(Path(receipt["candidate"]) / name, arcname=name, recursive=False)
        bundle.add(
            Path(receipt["candidate"]) / "artifacts/external-sources",
            arcname="artifacts/external-sources",
            recursive=False,
        )
    if (
        require_cpu(root, receipt_path) != receipt
        or sha256(receipt_path) != receipt_digest
        or {name: sha256(staged_helpers / name) for name in names} != hashes
        or {name: sha256(source) for name, source in sources.items()} != hashes
    ):
        raise ValueError(f"Production freeze changed source; preserve staging evidence at {staging}")
    release = {
        "status": "frozen_validated_launch_release",
        "root": str(root),
        "candidate": receipt["candidate"],
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "archive": str(archive),
        "archive_sha256": sha256(staged_archive),
        "validation_binding_sha256": sha256(root / "run-preparation/upstream-full-model-corrected-binding.json"),
        "validation_sha256": evidence["receipt_sha256"],
        "state_sha256": evidence["state_sha256"],
        "preparation": str(preparation),
        "preparation_sha256": sha256(preparation),
        "cpu_receipt": str(receipt_path),
        "cpu_receipt_sha256": sha256(receipt_path),
        "files_sha256": receipt["files_sha256"],
        "source_sha256": {
            name: digest
            for name, digest in receipt["files_sha256"].items()
            if name.startswith("experiment/") and name.endswith(".py")
        },
        "data_receipt_sha256": numerical["data_receipt_sha256"],
        "environment": ENVIRONMENT,
        "arms": ARMS,
        "nodes": NODES,
        "commands": upstream_run_history.commands(root, Path(receipt["candidate"])),
        "monitor_interval_seconds": 900,
        "training_timeout_seconds": 21600,
        "policy": (
            "Original base LR/full budget, no resume/retry; owner-only migration; all original gates reconstructed."
        ),
    }
    staged_release = staging / "release.json"
    write(staged_release, release)
    release = read(staged_release)
    publication = {
        "status": "prepared_identical_freeze_publication",
        "root": str(root),
        "staging": str(staging),
        "release": release,
        "release_sha256": sha256(staged_release),
    }
    write(publication_path, publication)
    publish_freeze(root, publication)
    return release


def publish_freeze(root: Path, publication: dict) -> None:
    """Finish only the identical hash-bound interrupted publication, without creating training intents."""
    release_path, helpers = paths(root)
    archive = root / "run-preparation/source-wave9-launch-20261117.tar"
    staging = Path(publication["staging"])
    release = publication["release"]
    staged_release = staging / "release.json"
    if (
        publication["status"] != "prepared_identical_freeze_publication"
        or publication["root"] != str(root)
        or not staging.resolve().is_relative_to((root / "run-preparation").resolve())
        or release["root"] != str(root)
        or release["helpers"] != str(helpers)
        or release["archive"] != str(archive)
        or sha256(staged_release) != publication["release_sha256"]
        or read(staged_release) != release
    ):
        raise ValueError("Interrupted Wave-9 publication identity differs")
    helper_source = helpers if helpers.exists() else staging / "tracking"
    archive_source = archive if archive.exists() else staging / "source.tar"
    if {name: sha256(helper_source / name) for name in release["helpers_sha256"]} != release["helpers_sha256"] or sha256(
        archive_source
    ) != release["archive_sha256"]:
        raise ValueError("Interrupted Wave-9 helper or archive changed")
    helpers.parent.mkdir(parents=True, exist_ok=True)
    if not helpers.exists():
        helper_source.rename(helpers)
    if not archive.exists():
        archive_source.rename(archive)
    if release_path.exists():
        if sha256(release_path) != publication["release_sha256"]:
            raise ValueError("Published Wave-9 release differs from original publication")
    else:
        temporary = release_path.with_suffix(".publication.tmp")
        shutil.copy2(staged_release, temporary)
        temporary.replace(release_path)


def binding(root: Path, *, observation: bool = False) -> dict:
    release = upstream_run_history.helper_binding(root) if observation else upstream_run_history.binding(root)
    if Path(__file__).resolve() != (Path(release["helpers"]) / "upstream_production.py").resolve():
        raise ValueError("Use the exact frozen Wave-9 controller")
    if not observation:
        require_cpu(root, Path(release["cpu_receipt"]))
    return release


def node_for(arm: str) -> str:
    return NODES[ARMS.index(arm)]


def worker_path(root: Path, arm: str) -> Path:
    return root / "run-preparation" / f"wave9-worker-{node_for(arm)}.json"


def worker_command(root: Path, release: dict, arm: str) -> list[str]:
    return [
        PYTHON,
        str(Path(release["helpers"]) / "upstream_production.py"),
        "--root",
        str(root),
        "--phase",
        "worker",
        "--arm",
        arm,
    ]


def healthy_keeper(keeper_file: Path = Path("/root/dl/runtime/gpu_matmul.pid")) -> dict:
    """Require eight identified keeper workers occupying eight distinct GPUs."""
    pid = int(keeper_file.read_text())
    generation = keeper_generation(pid)
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    occupancy = [tuple(field.strip() for field in row) for row in csv.reader(rows)]
    owned = {worker for worker, _, _ in generation if worker != pid}
    if (
        len(owned) != 8
        or len(occupancy) != 8
        or any(len(row) != 2 for row in occupancy)
        or {int(row[0]) for row in occupancy} != owned
        or len({row[1] for row in occupancy}) != 8
        or keeper_generation(pid) != generation
        or keeper_file.read_text().strip() != str(pid)
    ):
        raise RuntimeError("Wave-9 keeper generation or exact eight-GPU occupancy differs")
    return {"pid": pid, "generation": generation}


def allocation(root: Path, release: dict, arm: str) -> dict:
    if local_node() != node_for(arm):
        raise RuntimeError("Wave-9 arm is on the wrong original allocation")
    idle_training()
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(devices) != 8 or any("A100-SXM4-40GB" not in name for name in devices):
        raise RuntimeError("Wave 9 requires its original eight A100 GPUs")
    subprocess.run(
        [PYTHON, "-m", "experiment.bootstrap", "--check-only"], cwd=release["candidate"], check=True, capture_output=True
    )
    return healthy_keeper()


def run_child(root: Path, release: dict, state: dict, path: Path) -> int:
    command = release["commands"][state["node"]]
    marker, generation = uuid4().hex, {}
    log = root / "runs/wave9" / f"{state['arm']}.torchrun.log"
    state.update(command=command, child_marker=marker, log=str(log))
    write(path, state)
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=release["candidate"],
            env=os.environ | ENVIRONMENT | {PROCESS_MARKER: marker},
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            observed = observed_process(child.pid)
            if observed is None or observed[2] != command:
                raise RuntimeError("Wave-9 launcher has no matching generation acknowledgment")
            state.update(status="running", child_pid=child.pid, child_start_ticks=observed[1])
            write(path, state)
            deadline = time.monotonic() + release["training_timeout_seconds"]
            while time.monotonic() < deadline:
                generation.update(tagged_processes(marker))
                try:
                    return child.wait(timeout=min(1, max(0.001, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
            raise subprocess.TimeoutExpired(command, release["training_timeout_seconds"])
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
    while ("pid" not in intent or intent.get("start_ticks") is None) and time.monotonic() < deadline:
        time.sleep(0.1)
        intent = read(intent_path)
    launch = read(root / "run-preparation/wave9-launch-state.json")
    own = observed_process(os.getpid())
    if (
        own is None
        or own[1] != intent.get("start_ticks")
        or intent.get("pid") != os.getpid()
        or own[2] != worker_command(root, release, arm)
        or intent["release_binding_sha256"] != sha256(paths(root)[0])
        or intent["worker_marker"] != launch["worker_markers"][arm]
        or os.environ.get(WORKER_MARKER) != intent["worker_marker"]
        or launch["binding_sha256"] != sha256(paths(root)[0])
        or launch["status"] not in ("launching", "launched", "needs_attention")
        or path.exists()
    ):
        raise RuntimeError("Wave-9 worker lacks its exact one-shot launch acknowledgment")
    with node_guard(root, node_for(arm)) as signals:
        keeper = allocation(root, release, arm)
        state = {
            "status": "starting",
            "arm": arm,
            "node": node_for(arm),
            "worker_pid": os.getpid(),
            "worker_start_ticks": own[1],
            "worker_command": own[2],
            "worker_marker": intent["worker_marker"],
            "release_binding_sha256": sha256(paths(root)[0]),
            "out": str(root / "runs/wave9" / arm),
        }
        write(path, state)
        try:
            if Path(state["out"]).exists():
                raise FileExistsError("Preserve previous full-budget output without retry")
            if healthy_keeper() != keeper:
                raise RuntimeError("Wave-9 keeper changed after allocation preflight")
            state["keeper"] = release_keeper(state, path)
            code = run_child(root, release, state, path)
            state["exit_code"] = code
            if code:
                raise RuntimeError("Wave-9 full-budget command failed")
            state["completed_run"] = upstream_run_history.completed_run(root, arm, release)
            state["status"] = "complete"
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if state.get("keeper_release_attempted"):
                try:
                    if "cleanup_error" in state:
                        raise RuntimeError("Unverified child cleanup prevents keeper restoration")
                    node_action(root, "restore")
                    state["restored_keeper"] = healthy_keeper()
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
        raise RuntimeError("Wave-9 worker startup is on the wrong node")
    path, release_path = worker_path(root, arm), paths(root)[0]
    intent_path = path.with_suffix(".start.json")
    with path.with_suffix(".start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() or intent_path.exists():
            raise FileExistsError("Wave-9 worker has launch evidence; no retry")
        launch = read(root / "run-preparation/wave9-launch-state.json")
        if launch["status"] != "launching" or launch["binding_sha256"] != sha256(release_path):
            raise RuntimeError("Wave-9 launch state differs from its release")
        command = worker_command(root, release, arm)
        starter = observed_process(os.getpid())
        if starter is None or starter[0] == "Z":
            raise RuntimeError("Wave-9 startup controller has no live generation")
        intent = {
            "status": "starting",
            "arm": arm,
            "node": node_for(arm),
            "command": command,
            "release_binding_sha256": sha256(release_path),
            "worker_marker": launch["worker_markers"][arm],
            "starter_pid": os.getpid(),
            "starter_start_ticks": starter[1],
            "starter_command": starter[2],
        }
        write(intent_path, intent)
        try:
            with path.with_suffix(".log").open("x") as stream:
                child = subprocess.Popen(
                    command,
                    cwd=release["candidate"],
                    env=os.environ | ENVIRONMENT | {WORKER_MARKER: intent["worker_marker"]},
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        except BaseException as error:
            intent.update(status="spawn_failed", error=f"{type(error).__name__}: {error}")
            write(intent_path, intent)
            raise
        observed = observed_process(child.pid)
        intent.update(pid=child.pid, start_ticks=None if observed is None else observed[1])
        write(intent_path, intent)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if path.exists():
                state = read(path)
                if state["worker_pid"] != child.pid or state["worker_start_ticks"] != intent["start_ticks"]:
                    raise RuntimeError("Wave-9 worker acknowledgment changed generation")
                return state
            if child.poll() is not None:
                raise RuntimeError("Wave-9 worker exited before acknowledgment; no retry")
            time.sleep(0.1)
        raise TimeoutError("Wave-9 worker startup unknown; preserve original child without retry")


def remote(root: Path, release: dict, arm: str, phase: str) -> dict:
    command = [
        PYTHON,
        str(Path(release["helpers"]) / "upstream_production.py"),
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
        timeout=300,
    )
    return json.loads(result.stdout.splitlines()[-1])


def startup_observation(root: Path, release: dict, arm: str, path: Path) -> dict:
    """Keep unknown starts live until exact process absence is verified; never retry them."""
    intent = read(path)
    if "starter_pid" in intent:
        starter = observed_process(intent["starter_pid"])
        if starter is not None and starter[0] != "Z" and starter[1] == intent["starter_start_ticks"]:
            return {"arm": arm, "status": "unknown", "receipt": intent}
    # Only a quiescent starter makes a later worker/descendant absence scan meaningful.
    intent = read(path)
    if worker_path(root, arm).exists():
        return probe_arm(root, arm)
    launch = read(root / "run-preparation/wave9-launch-state.json")
    if (
        intent["arm"] != arm
        or intent["node"] != node_for(arm)
        or intent["command"] != worker_command(root, release, arm)
        or intent["release_binding_sha256"] != sha256(paths(root)[0])
        or intent["worker_marker"] != launch["worker_markers"][arm]
    ):
        raise ValueError("Wave-9 startup intent identity changed")
    observed = observed_process(intent["pid"]) if intent.get("pid") is not None else None
    if observed is not None and observed[0] != "Z":
        matches = observed[1] == intent.get("start_ticks") and observed[2] == intent["command"]
        return {"arm": arm, "status": "unknown" if matches else "worker_generation_changed", "receipt": intent}
    token = f"{WORKER_MARKER}={intent['worker_marker']}".encode()
    descendants = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if token not in (directory / "environ").read_bytes().split(b"\0"):
                continue
            child = observed_process(int(directory.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if child is not None and child[0] != "Z":
            descendants[int(directory.name)] = child[1]
    if descendants:
        return {"arm": arm, "status": "live_orphaned_training", "receipt": intent, "descendants": descendants}
    if (
        intent.get("pid") is not None
        or intent["status"] == "spawn_failed"
        or "starter_pid" in intent
        or launch["status"] == "needs_attention"
    ):
        return {"arm": arm, "status": "needs_attention", "receipt": intent, "startup_absence_verified": True}
    return {"arm": arm, "status": "unknown", "receipt": intent}


def probe_arm(root: Path, arm: str) -> dict:
    release = binding(root, observation=True)
    if local_node() != node_for(arm):
        raise RuntimeError("Wave-9 observation is on the wrong node")
    path = worker_path(root, arm)
    if not path.exists():
        if path.with_suffix(".start.json").exists():
            return startup_observation(root, release, arm, path.with_suffix(".start.json"))
        return {"arm": arm, "status": "not_launched"}
    state = read(path)
    if (
        state["arm"] != arm
        or state["node"] != node_for(arm)
        or state["release_binding_sha256"] != sha256(paths(root)[0])
    ):
        raise ValueError("Wave-9 saved worker identity differs")
    observed = observed_process(state["worker_pid"])
    children = tagged_processes(state["child_marker"]) if state.get("child_marker") else {}
    if observed is not None and observed[0] != "Z":
        if observed[1] != state["worker_start_ticks"] or observed[2] != worker_command(root, release, arm):
            return {"arm": arm, "status": "worker_generation_changed", "receipt": state, "descendants": children}
        return {"arm": arm, "status": "running", "receipt": state}
    if children:
        return {"arm": arm, "status": "live_orphaned_training", "receipt": state, "descendants": children}
    complete = (
        state["status"] == "complete"
        and "cleanup_error" not in state
        and all(
            type(state.get(k)) is int and state[k] == 0 for k in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
    )
    if complete and (
        state["completed_run"] != upstream_run_history.completed_run(root, arm, release)
        or state["log_sha256"] != sha256(Path(state["log"]))
    ):
        raise ValueError("Completed Wave-9 history or log changed")
    return {"arm": arm, "status": "complete" if complete else "needs_attention", "receipt": state}


def snapshot(root: Path) -> dict:
    release = binding(root, observation=True)
    workers, runs, launches = [], [], []
    launch = read(root / "run-preparation/wave9-launch-state.json")
    for arm in ARMS:
        try:
            worker = remote(root, release, arm, "probe")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            worker = {"arm": arm, "status": "unknown", "error": str(error)}
        if worker["status"] == "not_launched" and launch["status"] == "needs_attention":
            worker["status"] = "not_launched_after_launch_failure"
        workers.append(worker)
        state = worker.get("receipt", {})
        run = {
            "name": arm,
            "arm": arm,
            "node": node_for(arm),
            "status": worker["status"],
            "planned_steps": 5120,
            "latest_metrics": None,
            "observed_tokens": None,
            "keeper_watcher": {
                "status": "restored" if worker["status"] == "complete" else state.get("status", "unknown")
            },
        }
        directory = root / "runs/wave9" / arm
        if (directory / "run.json").exists():
            try:
                measured = upstream_run_history.measured_run(root, arm, release)
                run.update(
                    latest_metrics={"step": measured["observed_updates"]}, observed_tokens=measured["observed_tokens"]
                )
            except (ValueError, OSError, KeyError) as error:
                run["accounting_error"] = str(error)
        elif worker["status"] in ("not_launched", "not_launched_after_launch_failure"):
            run["observed_tokens"] = 0
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
    write(root / "run-preparation/wave9.json", {"expected_launches": 4, "launches": launches})
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
    path = root / "run-preparation/wave9-launch-state.json"
    directory = root / "run-preparation/monitor-wave9"
    if any(p.exists() for p in (path, directory, root / "runs/wave9")):
        raise FileExistsError("Wave-9 launch already has evidence; no retry")
    for arm in ARMS:
        if worker_path(root, arm).exists() or worker_path(root, arm).with_suffix(".start.json").exists():
            raise FileExistsError("Wave-9 worker already has launch evidence")
        remote(root, release, arm, "inspect")
    (root / "runs/wave9").mkdir(parents=True)
    directory.mkdir()
    state = {
        "status": "launching",
        "binding_sha256": sha256(paths(root)[0]),
        "workers": {},
        "worker_markers": {arm: uuid4().hex for arm in ARMS},
    }
    write(path, state)
    try:
        command = ["bash", str(Path(release["helpers"]) / "monitor_upstream_run.sh"), str(root)]
        with (directory / "monitor.log").open("x") as log:
            monitor = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (directory / "pid").exists():
                pid = int((directory / "pid").read_text())
                observed = observed_process(pid)
                if pid != monitor.pid or observed is None or observed[0] == "Z" or observed[2] != command:
                    raise RuntimeError("Wave-9 monitor acknowledgment differs from its spawned generation")
                break
            if monitor.poll() is not None:
                raise RuntimeError("Wave-9 monitor exited before acknowledgment")
            time.sleep(0.1)
        else:
            raise TimeoutError("Wave-9 monitor startup unknown; preserve original child without retry")
        state["monitor_pid"] = monitor.pid
        write(path, state)
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
                raise RuntimeError("Wave-9 worker acknowledgment differs; preserve partial launch")
        state["status"] = "launched"
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(path, state)
    return state


def main():
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
        parser.error("Preparation/CPU verification requires --candidate")
    if args.phase in ("cpu", "snapshot") and args.out is None:
        parser.error("CPU verification/snapshot requires --out")
    if args.phase == "freeze" and args.cpu_receipt is None:
        parser.error("Freeze requires --cpu-receipt")
    if args.phase in ("inspect", "start", "worker", "probe") and args.arm is None:
        parser.error("Node operation requires --arm")
    root = args.root.resolve()
    if args.phase == "prepare":
        result = prepare(root, args.candidate)
    elif args.phase == "cpu":
        result = cpu(root, args.candidate, args.out)
    elif args.phase == "freeze":
        with (root / "run-preparation/.wave9-freeze.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = freeze(root, args.cpu_receipt)
    elif args.phase == "check":
        result = binding(root)
    elif args.phase == "launch":
        with defer_signals():
            with (root / "run-preparation/.wave9-launch-controller.lock").open("a") as lock:
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
