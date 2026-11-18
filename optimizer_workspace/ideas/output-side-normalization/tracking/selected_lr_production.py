# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own one selected-LR full-budget wave after original validation and exact restoration."""

import argparse
import ast
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

import selected_lr_run_history as run_history
from launch_wave5 import NODES, PYTHON, node_action, read, sha256
from owned_output_handoff import keeper_observation, marker_processes, observed, require_absent, same_generation
from prepare_selected_lr_release import require_cpu
from prepare_selected_lr_validation import require_archive as require_controller_archive
from run_diagonal_diagnostic import finish_child, tagged_processes
from selected_lr_run_history import ARMS, ENVIRONMENT, EXTRA_HELPERS, STATUS, WAVE, WORKER_MARKER, commands, paths
from selected_lr_validation import (
    CHILD_MARKER,
    CONTROLLER_MARKER,
    MASK,
    NODE,
    controller_command,
    locations,
    validator_command,
)
from selected_lr_wave10_handoff import probe as wave10_handoff
from watch_wave5 import write
from wave7_release import defer_signals, idle_training, local_node, node_guard, release_keeper

CONTROLLER_TESTS = (
    "test_selected_lr_production.py",
    "test_prepare_selected_lr_release.py",
    "test_selected_lr_release_owner.py",
)
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)


def node_for(arm: str) -> str:
    return NODES[ARMS.index(arm)]


def worker_path(root: Path, arm: str) -> Path:
    return root / f"run-preparation/wave11-workers/{arm}.json"


def worker_command(root: Path, release: dict, arm: str) -> list[str]:
    return [
        PYTHON,
        str(Path(release["helpers"]) / Path(__file__).name),
        "--root",
        str(root),
        "--phase",
        "worker",
        "--arm",
        arm,
    ]


def evidence_identities(root: Path) -> dict:
    binding_path, helpers, state_path, _, receipt_path = locations(root)
    return {
        "validation_binding_sha256": sha256(binding_path),
        "state_sha256": sha256(state_path),
        "validation_sha256": sha256(receipt_path),
        "intent_sha256": sha256(helpers.parent / "controller.json"),
        "predecessor_release_sha256": sha256(root / "run-preparation/wave10-launch-binding.json"),
        "predecessor_analysis_sha256": sha256(root / "run-preparation/analysis-wave10.json"),
        "predecessor_cloud_sha256": sha256(root / "run-preparation/swanlab-upload-wave10.json"),
    }


def same_keeper(actual: dict, expected: dict) -> bool:
    generation = {int(pid): process for pid, process in expected["generation"].items()}
    current = {int(pid): process for pid, process in actual["generation"].items()}
    return (
        actual["pid"] == expected["pid"]
        and set(current) == set(generation)
        and all(same_generation(process, current[pid]) for pid, process in generation.items())
        and actual["devices"] == expected["devices"]
    )


def terminal_validation_observation(root: Path) -> dict:
    """Inspect the original validator's exact acknowledged generations and restored node."""
    binding_path, helpers, state_path, _, receipt_path = locations(root)
    original, state, intent = read(binding_path), read(state_path), read(helpers.parent / "controller.json")
    if (
        local_node() != NODE
        or state["status"] != "passed"
        or state["node"] != NODE
        or state["candidate"] != original["candidate"]
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or any(key in state for key in ("error", "cleanup_error"))
        or intent["status"] != "acknowledged"
        or intent["pid"] != state["controller_pid"]
        or intent["start_ticks"] != state["controller_start_ticks"]
        or intent["command"] != controller_command(root)
        or intent["binding_sha256"] != sha256(binding_path)
        or state["binding_sha256"] != sha256(binding_path)
        or state["validation_sha256"] != sha256(receipt_path)
        or state["command"] != validator_command(root, Path(original["candidate"]))
        or intent["generation"]["pid"] != intent["pid"]
        or intent["generation"]["start_ticks"] != intent["start_ticks"]
        or intent["generation"]["command"] != intent["command"]
        or intent["generation"]["group"] != intent["pid"]
        or intent["generation"]["session"] != intent["pid"]
        or not intent.get("marker")
        or not state.get("child_marker")
    ):
        raise RuntimeError("Original selected-LR validation acknowledgment or terminal evidence differs")
    require_absent(state["controller_pid"])
    require_absent(state["child_pid"])
    if marker_processes(CHILD_MARKER, state["child_marker"]):
        raise RuntimeError("Original selected-LR diagnostic descendants remain")
    keeper = keeper_observation()
    if not same_keeper(keeper, state["restored_keeper"]):
        raise RuntimeError("Selected-LR current keeper differs from its original restoration")
    marked = marker_processes(CONTROLLER_MARKER, intent["marker"])
    if any(
        pid not in keeper["generation"] or not same_generation(process, keeper["generation"][pid])
        for pid, process in marked.items()
    ):
        raise RuntimeError("Selected-LR controller marker exists outside its exact restored keeper")
    old_release = read(root / "run-preparation/wave10-launch-binding.json")
    old_arm = old_release["arms"][old_release["nodes"].index(NODE)]
    old_state = read(root / f"run-preparation/wave10-worker-{NODE}.json")
    if old_state["arm"] != old_arm or old_state["status"] != "complete":
        raise RuntimeError("Selected-LR predecessor node mapping differs")
    require_absent(old_state["worker_pid"])
    require_absent(old_state["child_pid"])
    require_absent(state["keeper"]["pid"])
    if marker_processes(CHILD_MARKER, old_state["child_marker"]):
        raise RuntimeError("Wave10 original training descendants remain on the validator node")
    inherited = marker_processes("OUTPUT_MARIN_WAVE10_WORKER_ID", old_state["worker_marker"])
    if any(
        pid not in keeper["generation"] or not same_generation(process, keeper["generation"][pid])
        for pid, process in inherited.items()
    ):
        raise RuntimeError("Wave10 worker marker exists outside the legitimate Wave11 keeper succession")
    if any(not same_generation(process, observed(pid)) for pid, process in keeper["generation"].items()):
        raise RuntimeError("Selected-LR restored keeper changed during inspection")
    return {
        "status": "selected_lr_original_validation_absent_exact_restored_keeper_verified",
        "node": NODE,
        "keeper": keeper,
    }


def validation_acceptance(root: Path, *, accepted: dict | None = None) -> dict:
    """Rederive original all-rank evidence under its node lock, then recheck identities."""
    original_path, helpers, state_path, _, receipt_path = locations(root)
    state = read(state_path)
    if state["status"] != "passed":
        raise RuntimeError("Original selected-LR validation has not passed")
    original = read(original_path)
    before = evidence_identities(root)
    with node_guard(root, NODE):
        observation = terminal_validation_observation(root)
        if accepted is not None and (
            before != {key: accepted[key] for key in before}
            or not same_keeper(observation["keeper"], accepted["keeper"])
        ):
            raise RuntimeError("Accepted original selected-LR evidence or allocation changed")
        if accepted is None:
            script = (
                "import json,sys\nfrom pathlib import Path\n"
                "from run_selected_lr_validation import binding\n"
                "from validate_selected_lr_full_budget import reconstruct\n"
                "root=Path(sys.argv[1]); frozen=binding(root)\n"
                "r=reconstruct(root,Path(frozen['candidate']),Path(frozen['receipt']))\n"
                "print(json.dumps({'status':r['status'],'owner':r['owner'],"
                "'commands':len(r['gpu_commands']),'flags':sum(v is True for t in r['trials'] "
                "for v in t['checks'].values())},allow_nan=False))"
            )
            result = json.loads(
                subprocess.run(
                    [PYTHON, "-P", "-c", script, str(root)],
                    cwd=helpers,
                    env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(helpers)},
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2400,
                ).stdout.splitlines()[-1]
            )
        else:
            receipt = read(receipt_path)
            result = {
                "status": receipt["status"],
                "owner": receipt["owner"],
                "commands": len(receipt["gpu_commands"]),
                "flags": sum(value is True for trial in receipt["trials"] for value in trial["checks"].values()),
            }
        if (
            result["status"] != "passed_actual_full_budget_config_gates"
            or result["owner"]
            != {
                "controller_pid": state["controller_pid"],
                "controller_start_ticks": state["controller_start_ticks"],
            }
            or result["commands"] != 12
            or result["flags"] != 30
            or before != evidence_identities(root)
            or not same_keeper(terminal_validation_observation(root)["keeper"], observation["keeper"])
            or original["predecessor_release_sha256"] != before["predecessor_release_sha256"]
        ):
            raise RuntimeError("Selected-LR independent original reconstruction or final allocation differs")
    return {
        "status": "selected_lr_validation_rederived_absent_exact_keeper_verified",
        **before,
        "binding_sha256": before["validation_binding_sha256"],
        "receipt_sha256": before["validation_sha256"],
        "commands": result["commands"],
        "checks": result["flags"],
        "keeper": observation["keeper"],
    }


def predecessor_results(root: Path) -> None:
    """Reproduce final Wave10 analysis/histories without reusing its superseded .9 keeper."""
    _, helpers, _, _, _ = locations(root)
    intent = read(helpers.parent / "controller.json")
    before = evidence_identities(root)
    original = intent["predecessor"]
    if any(
        original[key] != before[other]
        for key, other in (
            ("release_sha256", "predecessor_release_sha256"),
            ("analysis_sha256", "predecessor_analysis_sha256"),
            ("cloud_sha256", "predecessor_cloud_sha256"),
        )
    ):
        raise ValueError("Wave10 result/cloud identities differ from original validation acceptance")
    release = read(root / "run-preparation/wave10-launch-binding.json")
    source_helpers = Path(release["helpers"])
    script = (
        "import json,sys\nfrom pathlib import Path\nfrom owned_output_production import snapshot\n"
        "from owned_output_run_history import binding,completed_run\nfrom analyze_owned_output_run import analyze\n"
        "root=Path(sys.argv[1]); release=binding(root)\n"
        "print(json.dumps({'snapshot':snapshot(root),'analysis':analyze(root),"
        "'histories':{arm:completed_run(root,arm,release) for arm in release['arms']}},allow_nan=False))"
    )
    actual = json.loads(
        subprocess.run(
            [PYTHON, "-P", "-c", script, str(root)],
            cwd=source_helpers,
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(source_helpers)},
            check=True,
            capture_output=True,
            text=True,
            timeout=1500,
        ).stdout.splitlines()[-1]
    )
    saved, cloud = read(root / "run-preparation/analysis-wave10.json"), read(
        root / "run-preparation/swanlab-upload-wave10.json"
    )
    if (
        actual["snapshot"]["status"] != "complete"
        or actual["snapshot"]["all_workers_terminal"] is not True
        or actual["analysis"] != saved
        or cloud["status"] != "verified"
        or cloud["visibility"] != "PRIVATE"
        or cloud["project"] != "siyuc3141/output-side-normalization-marin"
        or set(cloud["runs"]) != set(release["arms"])
        or before != evidence_identities(root)
    ):
        raise RuntimeError("Wave10 final broad-window result or complete cloud evidence differs")
    for arm in release["arms"]:
        local, remote = actual["histories"][arm], cloud["runs"][arm]
        if remote["status"] != "verified" or any(
            remote[key] != local[key]
            for key in ("manifest_sha256", "metrics_sha256", "checkpoint_receipt_sha256", "checkpoint_rank_bytes")
        ):
            raise ValueError("Wave10 imported history differs from final original histories")


def allocation(root: Path, release: dict, arm: str) -> dict:
    """Match predecessor ownership by node, allowing only the designated validation succession."""
    node = node_for(arm)
    if local_node() != node or os.environ.get("CUDA_VISIBLE_DEVICES") != MASK:
        raise RuntimeError("Wave11 allocation requires its designated node and explicit eight-device mask")
    idle_training()
    if node == NODE:
        observation = terminal_validation_observation(root)
    else:
        old_release = read(root / "run-preparation/wave10-launch-binding.json")
        predecessor_arm = old_release["arms"][old_release["nodes"].index(node)]
        observation = wave10_handoff(root, predecessor_arm)
    accepted = release["allocation_by_node"][node]
    if not same_keeper(observation["keeper"], accepted["keeper"]):
        raise RuntimeError("Wave11 predecessor keeper changed after release acceptance")
    subprocess.run(
        [PYTHON, "-m", "experiment.bootstrap", "--check-only"],
        cwd=release["candidate"],
        env=os.environ | ENVIRONMENT,
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    current = keeper_observation()
    if not same_keeper(current, observation["keeper"]):
        raise RuntimeError("Wave11 predecessor keeper changed during bootstrap preflight")
    return current


def initial_allocation(root: Path, node: str) -> dict:
    if node == NODE:
        return terminal_validation_observation(root)
    old = read(root / "run-preparation/wave10-launch-binding.json")
    arm = old["arms"][old["nodes"].index(node)]
    frozen_helpers = locations(root)[1]
    command = [
        "env",
        "CUDA_VISIBLE_DEVICES=",
        PYTHON,
        str(frozen_helpers / "selected_lr_wave10_handoff.py"),
        "--root",
        str(root),
        "--arm",
        arm,
    ]
    return json.loads(
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"root@{node}", shlex.join(command)],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        ).stdout.splitlines()[-1]
    )


def helper_sources(original: dict) -> dict[str, Path]:
    """Freeze the transitive local import closure while retaining original validation helpers."""
    directory = Path(__file__).parent
    sources = {name: Path(original["helpers"]) / name for name in original["helpers_sha256"]}
    pending = set(EXTRA_HELPERS)
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        path = directory / name
        if Path(name).name != name or not path.is_file():
            raise ValueError(f"Wave11 production helper dependency missing: {name}")
        sources[name] = path
        if path.suffix != ".py":
            continue
        for item in ast.walk(ast.parse(path.read_text())):
            modules = (
                [alias.name for alias in item.names]
                if isinstance(item, ast.Import)
                else [item.module] if isinstance(item, ast.ImportFrom) and item.module else []
            )
            for module in modules:
                dependency = f"{module.split('.')[0]}.py"
                if (directory / dependency).is_file() and dependency not in sources:
                    pending.add(dependency)
    return dict(sorted(sources.items()))


def controller_cpu(cpu_receipt_path: Path, root: Path, out: Path) -> dict:
    """Test the exact helper closure in an independent hidden-CUDA archive before publication."""
    cpu = require_cpu(root, cpu_receipt_path)
    candidate = Path(cpu["candidate"])
    if out.exists() or out.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Wave11 controller CPU evidence needs an unused external directory")
    sources = helper_sources(read(locations(root)[0]))
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, source in sources.items():
        shutil.copy2(source, helpers / name)
    hashes = {name: sha256(helpers / name) for name in sources}
    log, archive = out / "controller.log", out / "controller-source.tar"
    command = [
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
        *(f"tracking/{name}" for name in CONTROLLER_TESTS),
    ]
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                command,
                cwd=out,
                env=os.environ
                | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers)))},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Controller CPU timeout at 600 seconds"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in hashes:
                bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        require_controller_archive(archive, {f"tracking/{name}": digest for name, digest in hashes.items()}, alias=False)
        if hashes != {name: sha256(helpers / name) for name in hashes} or require_cpu(root, cpu_receipt_path) != cpu:
            raise ValueError("Wave11 controller or candidate changed during CPU verification")
    except (ValueError, OSError, KeyError, tarfile.TarError, subprocess.SubprocessError) as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": "selected_lr_production_controller_cpu_verified" if code == 0 and error is None else "needs_attention",
        "candidate_cpu_receipt": str(cpu_receipt_path.resolve()),
        "candidate_cpu_receipt_sha256": sha256(cpu_receipt_path),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "command": command,
        "cwd": str(out),
        "exit_code": code,
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive) if archive.is_file() else None,
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != "selected_lr_production_controller_cpu_verified":
        raise RuntimeError(f"Wave11 controller CPU failure preserved at {out}")
    return result


def require_controller_cpu(root: Path, path: Path, candidate_cpu_path: Path) -> dict:
    receipt = read(path)
    sources = helper_sources(read(locations(root)[0]))
    helpers = Path(receipt["helpers"])
    command = [
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
        *(f"tracking/{name}" for name in CONTROLLER_TESTS),
    ]
    if (
        receipt["status"] != "selected_lr_production_controller_cpu_verified"
        or receipt["candidate_cpu_receipt"] != str(candidate_cpu_path.resolve())
        or receipt["candidate_cpu_receipt_sha256"] != sha256(candidate_cpu_path)
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["cuda_visible_devices"] != ""
        or receipt["command"] != command
        or receipt["cwd"] != str(helpers.parent)
        or receipt["helpers_sha256"] != {name: sha256(source) for name, source in sources.items()}
        or receipt["helpers_sha256"] != {name: sha256(helpers / name) for name in sources}
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
    ):
        raise ValueError("Wave11 controller CPU source, command or outcome differs")
    require_controller_archive(
        Path(receipt["archive"]),
        {f"tracking/{name}": digest for name, digest in receipt["helpers_sha256"].items()},
        alias=False,
    )
    return receipt


def freeze(root: Path, cpu_receipt_path: Path, controller_cpu_path: Path) -> dict:
    cpu = require_cpu(root, cpu_receipt_path)
    controller = require_controller_cpu(root, controller_cpu_path, cpu_receipt_path)
    original = read(locations(root)[0])
    sources = helper_sources(original)
    hashes = {name: sha256(source) for name, source in sources.items()}
    candidate = Path(cpu["candidate"])
    if hashes["selected_lr_release_owner.py"] != sha256(candidate / "experiment/selected_lr_release_owner.py"):
        raise ValueError("Wave11 production owner differs from the CPU-tested candidate")
    accepted = validation_acceptance(root)
    predecessor_results(root)
    allocations = {node: initial_allocation(root, node) for node in NODES}
    release_path, helpers = paths(root)
    archive = root / "run-preparation/source-wave11-launch-20261117.tar"
    if any(path.exists() for path in (release_path, helpers, archive)):
        raise FileExistsError("Wave11 production release already has evidence; preserve without retry")
    if require_cpu(root, cpu_receipt_path) != cpu or hashes != {name: sha256(path) for name, path in sources.items()}:
        raise ValueError("Wave11 CPU-tested source or helpers changed during acceptance")
    helpers.mkdir(parents=True)
    with tarfile.open(archive, "x") as bundle:
        for name, source in sources.items():
            shutil.copy2(source, helpers / name)
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
        for name in cpu["files_sha256"]:
            bundle.add(candidate / name, arcname=name, recursive=False)
        bundle.add(candidate / run_history.PREPARATION, arcname="preparation.json", recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    result = {
        "status": STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "files_sha256": cpu["files_sha256"],
        "source_sha256": {
            name: digest
            for name, digest in cpu["files_sha256"].items()
            if name.startswith("experiment/") and name.endswith(".py")
        },
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "cpu_receipt": str(cpu_receipt_path.resolve()),
        "cpu_receipt_sha256": sha256(cpu_receipt_path),
        "controller_cpu_receipt": str(controller_cpu_path.resolve()),
        "controller_cpu_receipt_sha256": sha256(controller_cpu_path),
        "preparation": str(candidate / run_history.PREPARATION),
        "preparation_sha256": sha256(candidate / run_history.PREPARATION),
        "data_receipt_sha256": original["data_receipt_sha256"],
        **{key: accepted[key] for key in evidence_identities(root)},
        "validation_reconstruction": accepted,
        "allocation_by_node": allocations,
        "arms": list(ARMS),
        "nodes": list(NODES),
        "environment": ENVIRONMENT,
        "commands": commands(root, candidate),
        "monitor_interval_seconds": 900,
        "training_timeout_seconds": 21600,
    }
    inventory = (
        cpu["files_sha256"]
        | {"preparation.json": result["preparation_sha256"]}
        | {f"tracking/{name}": digest for name, digest in hashes.items()}
    )
    run_history.require_archive(archive, inventory)
    if (
        require_cpu(root, cpu_receipt_path) != cpu
        or require_controller_cpu(root, controller_cpu_path, cpu_receipt_path) != controller
        or hashes != {name: sha256(helpers / name) for name in hashes}
    ):
        raise ValueError("Wave11 archived source changed before publication")
    validation_acceptance(root, accepted=accepted)
    for node, expected in allocations.items():
        if not same_keeper(initial_allocation(root, node)["keeper"], expected["keeper"]):
            raise RuntimeError("Wave11 all-node allocation changed before publication")
    if evidence_identities(root) != {key: accepted[key] for key in evidence_identities(root)}:
        raise ValueError("Wave11 accepted evidence changed before publication")
    with node_guard(root, NODE):
        final = terminal_validation_observation(root)
        if not same_keeper(final["keeper"], accepted["keeper"]) or evidence_identities(root) != {
            key: accepted[key] for key in evidence_identities(root)
        }:
            raise RuntimeError("Wave11 final locked validation allocation or evidence changed")
        write(release_path, result)
    return result


def binding(root: Path, *, observation: bool = False) -> dict:
    return run_history.helper_binding(root) if observation else run_history.binding(root)


def run_child(root: Path, release: dict, state: dict, path: Path) -> int:
    command, marker, generation = release["commands"][state["node"]], uuid4().hex, {}
    log = root / f"runs/{WAVE}/{state['arm']}.torchrun.log"
    state.update(command=command, child_marker=marker, log=str(log))
    write(path, state)
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=release["candidate"],
            env=os.environ | ENVIRONMENT | {CHILD_MARKER: marker},
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            own = observed(child.pid)
            if own is None or own["status"] in ("Z", "X", "x") or own["command"] != command:
                raise RuntimeError("Wave11 torchrun lacks its original live generation")
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
            except BaseException as error:
                state["cleanup_error"] = str(error)
                raise
            finally:
                state.update(child_generation=generation, log_sha256=sha256(log))
                write(path, state)


def run_arm(root: Path, arm: str) -> dict:
    release, path = binding(root), worker_path(root, arm)
    deadline = time.monotonic() + 30
    intent = read(path.with_suffix(".start.json"))
    while intent.get("status") == "starting" and time.monotonic() < deadline:
        time.sleep(0.05)
        intent = read(path.with_suffix(".start.json"))
    launch, own = read(root / "run-preparation/wave11-launch-state.json"), observed(os.getpid())
    expected = intent.get("generation")
    if (
        own is None
        or own["status"] in ("Z", "X", "x")
        or expected is None
        or any(own[key] != expected[key] for key in ("pid", "start_ticks", "group", "session", "command"))
        or own["command"] != worker_command(root, release, arm)
        or intent["status"] != "acknowledged"
        or intent["release_binding_sha256"] != sha256(paths(root)[0])
        or intent["worker_marker"] != launch["worker_markers"][arm]
        or os.environ.get(WORKER_MARKER) != intent["worker_marker"]
        or launch["binding_sha256"] != sha256(paths(root)[0])
        or launch["status"] not in ("launching", "launched")
        or path.exists()
    ):
        raise RuntimeError("Wave11 worker lacks its original one-shot generation acknowledgment")
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
                raise FileExistsError("Wave11 output already exists; preserve without retry")
            if not same_keeper(keeper_observation(), keeper):
                raise RuntimeError("Wave11 keeper changed after allocation preflight")
            state["keeper"] = release_keeper(state, path)
            code = run_child(root, release, state, path)
            state["exit_code"] = code
            if type(code) is not int or code != 0:
                raise RuntimeError("Wave11 full-budget training command failed")
            state["completed_run"] = run_history.completed_run(root, arm, release)
            state["status"] = "complete"
        except BaseException as error:
            state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if state.get("keeper_release_attempted"):
                try:
                    if "cleanup_error" in state or tagged_processes(state.get("child_marker", "")):
                        raise RuntimeError("Unverified Wave11 training descendants prevent restoration")
                    node_action(root, "restore")
                    state.update(restored_keeper=keeper_observation(), keeper_restore=0)
                except BaseException as error:
                    state.update(status="needs_attention", keeper_restore=f"refused: {error}")
            if signals:
                state["deferred_signals"] = signals
            write(path, state)
    return state


def start_arm(root: Path, arm: str) -> dict:
    release, path = binding(root), worker_path(root, arm)
    if local_node() != node_for(arm):
        raise RuntimeError("Wave11 startup requires its designated node")
    path.parent.mkdir(parents=True, exist_ok=True)
    intent_path = path.with_suffix(".start.json")
    with path.with_suffix(".start.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() or intent_path.exists():
            raise FileExistsError("Wave11 worker already has intent or evidence; no retry")
        launch = read(root / "run-preparation/wave11-launch-state.json")
        if launch["status"] != "launching" or launch["binding_sha256"] != sha256(paths(root)[0]):
            raise RuntimeError("Wave11 startup differs from its original release")
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
                    env=os.environ | ENVIRONMENT | {WORKER_MARKER: intent["worker_marker"]},
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            own = observed(child.pid)
            if own is None or own["status"] in ("Z", "X", "x") or own["command"] != command:
                raise RuntimeError("Wave11 startup generation is unknown; preserve original handle")
            intent.update(status="acknowledged", pid=child.pid, start_ticks=own["start_ticks"], generation=own)
            write(intent_path, intent)
        except BaseException as error:
            intent.update(status="unacknowledged", error=f"{type(error).__name__}: {error}")
            write(intent_path, intent)
            raise
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if path.exists():
                state = read(path)
                if state["worker_pid"] != child.pid or state["worker_start_ticks"] != own["start_ticks"]:
                    raise RuntimeError("Wave11 worker acknowledgment changed generation")
                return state
            if child.poll() is not None:
                raise RuntimeError("Wave11 worker exited before state acknowledgment; no retry")
            time.sleep(0.1)
        raise TimeoutError("Wave11 worker startup unknown; preserve original process")


def remote(root: Path, release: dict, arm: str, phase: str) -> dict:
    command = [
        "env",
        f"CUDA_VISIBLE_DEVICES={MASK}",
        PYTHON,
        str(Path(release["helpers"]) / Path(__file__).name),
        "--root",
        str(root),
        "--phase",
        phase,
        "--arm",
        arm,
    ]
    return json.loads(
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"root@{node_for(arm)}", shlex.join(command)],
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        ).stdout.splitlines()[-1]
    )


def probe_arm(root: Path, arm: str) -> dict:
    release, path = binding(root, observation=True), worker_path(root, arm)
    if local_node() != node_for(arm):
        raise RuntimeError("Wave11 observation requires its designated node")
    if not path.exists():
        intent_path = path.with_suffix(".start.json")
        if not intent_path.exists():
            return {"arm": arm, "status": "not_launched"}
        intent = read(intent_path)
        if intent.get("starter") and same_generation(intent["starter"], observed(intent["starter"]["pid"])):
            return {"arm": arm, "status": "unknown", "receipt": intent}
        if intent.get("pid") and observed(intent["pid"]) is not None:
            return {"arm": arm, "status": "unknown", "receipt": intent}
        descendants = marker_processes(WORKER_MARKER, intent["worker_marker"])
        return {"arm": arm, "status": "live_orphaned_training" if descendants else "needs_attention", "receipt": intent}
    state, intent = read(path), read(path.with_suffix(".start.json"))
    if (
        state["arm"] != arm
        or state["node"] != node_for(arm)
        or state["release_binding_sha256"] != sha256(paths(root)[0])
        or intent["status"] != "acknowledged"
        or intent["pid"] != state["worker_pid"]
        or intent["start_ticks"] != state["worker_start_ticks"]
        or intent["worker_marker"] != state["worker_marker"]
        or intent["command"] != state["worker_command"]
    ):
        raise ValueError("Wave11 saved original worker identity differs")
    own = observed(state["worker_pid"])
    children = tagged_processes(state["child_marker"]) if state.get("child_marker") else {}
    if own is not None and own["status"] not in ("Z", "X", "x"):
        if own["start_ticks"] != state["worker_start_ticks"] or own["command"] != worker_command(root, release, arm):
            return {"arm": arm, "status": "worker_generation_changed", "receipt": state}
        return {"arm": arm, "status": "running", "receipt": state}
    if children:
        return {"arm": arm, "status": "live_orphaned_training", "receipt": state, "descendants": children}
    complete = (
        state["status"] == "complete"
        and not any(key in state for key in ("error", "cleanup_error"))
        and all(
            type(state.get(key)) is int and state[key] == 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
    )
    if complete:
        if state["completed_run"] != run_history.completed_run(root, arm, release) or state["log_sha256"] != sha256(
            Path(state["log"])
        ):
            raise ValueError("Wave11 completed original history or log differs")
        keeper = keeper_observation()
        if not same_keeper(keeper, state["restored_keeper"]):
            raise ValueError("Wave11 completed worker restoration changed")
        marked = marker_processes(WORKER_MARKER, state["worker_marker"])
        if any(
            pid not in keeper["generation"] or not same_generation(process, keeper["generation"][pid])
            for pid, process in marked.items()
        ):
            raise RuntimeError("Wave11 worker marker exists outside its restored keeper")
    return {"arm": arm, "status": "complete" if complete else "needs_attention", "receipt": state}


def snapshot(root: Path) -> dict:
    release = binding(root, observation=True)
    launch = read(root / "run-preparation/wave11-launch-state.json")
    workers, runs, launches = [], [], []
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
    write(root / "run-preparation/wave11.json", {"expected_launches": 4, "launches": launches})
    return {
        "status": "complete" if statuses == {"complete"} else "needs_attention" if terminal else "active",
        "binding_sha256": sha256(paths(root)[0]),
        "workers": workers,
        "runs": runs,
        "all_workers_terminal": terminal,
    }


def launch(root: Path) -> dict:
    release = binding(root)
    validation_acceptance(root)
    predecessor_results(root)
    path, directory = root / "run-preparation/wave11-launch-state.json", root / "run-preparation/monitor-wave11"
    if any(item.exists() for item in (path, directory, root / f"runs/{WAVE}")):
        raise FileExistsError("Wave11 launch already has intent or evidence; preserve without retry")
    for arm in ARMS:
        worker = worker_path(root, arm)
        if worker.exists() or worker.with_suffix(".start.json").exists():
            raise FileExistsError("Wave11 worker already has evidence")
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
        command = ["bash", str(Path(release["helpers"]) / "monitor_selected_lr_run.sh"), str(root)]
        with (directory / "monitor.log").open("x") as log:
            monitor = subprocess.Popen(
                command,
                env=os.environ | ENVIRONMENT,
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
                    own is None
                    or own["status"] in ("Z", "X", "x")
                    or int((directory / "pid").read_text()) != monitor.pid
                    or own["command"] != command
                ):
                    raise RuntimeError("Wave11 monitor acknowledgment differs")
                state.update(monitor_pid=monitor.pid, monitor_start_ticks=own["start_ticks"])
                write(path, state)
                break
            if monitor.poll() is not None:
                raise RuntimeError("Wave11 monitor exited before acknowledgment")
            time.sleep(0.1)
        else:
            raise TimeoutError("Wave11 monitor startup unknown; preserve original process")
        for arm in ARMS:
            acknowledgment = remote(root, release, arm, "start")
            state["workers"][arm] = acknowledgment
            write(path, state)
            if (
                acknowledgment["arm"] != arm
                or acknowledgment["node"] != node_for(arm)
                or acknowledgment["worker_command"] != worker_command(root, release, arm)
                or acknowledgment["worker_marker"] != state["worker_markers"][arm]
                or acknowledgment["release_binding_sha256"] != state["binding_sha256"]
                or acknowledgment["status"] not in ("starting", "running", "complete")
            ):
                raise RuntimeError("Wave11 worker acknowledgment differs; preserve partial launch")
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
        choices=("cpu", "freeze", "check", "launch", "inspect", "start", "worker", "probe", "snapshot"),
        required=True,
    )
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--controller-cpu-receipt", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase in ("cpu", "freeze") and args.cpu_receipt is None:
        parser.error("cpu/freeze requires --cpu-receipt")
    if args.phase == "freeze" and args.controller_cpu_receipt is None:
        parser.error("freeze requires --controller-cpu-receipt")
    if args.phase in ("inspect", "start", "worker", "probe") and args.arm is None:
        parser.error("node phases require --arm")
    if args.phase in ("cpu", "snapshot") and args.out is None:
        parser.error("cpu/snapshot requires --out")
    root = args.root.resolve()
    if args.phase in ("freeze", "launch"):
        with (root / f"run-preparation/.wave11-{args.phase}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with defer_signals():
                result = (
                    freeze(root, args.cpu_receipt, args.controller_cpu_receipt)
                    if args.phase == "freeze"
                    else launch(root)
                )
    elif args.phase == "cpu":
        result = controller_cpu(args.cpu_receipt, root, args.out)
    elif args.phase == "check":
        result = binding(root)
    elif args.phase == "inspect":
        release = binding(root)
        with node_guard(root, node_for(args.arm)):
            keeper = allocation(root, release, args.arm)
        result = {"status": "allocation_preflight_passed", "arm": args.arm, "node": node_for(args.arm), "keeper": keeper}
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
