# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own a separate one-shot root diagnostic after CPU/archive and exact keeper gates."""

import argparse
import ast
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

import owned_output_handoff as handoff
import root_preparation_validation as validation
from fused_attribution_validation import require_keeper_shape
from fused_attribution_validation_controller import (
    acknowledged_process,
    finish_marked_child,
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from fused_attribution_validation_controller import (
    helper_names as attribution_helper_names,
)
from prepare_root_preparation_diagnostic import parent
from prepare_selected_lr_validation import require_archive
from root_preparation_acceptance import require_cpu
from root_preparation_contract import (
    CHILD_MARKER,
    CONTROLLER_MARKER,
    ENVIRONMENT,
    FROZEN_STATUS,
    NODE,
    PREPARATION,
    PYTHON,
    controller_command,
    locations,
    plan,
    validator_command,
)
from selected_lr_failure_handoff import (
    keeper_guard,
    release_accepted_keeper,
    restore_accepted_keeper,
    same_keeper,
)
from selected_lr_release_owner import candidate_files, read, sha256
from watch_wave5 import write
from wave7_release import idle_training, local_node, node_guard

TESTS = (
    "test_root_preparation_validation.py",
    "test_root_preparation_controller.py",
    "test_fused_attribution_validation_controller.py",
    "test_selected_lr_failure_handoff.py",
)
CPU_STATUS = "root_preparation_controller_cpu_archive_verified_gpu_unissued"
SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual"
)
MANAGER = "recovery_keeper_manager.sh"
TIMEOUT = 12 * validation.COMMAND_TIMEOUT + 2 * validation.PROBE_TIMEOUT + 14400
EXTERNAL = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout", "swanlab", "dotenv"}


def helper_sources(root: Path, directory: Path, cpu: dict) -> dict[str, Path]:
    inherited = Path(cpu["helpers"])
    result = {name: inherited / name for name in cpu["helpers_sha256"]}
    # Existing ownership fixtures copy modules by name, outside flat import discovery.
    pending = [Path(__file__).name, *TESTS, *attribution_helper_names(directory)]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        source = directory / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Root executor helper is absent or aliased: {name}")
        if name in result and sha256(source) != cpu["helpers_sha256"][name]:
            raise ValueError("Root executor attempted to replace accepted candidate helpers")
        result[name] = source
        visited.add(name)
        for node in ast.walk(ast.parse(source.read_bytes())):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                modules = [node.module] if node.module else []
            else:
                continue
            for module in modules:
                head = module.split(".")[0]
                dependency = directory / f"{head}.py"
                if dependency.is_file():
                    pending.append(dependency.name)
                elif head not in EXTERNAL | {"experiment"}:
                    raise ValueError(f"Unbound root executor dependency: {name}: {module}")
    prior = read(root / "run-preparation/fused-continuation-binding-20261118.json")
    manager = directory / MANAGER
    if not manager.is_file():
        manager = Path(prior["helpers"]) / MANAGER
    if manager.is_symlink() or sha256(manager) != prior["helpers_sha256"][MANAGER]:
        raise ValueError("Root executor requires the original serialized keeper manager")
    result[MANAGER] = manager
    return result


def cpu_command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        PYTHON,
        "--with",
        "pytest-timeout",
        "python",
        "-P",
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "--rootdir",
        str(out),
        "-p",
        "no:cacheprovider",
        "-p",
        "root_diagnostic_pytest_guard",
        "--timeout=60",
        "--session-timeout=600",
        "--import-mode=importlib",
        "-m",
        SAFE_MARKERS,
        "-q",
        *(f"tracking/{name}" for name in TESTS),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def controller_cpu(root: Path, candidate_receipt: Path, out: Path) -> dict:
    cpu = require_cpu(root, candidate_receipt)
    candidate = Path(cpu["candidate"])
    if out.exists() or out.is_relative_to(candidate) or candidate.is_relative_to(out):
        raise FileExistsError("Root executor CPU evidence requires a new external directory")
    sources = helper_sources(root, Path(__file__).parent, cpu)
    hashes = {name: sha256(source) for name, source in sources.items()}
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    for name, source in sources.items():
        shutil.copy2(source, helpers / name)
    fixtures = {}
    for path in (candidate / "experiment/configs").glob("*full_lr*.json"):
        name = str(path.relative_to(candidate))
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        fixtures[name] = sha256(target)
    log, archive = out / "controller.log", out / "source.tar"
    code, error = 124, None
    try:
        with log.open("x") as stream:
            code = subprocess.run(
                cpu_command(out),
                cwd=out,
                env=os.environ
                | {
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers))),
                    "UV_OFFLINE": "1",
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
    except (OSError, subprocess.TimeoutExpired) as failure:
        error = f"{type(failure).__name__}: {failure}"
    inventory = {f"tracking/{name}": value for name, value in hashes.items()} | fixtures
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            bundle.add(out / name, arcname=name, recursive=False)
    require_archive(archive, inventory, alias=False)
    if hashes != {name: sha256(source) for name, source in sources.items()}:
        error = "Executor source changed during CPU verification"
    result = {
        "status": CPU_STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate_cpu_receipt": str(candidate_receipt),
        "candidate_cpu_receipt_sha256": sha256(candidate_receipt),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "fixtures_sha256": fixtures,
        "command": cpu_command(out),
        "cwd": str(out),
        "exit_code": code,
        "cuda_visible_devices": "",
        **{name: str(path) for name, path in (("log", log), ("archive", archive), ("tests", out / "tests.xml"))},
        **{
            f"{name}_sha256": sha256(path) if path.is_file() else None
            for name, path in (("log", log), ("archive", archive), ("tests", out / "tests.xml"))
        },
    }
    if error:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CPU_STATUS:
        raise RuntimeError(f"Root executor CPU failure preserved at {out}")
    try:
        return require_controller_cpu(root, out / "receipt.json", candidate_receipt)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, tarfile.TarError) as failure:
        result.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        write(out / "receipt.json", result)
        raise RuntimeError(f"Root executor final acceptance failure preserved at {out}") from failure


def require_controller_cpu(root: Path, path: Path, candidate_receipt: Path) -> dict:
    raw, cpu = path.read_bytes(), require_cpu(root, candidate_receipt)
    receipt = json.loads(raw)
    helpers = Path(receipt["helpers"])
    require_helper_bytes(helpers, receipt["helpers_sha256"])
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    counts = {
        name: sum(int(suite.attrib[name]) for suite in suites) for name in ("tests", "failures", "errors", "skipped")
    }
    cases = [case for suite in suites for case in suite.findall("testcase")]
    expected_modules = {Path(name).stem for name in TESTS}
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root)
        or receipt["candidate_cpu_receipt"] != str(candidate_receipt)
        or receipt["candidate_cpu_receipt_sha256"] != sha256(candidate_receipt)
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or receipt["cuda_visible_devices"] != ""
        or "error" in receipt
        or helpers != path.parent / "tracking"
        or helpers != helpers.resolve()
        or receipt["command"] != cpu_command(helpers.parent)
        or receipt["cwd"] != str(helpers.parent)
        or receipt["helpers_sha256"]
        != {name: sha256(source) for name, source in helper_sources(root, helpers, cpu).items()}
        or any(receipt["helpers_sha256"].get(name) != value for name, value in cpu["helpers_sha256"].items())
        or receipt["fixtures_sha256"]
        != {
            str(source.relative_to(Path(cpu["candidate"]))): sha256(source)
            for source in (Path(cpu["candidate"]) / "experiment/configs").glob("*full_lr*.json")
        }
        or any(
            sha256(helpers.parent / name) != value or cpu["files_sha256"][name] != value
            for name, value in receipt["fixtures_sha256"].items()
        )
        or any(
            Path(receipt[name]) != helpers.parent / filename or receipt[f"{name}_sha256"] != sha256(Path(receipt[name]))
            for name, filename in (("log", "controller.log"), ("archive", "source.tar"), ("tests", "tests.xml"))
        )
        or counts["tests"] == 0
        or counts["tests"] != len(cases)
        or {case.attrib["classname"].split(".")[-1] for case in cases} != expected_modules
        or any(case.find(name) is not None for case in cases for name in ("failure", "error", "skipped"))
        or any(counts[name] for name in ("failures", "errors", "skipped"))
    ):
        raise ValueError("Root executor CPU/archive acceptance differs")
    require_archive(
        Path(receipt["archive"]),
        {f"tracking/{name}": value for name, value in receipt["helpers_sha256"].items()} | receipt["fixtures_sha256"],
        alias=False,
    )
    if path.read_bytes() != raw:
        raise ValueError("Root executor receipt changed during authentication")
    return receipt


def inventory(frozen: dict) -> dict:
    prepared = read(Path(frozen["candidate"]) / PREPARATION)
    return (
        frozen["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {
            PREPARATION: frozen["preparation_sha256"],
            "candidate-cpu-receipt.json": frozen["cpu_receipt_sha256"],
            "controller-cpu-receipt.json": frozen["controller_cpu_receipt_sha256"],
        }
        | {f"tracking/{name}": value for name, value in frozen["helpers_sha256"].items()}
    )


def freeze(root: Path, candidate_receipt: Path, controller_receipt: Path) -> dict:
    cpu = require_cpu(root, candidate_receipt)
    executor = require_controller_cpu(root, controller_receipt, candidate_receipt)
    candidate = Path(cpu["candidate"])
    prepared = validation.source(candidate, root)
    _, predecessor = parent(root)
    sources = helper_sources(root, Path(__file__).parent, cpu)
    hashes = {name: sha256(source) for name, source in sources.items()}
    path, helpers, state, work, receipt = locations(root)
    archive = root / "run-preparation/source-root-preparation-20261118.tar"
    if predecessor != prepared["allocation_predecessor"] or hashes != executor["helpers_sha256"]:
        raise ValueError("Root freeze needs exact accepted executor bytes and predecessor")
    with (root / "run-preparation/.root-preparation-freeze.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(target.exists() for target in (path, helpers.parent, archive, state, work, receipt)):
            raise FileExistsError("Root diagnostic identity already has evidence; preserve it")
        helpers.mkdir(parents=True)
        for name, source in sources.items():
            shutil.copy2(source, helpers / name)
        frozen = {
            "status": FROZEN_STATUS,
            "root": str(root),
            "candidate": str(candidate),
            "node": NODE,
            "python": PYTHON,
            "environment": ENVIRONMENT,
            "files_sha256": cpu["files_sha256"],
            "preparation_sha256": sha256(candidate / PREPARATION),
            "cpu_receipt": str(candidate_receipt),
            "cpu_receipt_sha256": sha256(candidate_receipt),
            "controller_cpu_receipt": str(controller_receipt),
            "controller_cpu_receipt_sha256": sha256(controller_receipt),
            "helpers": str(helpers),
            "helpers_sha256": hashes,
            "archive": str(archive),
            "data_receipt_sha256": prepared["data_receipt_sha256"],
            "allocation_predecessor": predecessor,
            "plan": plan(candidate, root),
            "commands": plan(candidate, root)["commands"],
            "validation_timeout": TIMEOUT,
            "policy": "One-shot diagnostic only; no production authorization or optimizer-quality claim.",
        }
        with tarfile.open(archive, "x") as bundle:
            for name in inventory(frozen):
                source = (
                    helpers / Path(name).name
                    if name.startswith("tracking/")
                    else (
                        candidate_receipt
                        if name == "candidate-cpu-receipt.json"
                        else controller_receipt if name == "controller-cpu-receipt.json" else candidate / name
                    )
                )
                bundle.add(source, arcname=name, recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
        frozen["archive_sha256"] = sha256(archive)
        require_archive(archive, inventory(frozen), alias=True)
        if (
            require_cpu(root, candidate_receipt) != cpu
            or require_controller_cpu(root, controller_receipt, candidate_receipt) != executor
            or hashes != {name: sha256(source) for name, source in sources.items()}
            or parent(root)[1] != predecessor
        ):
            raise ValueError("Root evidence changed while freezing")
        write(path, frozen)
    return frozen


def binding(root: Path) -> dict:
    path, helpers, _, _, _ = locations(root)
    raw = path.read_bytes()
    frozen = json.loads(raw)
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    for name in frozen["helpers_sha256"]:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError("Root executor imported mutable helpers")
    if Path(__file__).resolve() != helpers / Path(__file__).name:
        raise RuntimeError("Root executor must execute its own frozen bytes")
    cpu = require_cpu(root, Path(frozen["cpu_receipt"]))
    executor = require_controller_cpu(root, Path(frozen["controller_cpu_receipt"]), Path(frozen["cpu_receipt"]))
    candidate = Path(cpu["candidate"])
    prepared = validation.source(candidate, root)
    if (
        frozen["status"] != FROZEN_STATUS
        or frozen["root"] != str(root)
        or frozen["candidate"] != str(candidate)
        or frozen["node"] != NODE
        or frozen["python"] != PYTHON
        or frozen["environment"] != ENVIRONMENT
        or frozen["helpers"] != str(helpers)
        or frozen["helpers_sha256"] != executor["helpers_sha256"]
        or frozen["files_sha256"] != candidate_files(candidate)
        or frozen["preparation_sha256"] != sha256(candidate / PREPARATION)
        or frozen["cpu_receipt_sha256"] != sha256(Path(frozen["cpu_receipt"]))
        or frozen["controller_cpu_receipt_sha256"] != sha256(Path(frozen["controller_cpu_receipt"]))
        or frozen["allocation_predecessor"] != prepared["allocation_predecessor"]
        or frozen["allocation_predecessor"] != parent(root)[1]
        or frozen["plan"] != plan(candidate, root)
        or frozen["commands"] != frozen["plan"]["commands"]
        or frozen["validation_timeout"] != TIMEOUT
        or frozen["archive"] != str(root / "run-preparation/source-root-preparation-20261118.tar")
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
    ):
        raise ValueError("Root frozen executor/source/plan/predecessor differs")
    require_archive(Path(frozen["archive"]), inventory(frozen), alias=True)
    if path.read_bytes() != raw:
        raise ValueError("Root binding changed during authentication")
    return frozen


def probe(root: Path, expected: dict) -> dict:
    if parent(root)[1] != expected:
        raise RuntimeError("Root terminal predecessor changed")
    current = handoff.keeper_observation()
    if not same_keeper(current, expected["restored_keeper"]):
        raise RuntimeError("Root diagnostic exact restored keeper changed")
    if any(handoff.observed(pid) is not None for pid in (expected["controller_pid"], *expected["child_pids"])):
        raise RuntimeError("Root original replay controller or worker remains live")
    intent = read(root / "controllers/saved-ns-replay-20261118/controller.json")
    marked = handoff.marker_processes("OUTPUT_MARIN_SAVED_NS_REPLAY_CONTROLLER", intent["marker"])
    if not set(marked).issubset({int(pid) for pid in current["generation"]}):
        raise RuntimeError("Root predecessor marker has nonkeeper descendants")
    state = read(root / "run-preparation/saved-ns-replay-state-20261118.json")
    if any(
        handoff.marker_processes("OUTPUT_MARIN_SAVED_NS_REPLAY_CHILD", execution["marker"])
        for execution in state["executions"]
    ):
        raise RuntimeError("Root predecessor has live replay children")
    return {"status": "saved_ns_exact_restored_keeper_live_handoff_verified", "keeper": current}


def start(root: Path) -> dict:
    frozen = binding(root)
    path, helpers, state, work, receipt = locations(root)
    intent_path, log = helpers.parent / "controller.json", helpers.parent / "controller.log"
    if local_node() != NODE or any(os.environ.get(key) != value for key, value in ENVIRONMENT.items()):
        raise RuntimeError("Root executor start requires its exact node and eight-GPU environment")
    with (helpers.parent / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(target.exists() for target in (intent_path, log, state, work, receipt)):
            raise FileExistsError("Root diagnostic execution identity already has intent or evidence")
        with keeper_guard():
            idle_training()
            probe(root, frozen["allocation_predecessor"])
        intent = {
            "status": "starting",
            "binding_sha256": sha256(path),
            "command": controller_command(root),
            "marker": uuid4().hex,
            "allocation_predecessor": frozen["allocation_predecessor"],
        }
        write(intent_path, intent)
        with log.open("x") as stream:
            child = subprocess.Popen(
                intent["command"],
                cwd=helpers,
                env=os.environ
                | ENVIRONMENT
                | {CONTROLLER_MARKER: intent["marker"], "PYTHONPATH": str(helpers), "PYTHONDONTWRITEBYTECODE": "1"},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        try:
            process = acknowledged_process(
                child, intent["command"], helpers, CONTROLLER_MARKER, intent["marker"], ENVIRONMENT
            )
        except BaseException as error:
            intent.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
            generations = {}
            try:
                finish_marked_child(child, intent["marker"], generations, CONTROLLER_MARKER)
                intent.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
            except BaseException as cleanup_error:
                intent["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                raise
            finally:
                intent.update(child_generation=generations, log_sha256=sha256(log))
                write(intent_path, intent)
            raise
        intent.update(status="acknowledged", pid=child.pid, start_ticks=process["start_ticks"], generation=process)
        write(intent_path, intent)
    return intent


def bounded_validator(root: Path, frozen: dict, state: dict) -> int:
    _, helpers, state_path, _, _ = locations(root)
    candidate = Path(frozen["candidate"])
    command, marker = validator_command(root, candidate), uuid4().hex
    log, generations = helpers.parent / "validator.log", {}
    with log.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=candidate,
            env=os.environ
            | ENVIRONMENT
            | {
                CHILD_MARKER: marker,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": os.pathsep.join((str(candidate), str(helpers))),
            },
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process = acknowledged_process(child, command, candidate, CHILD_MARKER, marker, ENVIRONMENT)
            state.update(
                status="validating",
                spawn_status="acknowledged",
                command=command,
                child_pid=child.pid,
                child_start_ticks=process["start_ticks"],
                child_marker=marker,
                log=str(log),
            )
            write(state_path, state)
            return child.wait(timeout=TIMEOUT)
        finally:
            try:
                finish_marked_child(child, marker, generations, CHILD_MARKER)
                state.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
            except BaseException as error:
                state["cleanup_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                state.update(child_generation=generations, log_sha256=sha256(log))
                write(state_path, state)


def validate(root: Path) -> dict:
    frozen = binding(root)
    path, helpers, state_path, _, receipt = locations(root)
    intent_path = helpers.parent / "controller.json"
    deadline = time.monotonic() + 30
    intent = read(intent_path)
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.05)
        intent = read(intent_path)
    own = process_context(
        os.getpid(),
        controller_command(root),
        helpers,
        CONTROLLER_MARKER,
        os.environ.get(CONTROLLER_MARKER, ""),
        ENVIRONMENT,
    )
    require_acknowledged_controller(intent, own)
    if (
        intent["binding_sha256"] != sha256(path)
        or intent["command"] != controller_command(root)
        or intent["marker"] != os.environ.get(CONTROLLER_MARKER)
        or local_node() != NODE
        or state_path.exists()
        or intent["allocation_predecessor"] != frozen["allocation_predecessor"]
    ):
        raise RuntimeError("Root executor lacks exact acknowledged ownership")
    state = {
        "status": "preflight",
        "binding_sha256": sha256(path),
        "controller_pid": own["pid"],
        "controller_start_ticks": own["start_ticks"],
        "node": NODE,
        "candidate": frozen["candidate"],
        "allocation_predecessor": frozen["allocation_predecessor"],
    }
    write(state_path, state)
    signals = []
    try:
        with node_guard(root, NODE) as signals, keeper_guard() as manager_lock:
            idle_training()
            accepted = probe(root, frozen["allocation_predecessor"])["keeper"]
            subprocess.run(
                [PYTHON, "-m", "experiment.bootstrap", "--check-only"],
                cwd=Path(frozen["candidate"]),
                env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                check=True,
                timeout=240,
            )
            if binding(root) != frozen:
                raise RuntimeError("Root binding changed before keeper transfer")
            state["handoff"] = probe(root, frozen["allocation_predecessor"])
            try:
                release_accepted_keeper(accepted, state, state_path, manager_lock)
                code = bounded_validator(root, frozen, state)
                state["exit_code"] = code
                if type(code) is not int or code != 0:
                    raise RuntimeError("Root diagnostic validator failed")
                result = validation.reconstruct(root, Path(frozen["candidate"]), receipt)
                state.update(
                    status="diagnostic_complete",
                    validation_sha256=sha256(receipt),
                    intervention_accepted=result["analysis"]["diagnostic_intervention_accepted"],
                )
            finally:
                if state.get("keeper_release_attempted"):
                    if (
                        state.get("keeper_release_complete") is not True
                        or "cleanup_error" in state
                        or (state.get("child_marker") and handoff.marker_processes(CHILD_MARKER, state["child_marker"]))
                    ):
                        state["keeper_restore"] = "refused: incomplete transfer or live descendants"
                        raise RuntimeError("Root incomplete transfer prevents keeper restoration")
                    restore_accepted_keeper(manager_lock)
                    restored = handoff.keeper_observation()
                    require_keeper_shape(restored)
                    if restored["devices"] != accepted["devices"]:
                        raise RuntimeError("Root restored keeper physical GPUs differ")
                    state.update(restored_keeper=restored, keeper_restore=0)
    except BaseException as error:
        state.update(status="needs_attention", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if signals:
            state["deferred_signals"] = signals
        write(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cpu", "freeze", "check", "start", "validate"), required=True)
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--controller-cpu-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase in ("cpu", "freeze") and args.cpu_receipt is None:
        parser.error("CPU/freeze require --cpu-receipt")
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU requires --out")
    if args.phase == "freeze" and args.controller_cpu_receipt is None:
        parser.error("Freeze requires --controller-cpu-receipt")
    if args.phase == "cpu":
        result = controller_cpu(root, args.cpu_receipt.resolve(), args.out.resolve())
    elif args.phase == "freeze":
        result = freeze(root, args.cpu_receipt.resolve(), args.controller_cpu_receipt.resolve())
    elif args.phase == "check":
        result = binding(root)
    elif args.phase == "start":
        result = start(root)
    else:
        result = validate(root)
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
