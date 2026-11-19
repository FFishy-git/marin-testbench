# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wait for exact original owners, then invoke their sole issued read-only terminal."""

import argparse
import ast
import fcntl
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

import ns_training_path_contract as contract
import ns_training_path_gpu_controller as controller
import owned_output_handoff as handoff
from fused_attribution_validation_controller import (
    acknowledged_process,
    finish_marked_child,
    process_context,
    require_acknowledged_controller,
    require_helper_bytes,
)
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import SAFE_MARKERS, read, sha256
from watch_wave5 import write

BINDING_SHA256 = "20133ef8358e51cc97e83b866063843ccc10a314ac2b72aef5b7a8b4354f0064"
INTENT_SHA256 = "7b24631110346132bc10b93e539faf07ba071422582715f12f5f060d496213aa"
ARCHIVE_SHA256 = "6fc0ec29d5075f654210b54b61d4c2e1b2440eb35555cb1c5fa636388bccd5ad"
WORK = "run-preparation/ns-training-path-terminal-followup-20261119"
MARKER = "OUTPUT_MARIN_NS_TRAINING_PATH_FOLLOWUP"
CHILD_MARKER = "OUTPUT_MARIN_NS_TRAINING_PATH_TERMINAL_CHILD"
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONFAULTHANDLER": "1"}
INTERVAL = 900
TERMINAL_TIMEOUT = 14400
CPU_STATUS = "ns_training_path_read_only_followup_cpu_archive_verified"
TEST_MODULE = "test_ns_training_path_followup"
NEW_HELPERS = ("ns_training_path_followup.py", f"{TEST_MODULE}.py")
TEST_COUNT = 19
SCOPE = (
    "One read-only original-generation waiter and original issued canonical terminal verifier. "
    "No keeper transfer, GPU execution, new tensor reader, numerical waiver, production release, "
    "quality/efficiency claim or SwanLab authority. Rejected cases and consumed terminal claims "
    "remain retained. Check exact owners every 900 seconds; do not restart on observation timeout."
)


def frozen(root: Path) -> dict:
    """Authenticate original immutable execution sources, without replaying tensors."""
    binding, helpers, _, _, _ = contract.locations(root)
    if root != root.resolve(strict=True) or sha256(binding) != BINDING_SHA256:
        raise ValueError("Training-path terminal follow-up original binding changed")
    saved = read(binding)
    intent_path = helpers.parent / "controller.json"
    if sha256(intent_path) != INTENT_SHA256:
        raise ValueError("Training-path terminal follow-up original acknowledgment changed")
    intent = read(intent_path)
    if (
        saved["root"] != str(root)
        or saved["helpers"] != str(helpers)
        or saved["status"] != contract.FROZEN_STATUS
        or saved["archive_sha256"] != ARCHIVE_SHA256
        or sha256(Path(saved["archive"])) != ARCHIVE_SHA256
        or intent["status"] != "acknowledged"
        or intent["binding_sha256"] != BINDING_SHA256
        or intent["command"] != saved["controller_command"]
        or sha256(Path(controller.__file__)) != saved["helpers_sha256"]["ns_training_path_gpu_controller.py"]
        or {path.name for path in helpers.iterdir() if path.name != "__pycache__"} != set(saved["helpers_sha256"])
    ):
        raise ValueError("Training-path terminal follow-up issued source or intent differs")
    require_helper_bytes(helpers, saved["helpers_sha256"])
    return saved


def handle(expected: dict, cwd: Path, marker_name: str, marker: str, environment: dict) -> dict:
    """Inspect one exact original owner without importing an unissued queue."""
    actual = handoff.observed(expected["pid"])
    if actual is None:
        return {"status": "absent", "expected": expected}
    if any(actual[key] != expected[key] for key in ("pid", "start_ticks", "command", "group", "session")):
        raise RuntimeError("Training-path terminal follow-up original owner generation was substituted")
    if actual["status"] in ("Z", "X", "x"):
        return {"status": "exited_handle_present", "observed": actual}
    verified = process_context(expected["pid"], expected["command"], cwd, marker_name, marker, environment)
    if verified["start_ticks"] != expected["start_ticks"]:
        raise RuntimeError("Training-path terminal follow-up owner changed during observation")
    return {"status": "live_exact_generation", "observed": verified}


def readiness(root: Path, binding: dict) -> dict:
    """Wait on exact handles; a completed receipt or zombie alone cannot authorize work."""
    _, helpers, state_path, _, result_path = contract.locations(root)
    intent, state = read(helpers.parent / "controller.json"), read(state_path)
    child, original = state["child_generation"], intent["generation"]
    if (
        intent["status"] != "acknowledged"
        or intent["binding_sha256"] != BINDING_SHA256
        or state["binding_sha256"] != BINDING_SHA256
        or state["controller_pid"] != intent["pid"]
        or state["controller_start_ticks"] != intent["start_ticks"]
        or original["pid"] != intent["pid"]
        or original["start_ticks"] != intent["start_ticks"]
        or original["command"] != binding["controller_command"]
        or state["command"] != binding["validator_command"]
        or child["pid"] != state["child_pid"]
        or child["start_ticks"] != state["child_start_ticks"]
        or child["parent"] != original["pid"]
        or child["command"] != binding["validator_command"]
    ):
        raise ValueError("Training-path terminal follow-up original owner evidence differs")
    owners = {
        "controller": handle(
            original,
            helpers,
            contract.CONTROLLER_MARKER,
            intent["marker"],
            binding["controller_environment"],
        ),
        "validator": handle(
            child,
            Path(binding["candidate"]),
            contract.CHILD_MARKER,
            state["child_marker"],
            binding["environment"],
        ),
    }
    result = read(result_path)
    progress = {
        "completed_commands": len(result.get("gpu_commands", [])),
        "case_decisions": len(result.get("results", [])),
        "result_time": result.get("time"),
    }
    if any(owner["status"] != "absent" for owner in owners.values()):
        return {
            "status": "waiting_for_original_owners",
            "ready": False,
            "owners": owners,
            "saved_state": state["status"],
            **progress,
        }
    terminal, attempt, outcome, _ = controller.terminal_paths(root, intent)
    if any(path.exists() or path.is_symlink() for path in (terminal, attempt, outcome)):
        raise FileExistsError("Training-path original terminal already has a one-shot claim or outcome")
    if (
        state["status"] != "validation_complete"
        or any(
            type(state.get(key)) is not int or state[key] != 0
            for key in ("exit_code", "cleanup_exit_code", "keeper_restore")
        )
        or state.get("marked_descendants_absent") is not True
        or state.get("keeper_release_complete") is not True
        or not state.get("restored_keeper")
        or any(key in state for key in ("error", "cleanup_error", "deferred_signals"))
    ):
        raise RuntimeError("Training-path original owners exited without complete cleanup/restoration")
    if (
        result["status"] != "ns_training_path_actual_config_gates_reconstructed_not_production"
        or result["binding_sha256"] != BINDING_SHA256
        or state["result_sha256"] != sha256(result_path)
        or progress["completed_commands"] != len(binding["commands"])
        or progress["case_decisions"] != len(binding["plan"]["trials"])
        or any(state[key] != result[key] for key in ("accepted_cases", "rejected_cases", "all_cases_accepted"))
        or "error" in result
    ):
        raise ValueError("Training-path original completed results or retained decisions differ")
    descendants = handoff.marker_processes(contract.CHILD_MARKER, state["child_marker"])
    for entry in result["gpu_commands"]:
        descendants.update(handoff.marker_processes(contract.COMMAND_MARKER, entry["marker"]))
    if descendants:
        return {
            "status": "waiting_for_original_descendants",
            "ready": False,
            "owners": owners,
            "marked_pids": sorted(descendants),
            **progress,
        }
    return {"status": "terminal_ready", "ready": True, "owners": owners, "terminal": str(terminal), **progress}


def cpu_command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        contract.PYTHON,
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
        "--import-mode=prepend",
        "-m",
        SAFE_MARKERS,
        "-q",
        str(out / "tracking" / NEW_HELPERS[-1]),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def accepted(root: Path, path: Path) -> dict:
    """Authenticate the entire waiter archive and its issued dependency bytes."""
    binding, raw = frozen(root), path.read_bytes()
    receipt = read(path)
    directory, helpers = path.parent, path.parent / "tracking"
    files = receipt["helpers_sha256"]
    cases = list(ET.parse(receipt["tests"]).getroot().iter("testcase"))
    suites = list(ET.parse(receipt["tests"]).getroot().iter("testsuite"))
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root)
        or receipt["binding_sha256"] != BINDING_SHA256
        or receipt["original_intent_sha256"] != INTENT_SHA256
        or receipt["original_archive_sha256"] != ARCHIVE_SHA256
        or receipt["environment"] != ENVIRONMENT
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or "error" in receipt
        or receipt["command"] != cpu_command(directory)
        or receipt["helpers"] != str(helpers)
        or receipt["scope"] != SCOPE
        or {name: files.get(name) for name in binding["helpers_sha256"]} != binding["helpers_sha256"]
        or set(files) - set(binding["helpers_sha256"]) != set(NEW_HELPERS)
        or {source.name for source in helpers.iterdir()} != set(files)
        or len(cases) != TEST_COUNT
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {TEST_MODULE}
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != TEST_COUNT
        or sum(int(suite.attrib["tests"]) for suite in suites) != TEST_COUNT
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
        or any(
            receipt[name] != str(directory / filename) or receipt[f"{name}_sha256"] != sha256(directory / filename)
            for name, filename in (("archive", "source.tar"), ("tests", "tests.xml"), ("log", "cpu.log"))
        )
        or Path(__file__).resolve() != helpers / NEW_HELPERS[0]
    ):
        raise ValueError("Training-path read-only follow-up CPU/source acceptance differs")
    require_helper_bytes(helpers, files)
    require_archive(
        Path(receipt["archive"]), {f"tracking/{name}": digest for name, digest in files.items()}, alias=False
    )
    for name in files:
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and Path(module.__file__).resolve() != helpers / name:
            raise RuntimeError(f"Training-path terminal waiter imported unarchived helper: {name}")
    if path.read_bytes() != raw:
        raise ValueError("Training-path read-only follow-up receipt changed during checking")
    return receipt


def terminal_phase(root: Path, binding: dict, work: Path, state: dict) -> None:
    """Claim no output itself; invoke the original verifier only after rechecking readiness."""
    if state.get("terminal_command"):
        raise FileExistsError("Training-path follow-up already issued its one terminal child")
    observation = readiness(root, binding)
    if not observation["ready"]:
        raise RuntimeError("Training-path original owner or descendant remains live")
    helpers, output = Path(binding["helpers"]), Path(observation["terminal"])
    argv = [
        contract.PYTHON,
        "-P",
        str(helpers / "ns_training_path_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "terminal",
        "--out",
        str(output),
    ]
    marker, generations, child = uuid4().hex, {}, None
    log = work / "terminal.log"
    state.update(status="reconstructing_terminal", terminal_command=argv, terminal_marker=marker, terminal=str(output))
    write(work / "state.json", state)
    environment = os.environ | ENVIRONMENT | {"PYTHONPATH": str(helpers), CHILD_MARKER: marker}
    environment.pop(MARKER, None)
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(
                argv,
                cwd=helpers,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            own = acknowledged_process(child, argv, helpers, CHILD_MARKER, marker, ENVIRONMENT)
            state.update(terminal_pid=own["pid"], terminal_generation=own)
            write(work / "state.json", state)
            state["terminal_exit_code"] = child.wait(timeout=TERMINAL_TIMEOUT)
            if type(state["terminal_exit_code"]) is not int or state["terminal_exit_code"] != 0:
                raise RuntimeError("Training-path issued one-shot terminal verifier failed; claim remains consumed")
        receipt = read(output)
        original = read(helpers.parent / "controller.json")
        _, attempt, outcome, _ = controller.terminal_paths(root, original)
        finished = read(outcome)
        if (
            receipt["status"] != controller.TERMINAL_STATUS
            or receipt["binding_sha256"] != BINDING_SHA256
            or finished["status"] != "complete"
            or finished["output"] != str(output)
            or finished["terminal_sha256"] != sha256(output)
            or finished["reconstruction_intent_sha256"] != sha256(attempt)
            or receipt["reconstruction_intent_sha256"] != sha256(attempt)
            or receipt["production_authorized"] is not False
        ):
            raise ValueError("Training-path issued terminal outcome or scope differs")
        state.update(
            status="terminal_reconstructed",
            terminal_sha256=sha256(output),
            terminal_outcome_sha256=sha256(outcome),
            **{key: receipt[key] for key in ("accepted_cases", "rejected_cases", "all_cases_accepted")},
        )
    finally:
        if child is not None:
            finish_marked_child(child, marker, generations, CHILD_MARKER)
            state.update(terminal_cleanup_exit_code=child.returncode, terminal_descendants_absent=True)
        state.update(
            terminal_generations=generations,
            terminal_log=str(log),
            terminal_log_sha256=sha256(log) if log.is_file() else None,
        )
        write(work / "state.json", state)


def command(root: Path, cpu: Path) -> list[str]:
    return [
        contract.PYTHON,
        "-P",
        str(cpu.parent / "tracking" / NEW_HELPERS[0]),
        "--root",
        str(root),
        "--cpu",
        str(cpu),
        "--phase",
        "execute",
    ]


def start(root: Path, cpu: Path) -> dict:
    accepted(root, cpu)
    binding, work = frozen(root), root / WORK
    readiness(root, binding)
    if work.exists():
        raise FileExistsError("Training-path terminal follow-up already has one-shot evidence")
    work.mkdir()
    intent = {
        "status": "starting",
        "command": command(root, cpu),
        "marker": uuid4().hex,
        "runner_sha256": sha256(Path(__file__)),
        "cpu_receipt_sha256": sha256(cpu),
        "binding_sha256": BINDING_SHA256,
        "interval": INTERVAL,
        "scope": SCOPE,
    }
    write(work / "intent.json", intent)
    child = None
    try:
        with (work / "followup.log").open("x") as stream:
            child = subprocess.Popen(
                intent["command"],
                cwd=work,
                env=os.environ | ENVIRONMENT | {"PYTHONPATH": str(cpu.parent / "tracking"), MARKER: intent["marker"]},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        own = acknowledged_process(child, intent["command"], work, MARKER, intent["marker"], ENVIRONMENT)
        intent.update(status="acknowledged", pid=own["pid"], start_ticks=own["start_ticks"], generation=own)
    except BaseException as failure:
        intent.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        if child is not None:
            finish_marked_child(child, intent["marker"], {}, MARKER)
        raise
    finally:
        write(work / "intent.json", intent)
    return intent


def execute(root: Path, cpu: Path) -> dict:
    accepted(root, cpu)
    binding, work = frozen(root), root / WORK
    intent, deadline = read(work / "intent.json"), time.monotonic() + 30
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.01)
        intent = read(work / "intent.json")
    own = process_context(os.getpid(), command(root, cpu), work, MARKER, intent["marker"], ENVIRONMENT)
    require_acknowledged_controller(intent, own)
    if intent["runner_sha256"] != sha256(Path(__file__)) or intent["cpu_receipt_sha256"] != sha256(cpu):
        raise ValueError("Training-path terminal follow-up acknowledged source differs")
    state = {
        "status": "waiting_for_original_owners",
        "pid": own["pid"],
        "start_ticks": own["start_ticks"],
        "binding_sha256": BINDING_SHA256,
        "cpu_receipt_sha256": sha256(cpu),
        "runner_sha256": intent["runner_sha256"],
        "interval": INTERVAL,
        "scope": SCOPE,
    }
    with (work / "execution.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            index = 0
            while True:
                observation = readiness(root, frozen(root))
                write(work / f"observation-{index:06d}.json", observation)
                state.update(status=observation["status"], original_observation=observation, observations=index + 1)
                write(work / "state.json", state)
                if observation["ready"]:
                    terminal_phase(root, binding, work, state)
                    break
                index += 1
                time.sleep(INTERVAL)
        except BaseException as failure:
            state.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
            raise
        finally:
            write(work / "state.json", state)
    return state


def cpu_acceptance(root: Path, out: Path) -> dict:
    """Stage all issued helpers unchanged; test and archive only the new read-only waiter."""
    binding = frozen(root)
    if out.exists() or out.is_relative_to(Path(__file__).resolve().parents[1]):
        raise FileExistsError("Training-path follow-up CPU output must be unused and external")
    sources = {name: Path(binding["helpers"]) / name for name in binding["helpers_sha256"]}
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        for node in ast.walk(ast.parse(sources[name].read_bytes(), filename=name)):
            modules = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("Training-path follow-up import escapes its complete issued helper closure")
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    hashes = {name: sha256(path) for name, path in sources.items()}
    for name, path in sources.items():
        shutil.copy2(path, helpers / name)
    log = out / "cpu.log"
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(out),
                cwd=out,
                env=os.environ | ENVIRONMENT | {"UV_OFFLINE": "1", "PYTHONPATH": str(helpers)},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "Training-path follow-up CPU acceptance timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    suites = list(ET.parse(out / "tests.xml").getroot().iter("testsuite")) if (out / "tests.xml").is_file() else []
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if (
        len(cases) != TEST_COUNT
        or sum(int(suite.attrib["tests"]) for suite in suites) != TEST_COUNT
        or any(int(suite.attrib[key]) for suite in suites for key in ("failures", "errors", "skipped"))
        or any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
    ):
        error = error or "Training-path follow-up CPU XML does not contain the complete passing suite"
    archive = out / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in hashes:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    receipt = {
        "status": CPU_STATUS if type(code) is int and code == 0 and error is None else "needs_attention",
        "root": str(root),
        "exit_code": code,
        "binding_sha256": BINDING_SHA256,
        "original_intent_sha256": INTENT_SHA256,
        "original_archive_sha256": ARCHIVE_SHA256,
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "environment": ENVIRONMENT,
        "command": cpu_command(out),
        "scope": SCOPE,
        **{
            key: value
            for name, path in (("archive", archive), ("tests", out / "tests.xml"), ("log", log))
            for key, value in ((name, str(path)), (f"{name}_sha256", sha256(path) if path.is_file() else None))
        },
    }
    if hashes != {name: sha256(path) for name, path in sources.items()}:
        receipt.update(status="needs_attention", error="Follow-up sources changed during CPU acceptance")
    if error:
        receipt.update(status="needs_attention", error=error)
    require_helper_bytes(helpers, hashes)
    require_archive(archive, {f"tracking/{name}": digest for name, digest in hashes.items()}, alias=False)
    write(out / "receipt.json", receipt)
    if receipt["status"] != CPU_STATUS:
        raise RuntimeError("Training-path follow-up CPU failure preserved")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cpu", "check", "start", "execute"))
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in ENVIRONMENT.items()):
        raise RuntimeError("Training-path read-only follow-up requires fixed hidden CUDA")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("--out is required for CPU acceptance")
        result = cpu_acceptance(root, args.out.resolve())
    else:
        if args.cpu is None:
            parser.error("--cpu is required for archived follow-up")
        cpu = args.cpu.resolve(strict=True)
        result = (
            accepted(root, cpu) if args.phase == "check" else (start if args.phase == "start" else execute)(root, cpu)
        )
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
