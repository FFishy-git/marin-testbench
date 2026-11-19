# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wait for original rational restoration, then issue one accepted NS diagnostic."""

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

import ns_budget_gpu_owner as owner
from fused_attribution_validation_controller import acknowledged_process, finish_marked_child, require_helper_bytes
from root_start_gpu_owner import require_cpu_archive
from watch_wave5 import write

PARENT = "run-preparation/ns-budget-gpu-owner-cpu/20261119T0003/receipt.json"
PARENT_SHA256 = "18e0b21734ebdebca8934319753fc6dbed37e5e565c5e28df5d0d1f453f81c69"
WORK = "run-preparation/ns-budget-queue-20261119"
MARKER = "OUTPUT_MARIN_NS_BUDGET_QUEUE"
INTERVAL = 900
PHASE_TIMEOUT = 600
TERMINAL_TIMEOUT = 14400
TERMINAL = "run-preparation/ns-budget-terminal-independent-20261119.json"
CPU_STATUS = "ns_budget_queue_cpu_archive_verified_gpu_unissued"
NEW_HELPERS = ("ns_budget_queue.py", "test_ns_budget_queue.py")
SCOPE = (
    "One original-owner-gated NS diagnostic freeze/check/start and issued terminal reconstruction. "
    "No keeper operation by the queue, "
    "numerical gate waiver, continuation, training quality, production or SwanLab acceptance."
)


def parent(root: Path) -> dict:
    path = root / PARENT
    if sha256(path) != PARENT_SHA256:
        raise ValueError("NS queue accepted owner receipt changed")
    return owner.require_cpu(root, path)


def sha256(path: Path) -> str:
    return owner.sha256(path)


def helper_sources(root: Path) -> dict[str, Path]:
    accepted = parent(root)
    sources = {name: Path(accepted["helpers"]) / name for name in accepted["helpers_sha256"]}
    sources.update({name: Path(__file__).resolve().parent / name for name in NEW_HELPERS})
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    for name in NEW_HELPERS:
        source = sources[name]
        if source.is_symlink() or not source.is_file():
            raise ValueError("NS queue helper is absent or aliased")
        for node in ast.walk(ast.parse(source.read_bytes(), filename=name)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else []
            )
            if any(
                f"{module.split('.')[0]}.py" not in sources and module.split(".")[0] not in external
                for module in modules
            ):
                raise ValueError("NS queue import escapes its accepted closure")
    return sources


def cpu_command(out: Path) -> list[str]:
    return [
        "/opt/conda/envs/torch-base/bin/uv",
        "run",
        "--offline",
        "--no-project",
        "--python",
        owner.PYTHON,
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
        owner.preparation.SAFE_MARKERS,
        "-q",
        str(out / "tracking" / NEW_HELPERS[-1]),
        "--junitxml",
        str(out / "tests.xml"),
    ]


def accepted(root: Path, path: Path) -> dict:
    raw = path.read_bytes()
    receipt = require_cpu_archive(root, path, CPU_STATUS, None)
    sources = helper_sources(root)
    cases = list(ET.parse(receipt["tests"]).getroot().iter("testcase"))
    if (
        receipt["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or receipt["helpers_sha256"] != {name: sha256(source) for name, source in sources.items()}
        or receipt["fixtures_sha256"] != parent(root)["fixtures_sha256"]
        or receipt["command"] != cpu_command(path.parent)
        or receipt["cwd"] != str(path.parent)
        or receipt["environment"] != owner.ENVIRONMENT
        or {case.attrib["classname"].split(".")[-1] for case in cases} != {Path(NEW_HELPERS[-1]).stem}
        or len({(case.attrib["classname"], case.attrib["name"]) for case in cases}) != len(cases)
        or Path(__file__).resolve() != path.parent / "tracking/ns_budget_queue.py"
    ):
        raise ValueError("NS queue complete CPU/source acceptance changed")
    for name, source in sources.items():
        module = sys.modules.get(Path(name).stem) if name.endswith(".py") else None
        if module is not None and sha256(Path(module.__file__)) != sha256(source):
            raise RuntimeError("NS queue imports different accepted dependency bytes")
    if path.read_bytes() != raw:
        raise ValueError("NS queue CPU receipt changed during authentication")
    return receipt


def cpu(root: Path, out: Path) -> dict:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or out.exists()
        or out.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise RuntimeError("NS queue CPU acceptance requires hidden CUDA and unused external evidence")
    inherited, sources = parent(root), helper_sources(root)
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    hashes = {name: sha256(source) for name, source in sources.items()}
    for name, source in sources.items():
        shutil.copy2(source, helpers / name)
    for name in inherited["fixtures_sha256"]:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2((root / PARENT).parent / name, target)
    error, log = None, out / "cpu.log"
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                cpu_command(out),
                cwd=out,
                env=os.environ
                | owner.ENVIRONMENT
                | {
                    "UV_OFFLINE": "1",
                    "PYTHONPATH": f"{helpers}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
                },
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "NS queue CPU acceptance timed out"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    archive = out / "source.tar"
    inventory = {f"tracking/{name}": digest for name, digest in hashes.items()} | inherited["fixtures_sha256"]
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            bundle.add(out / name, arcname=name, recursive=False)
    owner.require_archive(archive, inventory, alias=False)
    unchanged = hashes == {name: sha256(source) for name, source in sources.items()}
    result = {
        "status": CPU_STATUS if type(code) is int and code == 0 and error is None and unchanged else "needs_attention",
        "root": str(root),
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "fixtures_sha256": inherited["fixtures_sha256"],
        "parent_cpu_receipt_sha256": PARENT_SHA256,
        "command": cpu_command(out),
        "cwd": str(out),
        "environment": owner.ENVIRONMENT,
        "cuda_visible_devices": "",
        "exit_code": code,
        **{name: str(path) for name, path in (("archive", archive), ("log", log), ("tests", out / "tests.xml"))},
        **{
            f"{name}_sha256": sha256(path)
            for name, path in (("archive", archive), ("log", log), ("tests", out / "tests.xml"))
            if path.is_file()
        },
        "scope": SCOPE,
    }
    if error:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CPU_STATUS:
        raise RuntimeError(f"NS queue CPU failure preserved at {out}")
    return result


def command(root: Path, cpu_receipt: Path) -> list[str]:
    return [
        owner.PYTHON,
        "-P",
        str(cpu_receipt.parent / "tracking/ns_budget_queue.py"),
        "--root",
        str(root),
        "--cpu-receipt",
        str(cpu_receipt),
        "--phase",
        "execute",
    ]


def handle(expected: dict, cwd: Path, marker_name: str, marker: str, environment: dict) -> dict:
    actual = owner.handoff.observed(expected["pid"])
    if actual is None:
        return {"status": "absent", "expected": expected}
    if any(actual[key] != expected[key] for key in ("pid", "start_ticks", "command", "group", "session")):
        raise RuntimeError("NS queue original predecessor handle was substituted")
    if actual["status"] in ("Z", "X", "x"):
        return {"status": "exited_handle_present", "observed": actual}
    verified = owner.process_context(expected["pid"], expected["command"], cwd, marker_name, marker, environment)
    if verified["start_ticks"] != expected["start_ticks"]:
        raise RuntimeError("NS queue original predecessor changed during context inspection")
    return {"status": "live_exact_generation", "observed": verified}


def readiness(root: Path) -> dict:
    """Observe original owners without treating stale state or timeout as termination."""
    binding_path, helpers, state_path, _, _ = owner.rational.locations(root)
    if (
        sha256(binding_path) != owner.preparation.BINDING_SHA256
        or sha256(root / owner.FOLLOWUP / "intent.json") != owner.FOLLOWUP_INTENT_SHA256
        or sha256(root / owner.FOLLOWUP_CPU) != owner.FOLLOWUP_CPU_SHA256
    ):
        raise ValueError("NS queue predecessor source or terminal-waiter identity changed")
    frozen = owner.read(binding_path)
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    intent, state = owner.read(helpers.parent / "controller.json"), owner.read(state_path)
    follow_intent = owner.read(root / owner.FOLLOWUP / "intent.json")
    if (
        intent["status"] != "acknowledged"
        or intent["binding_sha256"] != sha256(binding_path)
        or intent["command"] != frozen["controller_command"]
        or intent["pid"] != state["controller_pid"]
        or intent["start_ticks"] != state["controller_start_ticks"]
        or state["binding_sha256"] != sha256(binding_path)
        or state["command"] != frozen["validator_command"]
        or state["child_generation"]["pid"] != state["child_pid"]
        or state["child_generation"]["start_ticks"] != state["child_start_ticks"]
        or state["child_generation"]["parent"] != intent["pid"]
        or follow_intent["status"] != "acknowledged"
    ):
        raise RuntimeError("NS queue original predecessor ancestry changed")
    originals = (
        (
            "controller",
            intent["generation"],
            helpers,
            owner.rational.CONTROLLER_MARKER,
            intent["marker"],
            owner.rational.ENVIRONMENT,
        ),
        (
            "validator",
            state["child_generation"],
            Path(frozen["candidate"]),
            owner.rational.CHILD_MARKER,
            state["child_marker"],
            owner.rational.ENVIRONMENT,
        ),
        (
            "terminal_waiter",
            follow_intent["generation"],
            root / owner.FOLLOWUP,
            owner.FOLLOWUP_MARKER,
            follow_intent["marker"],
            {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONFAULTHANDLER": "1"},
        ),
    )
    handles = {
        name: handle(expected, cwd, marker_name, marker, environment)
        for name, expected, cwd, marker_name, marker, environment in originals
    }
    if any(value["status"] != "absent" for value in handles.values()):
        return {"status": "waiting_for_original_rational_terminal", "ready": False, "owners": handles}
    previous = owner.predecessor(root)
    return {"status": "original_terminal_ready", "ready": True, "owners": handles, "predecessor": previous}


def require_empty_launch(root: Path) -> None:
    if any(path.exists() for path in (*owner.locations(root), root / "run-preparation/source-ns-budget-20261118.tar")):
        raise FileExistsError("NS queue found existing one-shot owner evidence; launch retry denied")


def phases(root: Path, inherited: dict) -> list[tuple[str, list[str], Path]]:
    original_helpers = Path(inherited["helpers"])
    issued = owner.locations(root)[1]
    return [
        (
            "freeze",
            [
                owner.PYTHON,
                "-P",
                str(original_helpers / "ns_budget_gpu_controller.py"),
                "--root",
                str(root),
                "--phase",
                "freeze",
                "--cpu-receipt",
                str(root / PARENT),
            ],
            original_helpers,
        ),
        *(
            (
                phase,
                [
                    owner.PYTHON,
                    "-P",
                    str(issued / "ns_budget_gpu_controller.py"),
                    "--root",
                    str(root),
                    "--phase",
                    phase,
                ],
                issued,
            )
            for phase in ("check", "start")
        ),
    ]


def run_phase(root: Path, phase: str, argv: list[str], cwd: Path, state: dict) -> dict:
    work, marker, generations = root / WORK, uuid4().hex, {}
    log = work / f"{phase}.log"
    if log.exists() or any(entry["phase"] == phase for entry in state["phases"]):
        raise FileExistsError("NS queue phase already has one-shot evidence")
    entry = {"phase": phase, "command": argv, "cwd": str(cwd), "marker": marker, "spawn_status": "starting"}
    state["active_phase"] = entry
    write(work / "state.json", state)
    child = None
    environment = os.environ | owner.ENVIRONMENT | {"PYTHONPATH": str(cwd), owner.CONTROLLER_MARKER: marker}
    environment.pop(MARKER, None)
    # The detached NS owner replaces this marker, so phase cleanup cannot kill it.
    try:
        with log.open("x") as stream:
            child = subprocess.Popen(
                argv,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            own = acknowledged_process(child, argv, cwd, owner.CONTROLLER_MARKER, marker, owner.ENVIRONMENT)
            entry.update(pid=own["pid"], start_ticks=own["start_ticks"], generation=own, spawn_status="acknowledged")
            write(work / "state.json", state)
            entry["exit_code"] = child.wait(timeout=TERMINAL_TIMEOUT if phase == "terminal" else PHASE_TIMEOUT)
            if type(entry["exit_code"]) is not int or entry["exit_code"] != 0:
                raise RuntimeError(f"NS queue {phase} phase failed")
    finally:
        try:
            if child is not None:
                finish_marked_child(child, marker, generations, owner.CONTROLLER_MARKER)
                entry.update(cleanup_exit_code=child.returncode, marked_descendants_absent=True)
        except BaseException as failure:
            entry["cleanup_error"] = f"{type(failure).__name__}: {failure}"
            raise
        finally:
            entry.update(log=str(log), log_sha256=sha256(log) if log.is_file() else None, descendants=generations)
            write(work / "state.json", state)
    return entry


def launch(root: Path, inherited: dict, state: dict) -> dict:
    require_empty_launch(root)
    observation = readiness(root)
    if observation["ready"] is not True:
        raise RuntimeError("NS queue cannot issue while original owners remain")
    if state["phases"]:
        raise FileExistsError("NS queue already began its one-shot sequence")
    state.update(status="issuing_ns_diagnostic", original_observation=observation)
    write(root / WORK / "state.json", state)
    for phase, argv, cwd in phases(root, inherited):
        state["phases"].append(run_phase(root, phase, argv, cwd, state))
        write(root / WORK / "state.json", state)
    binding, helpers, _, _ = owner.locations(root)
    frozen, intent = owner.read(binding), owner.read(helpers.parent / "controller.json")
    own = owner.process_context(
        intent["pid"],
        owner.controller_command(root),
        helpers,
        owner.CONTROLLER_MARKER,
        intent["marker"],
        owner.ENVIRONMENT,
    )
    owner.require_acknowledged_controller(intent, own)
    if intent["binding_sha256"] != sha256(binding) or frozen["cpu_receipt_sha256"] != PARENT_SHA256:
        raise RuntimeError("NS queue issued controller has different accepted source")
    state.update(
        status="ns_diagnostic_controller_acknowledged",
        issued_binding_sha256=sha256(binding),
        issued_controller=intent,
    )
    write(root / WORK / "state.json", state)
    return state


def issued_readiness(root: Path, state: dict) -> dict:
    """Wait for actual issued owner absence before independent retained-result reconstruction."""
    binding, helpers, state_path, _ = owner.locations(root)
    if sha256(binding) != state["issued_binding_sha256"]:
        raise ValueError("NS queue issued binding changed")
    frozen, intent = owner.read(binding), owner.read(helpers.parent / "controller.json")
    inherited = parent(root)
    require_helper_bytes(helpers, frozen["helpers_sha256"])
    if (
        intent != state["issued_controller"]
        or frozen["cpu_receipt_sha256"] != PARENT_SHA256
        or frozen["helpers_sha256"] != inherited["helpers_sha256"]
        or frozen["controller_command"] != owner.controller_command(root)
        or frozen["environment"] != owner.ENVIRONMENT
        or intent["binding_sha256"] != sha256(binding)
    ):
        raise ValueError("NS queue issued controller or accepted source changed")
    controller = handle(intent["generation"], helpers, owner.CONTROLLER_MARKER, intent["marker"], owner.ENVIRONMENT)
    saved = owner.read(state_path) if state_path.is_file() else {}
    workers = {}
    entries = saved.get("executions", [])
    if saved.get("active_worker", {}).get("spawn_status") == "acknowledged":
        entries = [*entries, saved["active_worker"]]
    for entry in entries:
        workers[str(entry["index"])] = handle(
            entry["generation"],
            helpers,
            owner.CHILD_MARKER,
            entry["marker"],
            frozen["worker_environment"],
        )
    if controller["status"] != "absent" or any(value["status"] != "absent" for value in workers.values()):
        return {
            "status": "waiting_for_issued_ns_owner",
            "ready": False,
            "controller": controller,
            "workers": workers,
            "saved_state": saved.get("status"),
        }
    if (
        saved.get("status") != "diagnostic_complete"
        or saved["binding_sha256"] != state["issued_binding_sha256"]
        or saved["controller_pid"] != intent["pid"]
        or saved["controller_start_ticks"] != intent["start_ticks"]
        or any(key in saved for key in ("error", "cleanup_error", "deferred_signals"))
    ):
        raise RuntimeError("NS queue issued owner exited without completed evidence; failure preserved")
    return {
        "status": "issued_ns_terminal_ready",
        "ready": True,
        "controller": controller,
        "workers": workers,
        "saved_state": saved["status"],
    }


def terminal_phase(root: Path, state: dict) -> dict:
    if (root / TERMINAL).exists():
        raise FileExistsError("NS queue issued terminal verifier already has one-shot evidence")
    observation = issued_readiness(root, state)
    if observation["ready"] is not True:
        raise RuntimeError("NS queue issued owner remains present")
    helpers = owner.locations(root)[1]
    argv = [
        owner.PYTHON,
        "-P",
        str(helpers / "ns_budget_gpu_controller.py"),
        "--root",
        str(root),
        "--phase",
        "terminal",
        "--out",
        str(root / TERMINAL),
    ]
    state["phases"].append(run_phase(root, "terminal", argv, helpers, state))
    result = owner.read(root / TERMINAL)
    if (
        result["status"] != "ns_budget_terminal_independently_reconstructed_exact_restoration_verified"
        or result["binding_sha256"] != state["issued_binding_sha256"]
        or result["controller_pid_absent"] != state["issued_controller"]["pid"]
        or result["verifier_sha256"]
        != owner.read(owner.locations(root)[0])["helpers_sha256"]["ns_budget_gpu_controller.py"]
        or any(sha256(root / name) != digest for name, digest in result["evidence_sha256"].items())
    ):
        raise ValueError("NS queue independently reconstructed terminal identity changed")
    state.update(
        status="ns_diagnostic_terminal_reconstructed",
        terminal=str(root / TERMINAL),
        terminal_sha256=sha256(root / TERMINAL),
        historical_controls_interpretable=result["comparison"]["historical_controls_interpretable"],
    )
    write(root / WORK / "state.json", state)
    return state


def start(root: Path, cpu_receipt: Path) -> dict:
    receipt = accepted(root, cpu_receipt)
    readiness(root)
    require_empty_launch(root)
    work = root / WORK
    if work.exists():
        raise FileExistsError("NS queue already has one-shot installation evidence")
    if owner.NODE not in subprocess.check_output(["hostname", "-I"], text=True).split():
        raise RuntimeError("NS queue is outside the original allocation")
    work.mkdir()
    intent = {
        "status": "starting",
        "command": command(root, cpu_receipt),
        "marker": uuid4().hex,
        "cpu_receipt": str(cpu_receipt),
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "runner_sha256": receipt["helpers_sha256"]["ns_budget_queue.py"],
        "parent_cpu_receipt_sha256": PARENT_SHA256,
        "interval": INTERVAL,
        "scope": SCOPE,
    }
    write(work / "intent.json", intent)
    child = None
    try:
        with (work / "queue.log").open("x") as stream:
            child = subprocess.Popen(
                intent["command"],
                cwd=work,
                env=os.environ | owner.ENVIRONMENT | {"PYTHONPATH": receipt["helpers"], MARKER: intent["marker"]},
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        own = acknowledged_process(child, intent["command"], work, MARKER, intent["marker"], owner.ENVIRONMENT)
        intent.update(status="acknowledged", pid=own["pid"], start_ticks=own["start_ticks"], generation=own)
    except BaseException as failure:
        intent.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
        if child is not None:
            finish_marked_child(child, intent["marker"], {}, MARKER)
        raise
    finally:
        write(work / "intent.json", intent)
    return intent


def execute(root: Path, cpu_receipt: Path) -> dict:
    accepted(root, cpu_receipt)
    inherited, work = parent(root), root / WORK
    intent, deadline = owner.read(work / "intent.json"), time.monotonic() + 30
    while intent["status"] == "starting" and time.monotonic() < deadline:
        time.sleep(0.01)
        intent = owner.read(work / "intent.json")
    own = owner.process_context(
        os.getpid(), command(root, cpu_receipt), work, MARKER, intent["marker"], owner.ENVIRONMENT
    )
    owner.require_acknowledged_controller(intent, own)
    if (
        intent["cpu_receipt_sha256"] != sha256(cpu_receipt)
        or intent["runner_sha256"] != sha256(Path(__file__))
        or intent["command"] != command(root, cpu_receipt)
        or intent["parent_cpu_receipt_sha256"] != PARENT_SHA256
        or intent["interval"] != INTERVAL
        or owner.NODE not in subprocess.check_output(["hostname", "-I"], text=True).split()
    ):
        raise RuntimeError("NS queue executable identity or allocation changed")
    state = {
        "status": "waiting_for_original_rational_terminal",
        "pid": own["pid"],
        "start_ticks": own["start_ticks"],
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "parent_cpu_receipt_sha256": PARENT_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "interval": INTERVAL,
        "phases": [],
        "scope": SCOPE,
    }
    with (work / "execution.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while True:
                observation = readiness(root)
                state.update(status=observation["status"], original_observation=observation)
                write(work / "state.json", state)
                if observation["ready"]:
                    launch(root, inherited, state)
                    break
                time.sleep(INTERVAL)
            while True:
                observation = issued_readiness(root, state)
                state.update(status=observation["status"], issued_observation=observation)
                write(work / "state.json", state)
                if observation["ready"]:
                    terminal_phase(root, state)
                    break
                time.sleep(INTERVAL)
        except BaseException as failure:
            state.update(status="needs_attention", error=f"{type(failure).__name__}: {failure}")
            raise
        finally:
            write(work / "state.json", state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cpu", "check", "start", "execute"))
    parser.add_argument("--cpu-receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if any(os.environ.get(name) != value for name, value in owner.ENVIRONMENT.items()):
        raise RuntimeError("NS queue requires hidden CUDA and its fixed environment")
    root = args.root.resolve(strict=True)
    if args.phase == "cpu":
        if args.out is None:
            parser.error("CPU acceptance requires --out")
        result = cpu(root, args.out.resolve())
    else:
        if args.cpu_receipt is None:
            parser.error("Accepted queue execution requires --cpu-receipt")
        path = args.cpu_receipt.resolve(strict=True)
        result = (
            accepted(root, path) if args.phase == "check" else (start if args.phase == "start" else execute)(root, path)
        )
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
