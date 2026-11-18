# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU archive and process-boundary tests for the isolated attribution controller."""

import copy
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import fused_attribution_validation_controller as controller
import pytest
from fused_attribution_validation_test_support import record, write_failed_profiler

PARENT = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/candidates/fused-attribution-full-schedule-20261118T0257"
)
PARENT_VERIFIER = (
    "import json\n"
    "def require_cpu(root, receipt):\n"
    "    value = json.loads(receipt.read_bytes())\n"
    "    assert value['root'] == str(root)\n"
    "    return value\n"
)
SYNTHETIC_VALIDATOR = """
import json, os, select, signal, subprocess, sys
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--root') + 1])
state_path = root / 'run-preparation/fused-attribution-validation-state-20261118.json'
poll = select.poll()
while not state_path.exists() or json.loads(state_path.read_bytes()).get('child_pid') != os.getpid():
    poll.poll(1)
if os.environ.get('CPU_SYNTHETIC_TIMEOUT') == '1':
    signal.pause()
if os.environ.get('CPU_SYNTHETIC_ORPHAN') == '1':
    read_fd, write_fd = os.pipe()
    child = subprocess.Popen(
        [sys.executable, '-c',
         'import os,signal,sys;os.write(int(sys.argv[1]),b"R");signal.pause()', str(write_fd)],
        pass_fds=(write_fd,), start_new_session=True)
    os.close(write_fd)
    assert os.read(read_fd, 1) == b'R'
    os.close(read_fd)
    print(child.pid, flush=True)
    sys.exit(7)
print(json.dumps({'pid': os.getpid(), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}), flush=True)
"""


def proc_record(proc: Path, process: dict, cwd: Path, environment: dict) -> None:
    directory = proc / str(process["pid"])
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 20
    fields[:4] = [process["status"], str(process["parent"]), str(process["group"]), str(process["session"])]
    fields[19] = process["start_ticks"]
    (directory / "stat").write_text(f"{process['pid']} (cpu fixture) {' '.join(fields)}\n")
    (directory / "cmdline").write_bytes(("\0".join(process["command"]) + "\0").encode())
    (directory / "environ").write_bytes(
        ("\0".join(f"{key}={value}" for key, value in environment.items()) + "\0").encode()
    )
    link = directory / "cwd"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(cwd, target_is_directory=True)


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    root, previous, candidate = tmp_path / "root", tmp_path / "parent", tmp_path / "candidate"
    shutil.copytree(PARENT, previous, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    write_failed_profiler(root)
    old_binding, old_helpers, old_state, _ = controller.predecessor_locations(root)
    manager = old_helpers / controller.MANAGER
    sibling_manager = Path(controller.__file__).with_name(controller.MANAGER)
    if sibling_manager.is_file():
        shutil.copy2(sibling_manager, manager)
    else:
        manager.write_text("#!/bin/bash\nexit 0\n")
    predecessor = controller.read(old_binding)
    predecessor["helpers_sha256"][controller.MANAGER] = controller.sha256(manager)
    record(old_binding, predecessor)
    state = controller.read(old_state)
    state["binding_sha256"] = controller.sha256(old_binding)
    record(old_state, state)
    intent_path = old_helpers.parent / "controller.json"
    intent = controller.read(intent_path)
    intent["binding_sha256"] = controller.sha256(old_binding)
    record(intent_path, intent)
    record(root / "data/experiment_receipt.json", {"cpu_fixture": True})
    preparation = controller.preparation
    metadata = controller.read(previous / preparation.PARENT_PREPARATION)
    for trial, spec in zip(metadata["trials"], controller.trial_specs(), strict=True):
        trial["config"] = str(previous / spec["config_name"])
    metadata["candidate"] = str(previous)
    metadata["data_receipt_sha256"] = controller.sha256(root / "data/experiment_receipt.json")
    parent_helpers = tmp_path / "parent-helpers"
    parent_helpers.mkdir()
    parent_helper = parent_helpers / "prepare_fused_attribution.py"
    parent_helper.write_text(PARENT_VERIFIER)
    metadata["helpers_sha256"] = {parent_helper.name: controller.sha256(parent_helper)}
    record(previous / preparation.PARENT_PREPARATION, metadata)
    digest = controller.sha256(previous / preparation.PARENT_PREPARATION)
    archive = tmp_path / "parent-source.tar"
    expected = preparation.parent_archive_inventory(metadata, digest)
    with tarfile.open(archive, "x") as bundle:
        for name in expected:
            source = parent_helper if name.startswith("tracking/") else previous / name
            bundle.add(source, arcname=name, recursive=False)
        bundle.add(previous / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    parent_receipt = tmp_path / "parent-cpu.json"
    record(
        parent_receipt,
        {
            "root": str(root),
            "candidate": str(previous),
            "files_sha256": controller.candidate_inventory(previous),
            "preparation_sha256": digest,
            "helpers_sha256": metadata["helpers_sha256"],
            "archive": str(archive),
            "archive_sha256": controller.sha256(archive),
        },
    )
    real_run = subprocess.run
    execution = {"candidate_code": 0, "controller_code": 0, "controller_failure": None, "during_controller": None}

    def boundary(command, **kwargs):
        if command == preparation.cpu_command() or command == controller.controller_cpu_command():
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
            kwargs["stdout"].write("synthetic external CPU suite outcome\n")
            key = "candidate_code" if command == preparation.cpu_command() else "controller_code"
            if key == "controller_code":
                if execution["during_controller"] is not None:
                    execution["during_controller"](Path(kwargs["cwd"]))
                if execution["controller_failure"] is not None:
                    raise execution["controller_failure"]
            return subprocess.CompletedProcess(command, execution[key])
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", boundary)
    preparation.prepare(root, parent_receipt, candidate)
    cpu_out = tmp_path / "candidate-cpu"
    preparation.cpu(root, candidate, cpu_out)
    return SimpleNamespace(
        root=root,
        candidate=candidate,
        candidate_receipt=cpu_out / "receipt.json",
        execution=execution,
        temporary=tmp_path,
    )


@pytest.mark.parametrize("code", (0, 7))
def test_controller_cpu_archives_success_and_failure_without_binding_or_gpu(accepted, code):
    item = accepted
    item.execution["controller_code"] = code
    out = item.temporary / "controller-cpu"
    if code:
        with pytest.raises(RuntimeError):
            controller.controller_cpu(item.root, item.candidate_receipt, out)
        result = controller.read(out / "receipt.json")
        assert result["status"] == "needs_attention"
    else:
        result = controller.controller_cpu(item.root, item.candidate_receipt, out)
        assert controller.require_controller_cpu(item.root, out / "receipt.json", item.candidate_receipt) == result
        fixture = out / controller.fixture_names()[0]
        fixture.write_text("{}\n")
        with pytest.raises(ValueError):
            controller.require_controller_cpu(item.root, out / "receipt.json", item.candidate_receipt)
    assert result["exit_code"] == code
    assert result["cuda_visible_devices"] == ""
    assert result["archive_sha256"] == controller.sha256(Path(result["archive"]))
    assert not controller.locations(item.root)[0].exists()
    assert not controller.locations(item.root)[2].exists()
    assert not controller.locations(item.root)[3].exists()


@pytest.mark.parametrize("collect", (True, False), ids=("collect", "run"))
def test_controller_cpu_archive_supports_real_owner_suite_in_fresh_process(accepted, collect):
    item = accepted
    result = controller.controller_cpu(item.root, item.candidate_receipt, item.temporary / "controller-cpu")
    extracted = item.temporary / "extracted-controller"
    with tarfile.open(result["archive"]) as bundle:
        for member in bundle.getmembers():
            target = extracted / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source:
                target.write_bytes(source.read())
    helper = extracted / "tracking/selected_lr_release_owner.py"
    assert controller.sha256(helper) == controller.sha256(Path(controller.__file__).with_name(helper.name))
    checked = subprocess.run(
        [
            controller.PYTHON,
            "-P",
            "-B",
            "-m",
            "pytest",
            "-c",
            "/dev/null",
            "-p",
            "no:cacheprovider",
            "-q",
            "-m",
            controller.SAFE_MARKERS,
            *(["--collect-only"] if collect else []),
            "tracking/test_fused_attribution_validation_owner.py",
        ],
        cwd=extracted,
        env=os.environ
        | {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join((str(item.candidate), str(extracted / "tracking"))),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.parametrize("fault", ("spawn_oserror", "timeout", "archive_write", "helper_drift", "candidate_drift"))
def test_controller_cpu_persists_spawn_archive_and_posttest_verification_failures(accepted, fault):
    item = accepted
    out = item.temporary / "controller-cpu"
    receipt_digest = controller.sha256(item.candidate_receipt)
    if fault == "spawn_oserror":
        item.execution["controller_failure"] = OSError("CPU verifier executable unavailable")
    elif fault == "timeout":
        item.execution["controller_failure"] = subprocess.TimeoutExpired(
            controller.controller_cpu_command(), controller.CPU_TIMEOUT
        )
    else:

        def disrupt(directory):
            if fault == "archive_write":
                (directory / "controller-source.tar").mkdir()
            elif fault == "helper_drift":
                (directory / "tracking/fused_attribution_validation_contract.py").write_text("# changed helper\n")
            else:
                path = item.candidate / controller.fixture_names()[0]
                path.write_text("{}\n")

        item.execution["during_controller"] = disrupt
    with pytest.raises(RuntimeError):
        controller.controller_cpu(item.root, item.candidate_receipt, out)
    result = controller.read(out / "receipt.json")
    assert result["status"] == "needs_attention" and result["error"]
    assert result["exit_code"] == {"spawn_oserror": 127, "timeout": 124}.get(fault, 0)
    assert result["candidate_cpu_receipt_sha256"] == receipt_digest
    assert result["cuda_visible_devices"] == ""
    assert result["command"] == controller.controller_cpu_command()
    assert result["log_sha256"] == controller.sha256(Path(result["log"]))
    if fault == "archive_write":
        assert result["archive_sha256"] is None
    else:
        assert result["archive_sha256"] == controller.sha256(Path(result["archive"]))
    with pytest.raises(ValueError):
        controller.require_controller_cpu(item.root, out / "receipt.json", item.candidate_receipt)
    binding_path, _, state_path, work, receipt = controller.locations(item.root)
    assert not any(path.exists() for path in (binding_path, state_path, work, receipt))


def test_controller_cpu_rewrites_success_as_failure_if_log_changes_after_receipt_save(accepted, monkeypatch):
    item = accepted
    out = item.temporary / "controller-cpu"
    real_replace = Path.replace
    disrupted = []

    def replace(source, target):
        result = real_replace(source, target)
        if Path(target) == out / "receipt.json" and not disrupted:
            disrupted.append(True)
            (out / "controller.log").write_text("replaced after provisional receipt\n")
        return result

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(RuntimeError):
        controller.controller_cpu(item.root, item.candidate_receipt, out)
    result = controller.read(out / "receipt.json")
    assert result["status"] == "needs_attention" and result["error"]
    assert result["exit_code"] == 0
    assert result["log_sha256"] != controller.sha256(Path(result["log"]))
    with pytest.raises(ValueError):
        controller.require_controller_cpu(item.root, out / "receipt.json", item.candidate_receipt)


@pytest.fixture
def frozen(accepted):
    item = accepted
    out = item.temporary / "controller-cpu"
    controller.controller_cpu(item.root, item.candidate_receipt, out)
    item.controller_receipt = out / "receipt.json"
    item.frozen = controller.freeze(item.root, item.candidate_receipt, item.controller_receipt)
    return item


def test_freeze_preserves_parent_and_checks_canonical_archived_binding_in_fresh_process(frozen):
    item = frozen
    before = controller.candidate_inventory(item.candidate)
    _, helpers, state_path, work, receipt = controller.locations(item.root)
    checked = subprocess.run(
        [
            controller.PYTHON,
            "-P",
            "-B",
            str(helpers / Path(controller.__file__).name),
            "--root",
            str(item.root),
            "--phase",
            "check",
        ],
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["status"] == controller.FROZEN_STATUS
    assert len(item.frozen["commands"]) == 27
    assert item.frozen["plan"]["planned_validation_tokens"] == 226492416
    assert controller.candidate_inventory(item.candidate) == before
    assert not state_path.exists() and not work.exists() and not receipt.exists()
    with pytest.raises(FileExistsError):
        controller.freeze(item.root, item.candidate_receipt, item.controller_receipt)
    (helpers / "fused_attribution_validation_contract.py").write_text("# replaced frozen contract\n")
    checked = subprocess.run(
        [
            controller.PYTHON,
            "-P",
            "-B",
            str(helpers / Path(controller.__file__).name),
            "--root",
            str(item.root),
            "--phase",
            "check",
        ],
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert checked.returncode != 0
    assert not state_path.exists() and not work.exists() and not receipt.exists()


@pytest.mark.parametrize("drift", ("parent_exit", "other_parent", "reused_pid", "session", "argv"))
def test_detached_ack_accepts_only_original_generation_after_parent_exit(drift):
    generation = {
        "pid": 2001,
        "parent": 2000,
        "start_ticks": "original",
        "group": 2001,
        "session": 2001,
        "command": ["python", "-P", "controller.py"],
        "status": "S",
    }
    intent = {"status": "acknowledged", "pid": 2001, "start_ticks": "original", "generation": generation}
    own = generation | {"parent": 1, "status": "R"}
    if drift == "other_parent":
        own["parent"] = 999
    elif drift == "reused_pid":
        own["start_ticks"] = "replacement"
    elif drift == "session":
        own["session"] = 999
    elif drift == "argv":
        own["command"] = ["replacement"]
    if drift == "parent_exit":
        controller.require_acknowledged_controller(intent, own)
        assert own["parent"] == 1 and intent["generation"]["parent"] == 2000
    else:
        with pytest.raises(RuntimeError):
            controller.require_acknowledged_controller(intent, own)


@pytest.mark.parametrize("scenario", ("passed", "orphan", "timeout"))
def test_bounded_validator_ack_and_pidfd_cleanup_leave_unmarked_cpu_peer_alive(tmp_path, monkeypatch, scenario):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    candidate.mkdir()
    _, helpers, state_path, _, _ = controller.locations(root)
    helpers.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    (helpers / "fused_attribution_validation.py").write_text(SYNTHETIC_VALIDATOR)
    monkeypatch.setattr(controller, "ENVIRONMENT", controller.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""})
    monkeypatch.setenv("CPU_SYNTHETIC_ORPHAN", "1" if scenario == "orphan" else "0")
    monkeypatch.setenv("CPU_SYNTHETIC_TIMEOUT", "1" if scenario == "timeout" else "0")
    if scenario == "timeout":
        monkeypatch.setattr(controller, "VALIDATION_TIMEOUT", 0.1)
    peer = subprocess.Popen([controller.PYTHON, "-c", "import signal;signal.pause()"], start_new_session=True)
    handle = controller.pidfd_open(peer.pid)

    def forbid_numeric_signal(*args):
        raise AssertionError("Cleanup must use pidfds")

    monkeypatch.setattr(os, "kill", forbid_numeric_signal)
    monkeypatch.setattr(os, "killpg", forbid_numeric_signal)
    try:
        state = {}
        if scenario == "timeout":
            with pytest.raises(subprocess.TimeoutExpired):
                controller.bounded_validator(root, {"candidate": str(candidate)}, state)
            code = -signal.SIGTERM
        else:
            code = controller.bounded_validator(root, {"candidate": str(candidate)}, state)
        saved = controller.read(state_path)
        assert code == {"passed": 0, "orphan": 7, "timeout": -signal.SIGTERM}[scenario]
        assert saved["spawn_status"] == "acknowledged"
        assert saved["cleanup_exit_code"] == code and saved["marked_descendants_absent"] is True
        assert saved["log_sha256"] == controller.sha256(Path(saved["log"]))
        assert controller.handoff.observed(saved["child_pid"]) is None
        assert not controller.handoff.marker_processes(controller.CHILD_MARKER, saved["child_marker"])
        if scenario == "orphan":
            rank_pid = int(Path(saved["log"]).read_text())
            process = controller.handoff.observed(rank_pid)
            assert str(rank_pid) in saved["child_generation"]
            assert process is None or process["status"] in ("Z", "X", "x")
        elif scenario == "passed":
            assert json.loads(Path(saved["log"]).read_text())["cuda_visible_devices"] == ""
        assert peer.poll() is None
    finally:
        signal.pidfd_send_signal(handle, signal.SIGTERM)
        peer.wait(timeout=10)
        os.close(handle)


@pytest.mark.parametrize("fault", ("none", "rank_gpu_swap", "extra_marker", "child_marker", "original_live"))
def test_live_handoff_uses_exact_restored_keeper_and_inherited_marker_whitelist(tmp_path, monkeypatch, fault):
    root, proc = tmp_path / "root", tmp_path / "proc"
    write_failed_profiler(root)
    expected = controller.validation.failed_profiler_evidence(root)
    keeper = expected["restored_keeper"]
    proc.mkdir()
    for process in keeper["generation"].values():
        proc_record(proc, process, root, {expected["controller_marker_name"]: expected["controller_marker"]})
    monkeypatch.setattr(controller.handoff, "PROC_ROOT", proc)
    pointer = tmp_path / "keeper.pid"
    pointer.write_text(str(keeper["pid"]))
    monkeypatch.setattr(controller.handoff, "KEEPER_PID", pointer)
    # Match the persisted generation to the public keeper protocol rather than mocking its acceptance.
    launcher = [
        controller.handoff.PYTHON,
        controller.handoff.TORCHRUN,
        "--standalone",
        "--nproc-per-node=8",
        controller.handoff.WORKLOAD,
        *controller.handoff.WORKLOAD_ARGS,
    ]
    ranks = [controller.handoff.PYTHON, "-u", controller.handoff.WORKLOAD, *controller.handoff.WORKLOAD_ARGS]
    for key, process in keeper["generation"].items():
        process["command"] = launcher if int(key) == keeper["pid"] else ranks
        proc_record(proc, process, root, {expected["controller_marker_name"]: expected["controller_marker"]})
    state_path = controller.predecessor_locations(root)[2]
    state = controller.read(state_path)
    state["restored_keeper"] = keeper
    record(state_path, state)
    expected = controller.validation.failed_profiler_evidence(root)
    occupied = copy.deepcopy(keeper["occupied"])
    if fault == "rank_gpu_swap":
        occupied[0][1], occupied[1][1] = occupied[1][1], occupied[0][1]
    elif fault in ("extra_marker", "original_live"):
        pid = 999 if fault == "extra_marker" else expected["controller_pid"]
        process = keeper["generation"]["100"] | {"pid": pid, "parent": 1, "group": pid, "session": pid}
        proc_record(proc, process, root, {expected["controller_marker_name"]: expected["controller_marker"]})
    elif fault == "child_marker":
        process = keeper["generation"]["101"]
        proc_record(
            proc,
            process,
            root,
            {
                expected["controller_marker_name"]: expected["controller_marker"],
                expected["child_marker_name"]: expected["child_marker"],
            },
        )

    def devices(command, **kwargs):
        if "--query-gpu=index,uuid,name" in command:
            return "\n".join(",".join(row) for row in keeper["devices"]) + "\n"
        if "--query-compute-apps=pid,gpu_uuid" in command:
            return "\n".join(",".join(row) for row in occupied) + "\n"
        raise AssertionError(f"Unexpected device subprocess: {command}")

    monkeypatch.setattr(subprocess, "check_output", devices)
    if fault == "none":
        result = controller.probe_failed_profiler(root, expected)
        assert result["original_handles_absent"] and result["child_markers_absent"]
        assert result["inherited_controller_marker_pids"] == list(range(100, 109))
        assert pointer.read_text() == "100"
    else:
        with pytest.raises(RuntimeError):
            controller.probe_failed_profiler(root, expected)
        assert pointer.read_text() == "100"


@contextmanager
def frozen_import(helpers: Path):
    """Load the real archived module graph without allowing already loaded checkout modules to leak in."""
    names = {path.stem for path in helpers.glob("*.py")}
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    old_path = list(sys.path)
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(helpers))
        yield importlib.import_module("fused_attribution_validation_controller")
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_start_refuses_unacknowledged_prior_intent_without_process_spawn(frozen, monkeypatch):
    item = frozen
    _, helpers, state_path, work, receipt = controller.locations(item.root)
    record(helpers.parent / "controller.json", {"status": "starting", "ambiguous": True})

    def node(command, **kwargs):
        assert command == ["hostname", "-I"]
        return f"{controller.NODE}\n"

    real_spawn = subprocess.Popen

    def forbidden_spawn(command, **kwargs):
        if command[:3] == [controller.PYTHON, "-P", "-c"]:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            return real_spawn(command, **kwargs)
        raise AssertionError("Existing intent must refuse before controller spawn or keeper changes")

    monkeypatch.setattr(subprocess, "check_output", node)
    monkeypatch.setattr(subprocess, "Popen", forbidden_spawn)
    for name, value in controller.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    with frozen_import(helpers) as archived:
        with pytest.raises(FileExistsError):
            archived.start(item.root)
    assert controller.read(helpers.parent / "controller.json") == {"status": "starting", "ambiguous": True}
    assert not state_path.exists() and not work.exists() and not receipt.exists()


@pytest.mark.parametrize("fault", ("validator_failure", "rank_gpu_swap", "partial_release"))
def test_locked_validation_refuses_drift_partial_release_or_restores_failed_validator(frozen, monkeypatch, fault):
    item = frozen
    root = item.root
    binding_path, helpers, state_path, _, receipt = controller.locations(root)
    proc = item.temporary / "proc"
    proc.mkdir()
    pointer, manager_lock = item.temporary / "keeper.pid", item.temporary / "manager.lock"
    manager = item.temporary / "manager.sh"
    shutil.copy2(helpers / controller.MANAGER, manager)
    accepted = copy.deepcopy(item.frozen["allocation_predecessor"]["restored_keeper"])
    parent_marker = item.frozen["allocation_predecessor"]
    launcher = [
        controller.handoff.PYTHON,
        controller.handoff.TORCHRUN,
        "--standalone",
        "--nproc-per-node=8",
        controller.handoff.WORKLOAD,
        *controller.handoff.WORKLOAD_ARGS,
    ]
    rank_command = [controller.handoff.PYTHON, "-u", controller.handoff.WORKLOAD, *controller.handoff.WORKLOAD_ARGS]
    for pid, process in accepted["generation"].items():
        process["command"] = launcher if int(pid) == accepted["pid"] else rank_command
    old_state_path = controller.predecessor_locations(root)[2]
    old_state = controller.read(old_state_path)
    old_state["restored_keeper"] = accepted
    record(old_state_path, old_state)
    # Refresh the synthetic preparation/CPU/controller archives after binding the keeper's exact public commands.
    prepared = controller.read(item.candidate / controller.PREPARATION)
    prepared["allocation_predecessor"] = controller.validation.failed_profiler_evidence(root)
    record(item.candidate / controller.PREPARATION, prepared)
    cpu = controller.read(item.candidate_receipt)
    cpu["preparation_sha256"] = controller.sha256(item.candidate / controller.PREPARATION)
    expected = controller.preparation.archive_inventory(prepared, cpu["preparation_sha256"])
    archive = Path(cpu["archive"])
    archive.unlink()
    with tarfile.open(archive, "x") as bundle:
        for name in expected:
            source = Path(cpu["helpers"]) / Path(name).name if name.startswith("tracking/") else item.candidate / name
            bundle.add(source, arcname=name, recursive=False)
        bundle.add(item.candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    cpu["archive_sha256"] = controller.sha256(archive)
    record(item.candidate_receipt, cpu)
    controller_cpu = controller.read(item.controller_receipt)
    controller_cpu["candidate_cpu_receipt_sha256"] = controller.sha256(item.candidate_receipt)
    record(item.controller_receipt, controller_cpu)
    item.frozen.update(
        preparation_sha256=cpu["preparation_sha256"],
        cpu_receipt_sha256=controller.sha256(item.candidate_receipt),
        controller_cpu_receipt_sha256=controller.sha256(item.controller_receipt),
        allocation_predecessor=prepared["allocation_predecessor"],
    )
    archive = Path(item.frozen["archive"])
    archive.unlink()
    with tarfile.open(archive, "x") as bundle:
        for name in controller.archive_inventory(item.frozen):
            if name.startswith("tracking/"):
                source = helpers / Path(name).name
            elif name == "candidate-cpu-receipt.json":
                source = item.candidate_receipt
            elif name == "controller-cpu-receipt.json":
                source = item.controller_receipt
            else:
                source = item.candidate / name
            bundle.add(source, arcname=name, recursive=False)
        bundle.add(item.candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    item.frozen["archive_sha256"] = controller.sha256(archive)
    record(binding_path, item.frozen)
    occupied = copy.deepcopy(accepted["occupied"])
    if fault == "rank_gpu_swap":
        occupied[0][1], occupied[1][1] = occupied[1][1], occupied[0][1]
    events, fd_pids = [], {}
    real_spawn, real_run = subprocess.Popen, subprocess.run
    controller_pid = os.getpid()
    own = {
        "pid": controller_pid,
        "parent": 1,
        "status": "S",
        "group": controller_pid,
        "session": controller_pid,
        "start_ticks": "controller-original",
        "command": controller.controller_command(root),
    }
    environment = controller.ENVIRONMENT | {controller.CONTROLLER_MARKER: "owned-controller"}
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    proc_record(proc, own, helpers, environment)
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "binding_sha256": controller.sha256(binding_path),
            "command": own["command"],
            "marker": "owned-controller",
            "allocation_predecessor": prepared["allocation_predecessor"],
            "pid": controller_pid,
            "start_ticks": own["start_ticks"],
            "generation": own | {"parent": 999},
        },
    )
    for process in accepted["generation"].values():
        proc_record(proc, process, root, {parent_marker["controller_marker_name"]: parent_marker["controller_marker"]})
    pointer.write_text(str(accepted["pid"]))

    def output(command, **kwargs):
        if command == ["hostname", "-I"]:
            return f"{controller.NODE}\n"
        if command == ["ps", "-eo", "args="]:
            return "\n".join(" ".join(process["command"]) for process in accepted["generation"].values())
        if "--query-gpu=index,uuid,name" in command:
            return "\n".join(",".join(row) for row in accepted["devices"]) + "\n"
        if "--query-compute-apps=pid,gpu_uuid" in command:
            return "\n".join(",".join(row) for row in occupied) + "\n"
        if "--query-compute-apps=pid" in command:
            return "\n".join(row[0] for row in occupied)
        raise AssertionError(f"Unanticipated external observation: {command}")

    def syscall(number, pid, flags):
        handle, writer = os.pipe()
        os.write(writer, b"ready")
        os.close(writer)
        fd_pids[handle] = pid.value
        return handle

    def signal_fd(handle, signum, *args):
        assert signum == signal.SIGTERM
        pid = fd_pids[handle]
        events.append(("release", pid))
        if fault == "partial_release" and pid == 101:
            return
        shutil.rmtree(proc / str(pid))
        occupied[:] = [row for row in occupied if int(row[0]) != pid]

    class FailedValidator:
        def __init__(self, command, **kwargs):
            assert command == controller.validator_command(root, item.candidate)
            self.pid, self.returncode = 25001, None
            process = {
                "pid": self.pid,
                "parent": controller_pid,
                "status": "R",
                "group": self.pid,
                "session": self.pid,
                "start_ticks": "validator-original",
                "command": command,
            }
            proc_record(proc, process, item.candidate, kwargs["env"])
            events.append(("validator", self.pid))

        def poll(self):
            return self.returncode

        def wait(self, timeout):
            self.returncode = 7
            shutil.rmtree(proc / str(self.pid))
            return 7

    def spawn(command, **kwargs):
        if command[:3] == [controller.PYTHON, "-P", "-c"]:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            return real_spawn(command, **kwargs)
        return FailedValidator(command, **kwargs)

    def run(command, **kwargs):
        if command == [controller.PYTHON, "-m", "experiment.bootstrap", "--check-only"]:
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["bash", str(manager), "start"]:
            assert kwargs["pass_fds"] == (int(kwargs["env"]["GPU_RESOURCE_MANAGER_LOCK_FD"]),)
            events.append(("restore", 300))
            restored = copy.deepcopy(accepted)
            restored["pid"] = 300
            restored["generation"] = {}
            occupied.clear()
            for offset, old in enumerate(accepted["generation"].values()):
                pid = 300 + offset
                process = old | {
                    "pid": pid,
                    "parent": 1 if offset == 0 else 300,
                    "group": pid,
                    "session": pid,
                    "start_ticks": f"restored-{pid}",
                }
                restored["generation"][str(pid)] = process
                proc_record(proc, process, root, {})
                if offset:
                    occupied.append([str(pid), accepted["devices"][offset - 1][1]])
            pointer.write_text("300")
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, **kwargs)

    with frozen_import(helpers) as archived:
        failure = sys.modules["selected_lr_failure_handoff"]
        monkeypatch.setattr(archived.handoff, "PROC_ROOT", proc)
        monkeypatch.setattr(archived.handoff, "KEEPER_PID", pointer)
        monkeypatch.setattr(failure, "KEEPER_PID", pointer)
        monkeypatch.setattr(failure, "MANAGER_LOCK", manager_lock)
        monkeypatch.setattr(failure, "KEEPER", str(manager))
        monkeypatch.setattr(failure, "Path", lambda value: proc if value == "/proc" else Path(value))
        monkeypatch.setattr(failure.LIBC, "syscall", syscall)
        monkeypatch.setattr(subprocess, "check_output", output)
        monkeypatch.setattr(subprocess, "Popen", spawn)
        monkeypatch.setattr(subprocess, "run", run)
        monkeypatch.setattr(signal, "pidfd_send_signal", signal_fd)
        with pytest.raises(RuntimeError):
            archived.validate(root)
    saved = controller.read(state_path)
    assert saved["status"] == "needs_attention"
    assert not receipt.exists()
    if fault == "rank_gpu_swap":
        assert events == [] and pointer.read_text() == "100"
        assert "keeper_release_attempted" not in saved
    elif fault == "partial_release":
        assert saved["keeper_release_attempted"] is True
        assert saved["keeper_release_complete"] is False
        assert saved["keeper_restore"].startswith("refused:")
        assert not any(event[0] in ("validator", "restore") for event in events)
        assert pointer.read_text() == "100"
    else:
        assert saved["keeper_release_complete"] is True
        assert saved["exit_code"] == saved["cleanup_exit_code"] == 7
        assert saved["keeper_restore"] == 0 and saved["restored_keeper"]["pid"] == 300
        assert len([event for event in events if event[0] == "release"]) == 9
        assert events[-2:] == [("validator", 25001), ("restore", 300)]
        assert not any(201 == pid or 202 == pid for event, pid in events if event == "release")
