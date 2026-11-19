# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test full-model candidate preservation, variant receipts and real detached-child cleanup."""

import ast
import copy
import fcntl
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import prepare_rational_continuation as preparation
import pytest
import rational_continuation_contract as contract
import rational_continuation_controller as controller
import rational_continuation_validation as validation
import rational_root_start as installation
import torch
from fused_attribution_validation_test_support import keeper, record
from selected_lr_release_owner import candidate_files, read, sha256

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    frozen = read(ROOT / preparation.PARENT_BINDING)
    sources = preparation.helper_sources(ROOT)
    fixtures = preparation.fixture_sources(ROOT)
    record(root / preparation.PARENT_BINDING, frozen)
    data = root / "data/experiment_receipt.json"
    data.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "data/experiment_receipt.json", data)
    predecessor_path = root / "run-preparation/predecessor.json"
    record(predecessor_path, {"fixture": "completed independently reconstructed start trial"})
    predecessor = {
        "restored_keeper": keeper(),
        "evidence_sha256": {str(predecessor_path.relative_to(root)): sha256(predecessor_path)},
    }
    # Source/evidence lookup is the external fixture boundary; transformations and checks remain real.
    monkeypatch.setattr(preparation, "parent", lambda root: frozen)
    monkeypatch.setattr(preparation, "helper_sources", lambda root: sources)
    monkeypatch.setattr(preparation, "fixture_sources", lambda root: fixtures)
    monkeypatch.setattr(preparation.predecessor, "evidence", lambda root: predecessor)
    prepared = preparation.prepare(root, candidate)
    return root, candidate, prepared, sources


def test_full_model_candidate_retains_parent_numerics_and_denies_unowned_training(candidate, tmp_path):
    root, path, prepared, _ = candidate
    parent = Path(prepared["parent_candidate"])
    original = candidate_files(parent)
    assert preparation.require_preparation(root, path) == prepared
    assert {
        name
        for name in original.keys() | prepared["files_sha256"].keys()
        if original.get(name) != prepared["files_sha256"].get(name)
    } == preparation.CHANGES
    assert (path / contract.CONFIG).read_bytes() == (parent / contract.CONFIG).read_bytes()
    ast.parse((path / "experiment/train.py").read_bytes())
    commands = contract.validation_commands(path, root, contract.locations(root)[3])
    assert len(commands) == 12 and len({tuple(command) for command in commands}) == 12
    for coordinate, command in zip(contract.coordinates(), commands, strict=True):
        variant, protocol, mode = coordinate
        assert command[command.index("--root-start-variant") + 1] == variant
        assert ("--trace-continuation" in command) == (protocol == "trace")
        assert ("--resume" in command) == (mode == "resumed")
    target = tmp_path / "unowned-output"
    denied = subprocess.run(
        [
            contract.PYTHON,
            "-m",
            "experiment.train",
            "--config",
            str(path / contract.CONFIG),
            "--data",
            str(root / "data"),
            "--data-receipt",
            str(root / "data/experiment_receipt.json"),
            "--out",
            str(target),
            "--launch-release",
            str(contract.locations(root)[0]),
            "--root-start-variant",
            "rational",
            "--max-steps",
            "24",
        ],
        cwd=path,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(path), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert denied.returncode != 0 and not target.exists()
    assert candidate_files(parent) == original and candidate_files(path) == prepared["files_sha256"]


@pytest.mark.parametrize("fault", ("optimizer", "owner", "config"))
def test_rehashed_candidate_cannot_accept_changed_scope(candidate, fault):
    root, path, prepared, _ = candidate
    changed = (
        path
        / {
            "optimizer": "experiment/block_toner.py",
            "owner": "experiment/fused_attribution_owner.py",
            "config": contract.CONFIG,
        }[fault]
    )
    changed.write_bytes(changed.read_bytes() + b"\n")
    record(path / contract.PREPARATION, prepared | {"files_sha256": candidate_files(path)})
    with pytest.raises(ValueError, match="exact candidate transformation"):
        preparation.require_preparation(root, path)


@pytest.mark.parametrize("fault", ("none", "variant", "rank", "identity", "installer", "alias"))
def test_resume_requires_original_per_rank_start_receipt(tmp_path, fault):
    path = tmp_path / "root_start_rank_3_from_000000.json"
    identity = {"source": "candidate", "data": "unchanged"}
    receipt = {
        "status": "source_bound_start_installed_before_model_and_root_warmup",
        "variant": "rational",
        "pinned_roots_sha256": installation.ROOTS_SHA256,
        "rational_start_sha256": installation.RATIONAL_SHA256,
        "installer_sha256": installation.digest(Path(installation.__file__)),
        "compilation_preserved": True,
        "compile_options": {"dynamic": False},
        "root_graph_lifecycle_unchanged": True,
        "rank": 3,
        "start_step": 0,
        "identity": identity,
    }
    if fault in ("variant", "rank", "identity", "installer"):
        key, value = {
            "variant": ("variant", "pinned"),
            "rank": ("rank", 2),
            "identity": ("identity", {"source": "other"}),
            "installer": ("installer_sha256", "0" * 64),
        }[fault]
        receipt[key] = value
    record(path, receipt)
    if fault == "alias":
        target = tmp_path / "aliased.json"
        path.rename(target)
        path.symlink_to(target)
    if fault == "none":
        installation.require_resume_start(tmp_path, 3, "rational", identity)
        assert read(path) == receipt
    else:
        with pytest.raises(ValueError, match="Checkpoint start"):
            installation.require_resume_start(tmp_path, 3, "rational", identity)


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "timeout"))
def test_real_bounded_validator_cleans_detached_descendants_and_preserves_exits(tmp_path, monkeypatch, outcome):
    _, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    source = helpers / "cpu_validator.py"
    source.write_text(
        "import json,os,select,signal,subprocess,sys\n"
        "from pathlib import Path\n"
        "state=Path(sys.argv[1]);poll=select.poll()\n"
        "while not state.exists() or json.loads(state.read_text()).get('child_pid')!=os.getpid():poll.poll(1)\n"
        "if sys.argv[2]=='timeout':signal.pause()\n"
        "if sys.argv[2]=='detached':\n"
        " r,w=os.pipe()\n"
        " child=subprocess.Popen([sys.executable,'-c',"
        "'import os,signal,sys;os.write(int(sys.argv[1]),b\"R\");signal.pause()',str(w)],"
        "pass_fds=(w,),start_new_session=True)\n"
        " os.close(w);assert os.read(r,1)==b'R';os.close(r)\n"
        " Path(sys.argv[3]).write_text(str(child.pid))\n"
        "sys.exit(0 if sys.argv[2]=='zero' else 7)\n"
    )
    command = [sys.executable, str(source), str(state_path), outcome, str(tmp_path / "detached.pid")]
    monkeypatch.setattr(contract, "validator_command", lambda root, candidate: command)
    monkeypatch.setattr(controller, "TIMEOUT", 0.25 if outcome == "timeout" else 10)
    environment = contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""}
    monkeypatch.setattr(contract, "ENVIRONMENT", environment)
    state = {"controller_pid": os.getpid(), "controller_start_ticks": "cpu-fixture"}
    if outcome == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            controller.bounded_validator(tmp_path, {"candidate": str(tmp_path)}, state)
    else:
        result = controller.bounded_validator(tmp_path, {"candidate": str(tmp_path)}, state)
        assert result == (0 if outcome == "zero" else 7)
    saved = read(state_path)
    assert saved["spawn_status"] == "acknowledged" and saved["marked_descendants_absent"] is True
    assert saved["cleanup_exit_code"] == (-15 if outcome == "timeout" else 0 if outcome == "zero" else 7)
    assert saved["child_generation"]["parent"] == os.getpid()
    assert controller.handoff.observed(saved["child_pid"]) is None
    assert not controller.handoff.marker_processes(contract.CHILD_MARKER, saved["child_marker"])
    assert saved["log_sha256"] == sha256(helpers.parent / "validator.log")
    if outcome == "detached":
        process = controller.handoff.observed(int((tmp_path / "detached.pid").read_text()))
        assert process is None or process["status"] == "Z"


@pytest.mark.parametrize("fault", ("none", "missing_module", "skipped_case", "bool_exit", "changed_source"))
def test_cpu_acceptance_requires_complete_xml_and_preserves_failed_attempt(candidate, tmp_path, monkeypatch, fault):
    root, path, _prepared, _ = candidate
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    out = tmp_path / "cpu-acceptance"
    modules = {name.stem for name in (path / "experiment/tests").glob("test_*.py")}
    modules |= {Path(name).stem for name in preparation.CPU_TESTS}

    def run_cpu(command, **kwargs):
        selected = sorted(modules)
        if fault == "missing_module":
            selected.pop()
        cases = "".join(
            f'<testcase classname="{name}" name="test_fixture">'
            f'{("<skipped/>" if fault == "skipped_case" and i == 0 else "")}</testcase>'
            for i, name in enumerate(selected)
        )
        (out / "tests.xml").write_text(
            f'<testsuites><testsuite tests="{len(selected)}" errors="0" failures="0" skipped="0">'
            f"{cases}</testsuite></testsuites>"
        )
        kwargs["stdout"].write("synthetic external complete CPU outcome\n")
        if fault == "changed_source":
            changed = path / "experiment/block_toner.py"
            changed.write_bytes(changed.read_bytes() + b"\n")
        return subprocess.CompletedProcess(command, False if fault == "bool_exit" else 0)

    monkeypatch.setattr(preparation.subprocess, "run", run_cpu)
    if fault == "none":
        result = preparation.cpu(root, path, out)
        assert preparation.require_cpu(root, out / "receipt.json") == result
        assert result["status"] == contract.CPU_STATUS
    else:
        with pytest.raises(RuntimeError, match="CPU failure preserved"):
            preparation.cpu(root, path, out)
        result = read(out / "receipt.json")
        assert result["status"] == "needs_attention"
    assert (out / "source.tar").is_file() and (out / "cpu.log").is_file()
    assert not contract.locations(root)[0].exists() and not contract.locations(root)[2].exists()


@pytest.mark.parametrize("outcome", ("spawn_failure", "nonzero"))
def test_execute_restores_locked_allocation_after_real_validator_failure(tmp_path, monkeypatch, outcome):
    binding, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    record(binding, {"fixture": "CPU external binding"})
    accepted, restored = keeper(), keeper()
    restored["generation"]["100"]["start_ticks"] = "restored-fixture-generation"
    own = {
        "pid": os.getpid(),
        "parent": os.getppid(),
        "group": os.getpid(),
        "session": os.getpid(),
        "start_ticks": "CPU-controller",
        "command": contract.controller_command(tmp_path),
        "status": "R",
    }
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "pid": own["pid"],
            "start_ticks": own["start_ticks"],
            "generation": own,
            "command": own["command"],
            "binding_sha256": sha256(binding),
            "marker": "cpu-controller",
        },
    )
    frozen = {"candidate": str(tmp_path), "allocation_predecessor": {"restored_keeper": accepted}}
    source = helpers / "validator_failure.py"
    source.write_text("import select,sys\nselect.poll().poll(100)\nsys.exit(7)\n")
    command = [sys.executable, str(source)]
    monkeypatch.setattr(contract, "validator_command", lambda root, candidate: command)
    monkeypatch.setattr(contract, "ENVIRONMENT", contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""})
    monkeypatch.setattr(controller, "binding", lambda root: frozen)
    monkeypatch.setattr(controller, "process_context", lambda *args: own)
    monkeypatch.setattr(controller, "local_node", lambda: contract.NODE)
    monkeypatch.setenv(contract.CONTROLLER_MARKER, "cpu-controller")
    events = []
    manager_path = tmp_path / "manager.lock"

    @contextmanager
    def manager():
        with manager_path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield stream.fileno()

    @contextmanager
    def node(*args):
        with (tmp_path / "node.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield []

    def host(command, **kwargs):
        if command == ["ps", "-eo", "args="]:
            return ""
        raise AssertionError(command)

    def release(keeper, state, path, handle):
        assert keeper == accepted and os.readlink(f"/proc/{os.getpid()}/fd/{handle}") == str(manager_path)
        state.update(keeper_release_attempted=True, keeper_release_complete=True)
        events.append("released")

    def restore(handle):
        assert os.readlink(f"/proc/{os.getpid()}/fd/{handle}") == str(manager_path)
        events.append("restored")

    monkeypatch.setattr(controller, "keeper_guard", manager)
    monkeypatch.setattr(controller, "node_guard", node)
    monkeypatch.setattr(controller.predecessor, "live", lambda *args: accepted)
    monkeypatch.setattr(controller, "release_accepted_keeper", release)
    monkeypatch.setattr(controller, "restore_accepted_keeper", restore)
    monkeypatch.setattr(controller.handoff, "keeper_observation", lambda: restored)
    monkeypatch.setattr(controller.subprocess, "check_output", host)
    monkeypatch.setattr(controller.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0))
    if outcome == "spawn_failure":

        def fail(*args, **kwargs):
            raise OSError("CPU validator spawn failure")

        monkeypatch.setattr(controller.subprocess, "Popen", fail)
    with pytest.raises((OSError, RuntimeError)):
        controller.execute(tmp_path)
    saved = read(state_path)
    assert saved["status"] == "needs_attention" and saved["keeper_restore"] == 0
    assert saved["restored_keeper"] == restored and events == ["released", "restored"]
    if outcome == "nonzero":
        assert saved["exit_code"] == saved["cleanup_exit_code"] == 7
        assert saved["marked_descendants_absent"] is True
    assert not controller.handoff.marker_processes(contract.CHILD_MARKER, saved["child_marker"])
    with pytest.raises(RuntimeError):
        controller.execute(tmp_path)
    assert read(state_path) == saved


@pytest.mark.parametrize("fault", ("none", "missing_rank", "variant", "identity", "extra_receipt", "hardlink"))
def test_retained_start_receipts_bind_every_rank_and_resume_without_changing_prefix(
    candidate, tmp_path, monkeypatch, fault
):
    _root, path, prepared, _ = candidate
    out = tmp_path / "retained-start"
    identity = {"source_sha256": prepared["files_sha256"], "data_receipt_sha256": prepared["data_receipt_sha256"]}
    # The full model reader is the expensive source/data boundary; receipts and file authentication remain real.
    monkeypatch.setattr(
        validation,
        "numerical_modules",
        lambda source: (None, SimpleNamespace(checkpoint_identity=lambda *args: identity), None),
    )
    records = {}
    for step in (0, contract.BOUNDARY):
        for rank in range(contract.WORLD):
            receipt = {
                "status": "source_bound_start_installed_before_model_and_root_warmup",
                "variant": "rational",
                "pinned_roots_sha256": installation.ROOTS_SHA256,
                "rational_start_sha256": installation.RATIONAL_SHA256,
                "installer_sha256": sha256(path / "experiment/rational_root_start.py"),
                "compilation_preserved": True,
                "compile_options": {"dynamic": False},
                "root_graph_lifecycle_unchanged": True,
                "rank": rank,
                "start_step": step,
                "identity": identity,
            }
            target = out / f"root_start_rank_{rank}_from_{step:06d}.json"
            record(target, receipt)
            records[target.name] = sha256(target)
    target = out / f"root_start_rank_7_from_{contract.BOUNDARY:06d}.json"
    if fault == "missing_rank":
        target.unlink()
    elif fault in ("variant", "identity"):
        value = read(target)
        value[fault] = "pinned" if fault == "variant" else {"source": "another"}
        record(target, value)
    elif fault == "extra_receipt":
        record(out / "root_start_rank_0_from_000023.json", read(target))
    elif fault == "hardlink":
        os.link(target, tmp_path / "alias.json")
    if fault == "none":
        assert validation.start_receipts(out, path, "rational", "resumed") == records
        prefix = {name: value for name, value in records.items() if name.endswith("_from_000000.json")}
        assert validation.start_receipts(out, path, "rational", "interrupted") == prefix
        assert {name: sha256(out / name) for name in records} == records
    else:
        with pytest.raises(ValueError, match="Rational continuation"):
            validation.start_receipts(out, path, "rational", "resumed")


@pytest.mark.parametrize(
    "fault", ("none", "missing_command", "wrong_coordinate", "missing_rank", "changed_rank", "source")
)
def test_original_reconstruction_requires_complete_six_command_history_and_all_rank_bytes(candidate, fault):
    root, path, _, _ = candidate
    binding_path = root / preparation.PARENT_BINDING
    frozen = read(binding_path)
    parent = Path(frozen["candidate"])
    work = root / "validation/fused_continuation_20261118"
    commands = validation.original_commands(parent, root, work)
    sources = validation.executed_sources(parent, frozen)
    entries = []
    for (protocol, mode), command in zip(
        [(protocol, mode) for protocol in contract.PROTOCOLS for mode in contract.MODES], commands, strict=True
    ):
        out = work / f"{protocol}_{'continuous' if mode == 'continuous' else 'resumed'}"
        config = read(path / contract.CONFIG)
        manifest = {
            "config": config,
            "source_sha256": sources,
            "data_receipt_sha256": frozen["data_receipt_sha256"],
        }
        manifest_path = out / (validation.INTERRUPTED_MANIFEST if mode == "interrupted" else "run.json")
        record(manifest_path, manifest)
        record(out / "config.json", config)
        prefix = out / "metrics_from_000000.jsonl"
        record(prefix, {"fixture": "complete historical prefix", "train_loss": 6.0})
        history = {prefix.name: sha256(prefix)}
        if mode == "resumed":
            resumed = out / "metrics_from_000022.jsonl"
            record(resumed, {"fixture": "historical resumed suffix", "train_loss": 6.1})
            history[resumed.name] = sha256(resumed)
        checkpoints = {}
        for step in ((22,) if mode == "interrupted" else validation.STEPS):
            folder = out / f"checkpoints/step_{step:06d}"
            record(folder / "complete.json", {"step": step, "world": 8})
            ranks = {}
            for rank in range(8):
                rank_path = folder / f"rank_{rank}.pt"
                if not rank_path.exists():
                    torch.save({"rank": rank, "step": step, "state": torch.ones(2)}, rank_path)
                ranks[rank_path.name] = sha256(rank_path)
            checkpoints[str(step)] = {"receipt_sha256": sha256(folder / "complete.json"), "rank_sha256": ranks}
        log = work / f"{protocol}_{mode}.log"
        log.write_text("CPU historical output\n")
        entries.append(
            {
                "protocol": protocol,
                "mode": mode,
                "command": command,
                "exit_code": 0,
                "log": str(log),
                "log_sha256": sha256(log),
                "trajectory": {
                    "manifest_sha256": sha256(manifest_path),
                    "history_sha256": history,
                    "checkpoints": checkpoints,
                },
            }
        )
    receipt = root / "run-preparation/diagnostic-fused-continuation-gpu-20261118.json"
    result = {
        "status": validation.ORIGINAL_STATUS,
        "execution_binding_sha256": sha256(binding_path),
        "candidate_preparation_sha256": frozen["preparation_sha256"],
        "commands": commands,
        "gpu_commands": entries,
    }
    if fault == "missing_command":
        entries.pop()
    elif fault == "wrong_coordinate":
        entries[0]["protocol"] = "trace"
    elif fault == "missing_rank":
        entries[-1]["trajectory"]["checkpoints"]["24"]["rank_sha256"].pop("rank_7.pt")
    elif fault == "changed_rank":
        (work / "trace_resumed/checkpoints/step_000024/rank_7.pt").write_bytes(b"changed retained checkpoint")
    elif fault == "source":
        changed = work / "control_continuous/run.json"
        manifest = read(changed)
        manifest["source_sha256"] = {}
        record(changed, manifest)
        entries[0]["trajectory"]["manifest_sha256"] = sha256(changed)
    record(receipt, result)
    before = sha256(receipt)
    if fault == "none":
        assert validation.original(root, path) == result
    else:
        with pytest.raises(ValueError, match="Original full-model"):
            validation.original(root, path)
    assert sha256(receipt) == before


@pytest.mark.parametrize("fault", ("none", "boolean_exit", "wrong_parent", "live_owner", "chronology", "keeper", "log"))
def test_terminal_requires_original_owner_absence_zero_exits_and_exact_restoration(tmp_path, monkeypatch, fault):
    binding, helpers, state_path, _, result_path = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    frozen = {
        "candidate": str(tmp_path / "candidate"),
        "allocation_predecessor": {"fixture": "original keeper"},
        "scope": "diagnostic",
    }
    record(binding, frozen)
    controller_generation = {
        "pid": 900001,
        "parent": 1,
        "start_ticks": "cpu-controller",
        "group": 900001,
        "session": 900001,
        "command": contract.controller_command(tmp_path),
    }
    child = {
        "pid": 900002,
        "parent": 900001,
        "start_ticks": "cpu-validator",
        "group": 900002,
        "session": 900002,
        "command": contract.validator_command(tmp_path, Path(frozen["candidate"])),
    }
    intent = {
        "status": "acknowledged",
        "pid": controller_generation["pid"],
        "start_ticks": controller_generation["start_ticks"],
        "generation": controller_generation,
        "command": controller_generation["command"],
        "binding_sha256": sha256(binding),
        "marker": "CPU-terminal-controller",
        "time": "2026-11-18T20:00:00+08:00",
    }
    record(helpers.parent / "controller.json", intent)
    (helpers.parent / "controller.log").write_text("CPU controller output\n")
    log = helpers.parent / "validator.log"
    log.write_text("CPU validator output\n")
    result = {"time": "2026-11-18T20:01:00+08:00", "analysis": {"diagnostic_intervention_accepted": False}}
    record(result_path, result)
    restored = keeper()
    state = {
        "status": "diagnostic_complete",
        "keeper_restore": 0,
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_release_attempted": True,
        "keeper_release_complete": True,
        "marked_descendants_absent": True,
        "spawn_status": "acknowledged",
        "controller_pid": intent["pid"],
        "controller_start_ticks": intent["start_ticks"],
        "child_pid": child["pid"],
        "child_start_ticks": child["start_ticks"],
        "child_generation": child,
        "child_marker": "CPU-terminal-validator",
        "command": child["command"],
        "allocation_predecessor": frozen["allocation_predecessor"],
        "binding_sha256": sha256(binding),
        "log": str(log),
        "log_sha256": sha256(log),
        "result_sha256": sha256(result_path),
        "intervention_accepted": False,
        "restored_keeper": restored,
        "time": "2026-11-18T20:02:00+08:00",
    }
    actual_keeper = copy.deepcopy(restored)
    if fault == "boolean_exit":
        state["exit_code"] = False
    elif fault == "wrong_parent":
        child["parent"] = 900003
    elif fault == "chronology":
        state["time"] = "2026-11-18T19:00:00+08:00"
    elif fault == "keeper":
        actual_keeper["generation"]["101"]["start_ticks"] = "replacement"
    elif fault == "log":
        log.write_text("changed CPU validator output\n")
    record(state_path, state)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Expensive full-model reconstruction and physical host inspection are external evidence boundaries.
    monkeypatch.setattr(controller, "binding", lambda root: frozen)
    monkeypatch.setattr(controller.validation, "reconstruct", lambda *args: result)
    monkeypatch.setattr(controller, "local_node", lambda: contract.NODE)
    monkeypatch.setattr(
        controller.handoff, "observed", lambda pid: controller_generation if fault == "live_owner" else None
    )
    monkeypatch.setattr(controller.handoff, "marker_processes", lambda *args: {})
    monkeypatch.setattr(controller.handoff, "keeper_observation", lambda: actual_keeper)
    lock_path = tmp_path / "terminal-manager.lock"

    @contextmanager
    def manager():
        with lock_path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield stream.fileno()

    monkeypatch.setattr(controller, "keeper_guard", manager)
    out = tmp_path / "run-preparation/terminal.json"
    before = {name: sha256(source) for name, source in (("state", state_path), ("result", result_path))}
    if fault == "none":
        saved = controller.terminal(tmp_path, out)
        assert read(out) == saved and saved["analysis"]["diagnostic_intervention_accepted"] is False
        assert saved["keeper"] == restored and saved["controller_pid_absent"] == intent["pid"]
        assert all(sha256(tmp_path / name) == value for name, value in saved["evidence_sha256"].items())
    else:
        with pytest.raises((RuntimeError, ValueError)):
            controller.terminal(tmp_path, out)
        assert not out.exists()
    assert {name: sha256(source) for name, source in (("state", state_path), ("result", result_path))} == before
