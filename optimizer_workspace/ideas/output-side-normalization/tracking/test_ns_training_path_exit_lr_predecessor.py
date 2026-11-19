# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise persisted CPU reader boundaries, one-shot sealing and closed exit-drain gates."""

import copy
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ns_training_path_exit_lr_predecessor as predecessor
import prepare_ns_training_path_exit_lr_predecessor as preparation
import pytest

READER = """
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(path): return json.loads(Path(path).read_text())
def snapshot(): return read(ROOT/'reader-snapshot.json')
def accepted(root,path): return read(path)
def frozen(root): return snapshot()['frozen']
def command(root,path): return snapshot()['waiter_command']
class Contract:
 CONTROLLER_MARKER='probe_controller'
 CHILD_MARKER='probe_validator'
 COMMAND_MARKER='probe_command'
 def locations(self,root):
  value=snapshot()['paths']
  return tuple(Path(value[name]) for name in ('binding','helpers','state','work','result'))
contract=Contract()
class Controller:
 def terminal_paths(self,root,intent):
  value=snapshot()
  return tuple(Path(value[name]) for name in ('terminal','attempt','outcome'))+(ROOT/'unused',)
controller=Controller()
MARKER='probe_waiter'
CHILD_MARKER='probe_terminal'
WORK='run-preparation/ns-training-path-exit-drain-terminal-followup-20261119'
ENVIRONMENT={'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONFAULTHANDLER':'1'}
class Handoff:
 def marker_processes(self,name,value): return read(ROOT/'marker-snapshot.json').get(name,{})
 def keeper_observation(self): return read(ROOT/'current-keeper.json')
handoff=Handoff()
def handle(expected,cwd,marker,value,environment):
 directory=ROOT/'fixture-proc'/str(expected['pid'])
 if directory.exists():
  if not (directory/'generation.json').exists(): raise RuntimeError('Incomplete process evidence')
  actual=read(directory/'generation.json')
  if any(actual[key]!=expected[key] for key in ('pid','start_ticks','group','session')):
   raise RuntimeError('Substituted process generation')
  if actual['status'] in ('Z','X','x'): return dict(status='exited_handle_present',expected=expected)
  if actual['command']!=expected['command']: raise RuntimeError('Exec changed command')
  return dict(status='live_exact_generation',expected=expected)
 if (Path('/proc')/str(expected['pid'])).exists():
  return dict(status='live_exact_generation',expected=expected)
 return dict(status='absent',expected=expected)
"""
REPORTER = """
import json
class Report:
 def evidence(self,root): return json.loads((root/'report-snapshot.json').read_text())
def report_module(root): return Report()
"""


def record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generation(pid: int, parent: int, command: list[str]) -> dict:
    return {
        "pid": pid,
        "parent": parent,
        "start_ticks": str(pid + 100),
        "group": pid,
        "session": pid,
        "command": command,
        "status": "S",
    }


@pytest.fixture
def completed(tmp_path, monkeypatch):
    """Use isolated executable reader files as external accepted-evidence fixtures."""
    root = tmp_path
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(predecessor, "PYTHON", sys.executable)
    reader = root / "reader/tracking/ns_training_path_exit_followup.py"
    reader.parent.mkdir(parents=True)
    reader.write_text(READER)
    reporter = reader.with_name("ns_training_path_exit_reporting.py")
    reporter.write_text(REPORTER)
    monkeypatch.setattr(predecessor, "__file__", str(reader.with_name("ns_training_path_exit_lr_predecessor.py")))
    Path(predecessor.__file__).write_text("# CPU-only predecessor identity fixture\n")
    monkeypatch.setattr(predecessor, "REPORTER_SHA256", digest(reporter))
    monkeypatch.setattr(predecessor, "REPORT_SOURCES", {})
    monkeypatch.setattr(predecessor, "FOLLOWUP_SHA256", digest(reader))
    monkeypatch.setattr(predecessor, "FOLLOWUP_CPU", "reader/receipt.json")
    bootstrap = root / "bootstrap.json"
    record(
        bootstrap, {"root": str(root), "new_helpers_sha256": {reader.name: digest(reader)}, "issued_helpers_sha256": {}}
    )
    # Fill the external archive inventory; production requires exactly169 accepted sources.
    sources = {reader.name: digest(reader)}
    for index in range(168):
        path = reader.with_name(f"source_{index}.py")
        path.write_text("# inherited CPU source\n")
        sources[path.name] = digest(path)
    data = json.loads(bootstrap.read_text())
    data["issued_helpers_sha256"] = {name: value for name, value in sources.items() if name != reader.name}
    record(bootstrap, data)
    monkeypatch.setattr(predecessor, "BOOTSTRAP", str(bootstrap.relative_to(root)))
    monkeypatch.setattr(predecessor, "BOOTSTRAP_SHA256", digest(bootstrap))
    tests = root / "reader/tests.xml"
    tests.write_text(
        '<testsuites><testsuite tests="24" errors="0" failures="0" skipped="0">'
        + "".join(f'<testcase classname="external_reader" name="case_{index}"/>' for index in range(24))
        + "</testsuite></testsuites>"
    )
    archive, cpu_log = root / "reader/source.tar", root / "reader/cpu.log"
    archive.write_bytes(b"external accepted CPU archive boundary")
    cpu_log.write_text("CPU evidence fixture\n")
    cpu = {
        "root": str(root),
        "status": "ns_training_path_exit_drain_read_only_followup_cpu_archive_verified",
        "exit_code": 0,
        "helpers": str(reader.parent),
        "helpers_sha256": sources,
        "environment": predecessor.ENVIRONMENT,
        "binding_sha256": predecessor.BINDING_SHA256,
        "original_intent_sha256": predecessor.INTENT_SHA256,
        **{key: str(path) for key, path in (("archive", archive), ("tests", tests), ("log", cpu_log))},
        **{f"{key}_sha256": digest(path) for key, path in (("archive", archive), ("tests", tests), ("log", cpu_log))},
    }
    cpu_path = root / predecessor.FOLLOWUP_CPU
    record(cpu_path, cpu)
    monkeypatch.setattr(predecessor, "FOLLOWUP_CPU_SHA256", digest(cpu_path))
    helpers = root / "issued/tracking"
    helpers.mkdir(parents=True)
    controller = helpers / "ns_training_path_exit_gpu_controller.py"
    controller.write_text("# issued independent verifier fixture\n")
    paths = dict(
        binding=str(root / predecessor.BINDING),
        helpers=str(helpers),
        state=str(root / "state.json"),
        work=str(root / "trajectories"),
        result=str(root / "result.json"),
    )
    control = [sys.executable, "-c", "import signal; signal.pause()"]
    frozen = {
        "root": str(root),
        "helpers": str(helpers),
        "candidate": str(root / "candidate"),
        "helpers_sha256": {controller.name: digest(controller)},
        "environment": {"CUDA_VISIBLE_DEVICES": ""},
        "controller_environment": {"CUDA_VISIBLE_DEVICES": ""},
        "controller_command": control,
        "validator_command": [*control, "validator"],
    }
    keeper = {
        "pid": 91000000,
        "devices": [[str(rank), f"probe-{rank}", "A100 CPU fixture"] for rank in range(8)],
        "occupied": [[str(91000001 + rank), f"probe-{rank}"] for rank in range(8)],
        "generation": {
            str(91000000 + rank): generation(91000000 + rank, 1 if rank == 0 else 91000000, ["keeper"])
            for rank in range(9)
        },
    }
    record(root / "current-keeper.json", keeper)
    record(root / "marker-snapshot.json", {})
    lineage_path = root / "failed-evidence.json"
    record(lineage_path, {"original": "failed", "recovery": "failed"})
    lineage_payload = root / "failed-rank.pt"
    lineage_payload.write_bytes(b"retained failed checkpoint")
    failed = {
        "status": "both_failed_training_path_attempts_sealed_for_distinct_exit_drain_not_numerical_acceptance",
        "root": str(root),
        "production_authorized": False,
        "gpu_execution_authorized": False,
        "evidence": {
            "failed-evidence.json": {"sha256": digest(lineage_path), "identity": predecessor.file_identity(lineage_path)}
        },
        "checkpoint_payloads": {
            "failed-rank.pt": {"sha256": digest(lineage_payload), "identity": predecessor.file_identity(lineage_payload)}
        },
        "retained_summary": dict(
            completed_commands=2, case_decisions=0, coverage_complete=False, numerically_accepted=False
        ),
        "original_retained_summary": dict(
            completed_commands=44, case_decisions=14, coverage_complete=False, numerically_accepted=False
        ),
    }
    record(root / predecessor.FAILED_SEAL, failed)
    monkeypatch.setattr(predecessor, "FAILED_SEAL_SHA256", digest(root / predecessor.FAILED_SEAL))
    frozen["allocation_predecessor"] = failed
    frozen["predecessor_acceptance"] = {"seal_sha256": predecessor.FAILED_SEAL_SHA256}
    frozen["devices"] = keeper["devices"]
    intent = dict(
        status="acknowledged",
        pid=90000001,
        start_ticks="90000101",
        command=control,
        generation=generation(90000001, 1, control),
        marker="controller-fixture",
    )
    waiter_work = root / predecessor.FOLLOWUP_WORK
    waiter_command = [sys.executable, "-P", str(reader), "--phase", "execute"]
    waiter = dict(
        status="acknowledged",
        pid=90000003,
        start_ticks="90000103",
        command=waiter_command,
        generation=generation(90000003, 1, waiter_command),
        marker="waiter-fixture",
        runner_sha256=predecessor.FOLLOWUP_SHA256,
        cpu_receipt_sha256=predecessor.FOLLOWUP_CPU_SHA256,
    )
    cases, entries, commands = [], [], []
    for index in range(18):
        name = f"exit-case-{index}"
        directories = {mode: root / "trajectories" / f"{name}_{mode}" for mode in ("continuous", "resumed")}
        trajectories = {}
        for mode in predecessor.MODES:
            directory = directories["continuous" if mode == "continuous" else "resumed"]
            record(directory / "config.json", {"fixture": name})
            manifest = "attribution-interrupted-manifest.json" if mode == "interrupted" else "run.json"
            record(directory / manifest, {"fixture": mode})
            history = directory / f"{mode}.jsonl"
            history.write_text('{"scalar":1}\n')
            checkpoints = {}
            for step in sorted(predecessor.CHECKPOINTS[mode]):
                folder = directory / f"checkpoints/step_{int(step):06d}"
                record(folder / "complete.json", {"step": step})
                ranks = {}
                for rank in range(8):
                    path = folder / f"rank_{rank}.pt"
                    if not path.exists():
                        path.write_bytes(f"retained checkpoint {rank}".encode())
                    ranks[path.name] = digest(path)
                checkpoints[step] = dict(
                    receipt_sha256=digest(folder / "complete.json"),
                    rank_sha256=ranks,
                    rank_bytes=[(folder / rank).stat().st_size for rank in ranks],
                )
            trajectories[mode] = dict(
                manifest_sha256=digest(directory / manifest),
                history_sha256={history.name: digest(history)},
                checkpoints=checkpoints,
            )
            command = [sys.executable, "CPU-fixture", name, mode]
            commands.append(command)
            log = root / "logs" / f"{name}_{mode}.log"
            log.parent.mkdir(exist_ok=True)
            log.write_text("retained command fixture\n")
            entry = dict(
                command=command,
                trial=name,
                mode=mode,
                log=str(log),
                log_sha256=digest(log),
                marker=f"probe-{index}-{mode}",
                generation=generation(90000100 + len(entries), 90000002, command),
            )
            record(log.with_suffix(".execution.json"), entry)
            entries.append(entry)
        cases.append(
            dict(
                name=name,
                accepted=index != 0,
                continuous=str(directories["continuous"]),
                resumed=str(directories["resumed"]),
                trajectories=trajectories,
            )
        )
    decisions = dict(
        accepted_cases=[case["name"] for case in cases if case["accepted"]],
        rejected_cases=["exit-case-0"],
        all_cases_accepted=False,
    )
    frozen["commands"] = commands
    execution_cpu = {
        "helpers_sha256": frozen["helpers_sha256"],
        "files_sha256": {"config.json": "fixture-config"},
    }
    monkeypatch.setattr(predecessor, "EXIT_CPU", "execution-cpu.json")
    record(root / predecessor.EXIT_CPU, execution_cpu)
    monkeypatch.setattr(predecessor, "EXIT_CPU_SHA256", digest(root / predecessor.EXIT_CPU))
    frozen["files_sha256"] = execution_cpu["files_sha256"]
    frozen["scope"] = "CPU-only external numerical boundary fixture"
    frozen["cpu_receipt"] = str(root / predecessor.EXIT_CPU)
    frozen["cpu_receipt_sha256"] = predecessor.EXIT_CPU_SHA256
    frozen["archive"] = str(archive)
    frozen["archive_sha256"] = digest(archive)
    frozen["plan"] = dict(commands=commands, trials=[{"name": case["name"]} for case in cases])
    record(root / predecessor.BINDING, frozen)
    monkeypatch.setattr(predecessor, "BINDING_SHA256", digest(root / predecessor.BINDING))
    cpu["binding_sha256"] = predecessor.BINDING_SHA256
    record(cpu_path, cpu)
    monkeypatch.setattr(predecessor, "FOLLOWUP_CPU_SHA256", digest(cpu_path))
    waiter["cpu_receipt_sha256"] = predecessor.FOLLOWUP_CPU_SHA256
    for value in (intent, waiter):
        value["binding_sha256"] = predecessor.BINDING_SHA256
    record(helpers.parent / "controller.json", intent)
    monkeypatch.setattr(predecessor, "INTENT_SHA256", digest(helpers.parent / "controller.json"))
    cpu["original_intent_sha256"] = predecessor.INTENT_SHA256
    record(cpu_path, cpu)
    monkeypatch.setattr(predecessor, "FOLLOWUP_CPU_SHA256", digest(cpu_path))
    waiter["cpu_receipt_sha256"] = predecessor.FOLLOWUP_CPU_SHA256
    record(waiter_work / "intent.json", waiter)
    monkeypatch.setattr(predecessor, "FOLLOWUP_INTENT_SHA256", digest(waiter_work / "intent.json"))
    now = datetime.now(UTC) + timedelta(seconds=2)
    result = dict(
        status="ns_training_path_exit_drain_gates_reconstructed_not_production",
        results=cases,
        gpu_commands=entries,
        trials=frozen["plan"]["trials"],
        binding_sha256=predecessor.BINDING_SHA256,
        time=now.isoformat(),
        **decisions,
    )
    record(Path(paths["result"]), result)
    state = dict(
        status="validation_complete",
        spawn_status="acknowledged",
        controller_pid=intent["pid"],
        controller_start_ticks=intent["start_ticks"],
        child_pid=90000002,
        child_start_ticks="90000102",
        child_generation=generation(90000002, intent["pid"], frozen["validator_command"]),
        command=frozen["validator_command"],
        child_marker="validator-fixture",
        binding_sha256=predecessor.BINDING_SHA256,
        result_sha256=digest(Path(paths["result"])),
        exit_code=0,
        cleanup_exit_code=0,
        keeper_restore=0,
        marked_descendants_absent=True,
        keeper_release_complete=True,
        restored_keeper=keeper,
        time=now.isoformat(),
        **decisions,
    )
    record(Path(paths["state"]), state)
    keys = dict(
        binding_sha256=predecessor.BINDING_SHA256,
        controller_pid=intent["pid"],
        controller_start_ticks=intent["start_ticks"],
        controller_marker=intent["marker"],
    )
    token = hashlib.sha256(json.dumps(keys, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    terminal_path = root / "run-preparation" / f"ns-training-path-exit-drain-20261119-terminal-{token}.json"
    attempt_path, outcome_path = helpers.parent / "terminal-intent.json", helpers.parent / "terminal-outcome.json"
    claim = dict(
        status="reconstruction_claimed",
        reconstructor_pid=90000004,
        output=str(terminal_path),
        verifier_sha256=digest(controller),
        time=now.isoformat(),
        **keys,
    )
    record(attempt_path, claim)
    terminal = dict(
        status="ns_training_path_exit_drain_terminal_independently_reconstructed_not_production",
        binding_sha256=predecessor.BINDING_SHA256,
        controller_pid_absent=intent["pid"],
        controller_start_ticks=intent["start_ticks"],
        controller_marker=intent["marker"],
        validator_pid_absent=state["child_pid"],
        verifier_sha256=digest(controller),
        result_sha256=state["result_sha256"],
        terminal_time=state["time"],
        predecessor_acceptance=frozen["predecessor_acceptance"],
        reconstruction_intent_sha256=digest(attempt_path),
        production_authorized=False,
        scope=frozen["scope"],
        keeper=keeper,
        evidence_sha256={
            str(Path(path).relative_to(root)): digest(Path(path))
            for path in (paths["binding"], paths["state"], paths["result"])
        },
        time=now.isoformat(),
        **decisions,
    )
    record(terminal_path, terminal)
    outcome = dict(
        status="complete",
        output=str(terminal_path),
        terminal_sha256=digest(terminal_path),
        reconstruction_intent_sha256=digest(attempt_path),
        time=now.isoformat(),
    )
    record(outcome_path, outcome)
    terminal_command = [
        sys.executable,
        "-P",
        str(controller),
        "--root",
        str(root),
        "--phase",
        "terminal",
        "--out",
        str(terminal_path),
    ]
    terminal_log = waiter_work / "terminal.log"
    terminal_log.write_text("retained independent terminal fixture\n")
    (waiter_work / "followup.log").write_text("retained independent waiter fixture\n")
    waiter_state = dict(
        status="terminal_reconstructed",
        pid=waiter["pid"],
        start_ticks=waiter["start_ticks"],
        binding_sha256=predecessor.BINDING_SHA256,
        runner_sha256=predecessor.FOLLOWUP_SHA256,
        cpu_receipt_sha256=predecessor.FOLLOWUP_CPU_SHA256,
        terminal_command=terminal_command,
        terminal_generation=generation(90000004, waiter["pid"], terminal_command),
        terminal_pid=90000004,
        terminal_marker="terminal-fixture",
        terminal_exit_code=0,
        terminal_cleanup_exit_code=0,
        terminal_descendants_absent=True,
        terminal_generations={},
        terminal=str(terminal_path),
        terminal_sha256=digest(terminal_path),
        terminal_outcome_sha256=digest(outcome_path),
        terminal_log=str(terminal_log),
        terminal_log_sha256=digest(terminal_log),
        time=now.isoformat(),
        **decisions,
    )
    record(waiter_work / "state.json", waiter_state)
    snapshot = dict(
        cpu=cpu,
        frozen=frozen,
        intent=intent,
        state=state,
        result=result,
        waiter=waiter,
        waiter_state=waiter_state,
        paths=paths,
        terminal=str(terminal_path),
        attempt=str(attempt_path),
        outcome=str(outcome_path),
        waiter_command=waiter_command,
    )
    record(root / "reader-snapshot.json", snapshot)
    report = dict(
        cpu=execution_cpu,
        sealed=failed,
        progress=dict(
            status="validation_complete",
            completed_commands=54,
            decisions=cases,
            state=state,
            intent=intent,
            result=result,
            terminal=terminal,
            **decisions,
        ),
    )
    record(root / "report-snapshot.json", report)
    manager_lock = root / "cpu-manager.lock"
    manager_lock.touch()
    monkeypatch.setattr(predecessor, "MANAGER_LOCK", manager_lock)
    return root, snapshot


def test_evidence_preserves_rejections_failed_lineage_and_payloads_without_reads(completed, monkeypatch):
    root, snapshot = completed
    original = Path.read_bytes

    def read_bytes(path):
        if path.suffix == ".pt":
            raise AssertionError("Predecessor attempted to read a tensor payload")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    before = {path: predecessor.file_identity(path) for path in root.rglob("*") if path.is_file()}
    result = predecessor.evidence(root)
    assert result["original_decisions"]["rejected_cases"] == ["exit-case-0"]
    assert result["original_decisions"]["all_cases_accepted"] is False
    assert len(result["checkpoint_payloads"]) == 18 * 2 * 3 * 8
    assert result["failed_lineage"] == snapshot["frozen"]["allocation_predecessor"]
    assert result["failed_checkpoint_payloads"]
    assert result["restored_keeper"] == snapshot["state"]["restored_keeper"]
    assert not result["execution_closure_merged"] and not result["actual_lr_execution_authorized"]
    assert not result["production_authorized"]
    assert before == {path: predecessor.file_identity(path) for path in before}
    assert not (root / predecessor.SEAL).exists()


@pytest.mark.parametrize("role", ("controller", "validator", "terminal_waiter", "terminal_child"))
@pytest.mark.parametrize("process_state", ("live", "zombie", "incomplete"))
def test_existing_owner_never_authorizes_seal(completed, role, process_state):
    root, snapshot = completed
    contexts = dict(
        controller=snapshot["intent"]["generation"],
        validator=snapshot["state"]["child_generation"],
        terminal_waiter=snapshot["waiter"]["generation"],
        terminal_child=snapshot["waiter_state"]["terminal_generation"],
    )
    own = contexts[role]
    directory = root / "fixture-proc" / str(own["pid"])
    directory.mkdir(parents=True)
    if process_state != "incomplete":
        record(directory / "generation.json", own | {"status": "Z" if process_state == "zombie" else "S"})
    with pytest.raises(RuntimeError):
        predecessor.evidence(root)
    assert not (root / predecessor.SEAL).exists()


@pytest.mark.parametrize(
    "fault",
    (
        "partial",
        "claimed",
        "waiter_cleanup",
        "claim_verifier",
        "rejection",
        "history",
        "payload",
        "failed_lineage",
        "keeper",
    ),
)
def test_rebound_or_incomplete_evidence_refuses_without_publication(completed, fault):
    root, snapshot = completed
    report = json.loads((root / "report-snapshot.json").read_text())
    if fault == "partial":
        report["progress"]["completed_commands"] = 6
        report["progress"]["decisions"] = report["progress"]["decisions"][:1]
    elif fault == "claimed":
        record(Path(snapshot["outcome"]), {"status": "reconstruction_claimed"})
    elif fault == "waiter_cleanup":
        snapshot["waiter_state"]["terminal_cleanup_exit_code"] = 1
        record(root / predecessor.FOLLOWUP_WORK / "state.json", snapshot["waiter_state"])
        record(root / "reader-snapshot.json", snapshot)
    elif fault == "claim_verifier":
        claim = json.loads(Path(snapshot["attempt"]).read_text())
        claim["verifier_sha256"] = "0" * 64
        record(Path(snapshot["attempt"]), claim)
    elif fault == "rejection":
        report["progress"]["result"]["rejected_cases"] = []
    elif fault == "history":
        next((root / "trajectories").rglob("continuous.jsonl")).write_text('{"scalar":2}\n')
    elif fault == "payload":
        next((root / "trajectories").rglob("rank_7.pt")).write_bytes(b"changed payload")
    elif fault == "failed_lineage":
        record(root / "failed-evidence.json", {"original": "accepted"})
    else:
        keeper = copy.deepcopy(snapshot["state"]["restored_keeper"])
        keeper["pid"] += 1
        record(root / "current-keeper.json", keeper)
    record(root / "report-snapshot.json", report)
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        predecessor.evidence(root)
    assert not (root / predecessor.SEAL).exists()


def test_unpinned_waiter_refuses_before_reader_or_output(completed, monkeypatch):
    root, _ = completed
    monkeypatch.setattr(predecessor, "FOLLOWUP_INTENT_SHA256", None)
    (root / "reader-snapshot.json").unlink()
    with pytest.raises(RuntimeError, match="not pinned"):
        predecessor.readiness(root)
    assert not (root / predecessor.SEAL).exists()


def test_seal_one_shot_identity_checks_survive_keeper_transfer(completed):
    root, _ = completed
    sealed = predecessor.seal(root)
    path = root / predecessor.SEAL
    seal_digest = digest(path)
    record(root / "current-keeper.json", {"transferred": True})
    assert predecessor.require_seal(root, seal_digest) == sealed
    with pytest.raises(FileExistsError):
        predecessor.seal(root)
    next((root / "trajectories").rglob("rank_7.pt")).write_bytes(b"changed retained payload")
    with pytest.raises(ValueError):
        predecessor.require_seal(root, seal_digest)
    assert digest(path) == seal_digest
    assert sealed["original_decisions"]["rejected_cases"] == ["exit-case-0"]


def test_archive_merge_refusal_cannot_create_or_overwrite_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    existing = tmp_path / "old-accepted.json"
    record(existing, {"status": "consumed"})
    before = digest(existing)
    with pytest.raises(RuntimeError, match="deferred"):
        preparation.require_cpu(tmp_path, tmp_path / "unused-receipt.json")
    assert digest(existing) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == [existing.name]


def test_substituted_live_exec_is_not_absence(completed):
    root, snapshot = completed
    own = snapshot["state"]["child_generation"]
    record(root / "fixture-proc" / str(own["pid"]) / "generation.json", own | {"command": ["different-exec"]})
    with pytest.raises(RuntimeError):
        predecessor.readiness(root)
    assert not (root / predecessor.SEAL).exists()


def test_bounded_reader_timeout_reaps_owned_cpu_child(tmp_path, monkeypatch):
    monkeypatch.setattr(predecessor, "PYTHON", sys.executable)
    monkeypatch.setattr(predecessor, "READ_TIMEOUT", 1)
    program = (
        "import os,signal; from pathlib import Path; " "Path('child.pid').write_text(str(os.getpid())); signal.pause()"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        predecessor.run_saved(tmp_path, program, [])
    pid = int((tmp_path / "child.pid").read_text())
    assert not (Path("/proc") / str(pid)).exists()


def test_real_cpu_terminal_child_blocks_handoff_and_is_reaped(completed):
    root, snapshot = completed
    command = [sys.executable, "-c", "import os,sys; print(os.getpid(),flush=True); sys.stdin.read()"]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert int(child.stdout.readline()) == child.pid
        snapshot["waiter_state"]["terminal_generation"] = generation(child.pid, snapshot["waiter"]["pid"], command)
        snapshot["waiter_state"]["terminal_pid"] = child.pid
        record(root / predecessor.FOLLOWUP_WORK / "state.json", snapshot["waiter_state"])
        record(root / "reader-snapshot.json", snapshot)
        observation = predecessor.readiness(root)
        assert observation["ready"] is False
        assert observation["owners"]["terminal_child"]["status"] == "live_exact_generation"
        with pytest.raises(RuntimeError):
            predecessor.evidence(root)
        assert not (root / predecessor.SEAL).exists()
    finally:
        child.communicate(timeout=10)
    assert child.returncode == 0 and not (Path("/proc") / str(child.pid)).exists()


def test_source_plan_retains_conflicting_handoffs_in_separate_parents(completed, monkeypatch):
    root, snapshot = completed
    old_receipt = root / "old-numerical-cpu.json"
    reader = root / "reader/tracking/ns_training_path_exit_followup.py"
    record(old_receipt, {"helpers_sha256": {reader.name: "0" * 64}})
    monkeypatch.setattr(preparation, "PARENT", old_receipt.name)
    monkeypatch.setattr(preparation, "PARENT_SHA256", digest(old_receipt))
    monkeypatch.setattr(preparation, "NEW_HELPERS", ("ns_training_path_exit_followup.py",))
    monkeypatch.setattr(preparation, "__file__", str(reader))
    before = {path: predecessor.file_identity(path) for path in root.rglob("*") if path.is_file()}
    plan = preparation.source_plan(root)
    assert plan["conflicting_helpers"] == {
        reader.name: {
            "actual_lr_sha256": "0" * 64,
            "exit_waiter_sha256": snapshot["cpu"]["helpers_sha256"][reader.name],
        }
    }
    assert plan["execution_closure_merged"] is False
    assert before == {path: predecessor.file_identity(path) for path in before}


def test_cli_cpu_refusal_leaves_only_original_files(tmp_path):
    existing = tmp_path / "consumed.json"
    record(existing, {"status": "consumed"})
    command = [
        sys.executable,
        "-B",
        "-P",
        str(Path(preparation.__file__).resolve()),
        "--root",
        str(tmp_path),
        "--phase",
        "cpu",
    ]
    result = subprocess.run(
        command,
        env={
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(Path(preparation.__file__).resolve().parent),
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [existing.name]
