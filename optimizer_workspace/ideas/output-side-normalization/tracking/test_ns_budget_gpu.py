# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU ownership, child-cleanup and retained-file regressions for the distinct NS diagnostic."""

import copy
import fcntl
import json
import os
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path

import ns_budget_execution as execution
import ns_budget_gpu_controller as controller
import ns_budget_gpu_owner as owner
import ns_budget_gpu_worker as worker
import ns_budget_trial as trial
import pytest
import torch
from selected_lr_release_owner import sha256
from watch_wave5 import write


def test_unissued_ns_owner_denies_before_cuda_or_output_creation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unissued NS owner initialized CUDA")

    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    with pytest.raises(FileNotFoundError):
        worker.run(tmp_path, "0" * 64, owner.locations(tmp_path)[2], 0)
    assert list(tmp_path.iterdir()) == []


def test_ns_predecessor_rejects_live_original_rational_owner_before_missing_terminal_receipts(tmp_path, monkeypatch):
    _, helpers, state, _, _ = owner.rational.locations(tmp_path)
    helpers.mkdir(parents=True)
    state.parent.mkdir(parents=True, exist_ok=True)
    write(state, {"controller_pid": 11, "child_pid": 12})
    write(helpers.parent / "controller.json", {})
    monkeypatch.setattr(owner.handoff, "observed", lambda pid: {"pid": pid} if pid == 11 else None)
    with pytest.raises(RuntimeError, match="original rational owner handle"):
        owner.predecessor(tmp_path)
    assert not (tmp_path / owner.TERMINAL).exists()


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached"))
def test_real_ns_worker_acknowledges_and_cleans_original_descendants(tmp_path, monkeypatch, outcome):
    binding, helpers, state_path, work = owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    work.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    binding.write_text("{}")
    source = helpers / "ns_budget_gpu_worker.py"
    source.write_text(
        "import argparse,json,os,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--state');p.add_argument('--work');p.add_argument('--outcome');a=p.parse_args()\n"
        "deadline=time.monotonic()+10\n"
        "while True:\n"
        " s=json.loads(Path(a.state).read_text());w=s['active_worker']\n"
        " if w['spawn_status']=='acknowledged' and w['pid']==os.getpid():break\n"
        " if time.monotonic()>deadline:sys.exit(9)\n"
        " time.sleep(.01)\n"
        "if a.outcome=='detached':\n"
        " child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)\n"
        " Path(a.work,'detached.pid').write_text(str(child.pid))\n"
        "Path(a.work,'process-0.json').write_text(json.dumps({'acknowledged':True}))\n"
        "sys.exit(0 if a.outcome=='zero' else 7)\n"
    )
    command = [sys.executable, str(source), "--state", str(state_path), "--work", str(work), "--outcome", outcome]
    monkeypatch.setattr(owner, "worker_command", lambda root, digest, index: command)
    state = {"status": "replaying", "controller_pid": os.getpid(), "controller_start_ticks": "cpu-owner"}
    if outcome == "zero":
        result = controller.run_worker(tmp_path, {"worker_environment": owner.ENVIRONMENT}, state, 0)
        assert result["exit_code"] == result["cleanup_exit_code"] == 0
    else:
        with pytest.raises(RuntimeError, match="worker failed"):
            controller.run_worker(tmp_path, {"worker_environment": owner.ENVIRONMENT}, state, 0)
    saved = read_json(state_path)
    child = saved["active_worker"]
    assert child["spawn_status"] == "acknowledged"
    assert child["generation"]["parent"] == os.getpid()
    assert child["exit_code"] == (0 if outcome == "zero" else 7)
    assert child["marked_descendants_absent"] and "cleanup_error" not in saved
    assert owner.handoff.observed(child["pid"]) is None
    assert not owner.handoff.marker_processes(owner.CHILD_MARKER, child["marker"])
    if outcome == "detached":
        descendant = owner.handoff.observed(int((work / "detached.pid").read_text()))
        assert descendant is None or descendant["status"] == "Z"


def read_json(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("failure", ("spawn", "nonzero"))
def test_ns_controller_restores_exact_allocation_after_real_worker_failure(tmp_path, monkeypatch, failure):
    binding, helpers, state_path, work = owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    own = {
        "pid": os.getpid(),
        "parent": os.getppid(),
        "status": "R",
        "group": os.getpid(),
        "session": os.getpid(),
        "start_ticks": "cpu-owner",
        "command": owner.controller_command(tmp_path),
    }
    keeper = {"pid": 100, "devices": [["0", "GPU-cpu", "CPU fixture"]], "generation": {}}
    frozen = {"predecessor": {"restored_keeper": keeper}, "worker_environment": owner.ENVIRONMENT}
    write(binding, frozen)
    intent = {
        "status": "acknowledged",
        "binding_sha256": sha256(binding),
        "pid": own["pid"],
        "start_ticks": own["start_ticks"],
        "generation": own,
        "command": own["command"],
        "marker": "cpu-marker",
    }
    write(helpers.parent / "controller.json", intent)
    restored = {"pid": 101, "devices": keeper["devices"], "generation": {}}
    lock_path, events = tmp_path / "manager.lock", []

    @contextmanager
    def manager_boundary():
        with lock_path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield stream.fileno()

    def host_boundary(command, **kwargs):
        if command == ["hostname", "-I"]:
            return owner.NODE
        if command == ["ps", "-eo", "args="]:
            return ""
        raise AssertionError(command)

    def release_boundary(accepted, state, path, handle):
        owner.require_held_lock(os.getpid(), handle, lock_path)
        assert accepted == keeper
        state.update(keeper_release_attempted=True, keeper_release_complete=True)
        write(path, state)
        events.append("released")

    def restore_boundary(handle):
        owner.require_held_lock(os.getpid(), handle, lock_path)
        events.append("restored")

    monkeypatch.setattr(owner, "require_frozen", lambda root, digest: copy.deepcopy(frozen))
    monkeypatch.setattr(owner, "require_live_predecessor", lambda root, previous: copy.deepcopy(keeper))
    monkeypatch.setattr(controller, "process_context", lambda *args, **kwargs: own)
    monkeypatch.setattr(controller, "keeper_guard", manager_boundary)
    monkeypatch.setattr(controller, "release_accepted_keeper", release_boundary)
    monkeypatch.setattr(controller, "restore_accepted_keeper", restore_boundary)
    monkeypatch.setattr(owner.handoff, "keeper_observation", lambda: copy.deepcopy(restored))
    monkeypatch.setattr(owner, "require_keeper_shape", lambda value: None)
    monkeypatch.setattr(subprocess, "check_output", host_boundary)
    monkeypatch.setenv(owner.CONTROLLER_MARKER, intent["marker"])
    source = helpers / "ns_budget_gpu_worker.py"
    source.write_text("import sys;sys.exit(7)\n")
    monkeypatch.setattr(owner, "worker_command", lambda root, digest, index: [sys.executable, str(source)])
    if failure == "spawn":

        def spawn_failure(*args, **kwargs):
            raise OSError("synthetic NS worker spawn failure")

        monkeypatch.setattr(subprocess, "Popen", spawn_failure)
    with pytest.raises((RuntimeError, OSError)):
        controller.execute(tmp_path)
    state = read_json(state_path)
    assert state["status"] == "needs_attention" and state["keeper_restore"] == 0
    assert state["restored_keeper"] == restored and state["executions"] == []
    assert events == ["released", "restored"]
    assert work.is_dir() and not (work / "comparison.json").exists()
    with pytest.raises(FileExistsError):
        controller.execute(tmp_path)
    assert read_json(state_path) == state


def cpu_retainer(mode, steps, source, coordinate):
    output = execution.precision.reference_fp64(source, (3.4445, -4.7750, 2.0315), steps=steps, eps=1e-5).float()
    phases = [{"phase": phase, "output": output} for phase in execution.PHASES]
    evidence = trial.summarize_kernel(phases[1:], [0.5] * trial.TIMING_REPLAYS)
    evidence.update(
        source_shape=list(source.shape),
        source_stride=list(source.stride()),
        source_storage_offset=source.storage_offset(),
        rng_exact=True,
        compilation_preserved=True,
        graphs_preserved=True,
        graph_replays=3 + trial.TIMING_REPLAYS,
        input_exact_finite=execution.tracing.comparison(source, source, "input"),
    )
    return phases, evidence


@pytest.fixture
def retained_ns_execution(tmp_path, monkeypatch):
    binding, _, _, work = owner.locations(tmp_path)
    work.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    write(binding, {})
    helpers = Path(execution.__file__).resolve().parent
    hashes = {name: sha256(helpers / name) for name in execution.REQUIRED_HELPERS}
    task = helpers.parent
    manifest = read_json(task / "experiment/upstream_manifest.json")
    muon = task / "artifacts/external-sources" / f"Toner-{manifest['revision']}/src/toner/fast/muon.py"
    config_path, receipt_path = tmp_path / "config.json", tmp_path / "receipt.json"
    config_path.write_text(json.dumps({"runtime": {"deterministic": True}}))
    receipt_path.write_text(json.dumps({"config": str(config_path)}))
    provenance = {
        "pinned_muon": str(muon),
        "pinned_muon_sha256": execution.replay.MUON_SHA256,
        "config_sha256": sha256(config_path),
    }
    source = torch.tensor([[[1.0, 0.2], [-0.1, 0.3], [0.2, 0.4]]])
    groups = {name: {"source": source.clone(), "output": source.clone()} for name in execution.replay.analyzer.MODES}
    groups["resumed"]["source"][0, 2, 1] += 0.01
    frozen = {"devices": [["0", "GPU-00000000-0000-0000-0000-000000000001", "CPU fixture"]], "helpers_sha256": hashes}
    frozen["predecessor"] = {"restored_keeper": {"pid": 900, "generation": {}, "devices": frozen["devices"]}}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(owner, "require_frozen", lambda root, digest: frozen)
    monkeypatch.setattr(execution.replay, "ROOT", tmp_path)
    monkeypatch.setattr(execution.replay.analyzer, "RECEIPT_RELATIVE", receipt_path.name)
    monkeypatch.setattr(execution.replay, "RECEIPT_SHA256", sha256(receipt_path))
    monkeypatch.setattr(execution.replay, "load_saved_group", lambda stack: (groups, provenance))
    entries = []
    for index in (0, 1):
        retained, lifecycle = execution.retain_protocol(groups, index, cpu_retainer)
        log = work / f"process-{index}.log"
        log.write_text("cpu fixture\n")
        pid, argv = 200 + index, owner.worker_command(tmp_path, sha256(binding), index)
        entry = {
            "index": index,
            "spawn_status": "acknowledged",
            "marker": f"cpu-{index}",
            "pid": pid,
            "start_ticks": str(pid),
            "controller_pid": 100,
            "controller_start_ticks": "100",
            "command": argv,
            "exit_code": 0,
            "cleanup_exit_code": 0,
            "marked_descendants_absent": True,
            "log": str(log),
            "log_sha256": sha256(log),
            "report": str(work / f"process-{index}.json"),
            "generation": {
                "pid": pid,
                "start_ticks": str(pid),
                "parent": 100,
                "command": argv,
                "group": pid,
                "session": pid,
                "status": "R",
            },
        }
        origin = controller.expected_owner(tmp_path, frozen, entry, index)
        report = {
            "status": execution.STATUS,
            "worker_index": index,
            "owner": origin,
            "report_path": entry["report"],
            "helper_sources_sha256": hashes,
            "coordinates": execution.coordinates(index),
            "rng_exact": True,
            "compilation_preserved": True,
            "graphs_preserved": True,
            "compile_options": {"dynamic": False},
            "physical_uuid": origin["physical_uuid"],
            "actual_uuid": origin["physical_uuid"],
            "outputs": str(work / f"process-{index}.pt"),
            "provenance": provenance,
            "runtime": {"deterministic": True},
            "lifecycle": lifecycle,
            "analysis": execution.summarize(retained, groups, (3.4445, -4.7750, 2.0315), index),
            "timings": execution.reconstruct_lifecycle(retained, lifecycle, groups, index),
        }
        torch.save(retained, report["outputs"])
        report["outputs_sha256"] = sha256(Path(report["outputs"]))
        write(Path(entry["report"]), report)
        entry["report_sha256"] = sha256(Path(entry["report"]))
        entries.append(entry)
    return frozen, entries


@pytest.mark.parametrize("fault", ("none", "output", "wrong_parent", "duplicate_worker"))
def test_ns_controller_reconstructs_opposite_order_retained_files_and_rejects_changed_origin(
    tmp_path, retained_ns_execution, fault
):
    _, entries = retained_ns_execution
    if fault == "output":
        report_path = Path(entries[1]["report"])
        report = read_json(report_path)
        retained = torch.load(report["outputs"], weights_only=False, map_location="cpu")
        retained[0]["output"][0, 2, 1] += 0.01
        torch.save(retained, report["outputs"])
        report["outputs_sha256"] = sha256(Path(report["outputs"]))
        write(report_path, report)
        entries[1]["report_sha256"] = sha256(report_path)
    if fault == "wrong_parent":
        entries[1]["generation"]["parent"] = 999
    elif fault == "duplicate_worker":
        entries[1]["pid"], entries[1]["start_ticks"] = entries[0]["pid"], entries[0]["start_ticks"]
    if fault != "none":
        with pytest.raises(ValueError):
            controller.compare_results(tmp_path, entries)
        return
    result = controller.compare_results(tmp_path, entries)
    assert len(result["cross_process"]["comparisons"]) == 144
    assert result["cross_process"]["all_cross_process_exact_finite"]
    assert result["all_same_input_repeats_exact_finite"]
    assert result["all_compiled_phases_match_pinned_eager"]
    assert not result["all_ten_step_tpu_phases_reproduce_saved_outputs"]
    assert not result["historical_controls_interpretable"]


@pytest.mark.parametrize(
    "fault", ("none", "live_controller", "live_worker", "changed_report", "changed_comparison", "changed_keeper")
)
def test_ns_terminal_reconstructs_retained_files_and_refuses_unfinished_or_changed_evidence(
    tmp_path, monkeypatch, retained_ns_execution, fault
):
    frozen, entries = retained_ns_execution
    binding, helpers, state_path, work = owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    keeper = frozen["predecessor"]["restored_keeper"]
    intent = {
        "status": "acknowledged",
        "binding_sha256": sha256(binding),
        "pid": 100,
        "start_ticks": "100",
        "command": owner.controller_command(tmp_path),
        "marker": "original-controller",
        "generation": {
            "pid": 100,
            "start_ticks": "100",
            "command": owner.controller_command(tmp_path),
            "group": 100,
            "session": 100,
        },
    }
    write(helpers.parent / "controller.json", intent)
    (helpers.parent / "controller.log").write_text("CPU retained terminal fixture\n")
    comparison = controller.compare_results(tmp_path, entries)
    if fault == "changed_comparison":
        comparison["historical_controls_interpretable"] = True
    write(work / "comparison.json", comparison)
    if fault == "changed_report":
        report_path = Path(entries[1]["report"])
        report = read_json(report_path)
        report["owner"]["controller_pid"] = 999
        write(report_path, report)
        entries[1]["report_sha256"] = sha256(report_path)
    state = {
        "status": "diagnostic_complete",
        "binding_sha256": sha256(binding),
        "controller_pid": 100,
        "controller_start_ticks": "100",
        "keeper_release_complete": True,
        "keeper_restore": 0,
        "predecessor": frozen["predecessor"],
        "executions": entries,
        "active_worker": entries[-1],
        "comparison_sha256": sha256(work / "comparison.json"),
        "restored_keeper": keeper,
    }
    write(state_path, state)
    monkeypatch.setattr(controller, "local_node", lambda: owner.NODE)
    monkeypatch.setattr(
        owner.handoff,
        "observed",
        lambda pid: (
            {"pid": pid}
            if (fault == "live_controller" and pid == 100) or (fault == "live_worker" and pid == 201)
            else None
        ),
    )
    monkeypatch.setattr(owner.handoff, "marker_processes", lambda name, marker: {})
    monkeypatch.setattr(
        owner.handoff, "keeper_observation", lambda: keeper | {"pid": 901} if fault == "changed_keeper" else keeper
    )
    monkeypatch.setattr(owner, "require_keeper_shape", lambda value: None)
    lock_path = tmp_path / "manager.lock"

    @contextmanager
    def manager_boundary():
        with lock_path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield stream.fileno()

    monkeypatch.setattr(controller, "keeper_guard", manager_boundary)
    receipt = tmp_path / "run-preparation/terminal.json"
    if fault != "none":
        with pytest.raises((RuntimeError, ValueError)):
            controller.terminal(tmp_path, receipt)
        assert not receipt.exists()
        return
    result = controller.terminal(tmp_path, receipt)
    assert result["keeper"] == keeper and result["worker_pids_absent"] == [200, 201]
    assert result["comparison"]["cross_process"]["all_cross_process_exact_finite"]
    assert result["comparison"]["historical_controls_interpretable"] is False
    assert result["evidence_sha256"][str(state_path.relative_to(tmp_path))] == sha256(state_path)
    with pytest.raises(RuntimeError):
        controller.terminal(tmp_path, receipt)
    assert read_json(receipt) == result


@pytest.mark.parametrize("fault", ("none", "node_unlocked", "manager_unlocked", "uuid", "occupancy", "generation"))
def test_ns_worker_boundary_requires_exclusive_original_locks_and_exact_physical_owner(tmp_path, monkeypatch, fault):
    binding, helpers, state_path, _ = owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    node_lock, manager_lock = (
        tmp_path / "run-preparation" / f".wave7-node-{owner.NODE}.lock",
        tmp_path / "manager.lock",
    )
    command = [
        sys.executable,
        "-c",
        "import fcntl,json,os,sys;"
        "n=open(sys.argv[1],'a');m=open(sys.argv[2],'a');"
        "fcntl.flock(n,fcntl.LOCK_EX) if sys.argv[3]!='node_unlocked' else None;"
        "fcntl.flock(m,fcntl.LOCK_EX) if sys.argv[3]!='manager_unlocked' else None;"
        "print(json.dumps({'node':n.fileno(),'manager':m.fileno()}),flush=True);sys.stdin.read()",
        str(node_lock),
        str(manager_lock),
        fault,
    ]
    devices = [[str(index), f"GPU-cpu-{index}", "CPU device boundary"] for index in range(8)]
    frozen = {
        "predecessor": {},
        "devices": devices,
        "worker_environment": owner.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": devices[0][1]},
    }
    write(binding, frozen)
    digest = sha256(binding)
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as child:
        try:
            descriptors = json.loads(child.stdout.readline())
            parent = owner.handoff.observed(child.pid)
            parent["command"] = owner.controller_command(tmp_path)
            own = owner.handoff.observed(os.getpid())
            own["command"], own["parent"] = owner.worker_command(tmp_path, digest, 0), child.pid
            intent = {
                "status": "acknowledged",
                "binding_sha256": digest,
                "pid": parent["pid"],
                "start_ticks": parent["start_ticks"],
                "command": parent["command"],
                "generation": parent,
                "marker": "cpu-controller",
            }
            write(helpers.parent / "controller.json", intent)
            state = {
                "status": "replaying",
                "binding_sha256": digest,
                "controller_pid": parent["pid"],
                "controller_start_ticks": parent["start_ticks"],
                "keeper_release_complete": True,
                "predecessor": frozen["predecessor"],
                "lock_descriptors": descriptors,
                "active_worker": {
                    "spawn_status": "acknowledged",
                    "index": 0,
                    "command": own["command"],
                    "pid": own["pid"],
                    "controller_pid": parent["pid"],
                    "start_ticks": own["start_ticks"] if fault != "generation" else "changed",
                    "marker": "cpu-worker",
                },
            }
            write(state_path, state)
            actual_devices = copy.deepcopy(devices)
            if fault == "uuid":
                actual_devices[0][1] = "GPU-changed"
            occupied = [[str(os.getpid()), devices[0][1]]]
            if fault == "occupancy":
                occupied.append(["999999", devices[1][1]])

            def device_boundary(argv, **kwargs):
                if argv == ["hostname", "-I"]:
                    return owner.NODE
                if argv == ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"]:
                    return "\n".join(",".join(row) for row in actual_devices)
                if argv == ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]:
                    return "\n".join(",".join(row) for row in occupied)
                raise AssertionError(argv)

            monkeypatch.setattr(owner, "require_frozen", lambda root, value: frozen)
            monkeypatch.setattr(owner, "process_context", lambda pid, *args: parent if pid == parent["pid"] else own)
            monkeypatch.setattr(owner.handoff, "same_generation", lambda expected, actual: True)
            monkeypatch.setattr(owner.handoff, "KEEPER_PID", tmp_path / "keeper.pid")
            monkeypatch.setattr(owner, "MANAGER_LOCK", manager_lock)
            monkeypatch.setattr(os, "getppid", lambda: child.pid)
            monkeypatch.setattr(subprocess, "check_output", device_boundary)
            monkeypatch.chdir(helpers)
            for name, value in frozen["worker_environment"].items():
                monkeypatch.setenv(name, value)
            if fault != "none":
                with pytest.raises(RuntimeError):
                    owner.require_gpu_owner(tmp_path, digest, state_path, 0)
            else:
                result = owner.require_gpu_owner(tmp_path, digest, state_path, 0)
                assert result["physical_uuid"] == devices[0][1]
                assert result["worker_pid"] == os.getpid() and result["controller_pid"] == child.pid
            assert read_json(state_path)["status"] == "replaying"
            assert not owner.locations(tmp_path)[3].exists()
        finally:
            child.stdin.close()
            child.wait(timeout=5)
    for path in (node_lock, manager_lock):
        with path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_ns_owner_source_closure_preserves_every_accepted_numerical_helper():
    root = execution.replay.ROOT
    sources = owner.helper_sources(root)
    parent = read_json(root / owner.PARENT)
    assert all(sha256(sources[name]) == value for name, value in parent["helpers_sha256"].items())
    assert set(sources) == set(parent["helpers_sha256"]) | set(owner.NEW_HELPERS)


@pytest.mark.parametrize("fault", ("none", "terminal_analysis", "followup_exit", "evidence", "live_followup"))
def test_ns_predecessor_accepts_negative_intervention_with_exact_restoration_and_rejects_changed_terminal(
    tmp_path, monkeypatch, fault
):
    binding, helpers, state_path, _, result_path = owner.rational.locations(tmp_path)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    follow_path, cpu_path = tmp_path / owner.FOLLOWUP, tmp_path / owner.FOLLOWUP_CPU
    follow_path.mkdir(parents=True)
    cpu_path.parent.mkdir(parents=True)
    runner = cpu_path.parent / "rational_continuation_followup.py"
    runner.write_text("# CPU terminal fixture\n")
    write(cpu_path, {"files_sha256": {runner.name: sha256(runner)}})
    helper_path = helpers / "rational_continuation_controller.py"
    helper_path.write_text("# issued verifier fixture\n")
    command = owner.rational.controller_command(tmp_path)
    validator = [owner.PYTHON, "-P", str(helpers / "validator.py")]
    frozen = {
        "helpers_sha256": {helper_path.name: sha256(helper_path)},
        "controller_command": command,
        "validator_command": validator,
        "commands": [[f"trial-{index}"] for index in range(12)],
    }
    write(binding, frozen)
    monkeypatch.setattr(owner.preparation, "BINDING_SHA256", sha256(binding))
    monkeypatch.setattr(owner, "FOLLOWUP_CPU_SHA256", sha256(cpu_path))
    keeper = {"pid": 99, "devices": [["0", "GPU-cpu", "CPU fixture"]], "generation": {}}
    controller_generation = {"pid": 11, "start_ticks": "11", "command": command, "group": 11, "session": 11}
    child_generation = {"pid": 12, "start_ticks": "12", "parent": 11, "command": validator, "group": 12, "session": 12}
    intent = {
        "status": "acknowledged",
        "binding_sha256": sha256(binding),
        "pid": 11,
        "start_ticks": "11",
        "command": command,
        "marker": "original-controller",
        "generation": controller_generation,
    }
    write(helpers.parent / "controller.json", intent)
    for name in ("controller.log", "validator.log"):
        (helpers.parent / name).write_text("CPU fixture\n")
    executed = []
    for index, argv in enumerate(frozen["commands"]):
        log = binding.parent / f"trial-{index}.log"
        log.write_text("CPU fixture\n")
        executed.append({"command": argv, "exit_code": 0, "log": str(log), "log_sha256": sha256(log)})
    result = {
        "status": "completed_rational_start_full_model_diagnostic_not_production",
        "binding_sha256": sha256(binding),
        "owner": {"controller_pid": 11, "controller_start_ticks": "11"},
        "commands": frozen["commands"],
        "gpu_commands": executed,
        "analysis": {"diagnostic_intervention_accepted": False},
    }
    write(result_path, result)
    state = {
        "status": "diagnostic_complete",
        "binding_sha256": sha256(binding),
        "controller_pid": 11,
        "controller_start_ticks": "11",
        "child_pid": 12,
        "child_start_ticks": "12",
        "child_marker": "original-child",
        "child_generation": child_generation,
        "command": validator,
        "spawn_status": "acknowledged",
        "marked_descendants_absent": True,
        "keeper_release_attempted": True,
        "keeper_release_complete": True,
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "result_sha256": sha256(result_path),
        "log": str(helpers.parent / "validator.log"),
        "log_sha256": sha256(helpers.parent / "validator.log"),
        "restored_keeper": keeper,
        "intervention_accepted": False,
    }
    write(state_path, state)
    terminal = {
        "status": "rational_continuation_terminal_independently_reconstructed_exact_restoration",
        "binding_sha256": sha256(binding),
        "result_sha256": sha256(result_path),
        "terminal_time": read_json(state_path)["time"],
        "controller_pid_absent": 11,
        "validator_pid_absent": 12,
        "verifier_sha256": sha256(helper_path),
        "analysis": result["analysis"],
        "keeper": keeper,
        "evidence_sha256": {str(result_path.relative_to(tmp_path)): sha256(result_path)},
    }
    if fault == "terminal_analysis":
        terminal["analysis"] = {"diagnostic_intervention_accepted": True}
    elif fault == "evidence":
        terminal["evidence_sha256"][str(result_path.relative_to(tmp_path))] = "0" * 64
    write(tmp_path / owner.TERMINAL, terminal)
    follow_command = [owner.PYTHON, "-P", str(runner)]
    follow_intent = {
        "status": "acknowledged",
        "pid": 21,
        "start_ticks": "21",
        "command": follow_command,
        "marker": "followup",
        "binding_sha256": sha256(binding),
        "runner_sha256": sha256(runner),
        "cpu_receipt_sha256": sha256(cpu_path),
        "generation": {"pid": 21, "start_ticks": "21", "command": follow_command, "group": 21, "session": 21},
    }
    write(follow_path / "intent.json", follow_intent)
    monkeypatch.setattr(owner, "FOLLOWUP_INTENT_SHA256", sha256(follow_path / "intent.json"))
    terminal_command = [
        owner.PYTHON,
        "-P",
        str(helper_path),
        "--root",
        str(tmp_path),
        "--phase",
        "terminal",
        "--out",
        str(tmp_path / owner.TERMINAL),
    ]
    (follow_path / "terminal.log").write_text("CPU fixture\n")
    follow = {
        "status": "terminal_reconstructed",
        "pid": 21,
        "start_ticks": "21",
        "runner_sha256": sha256(runner),
        "cpu_receipt_sha256": sha256(cpu_path),
        "binding_sha256": sha256(binding),
        "terminal_command": terminal_command,
        "terminal_pid": 22,
        "terminal_marker": "read-only",
        "terminal_generation": {"pid": 22, "parent": 21, "command": terminal_command, "group": 22, "session": 22},
        "terminal_exit_code": 0 if fault != "followup_exit" else 7,
        "terminal_cleanup_exit_code": 0,
        "terminal_descendants_absent": True,
        "terminal_sha256": sha256(tmp_path / owner.TERMINAL),
        "terminal_status": terminal["status"],
        "terminal_log": str(follow_path / "terminal.log"),
        "terminal_log_sha256": sha256(follow_path / "terminal.log"),
    }
    write(follow_path / "state.json", follow)
    monkeypatch.setattr(
        owner.handoff, "observed", lambda pid: {"pid": pid} if fault == "live_followup" and pid == 21 else None
    )
    monkeypatch.setattr(owner.handoff, "marker_processes", lambda name, marker: {})
    monkeypatch.setattr(owner, "require_keeper_shape", lambda value: None)
    if fault != "none":
        with pytest.raises((RuntimeError, ValueError)):
            owner.predecessor(tmp_path)
        return
    accepted = owner.predecessor(tmp_path)
    assert accepted["restored_keeper"] == keeper
    assert accepted["intervention_accepted"] is False
    monkeypatch.setattr(owner.handoff, "keeper_observation", lambda: copy.deepcopy(keeper))
    assert owner.require_live_predecessor(tmp_path, accepted) == keeper
    monkeypatch.setattr(owner.handoff, "keeper_observation", lambda: keeper | {"pid": 100})
    with pytest.raises(RuntimeError, match="keeper generation"):
        owner.require_live_predecessor(tmp_path, accepted)


def test_ns_frozen_source_denies_changed_execution_even_with_rehashed_binding(tmp_path):
    root = execution.replay.ROOT
    binding, helpers, _, work = owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    inherited = read_json(root / owner.PARENT)
    for name in inherited["helpers_sha256"]:
        source = Path(inherited["helpers"]) / name
        (helpers / name).write_bytes(source.read_bytes())
    for name in owner.NEW_HELPERS:
        (helpers / name).write_bytes((Path(owner.__file__).parent / name).read_bytes())
    hashes = {path.name: sha256(path) for path in helpers.iterdir()}
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in hashes:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    cpu_path = tmp_path / "cpu.json"
    cpu_path.write_text("{}")
    previous = {"restored_keeper": {"devices": [["0", "GPU-cpu", "CPU fixture"]]}}
    frozen = {
        "status": owner.STATUS,
        "root": str(tmp_path),
        "helpers": str(helpers),
        "work": str(work),
        "node": owner.NODE,
        "helpers_sha256": hashes,
        "environment": owner.ENVIRONMENT,
        "worker_environment": owner.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": "GPU-cpu"},
        "devices": previous["restored_keeper"]["devices"],
        "predecessor": previous,
        "controller_command": owner.controller_command(tmp_path),
        "workers": 2,
        "worker_timeout": owner.WORKER_TIMEOUT,
        "coordinates": [execution.coordinates(index) for index in (0, 1)],
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "cpu_receipt": str(cpu_path),
        "cpu_receipt_sha256": sha256(cpu_path),
    }
    write(binding, frozen)
    program = (
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0,sys.argv[1]);import ns_budget_gpu_owner as owner\n"
        "owner.require_cpu=lambda root,path:{'helpers_sha256':owner.read(owner.locations(root)[0])['helpers_sha256']}\n"
        "owner.predecessor=lambda root:owner.read(owner.locations(root)[0])['predecessor']\n"
        "owner.require_frozen(Path(sys.argv[2]),owner.sha256(owner.locations(Path(sys.argv[2]))[0]))\n"
    )
    command = [sys.executable, "-P", "-c", program, str(helpers), str(tmp_path)]
    valid = subprocess.run(
        command, env=os.environ | {"CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=30
    )
    assert valid.returncode == 0, valid.stderr
    with (helpers / "ns_budget_execution.py").open("a") as stream:
        stream.write("\n# changed execution source\n")
    write(binding, frozen | {"scope": "rehashed unrelated metadata"})
    changed = subprocess.run(
        command, env=os.environ | {"CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=30
    )
    assert changed.returncode != 0
    assert "helper" in changed.stderr
