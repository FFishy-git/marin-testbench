# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test source-bound start lifecycle, real child cleanup and saved-result reconstruction on CPU."""

import copy
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import root_stage_trace as tracing
import root_start_accuracy as accuracy
import root_start_execution as execution
import root_start_gpu_controller as controller
import root_start_gpu_owner as owner
import root_start_gpu_worker as worker
import root_start_rational as rational
import torch
from benchmark_diagonal import source_definitions
from selected_lr_release_owner import sha256
from watch_wave5 import write


def test_unissued_start_owner_denies_before_cuda_or_outputs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unissued start owner initialized CUDA")

    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    with pytest.raises(FileNotFoundError):
        worker.run(tmp_path, "0" * 64, owner.locations(tmp_path)[2], 0)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ("zero", "nonzero", "detached"))
def test_real_start_worker_acknowledges_and_cleans_original_descendants(tmp_path, monkeypatch, failure):
    binding, helpers, state_path, work = owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    work.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    binding.write_text("{}")
    source = helpers / "root_start_gpu_worker.py"
    source.write_text(
        "import argparse,json,os,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "p=argparse.ArgumentParser();p.add_argument('--state');p.add_argument('--work');p.add_argument('--failure');a=p.parse_args()\n"
        "deadline=time.monotonic()+10\n"
        "while True:\n"
        " s=json.loads(Path(a.state).read_text());w=s['active_worker']\n"
        " if w['spawn_status']=='acknowledged' and w['pid']==os.getpid():break\n"
        " if time.monotonic()>deadline:sys.exit(9)\n"
        " time.sleep(.01)\n"
        "if a.failure=='detached':\n"
        " child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)\n"
        " Path(a.work,'detached.pid').write_text(str(child.pid))\n"
        "Path(a.work,'process-0.json').write_text(json.dumps({'acknowledged':True}))\n"
        "sys.exit(0 if a.failure=='zero' else 7)\n"
    )
    command = [sys.executable, str(source), "--state", str(state_path), "--work", str(work), "--failure", failure]
    monkeypatch.setattr(owner, "worker_command", lambda root, digest, index: command)
    state = {"status": "replaying", "controller_pid": os.getpid(), "controller_start_ticks": "cpu-owner"}
    if failure == "zero":
        result = controller.run_worker(tmp_path, {"worker_environment": owner.ENVIRONMENT}, state, 0)
        assert result["exit_code"] == result["cleanup_exit_code"] == 0
    else:
        with pytest.raises(RuntimeError, match="worker failed"):
            controller.run_worker(tmp_path, {"worker_environment": owner.ENVIRONMENT}, state, 0)
    saved = json.loads(state_path.read_text())
    child = saved["active_worker"]
    assert child["spawn_status"] == "acknowledged"
    assert child["generation"]["parent"] == os.getpid()
    assert child["exit_code"] == (0 if failure == "zero" else 7)
    assert child["marked_descendants_absent"]
    assert owner.handoff.observed(child["pid"]) is None
    assert not owner.handoff.marker_processes(owner.CHILD_MARKER, child["marker"])
    assert "cleanup_error" not in saved
    if failure == "detached":
        descendant = owner.handoff.observed(int((work / "detached.pid").read_text()))
        assert descendant is None or descendant["status"] == "Z"


@pytest.mark.parametrize("failure", ("spawn", "nonzero"))
def test_controller_restores_exact_allocation_after_worker_failure(tmp_path, monkeypatch, failure):
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
    lock_path = tmp_path / "manager.lock"
    events = []

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

    # Allocation and process identity are external boundaries; worker spawn/exit/cleanup run for real.
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
    source = helpers / "root_start_gpu_worker.py"
    source.write_text("import sys;sys.exit(7)\n")
    monkeypatch.setattr(owner, "worker_command", lambda root, digest, index: [sys.executable, str(source)])
    if failure == "spawn":

        def spawn_failure(*args, **kwargs):
            raise OSError("synthetic start-worker spawn failure")

        monkeypatch.setattr(subprocess, "Popen", spawn_failure)
    with pytest.raises((RuntimeError, OSError)):
        controller.execute(tmp_path)
    state = json.loads(state_path.read_text())
    assert state["status"] == "needs_attention"
    assert state["keeper_restore"] == 0
    assert state["restored_keeper"] == restored
    assert state["executions"] == []
    assert events == ["released", "restored"]
    assert work.is_dir() and not (work / "comparison.json").exists()
    with pytest.raises(FileExistsError):
        controller.execute(tmp_path)
    assert json.loads(state_path.read_text()) == state


@pytest.mark.parametrize("fault", ("none", "changed_output", "wrong_parent", "duplicate_worker"))
def test_comparison_independently_reconstructs_complete_opposite_start_outputs(tmp_path, monkeypatch, fault):
    binding, _, _, work = owner.locations(tmp_path)
    work.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    write(binding, {})
    task = Path(__file__).resolve().parents[1]
    manifest = json.loads((task / "experiment/upstream_manifest.json").read_text())
    roots = task / f"artifacts/external-sources/Toner-{manifest['revision']}/src/toner/fast/roots.py"
    pinned = source_definitions(
        roots, ("_start",), {"torch": torch, "EPS": 1e-3, "QUARTER": 0.25, "U32": 2.0**-24, "LOWP": torch.float16}
    )["_start"]
    B = torch.diag_embed(torch.tensor([[1.0, 1.000002, 2.0]])) + torch.tensor(
        [[0.0, 0.01, -0.02], [0.01, 0.0, 0.03], [-0.02, 0.03, 0.0]]
    )
    sources = {group: {mode: B.clone() for mode in accuracy.MODES} for group in accuracy.GROUPS}
    config = json.loads((task / "experiment/configs/inner_outer_leftdiag_0p5m_seed0.json").read_text())
    provenance = {group: {"pinned_roots": str(roots.resolve()), "config": config} for group in accuracy.GROUPS}
    frozen = {"devices": [["0", "GPU-00000000-0000-0000-0000-000000000001", "CPU fixture"]], "helpers_sha256": {}}
    monkeypatch.setattr(owner, "require_frozen", lambda root, digest: frozen)
    monkeypatch.setattr(
        execution.replay,
        "load_saved_roots",
        lambda stack, group: (
            {mode: {"saved_output": (None, value)} for mode, value in sources[group].items()},
            provenance[group],
        ),
    )
    entries = []
    for index in range(2):

        def retain(variant, value, coordinate):
            result = (pinned if variant == "pinned" else rational.rational_start)(value, 0.0)
            return [{"phase": phase, "output": result} for phase in accuracy.PHASES], {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "storage_offset": value.storage_offset(),
                "source_shape": list(value.shape),
                "source_stride": list(value.stride()),
                "source_storage_offset": value.storage_offset(),
                "rng_exact": True,
                "compilation_preserved": True,
                "graphs_preserved": True,
                "captured_graph_replays": 3,
                "input_exact_finite": tracing.comparison(value, value, "input"),
            }

        output, lifecycle = execution.retain_protocol(sources, index, retain)
        log = work / f"process-{index}.log"
        log.write_text("cpu fixture\n")
        pid = 200 + index
        argv = owner.worker_command(tmp_path, sha256(binding), index)
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
            "helper_sources_sha256": {},
            "coordinates": accuracy.coordinates(index),
            "rng_exact": True,
            "compilation_preserved": True,
            "graphs_preserved": True,
            "compile_options": {"dynamic": False},
            "physical_uuid": origin["physical_uuid"],
            "actual_uuid": origin["physical_uuid"],
            "outputs": str(work / f"process-{index}.pt"),
            "provenance": provenance,
            "runtime": config["runtime"],
            "freeze": 0.0,
            "lifecycle": lifecycle,
            "analysis": accuracy.summarize(output, sources, pinned, 0.0, index),
        }
        if index == 1 and fault == "changed_output":
            output[0]["output"][4].add_(0.01)
        torch.save(output, report["outputs"])
        report["outputs_sha256"] = sha256(Path(report["outputs"]))
        write(Path(entry["report"]), report)
        entry["report_sha256"] = sha256(Path(entry["report"]))
        entries.append(entry)
    if fault == "wrong_parent":
        entries[1]["generation"]["parent"] = 999
    elif fault == "duplicate_worker":
        entries[1]["pid"], entries[1]["start_ticks"] = entries[0]["pid"], entries[0]["start_ticks"]
    if fault != "none":
        with pytest.raises(ValueError):
            controller.compare_results(tmp_path, entries)
        return
    result = controller.compare_results(tmp_path, entries)
    assert len(result["cross_process"]) == 32
    assert result["all_cross_process_exact_finite"]
    assert result["all_repeat_outputs_exact_finite"]
    assert result["all_unaffected_outputs_exact_finite"]
    assert result["all_rational_O_within_fp64_tolerance"]


def test_new_owner_source_closure_preserves_every_accepted_numerical_helper():
    root = execution.replay.ROOT
    sources = owner.helper_sources(root)
    parent = json.loads((root / owner.PARENT).read_text())
    assert all(sha256(sources[name]) == value for name, value in parent["helpers_sha256"].items())
    assert set(sources) == set(parent["helpers_sha256"]) | set(owner.NEW_HELPERS)


def test_archived_start_owner_rejects_changed_source_even_when_binding_is_rehashed(tmp_path, monkeypatch):
    binding, helpers, _, work = owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    inherited = execution.replay.ROOT / owner.PARENT
    cpu = json.loads(inherited.read_text())
    for name in cpu["helpers_sha256"]:
        shutil.copy2(Path(cpu["helpers"]) / name, helpers / name)
    inventory = dict(cpu["helpers_sha256"])
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    receipt = tmp_path / "cpu.json"
    receipt.write_text("{}")
    previous = {"restored_keeper": {"devices": [["0", "GPU-cpu", "CPU fixture"]]}}
    frozen = {
        "status": owner.STATUS,
        "root": str(tmp_path),
        "helpers": str(helpers),
        "work": str(work),
        "node": owner.NODE,
        "helpers_sha256": inventory,
        "environment": owner.ENVIRONMENT,
        "worker_environment": owner.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": "GPU-cpu"},
        "devices": previous["restored_keeper"]["devices"],
        "predecessor": previous,
        "controller_command": owner.controller_command(tmp_path),
        "workers": 2,
        "worker_timeout": owner.WORKER_TIMEOUT,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "cpu_receipt": str(receipt),
        "cpu_receipt_sha256": sha256(receipt),
    }
    write(binding, frozen)
    monkeypatch.setattr(owner, "require_cpu", lambda root, path: {"helpers_sha256": inventory})
    monkeypatch.setattr(owner, "predecessor", lambda root: previous)
    # A fresh process uses only archived imports and persisted files, matching the launch boundary.
    program = (
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0,sys.argv[1]);import root_start_gpu_owner as owner\n"
        "owner.require_cpu=lambda root,path:{'helpers_sha256':owner.read(owner.locations(root)[0])['helpers_sha256']}\n"
        "owner.predecessor=lambda root:owner.read(owner.locations(root)[0])['predecessor']\n"
        "owner.require_frozen(Path(sys.argv[2]),owner.sha256(owner.locations(Path(sys.argv[2]))[0]))\n"
    )
    for name in owner.NEW_HELPERS:
        if name != "test_root_start_gpu.py":
            shutil.copy2(Path(owner.__file__).parent / name, helpers / name)
            inventory[name] = sha256(helpers / name)
    with tarfile.open(archive, "w") as bundle:
        for name in inventory:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    frozen["helpers_sha256"], frozen["archive_sha256"] = inventory, sha256(archive)
    write(binding, frozen)
    command = [sys.executable, "-P", "-c", program, str(helpers), str(tmp_path)]
    valid = subprocess.run(
        command, env=os.environ | {"CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=30
    )
    assert valid.returncode == 0, valid.stderr
    (helpers / "root_start_execution.py").write_text("# changed source\n")
    changed = subprocess.run(
        command, env=os.environ | {"CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=30
    )
    assert changed.returncode != 0
    assert "helper" in changed.stderr
