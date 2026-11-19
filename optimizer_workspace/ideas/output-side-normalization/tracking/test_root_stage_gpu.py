# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU evidence reconstruction and real subprocess cleanup for the unissued root replay."""

import copy
import fcntl
import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from xml.etree import ElementTree as ET

import fused_attribution_validation_controller as acknowledgment
import pytest
import root_stage_failed_predecessor as failed_previous
import root_stage_gpu_controller as controller
import root_stage_gpu_owner as owner
import root_stage_gpu_worker as worker
import root_stage_trace as tracing
import torch
from fused_continuation_test_support import write_process
from selected_lr_release_owner import candidate_files, sha256
from watch_wave5 import write


def candidate(value: float) -> tuple:
    matrix = torch.full((2, 3, 3), value)
    return (
        (matrix.clone(), matrix.clone()),
        matrix.clone(),
        torch.zeros(2, dtype=torch.bool),
        (torch.ones(2, 3), matrix.clone(), torch.ones(2)),
    )


def saved_and_retained(index: int) -> tuple[dict, list]:
    saved = {
        group: {mode: {"saved_output": candidate(1.0)} for mode in ("continuous", "resumed")} for group in ("d", "f")
    }
    retained = []
    for coordinate in worker.coordinates(index):
        output = candidate(1.0)
        stages = None
        if coordinate["protocol"] == "trace":
            counts, records = Counter(), []
            for stage in worker.stage_sequence(3):
                values = {name: output[1].clone() for name in tracing.OUTPUT_FIELDS[stage]}
                if stage in tracing.PRODUCTS:
                    values.update(left=output[1].clone(), right=output[1].clone())
                records.append({"stage": stage, "occurrence": counts[stage], "values": values})
                counts[stage] += 1
            stages = {
                "schema": 1,
                "records": records,
                "output": tracing.cpu_tree(output),
            }
        retained.append({**coordinate, "output": output, "stages": stages})
    return saved, retained


@pytest.mark.parametrize("index", (0, 1))
def test_retained_reconstruction_checks_complete_opposite_protocol_order_and_historical_outputs(index):
    saved, retained = saved_and_retained(index)
    result = worker.summarize(retained, saved, index)
    assert result["trace_transparency_exact"]
    assert result["capture_repeat_safety_exact"]
    assert result["intermediates_interpretable"]
    assert result["historical_outputs_within_tolerance"]
    assert len(result["trace_transparency"]) == 12
    assert len(result["capture_repeat_safety"]) == len(result["historical_output_comparisons"]) == 24
    assert all(value["complete_stage_coverage"] for value in result["repeated_stages"])
    assert result["coordinates"][0]["protocol"] == ("control" if index == 0 else "trace")
    saved["f"]["continuous"]["saved_output"][0][0][-1, -1, -1] += 0.25
    failing = worker.summarize(retained, saved, index)
    assert failing["intermediates_interpretable"]
    assert not failing["historical_outputs_within_tolerance"]
    failures = [item for item in failing["historical_output_comparisons"] if not item["within_tolerance"]]
    assert len(failures) == 6
    assert all(item["group"] == "f" and item["mode"] == "continuous" for item in failures)
    assert all(item["changed_tensors"][0]["mismatched_elements"] == 1 for item in failures)


def test_transparency_failure_prevents_interpretation_despite_exact_trace_repeats():
    saved, retained = saved_and_retained(0)
    target = next(item for item in retained if item["protocol"] == "control" and item["group"] == "f")
    target["output"][0][0][-1, -1, -1] += 0.25
    result = worker.summarize(retained, saved, 0)
    assert not result["trace_transparency_exact"]
    assert not result["intermediates_interpretable"]
    assert all(value["output"]["exact"] for value in result["repeated_stages"])
    assert any(not item["within_tolerance"] for item in result["trace_transparency"])
    assert json.loads(json.dumps(result, allow_nan=False))["intermediates_interpretable"] is False


def test_same_input_stage_change_prevents_interpretation_despite_equal_final_outputs():
    saved, retained = saved_and_retained(0)
    target = next(item for item in retained if item["protocol"] == "trace" and item["phase"] == "replay")
    target["stages"]["records"][0]["values"]["B"][-1, -1, -1] += 0.25
    result = worker.summarize(retained, saved, 0)
    assert result["trace_transparency_exact"]
    assert result["capture_repeat_safety_exact"]
    assert not result["capture_repeat_stages_exact"]
    assert not result["intermediates_interpretable"]
    changed = result["repeated_stages"][0]["stages"][0]
    assert changed["changed_tensors"][0]["mismatched_elements"] == 1
    assert changed["changed_tensors"][0]["max_absolute_error"] == 0.25


def test_matching_nonfinite_intermediates_never_pass_interpretation():
    saved, retained = saved_and_retained(0)
    for item in retained:
        if item["stages"] is not None:
            item["stages"]["records"][0]["values"]["B"].fill_(float("inf"))
    result = worker.summarize(retained, saved, 0)
    assert result["trace_transparency_exact"]
    assert not result["trace_stages_finite"]
    assert not result["intermediates_interpretable"]


def test_missing_phase_or_output_tensors_cannot_be_reconstructed_as_complete():
    saved, retained = saved_and_retained(0)
    with pytest.raises(ValueError, match="coverage"):
        worker.summarize(retained[:-1], saved, 0)
    retained[0]["output"] = retained[0]["output"][:-1]
    with pytest.raises(ValueError, match="seven"):
        worker.summarize(retained, saved, 0)


def test_cpu_staging_preserves_full_split_input_values_and_storage_layout_and_detects_changes():
    values = {
        "C": torch.arange(18, dtype=torch.float32).reshape(2, 3, 3).mT,
        "tol": torch.ones(2, 3),
        "SB": torch.ones(2, 6, 3, dtype=torch.float16),
        "S2": torch.ones(2, 3, 6, dtype=torch.float16),
    }
    group, layouts = worker.stage_group(values, "cpu")
    worker.require_group_unchanged(group, values, layouts)
    assert group.C.stride() == values["C"].stride()
    assert group.basis.SB.dtype == torch.float16
    group.C[-1, -1, -1] += 0.25
    with pytest.raises(ValueError, match="input values"):
        worker.require_group_unchanged(group, values, layouts)
    assert values["C"][-1, -1, -1] == 17


def test_live_root_predecessor_refuses_before_future_terminal_evidence_or_gpu_inventory(tmp_path):
    _, helpers, state, _, _ = owner.previous.locations(tmp_path)
    helpers.mkdir(parents=True)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"status": "validating"}))
    (helpers.parent / "controller.json").write_text(json.dumps({"status": "acknowledged"}))
    with pytest.raises(RuntimeError, match="incomplete"):
        owner.predecessor(tmp_path)
    assert not owner.terminal_observation(tmp_path).exists()
    assert not owner.locations(tmp_path)[0].exists()


def test_unissued_worker_refuses_before_cuda_initialization_or_output_creation(tmp_path, monkeypatch):
    def forbid_cuda(*args, **kwargs):
        pytest.fail("Unissued root worker touched CUDA")

    monkeypatch.setattr(torch.cuda, "device_count", forbid_cuda)
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbid_cuda)
    with pytest.raises(FileNotFoundError):
        worker.run(tmp_path, "0" * 64, owner.locations(tmp_path)[2], 0)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ("zero", "nonzero", "detached"))
def test_real_owned_worker_acknowledges_exit_and_cleans_detached_descendants(tmp_path, monkeypatch, failure):
    binding, helpers, state_path, work = owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    work.mkdir(parents=True)
    binding.parent.mkdir(parents=True, exist_ok=True)
    binding.write_text("{}")
    source = helpers / "root_stage_gpu_worker.py"
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
    command = [
        sys.executable,
        str(source),
        "--state",
        str(state_path),
        "--work",
        str(work),
        "--failure",
        failure,
    ]
    monkeypatch.setattr(owner, "worker_command", lambda root, digest, index: command)
    environment = dict(owner.ENVIRONMENT)
    state = {"status": "replaying", "controller_pid": os.getpid(), "controller_start_ticks": "synthetic"}
    if failure == "zero":
        execution = controller.run_worker(tmp_path, {"worker_environment": environment}, state, 0)
        assert execution["exit_code"] == execution["cleanup_exit_code"] == 0
        assert execution["marked_descendants_absent"]
    else:
        with pytest.raises(RuntimeError, match="worker failed"):
            controller.run_worker(tmp_path, {"worker_environment": environment}, state, 0)
    saved = json.loads(state_path.read_text())
    execution = saved["active_worker"]
    assert execution["spawn_status"] == "acknowledged"
    assert execution["exit_code"] == (0 if failure == "zero" else 7)
    assert execution["marked_descendants_absent"]
    assert owner.handoff.observed(execution["pid"]) is None
    assert not owner.handoff.marker_processes(owner.CHILD_MARKER, execution["marker"])
    if failure == "detached":
        pid = int((work / "detached.pid").read_text())
        process = owner.handoff.observed(pid)
        assert process is None or process["status"] == "Z"


def test_parent_lock_probe_requires_held_exclusive_lock_and_preserves_it(tmp_path):
    path = tmp_path / "allocation.lock"
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner.require_held_lock(os.getpid(), stream.fileno(), path)
        with path.open("r+") as independent, pytest.raises(BlockingIOError):
            fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with path.open("r+") as unlocked:
        with pytest.raises(RuntimeError, match="exclusive flock"):
            owner.require_held_lock(os.getpid(), unlocked.fileno(), path)


@contextmanager
def archived_controller(helpers):
    names = {path.stem for path in helpers.glob("*.py")}
    loaded = {name: sys.modules[name] for name in names if name in sys.modules}
    previous_path = list(sys.path)
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(helpers))
        yield importlib.import_module("root_stage_gpu_controller")
    finally:
        sys.path[:] = previous_path
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(loaded)


@pytest.fixture
def terminal_fixture(tmp_path, monkeypatch):
    external_root = worker.replay.ROOT
    actual_binding = owner.read(owner.previous.locations(external_root)[0])
    sources = owner.helper_sources(external_root)
    root = tmp_path / "output"
    binding_path, helpers, state_path, _, gpu_path = owner.previous.locations(root)
    helpers.mkdir(parents=True)
    for name in actual_binding["helpers_sha256"]:
        shutil.copy2(sources[name], helpers / name)
    candidate_path = root / "candidate"
    experiment = candidate_path / "experiment"
    experiment.mkdir(parents=True)
    (experiment / "fixture.py").write_text("VALUE = 1\n")
    pinned = candidate_path / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    (pinned.parent / "external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    preparation = candidate_path / owner.previous.PREPARATION
    write(preparation, {"parent_metadata_sha256": {}})
    devices = [[str(index), f"GPU-00000000-0000-0000-0000-{index:012d}", "NVIDIA A100-SXM4-40GB"] for index in range(8)]
    keeper_pid = 920001
    generation = {
        str(keeper_pid): {
            "pid": keeper_pid,
            "parent": 1,
            "status": "S",
            "group": keeper_pid,
            "session": keeper_pid,
            "start_ticks": "keeper-parent",
            "command": ["synthetic-keeper"],
        }
    }
    for index in range(8):
        pid = keeper_pid + index + 1
        generation[str(pid)] = {
            "pid": pid,
            "parent": keeper_pid,
            "status": "R",
            "group": pid,
            "session": pid,
            "start_ticks": f"keeper-rank-{index}",
            "command": ["synthetic-keeper-rank"],
        }
    keeper = {
        "pid": keeper_pid,
        "generation": generation,
        "devices": devices,
        "occupied": [[str(keeper_pid + index + 1), device[1]] for index, device in enumerate(devices)],
    }
    frozen = {
        "root": str(root),
        "helpers": str(helpers),
        "candidate": str(candidate_path),
        "files_sha256": candidate_files(candidate_path),
        "preparation_sha256": sha256(preparation),
        "cpu_receipt_sha256": "1" * 64,
        "controller_cpu_receipt_sha256": "2" * 64,
        "helpers_sha256": {name: sha256(helpers / name) for name in actual_binding["helpers_sha256"]},
        "commands": [["synthetic-command", str(index)] for index in range(12)],
    }
    cpu_path = candidate_path / "candidate-cpu-receipt.json"
    controller_cpu = candidate_path / "controller-cpu-receipt.json"
    write(cpu_path, {"fixture": True})
    write(controller_cpu, {"fixture": True})
    frozen["cpu_receipt_sha256"], frozen["controller_cpu_receipt_sha256"] = sha256(cpu_path), sha256(controller_cpu)
    archive_path = root / "run-preparation/previous-source.tar"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    frozen["archive"] = str(archive_path)
    with tarfile.open(archive_path, "x") as archive:
        for name in owner.previous_inventory(frozen):
            path = helpers / name.removeprefix("tracking/") if name.startswith("tracking/") else candidate_path / name
            archive.add(path, arcname=name, recursive=False)
        archive.add(pinned.parent / "external-sources", arcname="artifacts/external-sources", recursive=False)
    frozen["archive_sha256"] = sha256(archive_path)
    write(binding_path, frozen)
    binding_hash = sha256(binding_path)
    monkeypatch.setattr(owner, "PREVIOUS_BINDING_SHA256", binding_hash)
    write(helpers.parent / "controller.log", {"fixture": True})
    log = helpers.parent / "validator.log"
    log.write_text("synthetic-terminal-validation\n")
    intent = {
        "status": "acknowledged",
        "binding_sha256": binding_hash,
        "pid": 920101,
        "start_ticks": "original-controller",
        "command": owner.previous.controller_command(root),
        "marker": "original-root-controller",
    }
    write(helpers.parent / "controller.json", intent)
    gates = {"trace_transparency": True, "prepared_all_rank_continuation": False}
    write(
        gpu_path,
        {
            "status": "completed_root_preparation_diagnostic_not_production_gate",
            "execution_binding_sha256": binding_hash,
            "owner": {"controller_pid": intent["pid"], "controller_start_ticks": intent["start_ticks"]},
            "tolerance": {"rtol": 1e-5, "atol": 1e-6},
            "analysis": {"gates": gates, "diagnostic_intervention_accepted": False},
            "gpu_commands": [{"command": value, "exit_code": 0} for value in frozen["commands"]],
        },
    )
    state = {
        "status": "diagnostic_complete",
        "binding_sha256": binding_hash,
        "validation_sha256": sha256(gpu_path),
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "keeper_restore": 0,
        "keeper_release_complete": True,
        "marked_descendants_absent": True,
        "controller_pid": intent["pid"],
        "controller_start_ticks": intent["start_ticks"],
        "child_pid": 920102,
        "child_start_ticks": "original-validator",
        "child_marker": "original-root-child",
        "restored_keeper": keeper,
        "log": str(log),
        "log_sha256": sha256(log),
    }
    write(state_path, state)
    paths = (binding_path, state_path, gpu_path, helpers.parent / "controller.json", log)
    observation = {
        "status": "root_diagnostic_terminal_independently_reconstructed_exact_restoration_verified",
        "probe_sha256": owner.TERMINAL_PROBE_SHA256,
        "controller_pid_absent": intent["pid"],
        "validator_pid_absent": state["child_pid"],
        "marked_descendants_absent": True,
        "commands": 12,
        "terminal_time": state["time"],
        "keeper": keeper,
        "gates": gates,
        "diagnostic_intervention_accepted": False,
        "evidence_sha256": {str(path.relative_to(root)): sha256(path) for path in paths},
    }
    write(owner.terminal_observation(root), observation)
    cpu_dir = root / "cpu"
    cpu_helpers = cpu_dir / "tracking"
    cpu_helpers.mkdir(parents=True)
    for name, path in sources.items():
        shutil.copy2(path, cpu_helpers / name)
    cpu_hashes = {name: sha256(cpu_helpers / name) for name in sources}
    xml = ET.Element("testsuites")
    suite = ET.SubElement(xml, "testsuite", tests=str(len(owner.TESTS)), failures="0", errors="0", skipped="0")
    for name in owner.TESTS:
        ET.SubElement(suite, "testcase", classname=Path(name).stem, name="synthetic-fixture")
    ET.ElementTree(xml).write(cpu_dir / "tests.xml")
    (cpu_dir / "cpu.log").write_text("synthetic-fixture-only\n")
    cpu_archive = cpu_dir / "source.tar"
    with tarfile.open(cpu_archive, "x") as archive:
        for name in cpu_hashes:
            archive.add(cpu_helpers / name, arcname=f"tracking/{name}", recursive=False)
    cpu_receipt = cpu_dir / "receipt.json"
    write(
        cpu_receipt,
        {
            "status": owner.CPU_STATUS,
            "root": str(root),
            "helpers": str(cpu_helpers),
            "helpers_sha256": cpu_hashes,
            "fixtures_sha256": {},
            "command": owner.cpu_command(cpu_dir),
            "exit_code": 0,
            "cuda_visible_devices": "",
            **{
                key: str(cpu_dir / filename)
                for key, filename in (("log", "cpu.log"), ("tests", "tests.xml"), ("archive", "source.tar"))
            },
            **{
                f"{key}_sha256": sha256(cpu_dir / filename)
                for key, filename in (("log", "cpu.log"), ("tests", "tests.xml"), ("archive", "source.tar"))
            },
        },
    )
    return root, cpu_receipt, keeper, binding_hash


def isolated_owner_boundary(archived, monkeypatch, keeper, binding_hash, lock_path):
    monkeypatch.setattr(archived.owner, "PREVIOUS_BINDING_SHA256", binding_hash)
    monkeypatch.setattr(archived.owner.handoff, "keeper_observation", lambda: copy.deepcopy(keeper))
    monkeypatch.setattr(archived.owner.handoff, "observed", lambda pid: None)
    monkeypatch.setattr(archived.owner.handoff, "marker_processes", lambda key, value: [])
    failure = sys.modules["selected_lr_failure_handoff"]
    monkeypatch.setattr(failure, "MANAGER_LOCK", lock_path)
    monkeypatch.setattr(failure, "require_manager_source", lambda: None)


@pytest.fixture
def failed_terminal_fixture(terminal_fixture):
    root, cpu_receipt, keeper, binding_hash = terminal_fixture
    binding_path, helpers, state_path, _, gpu_path = owner.previous.locations(root)
    gpu, state = (owner.read(path) for path in (gpu_path, state_path))
    gpu.update(status="needs_attention", error=failed_previous.reconstruction.FAILURE)
    gpu.pop("analysis")
    gpu["cold_probe_evidence"] = {"within_tolerance": True}
    for index, execution in enumerate(gpu["gpu_commands"]):
        log = helpers.parent / f"command-{index}.log"
        log.write_text(f"synthetic-command-{index}\n")
        execution.update(log=str(log), log_sha256=sha256(log))
    write(gpu_path, gpu)
    state.update(status="needs_attention", exit_code=1, cleanup_exit_code=1)
    state["error"] = "RuntimeError: Root diagnostic validator failed"
    write(state_path, state)
    original = {
        path: sha256(path)
        for path in (
            binding_path,
            state_path,
            gpu_path,
            helpers.parent / "controller.json",
            helpers.parent / "controller.log",
            Path(state["log"]),
        )
    }
    write(cpu_receipt, owner.read(cpu_receipt))

    def checkpoint_pairs(actual, expected, failing):
        pairs = {}
        for step in (22, 23, 24):
            ranks = []
            for rank in range(8):
                files = {}
                for directory in (actual, expected):
                    path = directory / f"checkpoints/step_{step:06d}/rank_{rank}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        write(path, {"rank": rank, "step": step})
                    files[str(path)] = sha256(path)
                changed = (
                    [{"name": f"rank_{rank}.model.weight", "within_tolerance": False}] if failing and step > 22 else []
                )
                ranks.append(
                    {
                        "rank": rank,
                        "tensor_comparisons": 1,
                        "failed_tensor_count": len(changed),
                        "changed_tensors": changed,
                        "scalar_or_structure_differences": [],
                        "checkpoint_sha256": files,
                    }
                )
            pairs[str(step)] = {"step": step, "ranks": ranks, "within_tolerance": not (failing and step > 22)}
        return {"checkpoints": pairs, "loss": {"comparisons": 39, "failures": [], "within_tolerance": True}}

    analysis = {
        "continuation": {
            regime: {
                protocol: checkpoint_pairs(
                    failed_previous.validation.directory(root, regime, protocol, "resumed"),
                    failed_previous.validation.directory(root, regime, protocol, "continuous"),
                    True,
                )
                for protocol in ("control", "trace")
            }
            for regime in ("unprepared", "prepared")
        },
        "transparency": {
            regime: {
                mode: checkpoint_pairs(
                    failed_previous.validation.directory(root, regime, "trace", mode),
                    failed_previous.validation.directory(root, regime, "control", mode),
                    False,
                )
                for mode in ("continuous", "resumed")
            }
            for regime in ("unprepared", "prepared")
        },
        "continuous_controls": {
            protocol: checkpoint_pairs(
                failed_previous.validation.directory(root, "prepared", protocol, "continuous"),
                failed_previous.validation.directory(root, "unprepared", protocol, "continuous"),
                False,
            )
            for protocol in ("control", "trace")
        },
        "unprepared_original_reproduction": {
            f"{protocol}_{mode}": checkpoint_pairs(
                failed_previous.validation.directory(root, "unprepared", protocol, mode),
                root / "validation/fused_continuation_20261118" / f"{protocol}_{mode}",
                False,
            )
            for protocol in ("control", "trace")
            for mode in ("continuous", "resumed")
        },
        "traces": {},
        "trace_root_lifecycle": {},
        "cold_root": gpu["cold_probe_evidence"],
        "gates": {name: name != "prepared_all_rank_continuation" for name in failed_previous.GATES},
        "diagnostic_intervention_accepted": False,
    }
    for regime in ("unprepared", "prepared"):
        for mode in ("continuous", "resumed"):
            key, hashes, lifecycle = f"{regime}_{mode}", {}, {}
            directory = failed_previous.validation.directory(root, regime, "trace", mode)
            for rank in range(8):
                for step in (22, 23, 24):
                    for suffix in (".pt", ".json"):
                        path = directory / f"continuation-trace/rank_{rank}/update_{step:06d}{suffix}"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        write(path, {"rank": rank, "step": step, "regime": regime, "mode": mode})
                        hashes[str(path)] = sha256(path)
                    flags = {
                        "warm": not (regime == "unprepared" and mode == "resumed" and step == 23),
                        "graph_active": (
                            step != 22 and not (mode == "resumed" and (regime == "unprepared" or step == 23))
                        ),
                    }
                    lifecycle[f"{rank}:{step}"] = {group: dict(flags) for group in ("d", "f")}
            analysis["traces"][key] = {"files_sha256": hashes}
            analysis["trace_root_lifecycle"][key] = lifecycle
    runner = root / failed_previous.RUNNER
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("# synthetic read-only runner\n")
    work = root / failed_previous.WORK
    work.mkdir(parents=True)
    command = [owner.PYTHON, "-P", str(runner), "--phase", "execute"]
    intent = {
        "status": "acknowledged",
        "pid": 920901,
        "start_ticks": "read-only-runner",
        "command": command,
        "marker": "read-only-runner-marker",
        "runner_sha256": sha256(runner),
        "binding_sha256": binding_hash,
        "cpu_receipt_sha256": sha256(cpu_receipt),
        "out": str(root / failed_previous.RESULT),
    }
    intent["generation"] = {
        "pid": intent["pid"],
        "start_ticks": intent["start_ticks"],
        "parent": 1,
        "group": intent["pid"],
        "session": intent["pid"],
        "command": command,
        "status": "R",
    }
    write(work / "intent.json", intent)
    result = {
        "status": failed_previous.reconstruction.STATUS,
        "original_status": state["status"],
        "original_failure": gpu["error"],
        "terminal_time": state["time"],
        "evidence_sha256": {str(path.relative_to(root)): value for path, value in original.items()},
        "binding_sha256": binding_hash,
        "verifier_sha256": owner.read(cpu_receipt)["helpers_sha256"]["root_preparation_reconstruction.py"],
        "controller_pid_absent": state["controller_pid"],
        "validator_pid_absent": state["child_pid"],
        "marked_descendants_absent": True,
        "keeper": keeper,
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "gpu_commands": gpu["gpu_commands"],
        "analysis": analysis,
    }
    write(root / failed_previous.RESULT, result)
    read_only_state = {key: value for key, value in intent.items() if key not in ("generation", "time", "marker")}
    read_only_state.update(
        status="reconstructed",
        result_status=result["status"],
        result_sha256=sha256(root / failed_previous.RESULT),
        exit_code=0,
    )
    write(work / "state.json", read_only_state)
    (work / "runner.log").write_text("synthetic read-only terminal runner\n")
    return root, cpu_receipt, keeper, binding_hash, original


def isolated_failed_owner_boundary(archived, boundary, fixture, lock_path):
    root, cpu_receipt, keeper, binding_hash, original = fixture
    isolated_owner_boundary(archived, boundary, keeper, binding_hash, lock_path)
    failed = archived.owner.failed_predecessor
    previous_binding, _, previous_state, _, previous_gpu = archived.owner.previous.locations(root)
    boundary.setattr(failed.reconstruction, "BINDING_SHA256", binding_hash)
    boundary.setattr(failed.reconstruction, "FAILED_STATE_SHA256", original[previous_state])
    boundary.setattr(failed.reconstruction, "FAILED_RECEIPT_SHA256", original[previous_gpu])
    boundary.setattr(failed, "CPU", str(cpu_receipt.relative_to(root)))
    boundary.setattr(failed, "CPU_SHA256", sha256(cpu_receipt))
    boundary.setattr(failed, "RUNNER_SHA256", sha256(root / failed.RUNNER))
    assert original[previous_binding] == binding_hash


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "pending",
        "live_runner",
        "changed_output",
        "missing_rank",
        "false_gate",
        "changed_keeper",
        "postseal_checkpoint",
        "false_seal",
    ),
)
def test_archived_failed_predecessor_handoff_requires_reconstructed_result_and_exact_restoration(
    failed_terminal_fixture, monkeypatch, tmp_path, fault
):
    fixture = failed_terminal_fixture
    root, cpu_receipt, keeper, _, original = fixture
    with archived_controller(cpu_receipt.parent / "tracking") as archived, monkeypatch.context() as boundary:
        isolated_failed_owner_boundary(archived, boundary, fixture, tmp_path / "manager.lock")
        failed = archived.owner.failed_predecessor
        work = root / failed.WORK
        state = archived.read(work / "state.json")
        result_path = root / failed.RESULT
        if fault == "pending":
            state["status"] = "reconstructing"
            write(work / "state.json", state)
        elif fault == "live_runner":
            boundary.setattr(
                archived.owner.handoff, "observed", lambda pid: {"pid": pid} if pid == state["pid"] else None
            )
        elif fault == "changed_output":
            result_path.write_text(result_path.read_text() + "\n")
        elif fault in ("missing_rank", "false_gate"):
            result = archived.read(result_path)
            if fault == "missing_rank":
                result["analysis"]["continuation"]["prepared"]["control"]["checkpoints"]["23"]["ranks"].pop()
            else:
                result["analysis"]["gates"]["prepared_all_rank_continuation"] = True
            write(result_path, result)
            state["result_sha256"] = sha256(result_path)
            write(work / "state.json", state)
        elif fault == "changed_keeper":
            changed = copy.deepcopy(keeper)
            changed["generation"][str(keeper["pid"])]["start_ticks"] = "changed-keeper"
            boundary.setattr(archived.owner.handoff, "keeper_observation", lambda: changed)
        if fault not in ("none", "postseal_checkpoint", "false_seal"):
            with pytest.raises((RuntimeError, ValueError)):
                archived.owner.failed_predecessor.seal(root)
            assert not archived.owner.locations(root)[0].exists()
        else:
            boundary.setenv("CUDA_VISIBLE_DEVICES", "")
            sealed = archived.owner.failed_predecessor.seal(root)
            if fault in ("postseal_checkpoint", "false_seal"):
                if fault == "postseal_checkpoint":
                    path = Path(next(iter(sealed["retained_file_identity"])))
                    path.write_text(path.read_text() + "\n")
                else:
                    sealed["predecessor"]["diagnostic_intervention_accepted"] = True
                    write(root / failed.SEAL, sealed)
                with pytest.raises(ValueError, match="terminal seal"):
                    archived.freeze(root, cpu_receipt)
                assert not archived.owner.locations(root)[0].exists()
                assert original == {path: sha256(path) for path in original}
                return
            frozen = archived.freeze(root, cpu_receipt)
            assert frozen["predecessor"]["diagnostic_intervention_accepted"] is False
            assert frozen["predecessor"]["gates"]["prepared_all_rank_continuation"] is False
    if fault == "none":
        _, helpers, _, _ = owner.locations(root)
        with archived_controller(helpers) as archived, monkeypatch.context() as boundary:
            isolated_failed_owner_boundary(archived, boundary, fixture, tmp_path / "manager.lock")
            assert archived.owner.require_frozen(root, sha256(owner.locations(root)[0])) == frozen
    assert original == {path: sha256(path) for path in original}


def test_archived_freeze_and_fresh_checker_bind_terminal_evidence_and_refuse_changed_source(
    terminal_fixture, monkeypatch, tmp_path
):
    root, cpu_receipt, keeper, binding_hash = terminal_fixture
    with archived_controller(cpu_receipt.parent / "tracking") as archived, monkeypatch.context() as boundary:
        isolated_owner_boundary(archived, boundary, keeper, binding_hash, tmp_path / "manager.lock")
        frozen = archived.freeze(root, cpu_receipt)
    binding, helpers, _, _ = owner.locations(root)
    with archived_controller(helpers) as archived, monkeypatch.context() as boundary:
        isolated_owner_boundary(archived, boundary, keeper, binding_hash, tmp_path / "manager.lock")
        assert archived.owner.require_frozen(root, sha256(binding)) == frozen
        assert frozen["predecessor"]["restored_keeper"]["pid"] == keeper["pid"]
        assert frozen["workers"] == 2
        source = helpers / "root_stage_gpu_worker.py"
        source.write_text(source.read_text() + "\n")
        with pytest.raises(ValueError, match="helper bytes"):
            archived.owner.require_frozen(root, sha256(binding))


def test_archived_controller_restores_keeper_after_child_spawn_failure(terminal_fixture, monkeypatch, tmp_path):
    root, cpu_receipt, keeper, binding_hash = terminal_fixture
    lock_path = tmp_path / "manager.lock"
    with archived_controller(cpu_receipt.parent / "tracking") as archived, monkeypatch.context() as boundary:
        isolated_owner_boundary(archived, boundary, keeper, binding_hash, lock_path)
        archived.freeze(root, cpu_receipt)
    binding, helpers, state_path, work = owner.locations(root)
    own = {
        "pid": os.getpid(),
        "parent": os.getppid(),
        "status": "R",
        "group": os.getpid(),
        "session": os.getpid(),
        "start_ticks": "cpu-controller",
        "command": owner.controller_command(root),
    }
    intent = {
        "status": "acknowledged",
        "binding_sha256": sha256(binding),
        "pid": own["pid"],
        "start_ticks": own["start_ticks"],
        "command": own["command"],
        "marker": "synthetic-owner",
        "generation": own,
    }
    write(helpers.parent / "controller.json", intent)
    restored = copy.deepcopy(keeper)
    restored["generation"][str(keeper["pid"])]["start_ticks"] = "fresh-restored-generation"
    calls = []
    with archived_controller(helpers) as archived, monkeypatch.context() as boundary:
        isolated_owner_boundary(archived, boundary, keeper, binding_hash, lock_path)
        real_process = archived.process_context

        def process_boundary(pid, *args, **kwargs):
            return own if pid == own["pid"] else real_process(pid, *args, **kwargs)

        def host_boundary(command, **kwargs):
            if command == ["hostname", "-I"]:
                return owner.NODE
            if command == ["ps", "-eo", "args="]:
                return ""
            raise AssertionError(command)

        def release_boundary(accepted, state, path, handle):
            assert accepted["pid"] == keeper["pid"]
            archived.owner.require_held_lock(os.getpid(), handle, lock_path)
            state.update(keeper_release_attempted=True, keeper_release_complete=True)
            archived.write(path, state)
            calls.append("released")

        def restore_boundary(handle):
            archived.owner.require_held_lock(os.getpid(), handle, lock_path)
            boundary.setattr(archived.owner.handoff, "keeper_observation", lambda: copy.deepcopy(restored))
            calls.append("restored")

        def spawn_failure(command, **kwargs):
            raise OSError("synthetic root child spawn failure")

        boundary.setattr(archived, "process_context", process_boundary)
        boundary.setattr(subprocess, "check_output", host_boundary)
        boundary.setattr(subprocess, "Popen", spawn_failure)
        boundary.setattr(archived, "release_accepted_keeper", release_boundary)
        boundary.setattr(archived, "restore_accepted_keeper", restore_boundary)
        boundary.setenv(owner.CONTROLLER_MARKER, intent["marker"])
        with pytest.raises(OSError, match="child spawn"):
            archived.execute(root)
    state = json.loads(state_path.read_text())
    assert calls == ["released", "restored"]
    assert state["status"] == "needs_attention"
    assert state["keeper_restore"] == 0
    assert state["restored_keeper"]["generation"][str(keeper["pid"])]["start_ticks"] == "fresh-restored-generation"
    assert state["executions"] == []
    assert work.is_dir() and not (work / "comparison.json").exists()


@pytest.mark.parametrize("failure", (False, True))
def test_archived_start_persists_acknowledgment_or_spawn_failure_without_retry(
    terminal_fixture, monkeypatch, tmp_path, failure
):
    root, cpu_receipt, keeper, binding_hash = terminal_fixture
    lock_path = tmp_path / "manager.lock"
    with archived_controller(cpu_receipt.parent / "tracking") as archived, monkeypatch.context() as boundary:
        isolated_owner_boundary(archived, boundary, keeper, binding_hash, lock_path)
        archived.freeze(root, cpu_receipt)
    _, helpers, state_path, work = owner.locations(root)
    proc = tmp_path / "proc"

    class FakeController:
        pid = 920201

        def poll(self):
            return None

    def host_boundary(command, **kwargs):
        if command == ["hostname", "-I"]:
            return owner.NODE
        if command == ["ps", "-eo", "args="]:
            return ""
        raise AssertionError(command)

    def spawn_boundary(command, **kwargs):
        if failure:
            raise OSError("synthetic controller spawn failure")
        process = {
            "pid": FakeController.pid,
            "parent": os.getpid(),
            "status": "S",
            "group": FakeController.pid,
            "session": FakeController.pid,
            "start_ticks": "920201",
            "command": command,
        }
        write_process(proc, process, helpers, kwargs["env"])
        return FakeController()

    with archived_controller(helpers) as archived, monkeypatch.context() as boundary:
        isolated_owner_boundary(archived, boundary, keeper, binding_hash, lock_path)
        boundary.setattr(archived.owner.handoff, "PROC_ROOT", proc)
        boundary.setattr(acknowledgment.handoff, "PROC_ROOT", proc)
        boundary.setattr(archived.owner.handoff, "observed", acknowledgment.handoff.observed)
        boundary.setattr(subprocess, "check_output", host_boundary)
        boundary.setattr(subprocess, "Popen", spawn_boundary)
        if failure:
            with pytest.raises(OSError, match="controller spawn"):
                archived.start(root)
            intent = archived.read(helpers.parent / "controller.json")
            assert intent["status"] == "needs_attention"
            assert "pid" not in intent
            assert intent["log_sha256"] == sha256(helpers.parent / "controller.log")
        else:
            intent = archived.start(root)
            assert intent["status"] == "acknowledged"
            assert intent["pid"] == FakeController.pid
            assert intent["start_ticks"] == "920201"
            assert intent["generation"]["command"] == owner.controller_command(root)
        with pytest.raises(FileExistsError):
            archived.start(root)
        assert archived.read(helpers.parent / "controller.json") == intent
    assert not state_path.exists() and not work.exists()


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "false_metrics",
        "replaced_output",
        "changed_cross_process",
        "changed_stage",
        "duplicate_worker",
        "wrong_parent",
        "live_worker",
        "changed_keeper",
        "false_comparison",
        "unfinished_keeper",
    ),
)
def test_archived_comparison_reconstructs_retained_outputs_and_rejects_or_exposes_changes(
    terminal_fixture, monkeypatch, tmp_path, fault
):
    root, cpu_receipt, keeper, binding_hash = terminal_fixture
    lock_path = tmp_path / "manager.lock"
    with archived_controller(cpu_receipt.parent / "tracking") as archived, monkeypatch.context() as boundary:
        isolated_owner_boundary(archived, boundary, keeper, binding_hash, lock_path)
        archived.freeze(root, cpu_receipt)
    binding, helpers, state_path, work = owner.locations(root)
    work.mkdir(parents=True)
    saved, _ = saved_and_retained(0)
    provenance = {
        name: {
            "config": {"runtime": {"deterministic": True}},
            "trace_pair": {
                mode: {
                    "input_layouts": {
                        "C": {"shape": [2, 3, 3], "dtype": "torch.float32", "stride": [9, 3, 1], "storage_offset": 0}
                    }
                }
                for mode in ("continuous", "resumed")
            },
        }
        for name in worker.GROUPS
    }
    executions = []
    with archived_controller(helpers) as archived, monkeypatch.context() as boundary:
        isolated_owner_boundary(archived, boundary, keeper, binding_hash, lock_path)
        # This test owns persisted worker reports; historical checkpoint loading is the external I/O boundary.
        boundary.setattr(archived.replay, "load_saved_roots", lambda stack, name: (saved[name], provenance[name]))
        frozen = archived.owner.require_frozen(root, sha256(binding))
        for index in range(2):
            _, retained = saved_and_retained(index)
            if index == 1 and fault == "changed_cross_process":
                target = next(item for item in retained if item["group"] == "f" and item["mode"] == "resumed")
                target["output"][0][0][-1, -1, -1] += 0.125
                target["stages"]["output"] = tracing.cpu_tree(target["output"])
            if index == 1 and fault == "changed_stage":
                for item in retained:
                    if item["protocol"] == "trace":
                        item["stages"]["records"][0]["values"]["B"][-1, -1, -1] += 0.125
            log, output_path, report_path = (work / f"process-{index}{suffix}" for suffix in (".log", ".pt", ".json"))
            log.write_text("synthetic root-stage worker\n")
            torch.save(retained, output_path)
            execution = {
                "index": index,
                "command": archived.owner.worker_command(root, sha256(binding), index),
                "controller_pid": 920301,
                "controller_start_ticks": "synthetic-controller",
                "pid": 920302 + index,
                "start_ticks": f"synthetic-worker-{index}",
                "marker": f"synthetic-worker-marker-{index}",
                "spawn_status": "acknowledged",
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "log": str(log),
                "log_sha256": sha256(log),
                "report": str(report_path),
            }
            execution["generation"] = {
                "pid": execution["pid"],
                "parent": execution["controller_pid"],
                "status": "R",
                "group": execution["pid"],
                "session": execution["pid"],
                "start_ticks": execution["start_ticks"],
                "command": execution["command"],
            }
            if fault == "wrong_parent" and index == 1:
                execution["generation"]["parent"] += 1
            report = {
                "status": archived.worker.STATUS,
                "worker_index": index,
                "binding_sha256": sha256(binding),
                "owner": {
                    "binding": frozen,
                    "device": "cuda:0",
                    "physical_uuid": keeper["devices"][0][1],
                    "controller_pid": execution["controller_pid"],
                    "controller_start_ticks": execution["controller_start_ticks"],
                    "worker_pid": execution["pid"],
                    "worker_start_ticks": execution["start_ticks"],
                },
                "outputs": str(output_path),
                "outputs_sha256": sha256(output_path),
                "report_path": str(report_path),
                "environment": frozen["worker_environment"],
                "provenance": provenance,
                "runtime": provenance["d"]["config"]["runtime"],
                "tolerance": {"rtol": 1e-5, "atol": 1e-6},
                "helper_sources_sha256": frozen["helpers_sha256"],
                "physical_uuid": keeper["devices"][0][1],
                "actual_uuid": keeper["devices"][0][1].removeprefix("GPU-"),
                "compilation_preserved": True,
                "graphs_preserved": True,
                "lifecycle": [{**item, "graph_active": item["phase"] != "eager"} for item in worker.coordinates(index)],
                "source_layouts": {
                    f"{protocol}:{group}:{mode}": provenance[group]["trace_pair"][mode]["input_layouts"]
                    for protocol in worker.PROTOCOLS
                    for group in worker.GROUPS
                    for mode in ("continuous", "resumed")
                },
                "analysis": archived.worker.summarize(retained, saved, index),
            }
            if fault == "false_metrics" and index == 1:
                report["analysis"]["trace_transparency_exact"] = False
            archived.write(report_path, report)
            execution["report_sha256"] = sha256(report_path)
            executions.append(execution)
        if fault == "replaced_output":
            torch.save(retained[:-1], work / "process-1.pt")
        if fault == "duplicate_worker":
            executions[1]["pid"] = executions[0]["pid"]
            executions[1]["start_ticks"] = executions[0]["start_ticks"]
            with pytest.raises(ValueError, match="distinct acknowledged"):
                archived.compare_results(root, executions)
            return
        if fault == "wrong_parent":
            with pytest.raises(ValueError, match="acknowledged origin"):
                archived.compare_results(root, executions)
            return
        if fault in ("false_metrics", "replaced_output"):
            with pytest.raises(ValueError, match=r"retained-output reconstruction|digest differs"):
                archived.compare_results(root, executions)
            assert not (work / "comparison.json").exists()
            return
        result = archived.compare_results(root, executions)
        intent = {
            "status": "acknowledged",
            "binding_sha256": sha256(binding),
            "pid": executions[0]["controller_pid"],
            "start_ticks": executions[0]["controller_start_ticks"],
            "command": archived.owner.controller_command(root),
            "marker": "synthetic-controller-marker",
        }
        intent["generation"] = {
            "pid": intent["pid"],
            "parent": 1,
            "status": "S",
            "group": intent["pid"],
            "session": intent["pid"],
            "start_ticks": intent["start_ticks"],
            "command": intent["command"],
        }
        archived.write(helpers.parent / "controller.json", intent)
        (helpers.parent / "controller.log").write_text("synthetic root-stage controller\n")
        archived.write(work / "comparison.json", result)
        if fault == "false_comparison":
            result["all_cross_process_outputs_exact"] = False
            archived.write(work / "comparison.json", result)
        state = {
            "status": "diagnostic_complete",
            "binding_sha256": sha256(binding),
            "controller_pid": intent["pid"],
            "controller_start_ticks": intent["start_ticks"],
            "predecessor": frozen["predecessor"],
            "executions": executions,
            "active_worker": executions[-1],
            "keeper_release_complete": True,
            "keeper_restore": 0 if fault != "unfinished_keeper" else "not restored",
            "restored_keeper": keeper,
            "comparison_sha256": sha256(work / "comparison.json"),
            "intermediates_interpretable": result["intermediates_interpretable"],
        }
        archived.write(state_path, state)
        boundary.setenv("CUDA_VISIBLE_DEVICES", "")
        boundary.setattr(subprocess, "check_output", lambda command, **kwargs: owner.NODE)
        if fault == "changed_keeper":
            changed = copy.deepcopy(keeper)
            changed["generation"][str(keeper["pid"])]["start_ticks"] = "different-live-generation"
            boundary.setattr(archived.owner.handoff, "keeper_observation", lambda: changed)
        if fault == "live_worker":
            boundary.setattr(
                archived.owner.handoff, "observed", lambda pid: {"pid": pid} if pid == executions[0]["pid"] else None
            )
        observation = root / "run-preparation/test-root-stage-terminal-observation.json"
        original_hashes = {path: sha256(path) for path in (binding, state_path, work / "comparison.json")}
        if fault in ("live_worker", "changed_keeper", "false_comparison", "unfinished_keeper"):
            exception = ValueError if fault == "false_comparison" else RuntimeError
            with pytest.raises(exception):
                archived.terminal(root, observation)
            assert not observation.exists()
            assert original_hashes == {path: sha256(path) for path in original_hashes}
            return
        terminal = archived.terminal(root, observation)
        assert archived.read(observation)["comparison"] == {key: value for key, value in result.items() if key != "time"}
        assert terminal["comparison_time"] == result["time"]
        assert terminal["worker_pids_absent"] == [execution["pid"] for execution in executions]
        assert terminal["intermediates_interpretable"] is (fault == "none")
        assert terminal["keeper"]["pid"] == keeper["pid"]
        assert original_hashes == {path: sha256(path) for path in original_hashes}
        with pytest.raises(FileExistsError):
            archived.terminal(root, observation)
    assert result["all_cross_process_outputs_exact"] is (fault != "changed_cross_process")
    assert result["all_cross_process_stages_exact"] is (fault not in ("changed_cross_process", "changed_stage"))
    assert result["intermediates_interpretable"] is (fault == "none")
    if fault == "changed_cross_process":
        changed = [row for row in result["cross_process"] if not row["output"]["exact"]]
        assert len(changed) == 1
        assert changed[0]["output"]["changed_tensors"][0]["mismatched_elements"] == 1
    if fault == "changed_stage":
        assert all(report["analysis"]["intermediates_interpretable"] for report in result["reports"])
        changed = [row for row in result["cross_process"] if row["stages"] and not row["stages"]["stages"][0]["exact"]]
        assert len(changed) == 12
    assert len(result["cross_process"]) == 24
    assert state_path.exists()
