# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify distinct executor ownership and cleanup without transferring GPU owners."""

import os
import subprocess
import tarfile
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import root_preparation_controller as controller
from fused_attribution_validation_test_support import keeper, record
from selected_lr_release_owner import sha256


@pytest.mark.parametrize("fault", ("none", "keeper", "original_child", "marker", "source"))
def test_live_predecessor_rejects_replacement_generations_and_live_replay_descendants(tmp_path, monkeypatch, fault):
    root = tmp_path / "root"
    restored = keeper()
    replacement = keeper()
    replacement["generation"]["100"]["start_ticks"] = "different-generation"
    expected = {"restored_keeper": restored, "controller_pid": 400, "child_pids": [410, 420]}
    record(root / "controllers/saved-ns-replay-20261118/controller.json", {"marker": "controller"})
    record(
        root / "run-preparation/saved-ns-replay-state-20261118.json",
        {"executions": [{"marker": "first"}, {"marker": "second"}]},
    )
    monkeypatch.setattr(controller, "parent", lambda root: (None, {} if fault == "source" else expected))
    monkeypatch.setattr(controller.handoff, "keeper_observation", lambda: replacement if fault == "keeper" else restored)
    monkeypatch.setattr(
        controller.handoff, "observed", lambda pid: {"pid": pid} if fault == "original_child" and pid == 410 else None
    )
    monkeypatch.setattr(
        controller.handoff,
        "marker_processes",
        lambda name, marker: {999: {"pid": 999}} if fault == "marker" and marker == "first" else {},
    )
    if fault == "none":
        result = controller.probe(root, expected)
        assert (
            result["keeper"] == restored and result["status"] == "saved_ns_exact_restored_keeper_live_handoff_verified"
        )
    else:
        with pytest.raises(RuntimeError):
            controller.probe(root, expected)
    assert not controller.locations(root)[0].exists()
    assert not controller.locations(root)[2].exists()


@pytest.mark.parametrize("outcome", ("zero", "failure", "timeout", "acknowledgment", "cleanup"))
def test_bounded_validator_records_cleanup_on_success_failure_and_timeout(tmp_path, monkeypatch, outcome):
    root = tmp_path / "root"
    candidate = tmp_path / "candidate"
    _, helpers, state_path, _, _ = controller.locations(root)
    helpers.mkdir(parents=True)
    state_path.parent.mkdir(parents=True)
    state = {}
    result = 0 if outcome == "zero" else 1
    child = SimpleNamespace(pid=900, returncode=None)

    def wait(timeout):
        if outcome == "timeout":
            raise subprocess.TimeoutExpired("fixture", timeout)
        child.returncode = result
        return result

    child.wait = wait
    monkeypatch.setattr(controller.subprocess, "Popen", lambda *args, **kwargs: child)

    def acknowledge(*args):
        if outcome == "acknowledgment":
            raise RuntimeError("fixture acknowledgment")
        return {"start_ticks": "123"}

    monkeypatch.setattr(controller, "acknowledged_process", acknowledge)

    def cleanup(child, marker, generations, marker_name):
        generations["bound"] = {"pid": child.pid}
        if outcome == "cleanup":
            raise RuntimeError("fixture cleanup")
        child.returncode = -15 if outcome == "timeout" else result

    monkeypatch.setattr(controller, "finish_marked_child", cleanup)
    if outcome == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            controller.bounded_validator(root, {"candidate": str(candidate)}, state)
    elif outcome in ("acknowledgment", "cleanup"):
        with pytest.raises(RuntimeError, match=f"fixture {outcome}"):
            controller.bounded_validator(root, {"candidate": str(candidate)}, state)
    else:
        assert controller.bounded_validator(root, {"candidate": str(candidate)}, state) == result
    saved = controller.read(state_path)
    if outcome == "cleanup":
        assert saved["cleanup_error"] == "RuntimeError: fixture cleanup"
        assert "marked_descendants_absent" not in saved
    else:
        assert saved["marked_descendants_absent"] is True
        assert saved["cleanup_exit_code"] == (-15 if outcome == "timeout" else result)
    assert saved["child_generation"] == {"bound": {"pid": 900}}
    assert saved["log_sha256"] == sha256(helpers.parent / "validator.log")


def test_unissued_real_controller_check_denies_without_state_or_gpu_mutation(tmp_path):
    root = tmp_path / "unused"
    root.mkdir()
    result = subprocess.run(
        [controller.PYTHON, "-P", str(Path(controller.__file__)), "--root", str(root), "--phase", "check"],
        env=os.environ
        | {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(Path(controller.__file__).parent),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "root-preparation-binding-20261118.json" in result.stderr
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("fault", ("none", "occupied", "predecessor", "acknowledgment", "cleanup"))
def test_start_requires_fresh_handoff_and_cleans_unacknowledged_controller(tmp_path, monkeypatch, fault):
    root = tmp_path / "root"
    path, helpers, state, work, receipt = controller.locations(root)
    helpers.mkdir(parents=True)
    record(path, {"fixture": "binding"})
    frozen = {"allocation_predecessor": {"fixture": "accepted"}}
    monkeypatch.setattr(controller, "binding", lambda root: frozen)
    monkeypatch.setattr(controller, "local_node", lambda: controller.NODE)
    monkeypatch.setattr(controller, "keeper_guard", lambda: nullcontext())
    for name, value in controller.ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    events = []

    def idle():
        events.append("idle")
        if fault == "occupied":
            raise RuntimeError("fixture occupied")

    def handoff(root, expected):
        events.append("handoff")
        if fault == "predecessor":
            raise RuntimeError("fixture predecessor")
        return {}

    child = SimpleNamespace(pid=900, returncode=None)

    def spawn(*args, **kwargs):
        events.append("spawn")
        return child

    def acknowledge(*args):
        events.append("acknowledgment")
        if fault in ("acknowledgment", "cleanup"):
            raise RuntimeError("fixture acknowledgment")
        return {"start_ticks": "123"}

    def cleanup(child, marker, generations, marker_name):
        events.append("cleanup")
        generations["bound"] = {"pid": child.pid}
        if fault == "cleanup":
            raise RuntimeError("fixture cleanup")
        child.returncode = -15

    monkeypatch.setattr(controller, "idle_training", idle)
    monkeypatch.setattr(controller, "probe", handoff)
    monkeypatch.setattr(controller.subprocess, "Popen", spawn)
    monkeypatch.setattr(controller, "acknowledged_process", acknowledge)
    monkeypatch.setattr(controller, "finish_marked_child", cleanup)
    if fault == "none":
        result = controller.start(root)
        assert result["status"] == "acknowledged" and result["pid"] == 900
    else:
        with pytest.raises(RuntimeError, match=f"fixture {fault}"):
            controller.start(root)
    intent = helpers.parent / "controller.json"
    if fault in ("occupied", "predecessor"):
        assert "spawn" not in events and not intent.exists()
    elif fault in ("acknowledgment", "cleanup"):
        saved = controller.read(intent)
        assert saved["status"] == "needs_attention" and "cleanup" in events
        if fault == "cleanup":
            assert "cleanup_error" in saved
        else:
            assert saved["marked_descendants_absent"] is True and saved["cleanup_exit_code"] == -15
    assert not state.exists() and not work.exists() and not receipt.exists()


@pytest.mark.parametrize("fault", ("none", "validator", "restore", "live_child"))
def test_validator_restores_after_failure_but_refuses_live_descendants(tmp_path, monkeypatch, fault):
    root = tmp_path / "root"
    path, helpers, state_path, _, receipt = controller.locations(root)
    helpers.mkdir(parents=True)
    record(path, {"fixture": "binding"})
    restored = keeper()
    frozen = {"candidate": str(tmp_path / "candidate"), "allocation_predecessor": {}}
    own = {"pid": 900, "start_ticks": "123"}
    marker = "controller"
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "binding_sha256": sha256(path),
            "command": controller.controller_command(root),
            "marker": marker,
            "allocation_predecessor": {},
        },
    )
    monkeypatch.setenv(controller.CONTROLLER_MARKER, marker)
    monkeypatch.setattr(controller, "binding", lambda root: frozen)
    monkeypatch.setattr(controller, "local_node", lambda: controller.NODE)
    monkeypatch.setattr(controller, "process_context", lambda *args: own)
    monkeypatch.setattr(controller, "require_acknowledged_controller", lambda *args: None)
    monkeypatch.setattr(controller, "node_guard", lambda *args: nullcontext([]))
    monkeypatch.setattr(controller, "keeper_guard", lambda: nullcontext("manager-lock"))
    monkeypatch.setattr(controller, "idle_training", lambda: None)
    monkeypatch.setattr(controller, "probe", lambda *args: {"keeper": restored})
    monkeypatch.setattr(controller.subprocess, "run", lambda *args, **kwargs: None)
    events = []

    def release(accepted, state, path, manager):
        events.append("release")
        state.update(keeper_release_attempted=True, keeper_release_complete=True)

    def validate(root, frozen, state):
        events.append("validate")
        state.update(child_marker="validator", marked_descendants_absent=fault != "live_child")
        record(receipt, {"fixture": "result"})
        return 1 if fault == "validator" else 0

    def restore(manager):
        events.append("restore")
        if fault == "restore":
            raise RuntimeError("fixture restore")

    monkeypatch.setattr(controller, "release_accepted_keeper", release)
    monkeypatch.setattr(controller, "bounded_validator", validate)
    monkeypatch.setattr(
        controller.validation, "reconstruct", lambda *args: {"analysis": {"diagnostic_intervention_accepted": False}}
    )
    monkeypatch.setattr(controller, "restore_accepted_keeper", restore)
    monkeypatch.setattr(controller.handoff, "keeper_observation", lambda: restored)
    monkeypatch.setattr(controller, "require_keeper_shape", lambda value: None)
    monkeypatch.setattr(controller.handoff, "marker_processes", lambda *args: {999: {}} if fault == "live_child" else {})
    if fault == "none":
        result = controller.validate(root)
        assert result["status"] == "diagnostic_complete" and result["intervention_accepted"] is False
    else:
        with pytest.raises(RuntimeError):
            controller.validate(root)
    state = controller.read(state_path)
    if fault == "live_child":
        assert "restore" not in events and state["keeper_restore"].startswith("refused")
    elif fault != "restore":
        assert state["keeper_restore"] == 0 and events[-1] == "restore"
    else:
        assert state["status"] == "needs_attention" and "fixture restore" in state["error"]


def test_executor_closure_includes_dynamic_owner_fixtures_and_preserves_inherited_helpers():
    root = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    cpu = controller.read(root / "run-preparation/root-preparation-cpu/testroot-20261118T1233/receipt.json")
    sources = controller.helper_sources(root, Path(controller.__file__).parent, cpu)
    assert set(controller.attribution_helper_names(Path(controller.__file__).parent)).issubset(sources)
    assert all(sha256(sources[name]) == value for name, value in cpu["helpers_sha256"].items())
    for name in ("fused_attribution_validation_owner.py", "test_fused_attribution_validation_owner.py"):
        assert name in sources


@pytest.mark.parametrize("fault", ("none", "executor_drift", "parent_drift", "reuse"))
def test_freeze_retains_exact_inventory_and_denies_unaccepted_or_existing_identity(tmp_path, monkeypatch, fault):
    root, candidate, source = tmp_path / "root", tmp_path / "candidate", tmp_path / "source"
    source.mkdir()
    record(root / "data/experiment_receipt.json", {"fixture": "data"})
    (root / "run-preparation").mkdir()
    record(candidate / "experiment/configs/fixture.json", {"fixture": "schedule"})
    pinned = candidate / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    (pinned / "fixture.txt").write_text("pinned fixture\n")
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    helper = source / "fixture.py"
    helper.write_text("# unchanged helper fixture\n")
    files = controller.candidate_files(candidate)
    predecessor = {"fixture": "accepted terminal"}
    prepared = {
        "parent_metadata_sha256": {},
        "allocation_predecessor": predecessor,
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
    }
    record(candidate / controller.PREPARATION, prepared)
    candidate_receipt, executor_receipt = tmp_path / "candidate-cpu.json", tmp_path / "executor-cpu.json"
    cpu = {"candidate": str(candidate), "files_sha256": files}
    executor = {"helpers_sha256": {helper.name: sha256(helper)}}
    record(candidate_receipt, cpu)
    record(executor_receipt, executor)
    monkeypatch.setattr(controller, "require_cpu", lambda *args: cpu)
    monkeypatch.setattr(controller, "require_controller_cpu", lambda *args: executor)
    monkeypatch.setattr(controller, "helper_sources", lambda *args: {helper.name: helper})
    monkeypatch.setattr(controller.validation, "source", lambda *args: prepared)
    monkeypatch.setattr(controller, "parent", lambda root: (None, {} if fault == "parent_drift" else predecessor))
    monkeypatch.setattr(controller, "plan", lambda candidate, root: {"commands": [["fixture", "24"]]})
    path, helpers, state, work, receipt = controller.locations(root)
    if fault == "executor_drift":
        executor["helpers_sha256"][helper.name] = "0" * 64
    elif fault == "reuse":
        record(path, {"fixture": "original immutable intent"})
    if fault != "none":
        with pytest.raises((ValueError, FileExistsError)):
            controller.freeze(root, candidate_receipt, executor_receipt)
        assert not helpers.exists() and not state.exists() and not work.exists() and not receipt.exists()
        return
    result = controller.freeze(root, candidate_receipt, executor_receipt)
    assert result == controller.read(path)
    inventory = controller.inventory(result)
    controller.require_archive(Path(result["archive"]), inventory, alias=True)
    with tarfile.open(result["archive"]) as archive:
        assert {member.name for member in archive} == set(inventory) | {"artifacts/external-sources"}
    assert controller.candidate_files(candidate) == files
    assert not state.exists() and not work.exists() and not receipt.exists()
    with pytest.raises(FileExistsError):
        controller.freeze(root, candidate_receipt, executor_receipt)
    (helpers / helper.name).write_text("# changed frozen helper\n")
    with pytest.raises(ValueError):
        controller.require_helper_bytes(helpers, result["helpers_sha256"])


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "bool_exit",
        "zero_xml",
        "missing_module",
        "xml_child_failure",
        "missing_fixture",
        "archive_member",
        "log_substitution",
        "candidate_helper",
    ),
)
def test_cpu_receipt_authenticates_complete_xml_fixtures_archive_and_inherited_bytes(tmp_path, monkeypatch, fault):
    root, candidate, out = tmp_path / "root", tmp_path / "candidate", tmp_path / "cpu"
    helpers = out / "tracking"
    helpers.mkdir(parents=True)
    fixture = candidate / "experiment/configs/outer_leftdiag_fused_full_lr_lower_fixture.json"
    record(fixture, {"fixture": "accepted schedule"})
    target = out / fixture.relative_to(candidate)
    target.parent.mkdir(parents=True)
    target.write_bytes(fixture.read_bytes())
    helper = helpers / "fixture.py"
    helper.write_text("# inherited candidate helper\n")
    hashes = {helper.name: sha256(helper)}
    cpu = {
        "candidate": str(candidate),
        "files_sha256": {str(fixture.relative_to(candidate)): sha256(fixture)},
        "helpers_sha256": hashes,
    }
    candidate_receipt = tmp_path / "candidate-cpu.json"
    record(candidate_receipt, cpu)
    monkeypatch.setattr(controller, "require_cpu", lambda *args: cpu)
    monkeypatch.setattr(controller, "helper_sources", lambda *args: {helper.name: helper})
    tests = out / "tests.xml"
    cases = "".join(f'<testcase classname="tracking.{Path(name).stem}" name="fixture"/>' for name in controller.TESTS)
    tests.write_text(
        f'<testsuites><testsuite tests="4" failures="0" errors="0" skipped="0">{cases}</testsuite></testsuites>'
    )
    log = out / "controller.log"
    log.write_text("complete fixture suite\n")
    archive = out / "source.tar"
    inventory = {f"tracking/{helper.name}": sha256(helper), str(fixture.relative_to(candidate)): sha256(fixture)}
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            bundle.add(out / name, arcname=name, recursive=False)
    receipt = {
        "status": controller.CPU_STATUS,
        "root": str(root),
        "candidate_cpu_receipt": str(candidate_receipt),
        "candidate_cpu_receipt_sha256": sha256(candidate_receipt),
        "helpers": str(helpers),
        "helpers_sha256": deepcopy(hashes),
        "fixtures_sha256": {str(fixture.relative_to(candidate)): sha256(fixture)},
        "command": controller.cpu_command(out),
        "cwd": str(out),
        "exit_code": 0,
        "cuda_visible_devices": "",
        **{name: str(path) for name, path in (("log", log), ("archive", archive), ("tests", tests))},
        **{f"{name}_sha256": sha256(path) for name, path in (("log", log), ("archive", archive), ("tests", tests))},
    }
    if fault == "bool_exit":
        receipt["exit_code"] = False
    elif fault == "zero_xml":
        tests.write_text('<testsuites><testsuite tests="0" failures="0" errors="0" skipped="0"/></testsuites>')
        receipt["tests_sha256"] = sha256(tests)
    elif fault == "missing_module":
        tests.write_text(tests.read_text().replace("test_root_preparation_controller", "wrong_module"))
        receipt["tests_sha256"] = sha256(tests)
    elif fault == "xml_child_failure":
        tests.write_text(tests.read_text().replace('name="fixture"/>', 'name="fixture"><failure/></testcase>', 1))
        receipt["tests_sha256"] = sha256(tests)
    elif fault == "missing_fixture":
        receipt["fixtures_sha256"] = {}
    elif fault == "archive_member":
        archive.unlink()
        with tarfile.open(archive, "x") as bundle:
            bundle.add(helper, arcname=f"tracking/{helper.name}", recursive=False)
        receipt["archive_sha256"] = sha256(archive)
    elif fault == "log_substitution":
        alternate = out / "alternate.log"
        alternate.write_bytes(log.read_bytes())
        receipt["log"] = str(alternate)
    elif fault == "candidate_helper":
        cpu["helpers_sha256"] = {helper.name: "0" * 64}
    path = out / "receipt.json"
    record(path, receipt)
    before = sha256(path)
    if fault == "none":
        result = controller.require_controller_cpu(root, path, candidate_receipt)
        assert result == controller.read(path)
    else:
        with pytest.raises(ValueError):
            controller.require_controller_cpu(root, path, candidate_receipt)
    assert sha256(path) == before
