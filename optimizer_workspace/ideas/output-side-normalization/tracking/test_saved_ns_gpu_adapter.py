# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind saved-input replay reports to acknowledged executions without GPU work."""

import copy
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pytest
import saved_ns_gpu_adapter as adapter
import saved_ns_gpu_owner as owner
import torch
from fused_continuation_test_support import write_process
from saved_ns_test_support import saved_evidence


@contextmanager
def archived_adapter(helpers):
    names = {path.stem for path in helpers.glob("*.py")}
    loaded = {name: sys.modules[name] for name in names if name in sys.modules}
    previous = list(sys.path)
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(helpers))
        yield importlib.import_module("saved_ns_gpu_adapter")
    finally:
        sys.path[:] = previous
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(loaded)


@pytest.fixture
def terminal_archive(tmp_path):
    root = tmp_path / "root"
    binding, helpers, state_path, _, receipt_path = owner.previous_locations(root)
    helpers.mkdir(parents=True)
    sources = owner.helper_sources(adapter.replay.ROOT)
    for name, source in sources.items():
        shutil.copy2(source, helpers / name)
    saved_evidence(root)
    candidate = root / "candidate"
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    preparation = candidate / "fused-continuation-preparation.json"
    adapter.write(preparation, {"parent_metadata_sha256": {}})
    for name in ("candidate-cpu-receipt.json", "controller-cpu-receipt.json"):
        adapter.write(candidate / name, {"synthetic_external_cpu_acceptance": True})
    predecessor_names = set(sources) - {"saved_ns_gpu_owner.py"}
    source_hashes = {name: adapter.sha256(helpers / name) for name in predecessor_names}
    inventory = owner.candidate_files(candidate) | {
        name: adapter.sha256(candidate / name)
        for name in (
            "fused-continuation-preparation.json",
            "candidate-cpu-receipt.json",
            "controller-cpu-receipt.json",
        )
    }
    inventory.update({f"tracking/{name}": digest for name, digest in source_hashes.items()})
    archive = root / "run-preparation/terminal-source.tar"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "x") as bundle:
        for name in inventory:
            path = helpers / name.removeprefix("tracking/") if name.startswith("tracking/") else candidate / name
            bundle.add(path, arcname=name, recursive=False)
        bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
    frozen = {
        "root": str(root),
        "helpers": str(helpers),
        "candidate": str(candidate),
        "files_sha256": owner.candidate_files(candidate),
        "preparation_sha256": adapter.sha256(preparation),
        "cpu_receipt_sha256": adapter.sha256(candidate / "candidate-cpu-receipt.json"),
        "controller_cpu_receipt_sha256": adapter.sha256(candidate / "controller-cpu-receipt.json"),
        "helpers_sha256": source_hashes,
        "archive": str(archive),
        "archive_sha256": adapter.sha256(archive),
        "commands": adapter.read(receipt_path)["commands"],
    }
    adapter.write(binding, frozen)
    predecessor_hash = adapter.sha256(binding)
    controller = {"controller_pid": 910001, "controller_start_ticks": "100"}
    adapter.write(
        receipt_path,
        adapter.read(receipt_path)
        | {
            "status": "completed_fused_continuation_diagnostic_not_production_gate",
            "owner": controller,
            "execution_binding_sha256": predecessor_hash,
        },
    )
    # Fixture identities replace only fixed evidence digests in the isolated helper tree.
    owner_path = helpers / "saved_ns_gpu_owner.py"
    text = owner_path.read_text()
    text = text.replace(owner.PREVIOUS_BINDING_SHA256, predecessor_hash)
    owner_path.write_text(text.replace(owner.RECEIPT_SHA256, adapter.sha256(receipt_path)))
    device = ["0", "GPU-12345678-1234-1234-1234-123456789abc", "NVIDIA A100"]
    keeper_process = {
        "pid": 910003,
        "start_ticks": "300",
        "parent": 1,
        "group": 910003,
        "session": 910003,
        "status": "S",
        "command": ["synthetic-keeper"],
    }
    restored = {"pid": 910003, "generation": {"910003": keeper_process}, "devices": [device]}
    log = helpers.parent / "validator.log"
    log.write_text("synthetic completed validator\n")
    (helpers.parent / "controller.log").write_text("synthetic completed controller\n")
    state = controller | {
        "status": "diagnostic_complete",
        "binding_sha256": predecessor_hash,
        "validation_sha256": adapter.sha256(receipt_path),
        "candidate": str(candidate),
        "child_pid": 910002,
        "child_start_ticks": "200",
        "child_marker": "synthetic-prior-child",
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "marked_descendants_absent": True,
        "keeper_release_complete": True,
        "keeper_restore": 0,
        "restored_keeper": restored,
        "log": str(log),
        "log_sha256": adapter.sha256(log),
    }
    adapter.write(state_path, state)
    adapter.write(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "pid": controller["controller_pid"],
            "start_ticks": controller["controller_start_ticks"],
            "marker": "synthetic-prior-controller",
            "binding_sha256": predecessor_hash,
        },
    )
    out = tmp_path / "cpu"
    copied = out / "tracking"
    shutil.copytree(helpers, copied)
    hashes = {name: adapter.sha256(copied / name) for name in sources}
    cpu_log = out / "cpu.log"
    cpu_log.write_text("synthetic external CPU suite acceptance; no CUDA execution\n")
    cpu_archive = out / "source.tar"
    with tarfile.open(cpu_archive, "x") as bundle:
        for name in hashes:
            bundle.add(copied / name, arcname=f"tracking/{name}", recursive=False)
    cpu_receipt = out / "receipt.json"
    adapter.write(
        cpu_receipt,
        {
            "status": owner.CPU_STATUS,
            "root": str(root),
            "helpers": str(copied),
            "helpers_sha256": hashes,
            "command": owner.cpu_command(copied),
            "exit_code": 0,
            "cuda_visible_devices": "",
            "fixtures_sha256": {},
            "log": str(cpu_log),
            "log_sha256": adapter.sha256(cpu_log),
            "archive": str(cpu_archive),
            "archive_sha256": adapter.sha256(cpu_archive),
        },
    )
    proc = tmp_path / "proc"
    proc.mkdir()
    write_process(proc, keeper_process, root, {owner.PREVIOUS_CONTROLLER_MARKER: "synthetic-prior-controller"})
    return root, copied, cpu_receipt, proc, restored


def isolated_keeper_boundary(archived, monkeypatch, tmp_path, proc, keeper):
    failure = sys.modules["selected_lr_failure_handoff"]
    manager = Path(archived.__file__).with_name("recovery_keeper_manager.sh")
    monkeypatch.setattr(failure, "MANAGER_LOCK", tmp_path / "manager.lock")
    monkeypatch.setattr(failure, "KEEPER", str(manager))
    monkeypatch.setattr(archived.owner.handoff, "PROC_ROOT", proc)
    monkeypatch.setattr(archived.owner.handoff, "keeper_observation", lambda: copy.deepcopy(keeper))


@pytest.mark.parametrize("fault", ("none", "incomplete", "source"))
def test_archived_freeze_checks_terminal_sources_and_fresh_checker_without_gpu_work(
    terminal_archive, monkeypatch, tmp_path, fault
):
    root, cpu_helpers, cpu_receipt, proc, keeper = terminal_archive
    if fault == "incomplete":
        state_path = owner.previous_locations(root)[2]
        adapter.write(state_path, adapter.read(state_path) | {"keeper_restore": None})
    elif fault == "source":
        (cpu_helpers / "saved_ns_gpu_worker.py").write_text("# substituted untested worker\n")
    with archived_adapter(cpu_helpers) as archived, monkeypatch.context() as boundary:
        isolated_keeper_boundary(archived, boundary, tmp_path, proc, keeper)
        if fault != "none":
            with pytest.raises((RuntimeError, ValueError)):
                archived.freeze(root, cpu_receipt)
            assert not any(path.exists() for path in owner.locations(root))
            assert not (root / "run-preparation/source-saved-ns-replay-20261118.tar").exists()
            return
        frozen = archived.freeze(root, cpu_receipt)
        with pytest.raises(FileExistsError):
            archived.freeze(root, cpu_receipt)
    binding, helpers, state, work = owner.locations(root)
    assert frozen["helpers_sha256"] == adapter.read(cpu_receipt)["helpers_sha256"]
    assert frozen["predecessor"]["restored_keeper"] == keeper
    command = [sys.executable, "-P", str(helpers / "saved_ns_gpu_adapter.py"), "--root", str(root), "--phase", "check"]
    checked = subprocess.run(
        command,
        cwd=helpers,
        env=os.environ | owner.ENVIRONMENT | {"PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout)["status"] == owner.STATUS
    before = binding.read_bytes()
    (helpers / "saved_ns_gpu_worker.py").write_text("# altered issued worker\n")
    checked = subprocess.run(
        command,
        cwd=helpers,
        env=os.environ | owner.ENVIRONMENT | {"PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert checked.returncode != 0
    assert binding.read_bytes() == before and not state.exists() and not work.exists()


def test_archived_start_persists_exact_acknowledgment_and_never_retries_existing_intent(
    terminal_archive, monkeypatch, tmp_path
):
    root, cpu_helpers, cpu_receipt, proc, keeper = terminal_archive
    with archived_adapter(cpu_helpers) as archived, monkeypatch.context() as boundary:
        isolated_keeper_boundary(archived, boundary, tmp_path, proc, keeper)
        archived.freeze(root, cpu_receipt)
    _, helpers, state_path, work = owner.locations(root)
    real_spawn = subprocess.Popen

    def device_boundary(command, **kwargs):
        if command == ["hostname", "-I"]:
            return owner.NODE
        if command == ["ps", "-eo", "args="]:
            return ""
        raise AssertionError(f"Unexpected start boundary: {command}")

    def controller_boundary(command, **kwargs):
        if command != owner.controller_command(root):
            return real_spawn(command, **kwargs)
        process = {
            "pid": 910004,
            "parent": os.getpid(),
            "status": "S",
            "group": 910004,
            "session": 910004,
            "start_ticks": "400",
            "command": command,
        }
        write_process(proc, process, helpers, kwargs["env"])
        return FakeController(process["pid"])

    class FakeController:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

    with archived_adapter(helpers) as archived, monkeypatch.context() as boundary:
        isolated_keeper_boundary(archived, boundary, tmp_path, proc, keeper)
        boundary.setattr(subprocess, "check_output", device_boundary)
        boundary.setattr(subprocess, "Popen", controller_boundary)
        intent = archived.start(root)
        assert intent["status"] == "acknowledged"
        assert intent["pid"] == 910004 and intent["start_ticks"] == "400"
        assert intent["generation"]["command"] == owner.controller_command(root)
        assert archived.read(helpers.parent / "controller.json") == intent
        with pytest.raises(FileExistsError):
            archived.start(root)
        assert archived.read(helpers.parent / "controller.json") == intent
    assert not state_path.exists() and not work.exists()


def test_archived_execute_creates_diagnostic_parent_and_restores_after_failed_child_spawn(
    terminal_archive, monkeypatch, tmp_path
):
    root, cpu_helpers, cpu_receipt, proc, keeper = terminal_archive
    with archived_adapter(cpu_helpers) as archived, monkeypatch.context() as boundary:
        isolated_keeper_boundary(archived, boundary, tmp_path, proc, keeper)
        archived.freeze(root, cpu_receipt)
    binding, helpers, state_path, work = owner.locations(root)
    own = {
        "pid": os.getpid(),
        "parent": 1,
        "status": "R",
        "group": os.getpid(),
        "session": os.getpid(),
        "start_ticks": "cpu-controller",
        "command": owner.controller_command(root),
    }
    environment = owner.ENVIRONMENT | {owner.CONTROLLER_MARKER: "synthetic-replay-controller"}
    write_process(proc, own, helpers, environment)
    for name in ("fd", "fdinfo"):
        (proc / str(os.getpid()) / name).symlink_to(Path("/proc/self") / name, target_is_directory=True)
    adapter.write(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "pid": own["pid"],
            "start_ticks": own["start_ticks"],
            "generation": own,
            "marker": environment[owner.CONTROLLER_MARKER],
            "binding_sha256": adapter.sha256(binding),
            "command": own["command"],
        },
    )
    pointer = tmp_path / "keeper.pid"
    pointer.write_text(str(keeper["pid"]))
    occupied = [str(keeper["pid"])]
    handles = {}
    restored = copy.deepcopy(keeper)
    restored["pid"] = 910005
    restored["generation"] = {
        "910005": (
            keeper["generation"]["910003"] | {"pid": 910005, "start_ticks": "500", "group": 910005, "session": 910005}
        )
    }
    live_keeper = {"value": copy.deepcopy(keeper)}

    def device_boundary(command, **kwargs):
        if command == ["hostname", "-I"]:
            return owner.NODE
        if command == ["ps", "-eo", "args="]:
            return ""
        if "--query-compute-apps=pid" in command:
            return "\n".join(occupied)
        raise AssertionError(f"Unexpected execute boundary: {command}")

    def syscall(number, pid, flags):
        reader, writer = os.pipe()
        os.write(writer, b"exited")
        os.close(writer)
        handles[reader] = pid.value
        return reader

    def signal_boundary(descriptor, signum):
        assert signum == signal.SIGTERM
        pid = handles[descriptor]
        shutil.rmtree(proc / str(pid))
        occupied.remove(str(pid))

    def spawn_failure(command, **kwargs):
        assert command == owner.worker_command(root, adapter.sha256(binding), 0)
        raise OSError("synthetic child spawn failure")

    with archived_adapter(helpers) as archived, monkeypatch.context() as boundary:
        isolated_keeper_boundary(archived, boundary, tmp_path, proc, keeper)
        failure = sys.modules["selected_lr_failure_handoff"]
        boundary.setattr(archived.owner.handoff, "KEEPER_PID", pointer)
        boundary.setattr(failure, "KEEPER_PID", pointer)
        boundary.setattr(failure, "keeper_observation", lambda: copy.deepcopy(live_keeper["value"]))
        boundary.setattr(archived.owner.handoff, "keeper_observation", lambda: copy.deepcopy(live_keeper["value"]))
        boundary.setattr(failure.LIBC, "syscall", syscall)
        boundary.setattr(signal, "pidfd_send_signal", signal_boundary)
        boundary.setattr(subprocess, "check_output", device_boundary)
        boundary.setattr(subprocess, "Popen", spawn_failure)
        for name, value in environment.items():
            boundary.setenv(name, value)

        def restored_manager(command, **kwargs):
            assert command == ["bash", failure.KEEPER, "start"]
            assert kwargs["pass_fds"] == (int(kwargs["env"]["GPU_RESOURCE_MANAGER_LOCK_FD"]),)
            live_keeper["value"] = restored
            pointer.write_text(str(restored["pid"]))
            occupied.append(str(restored["pid"]))
            return subprocess.CompletedProcess(command, 0)

        boundary.setattr(subprocess, "run", restored_manager)
        assert not work.parent.exists()
        with pytest.raises(OSError, match="synthetic child spawn failure"):
            archived.execute(root)
    state = adapter.read(state_path)
    assert state["status"] == "needs_attention"
    assert state["keeper_release_attempted"] is state["keeper_release_complete"] is True
    assert state["keeper_restore"] == 0 and state["restored_keeper"] == restored
    assert state["executions"] == []
    assert work.is_dir() and not (work / "comparison.json").exists()
    assert pointer.read_text() == str(restored["pid"])


@pytest.mark.parametrize("fault", ("none", "false_metrics", "replaced_output", "changed_cross_process"))
def test_archived_comparison_reconstructs_retained_outputs_and_exposes_cross_process_errors(
    terminal_archive, monkeypatch, tmp_path, fault
):
    root, cpu_helpers, cpu_receipt, proc, keeper = terminal_archive
    with archived_adapter(cpu_helpers) as archived, monkeypatch.context() as boundary:
        isolated_keeper_boundary(archived, boundary, tmp_path, proc, keeper)
        archived.freeze(root, cpu_receipt)
    binding, helpers, _, work = owner.locations(root)
    work.mkdir(parents=True)
    with archived_adapter(helpers) as archived, monkeypatch.context() as boundary:
        boundary.setattr(archived.replay, "ROOT", root)
        boundary.setattr(archived.replay, "RECEIPT_SHA256", archived.owner.RECEIPT_SHA256)
        boundary.setattr(archived.replay, "BINDING_SHA256", archived.owner.PREVIOUS_BINDING_SHA256)
        prior = archived.read(owner.previous_locations(root)[0])
        muon = Path(prior["candidate"]) / "artifacts/pinned-sources-materialized/Toner-synthetic/src/toner/fast/muon.py"
        boundary.setattr(archived.replay, "MUON_SHA256", archived.sha256(muon))
        digest = archived.sha256(binding)
        frozen = archived.owner.require_frozen(root, digest)
        with ExitStack() as stack:
            groups, provenance = archived.replay.load_saved_group(stack)
        executions = []
        for index in range(2):
            result = archived.replay.replay_pair(groups, lambda source: source.clone())
            outputs = result.outputs
            if fault == "changed_cross_process" and index == 1:
                outputs[3][-1, -1, -1] += 0.125
                result.report["repeats"]["resumed"] = archived.replay.compare_outputs(outputs[3], outputs[1])
                result.report["cross_outputs"][1] = archived.replay.compare_outputs(outputs[3], outputs[2])
                for key, mode in (("own_saved_output", "resumed"), ("other_saved_output", "continuous")):
                    result.report["saved_comparisons"][3][key] = archived.replay.compare_outputs(
                        outputs[3], groups[mode]["output"]
                    )
            report, execution, _ = report_pair(root)
            report.update(result.report)
            execution.update(
                index=index,
                pid=123 + index,
                command=archived.owner.worker_command(root, digest, index),
                report=str(work / f"process-{index}.json"),
            )
            context, _, runtime, _ = report_context(root)
            report.update(
                status="saved_ns_cuda_replay_not_production_gate",
                worker_index=index,
                binding_sha256=digest,
                report_path=execution["report"],
                outputs=str(work / f"process-{index}.pt"),
                runtime=runtime,
                environment=frozen["worker_environment"],
                device=context["device"],
                numerical_settings=context["numerical_settings"],
                source_layouts=context["source_layouts"],
                worker_sources_sha256=archived.owner.worker_sources(frozen),
                provenance=provenance,
            )
            report["owner"].update(worker_pid=execution["pid"], binding=frozen)
            torch.save(outputs, Path(report["outputs"]))
            report["outputs_sha256"] = archived.sha256(Path(report["outputs"]))
            if fault == "false_metrics" and index == 1:
                report["repeats"]["continuous"]["changed_elements"] = 1
            archived.write(Path(execution["report"]), report)
            execution["report_sha256"] = archived.sha256(Path(execution["report"]))
            executions.append(execution)
        if fault == "replaced_output":
            torch.save(outputs[:3], work / "process-1.pt")
        if fault in ("false_metrics", "replaced_output"):
            with pytest.raises(ValueError, match=r"retained output reconstruction|digest differs"):
                archived.compare_results(root, executions)
            assert not (work / "comparison.json").exists()
            return
        comparison = archived.compare_results(root, executions)
    assert comparison["status"] == "saved_ns_cuda_replay_compared_not_production_gate"
    assert comparison["all_cross_process_exact"] is (fault == "none")
    assert all(row["exact"] for row in comparison["cross_process"][:3])
    final = comparison["cross_process"][3]
    if fault == "changed_cross_process":
        assert final["changed_elements"] == final["mismatched_elements"] == 1
        assert final["max_absolute_error"] == 0.125
        assert final["within_tolerance"] is False
    else:
        assert all(row["exact"] and row["finite"] for row in comparison["cross_process"])
        assert all(
            row["own_saved_output"]["exact"] for report in comparison["reports"] for row in report["saved_comparisons"]
        )
    assert not owner.locations(root)[2].exists()


def report_pair(root):
    digest = "b" * 64
    work = owner.locations(root)[3]
    execution = {
        "index": 0,
        "pid": 123,
        "start_ticks": "456",
        "controller_pid": 789,
        "controller_start_ticks": "101",
        "command": owner.worker_command(root, digest, 0),
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "marked_descendants_absent": True,
        "report": str(work / "process-0.json"),
    }
    report = {
        "status": "saved_ns_cuda_replay_not_production_gate",
        "worker_index": 0,
        "binding_sha256": digest,
        "order": list(adapter.replay.ORDER),
        "tolerance": dict(adapter.replay.MOMENT_TOLERANCE),
        "replay_mode": "tpu",
        "ns_steps": 10,
        "ns_eps": 1e-5,
        "owner": {
            "worker_pid": 123,
            "worker_start_ticks": "456",
            "controller_pid": 789,
            "controller_start_ticks": "101",
        },
        "report_path": str(work / "process-0.json"),
        "outputs": str(work / "process-0.pt"),
    }
    return report, execution, digest


@pytest.mark.parametrize(
    "fault",
    (
        ("binding_sha256", "a" * 64),
        ("order", ["resumed", "continuous", "resumed", "continuous"]),
        ("outputs", "/tmp/copied-process.pt"),
        ("report_path", "/tmp/copied-report.json"),
        ("worker_index", 1),
    ),
)
def test_copied_or_reordered_report_is_not_accepted_as_this_execution(tmp_path, fault):
    report, execution, digest = report_pair(tmp_path)
    adapter.require_report_origin(report, execution, tmp_path, digest, 0)
    with pytest.raises(ValueError, match="acknowledged"):
        adapter.require_report_origin(report | {fault[0]: fault[1]}, execution, tmp_path, digest, 0)


@pytest.mark.parametrize("field", ("worker_start_ticks", "controller_start_ticks", "worker_pid"))
def test_report_owner_generation_must_match_acknowledged_process(tmp_path, field):
    report, execution, digest = report_pair(tmp_path)
    changed = copy.deepcopy(report)
    changed["owner"][field] = "reused-generation"
    with pytest.raises(ValueError, match="acknowledged"):
        adapter.require_report_origin(changed, execution, tmp_path, digest, 0)


def test_zero_exit_without_marked_cleanup_is_not_a_complete_replay_process(tmp_path):
    report, execution, digest = report_pair(tmp_path)
    with pytest.raises(ValueError, match="acknowledged"):
        adapter.require_report_origin(report, execution | {"marked_descendants_absent": False}, tmp_path, digest, 0)


def test_partial_process_set_cannot_publish_comparison_or_completion(tmp_path):
    with pytest.raises(ValueError, match="two complete"):
        adapter.compare_results(tmp_path, [])
    assert list(tmp_path.iterdir()) == []


def report_context(root):
    helpers = root / "tracking"
    names = (
        "saved_ns_gpu_worker.py",
        "saved_ns_gpu_owner.py",
        "replay_saved_ns.py",
        "analyze_fused_continuation_stages.py",
        "diagnose_continuation.py",
        "diagnose_fused_diagonal.py",
        "benchmark_diagonal.py",
    )
    hashes = {name: f"{index:064x}" for index, name in enumerate(names)}
    uuid = "12345678-1234-1234-1234-123456789abc"
    frozen = {
        "helpers": str(helpers),
        "helpers_sha256": hashes,
        "worker_environment": {"CUDA_VISIBLE_DEVICES": f"GPU-{uuid}"},
        "devices": [["0", f"GPU-{uuid}", "NVIDIA A100"]],
    }
    runtime = {"deterministic": True}
    raw = {"shape": [16, 512, 512], "stride": [262144, 512, 1], "storage_offset": 0}
    provenance = {"raw_layouts": {mode: {"source": dict(raw)} for mode in ("continuous", "resumed")}}
    report = {
        "owner": {"binding": copy.deepcopy(frozen)},
        "environment": dict(frozen["worker_environment"]),
        "runtime": dict(runtime),
        "device": {"physical_uuid": f"GPU-{uuid}", "actual_uuid": uuid, "logical": "cuda:0"},
        "replay_mode": "tpu",
        "numerical_settings": {
            "deterministic_algorithms": True,
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "autocast": False,
            "cuda_source_uploads": 2,
            "source_values_checked_after_each_call": True,
        },
        "worker_sources_sha256": {str(helpers / name): digest for name, digest in hashes.items()},
        "source_layouts": {mode: {place: dict(raw) for place in ("cpu", "cuda")} for mode in ("continuous", "resumed")},
        "provenance": copy.deepcopy(provenance),
    }
    return report, frozen, runtime, provenance


@pytest.mark.parametrize("fault", ("empty", "truncated", "copied_path", "changed_digest", "extra"))
def test_report_context_rejects_missing_or_substituted_worker_source_evidence(tmp_path, fault):
    report, frozen, runtime, provenance = report_context(tmp_path)
    adapter.require_report_context(report, frozen, runtime, provenance)
    sources = report["worker_sources_sha256"]
    path = str(tmp_path / "tracking/saved_ns_gpu_owner.py")
    if fault == "empty":
        sources.clear()
    elif fault == "truncated":
        del sources[path]
    elif fault == "copied_path":
        sources[str(tmp_path / "copied/saved_ns_gpu_owner.py")] = sources.pop(path)
    elif fault == "changed_digest":
        sources[path] = "f" * 64
    else:
        sources[str(tmp_path / "tracking/unbound.py")] = "f" * 64
    with pytest.raises(ValueError, match="executed helper identity"):
        adapter.require_report_context(report, frozen, runtime, provenance)


def test_report_context_accepts_uuid_serialization_but_rejects_a_different_actual_device(tmp_path):
    report, frozen, runtime, provenance = report_context(tmp_path)
    adapter.require_report_context(report, frozen, runtime, provenance)
    report["device"]["actual_uuid"] = report["device"]["physical_uuid"]
    adapter.require_report_context(report, frozen, runtime, provenance)
    report["device"]["actual_uuid"] = "GPU-ffffffff-1234-1234-1234-123456789abc"
    with pytest.raises(ValueError, match="physical environment"):
        adapter.require_report_context(report, frozen, runtime, provenance)


SYNTHETIC_WORKER = """
import json,os,select,signal,subprocess,sys
from pathlib import Path
state_path = Path(sys.argv[1])
index = int(sys.argv[2])
poll = select.poll()
while True:
    state = json.loads(state_path.read_text())
    worker = state["active_worker"]
    if worker.get("spawn_status") == "acknowledged" and worker.get("pid") == os.getpid():
        break
    poll.poll(1)
if os.environ.get("SAVED_NS_CPU_FAILURE") == "timeout":
    signal.pause()
if os.environ.get("SAVED_NS_CPU_FAILURE") == "orphan":
    reader,writer = os.pipe()
    child = subprocess.Popen(
        [sys.executable,"-c",
         "import os,signal,sys;os.write(int(sys.argv[1]),b'R');signal.pause()",str(writer)],
        pass_fds=(writer,),start_new_session=True)
    os.close(writer)
    assert os.read(reader,1) == b"R"
    Path(sys.argv[3]).write_text(str(child.pid))
    sys.exit(7)
report = {"worker_pid":os.getpid(),"index":index,
          "mask":os.environ["CUDA_VISIBLE_DEVICES"]}
Path(sys.argv[3]).write_text(json.dumps(report))
"""


@pytest.mark.parametrize("failure", ("none", "timeout", "orphan"))
def test_real_worker_records_acknowledged_exit_and_cleans_detached_failure_children(tmp_path, monkeypatch, failure):
    binding, helpers, state_path, work = owner.locations(tmp_path)
    helpers.mkdir(parents=True)
    work.mkdir(parents=True)
    binding.parent.mkdir(parents=True)
    binding.write_text("bound CPU orchestration fixture\n")
    command = [
        sys.executable,
        "-P",
        "-c",
        SYNTHETIC_WORKER,
        str(state_path),
        "0",
        str(work / "process-0.json"),
    ]
    monkeypatch.setattr(owner, "worker_command", lambda root, digest, index: command)
    monkeypatch.setattr(adapter, "WORKER_TIMEOUT", 0.25 if failure == "timeout" else 5)
    monkeypatch.setenv("SAVED_NS_CPU_FAILURE", failure)
    state = {
        "controller_pid": os.getpid(),
        "controller_start_ticks": owner.handoff.observed(os.getpid())["start_ticks"],
    }
    adapter.write(state_path, state)
    frozen = {"worker_environment": dict(owner.ENVIRONMENT)}
    if failure == "none":
        result = adapter.run_worker(tmp_path, frozen, state, 0)
        report = json.loads(Path(result["report"]).read_text())
        assert report == {"worker_pid": result["pid"], "index": 0, "mask": ""}
        assert result["report_sha256"] == adapter.sha256(Path(result["report"]))
        assert result["exit_code"] == result["cleanup_exit_code"] == 0
    else:
        expected = subprocess.TimeoutExpired if failure == "timeout" else RuntimeError
        with pytest.raises(expected):
            adapter.run_worker(tmp_path, frozen, state, 0)
    saved = adapter.read(state_path)["active_worker"]
    assert saved["spawn_status"] == "acknowledged"
    assert saved["marked_descendants_absent"] is True
    assert type(saved["cleanup_exit_code"]) is int
    assert owner.handoff.observed(saved["pid"]) is None
    assert owner.handoff.marker_processes(owner.CHILD_MARKER, saved["marker"]) == {}
    assert saved["log_sha256"] == adapter.sha256(Path(saved["log"]))
    if failure == "orphan":
        orphan = owner.handoff.observed(int((work / "process-0.json").read_text()))
        assert orphan is None or orphan["status"] in owner.EXITED
    assert adapter.read(state_path).get("keeper_release_attempted") is None
