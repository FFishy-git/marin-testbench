# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise wave-6 evidence contracts without touching CUDA or live allocations."""

import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
from pathlib import Path

import analyze_wave
import analyze_wave6
import pytest
import watch_wave5
import watch_wave6
import wave6
from launch_wave5 import WORKLOAD


def record(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path
    candidate = root / "candidate"
    candidate.mkdir()
    helper = root / "controller"
    helper.mkdir()
    for name in wave6.HELPERS:
        (helper / name).write_text(f"bound {name}")
    archive = root / "source.tar"
    archive.write_bytes(b"preserved source")
    preparation = root / "run-preparation"
    record(preparation / "validation-wave6-cpu.json", {"status": "passed"})
    record(root / "data/experiment_receipt.json", {"identity": "data"})
    for arm in wave6.ARMS:
        path = candidate / "experiment/configs" / f"{arm}_0p5m_seed0.json"
        record(path, {"experiment": {"arm": arm}, "training": {"steps": 5120}})
    (candidate / "experiment/train.py").write_text("source")
    pinned = candidate / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    (pinned / "reference.py").write_text("reference")
    revision = "reviewed-revision"
    for name in ("reproduce/marin/train.py", "src/toner/optimizer.py"):
        path = pinned / f"Toner-{revision}" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"upstream {name}")
    record(
        candidate / "experiment/upstream_manifest.json",
        {
            "revision": revision,
            "files": {
                str(path.relative_to(pinned / f"Toner-{revision}")): wave6.sha256(path)
                for path in (pinned / f"Toner-{revision}").rglob("*.py")
            },
        },
    )
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    source_hashes = wave6.sources(candidate)
    frozen = {
        "candidate": str(candidate),
        "helpers": str(helper),
        "arms": list(wave6.ARMS),
        "required_gates": list(wave6.GATES),
        "files_sha256": wave6.candidate_inventory(candidate),
        "helpers_sha256": {name: wave6.sha256(helper / name) for name in wave6.HELPERS},
        "archive": str(archive),
        "archive_sha256": wave6.sha256(archive),
        "cpu_receipt_sha256": wave6.sha256(preparation / "validation-wave6-cpu.json"),
        "data_receipt_sha256": wave6.sha256(root / "data/experiment_receipt.json"),
        "source_sha256": source_hashes,
        "predecessors": list(wave6.PREDECESSORS),
    }
    record(preparation / "wave6-candidate.json", frozen)
    monkeypatch.setattr(wave6, "__file__", str(helper / "wave6.py"))
    monkeypatch.setattr(watch_wave6, "__file__", str(helper / "watch_wave6.py"))
    log = root / "validation.log"
    log.write_text("passed")
    receipt = preparation / "validation-wave6-gpu.json"
    record(
        receipt,
        {
            "checks": {name: True for name in wave6.GATES},
            "source_sha256": source_hashes,
            "data_receipt_sha256": frozen["data_receipt_sha256"],
            "environment": wave6.ENVIRONMENT,
            "config_sha256": {
                arm: wave6.sha256(candidate / "experiment/configs" / f"{arm}_0p5m_seed0.json") for arm in wave6.ARMS
            },
            "evidence": {"gpu_commands": [{"exit_code": 0, "log": str(log), "log_sha256": wave6.sha256(log)}]},
        },
    )
    record(
        preparation / "wave6-validation-state.json",
        {"status": "passed", "keeper_restore": 0, "validation_sha256": wave6.sha256(receipt)},
    )
    run_names = (
        "wave2/inner_outer_mlpdiag",
        "wave4/inner_outer_mlpdiag_lag",
        "wave4/muont_1m",
        "wave4/inner_outer_mlpdiag_1m",
        "wave4/inner_outer_mlpdiag_seed2",
        "wave4-baseline/muont_seed2",
    )
    for name in run_names:
        run = root / "runs" / name
        manifest = {
            "status": "complete",
            "completed_steps": 5,
            "data_receipt_sha256": "data",
            "config": {
                "training": {
                    "steps": 5,
                    "tokens": 5 * analyze_wave.WINDOW,
                    "batch_tokens": analyze_wave.WINDOW,
                    "seed": 2 if "seed2" in name else 0,
                    "heldout_windows": 64,
                },
                "model": {"sequence_length": 4096},
                "data": {"mixture": "same"},
            },
        }
        record(run / "run.json", manifest)
        rows = [
            {"step": step, "tokens": step * analyze_wave.WINDOW, "train_loss": 6.0 - step, "selection_loss": 3.0}
            for step in range(1, 6)
        ]
        (run / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        record(run / "checkpoints/step_000005/complete.json", {"step": 5, "world": 8})
    for previous in wave6.PREDECESSORS:
        names = [name for name in run_names if name.startswith(previous + "/")]
        entries = [
            {
                "node": node,
                "pid": 99000 + index,
                "out": str(root / "runs" / name),
                "arm": name.split("/")[-1],
            }
            for index, (node, name) in enumerate(
                zip(wave6.NODES if previous == "wave4" else (wave6.NODES[2],), names, strict=True)
            )
        ]
        record(preparation / f"{previous}.json", {"launches": entries})
        for entry in entries:
            (Path(entry["out"]).parent / f"keeper-restore-{entry['node']}.log").write_text("keeper start exit 0")
        record(
            preparation / f"swanlab-upload-{previous}.json",
            {
                "status": "verified",
                "runs": {
                    Path(entry["out"]).name: {
                        "status": "verified",
                        "manifest_sha256": wave6.sha256(Path(entry["out"]) / "run.json"),
                        "metrics_sha256": wave6.sha256(Path(entry["out"]) / "metrics_from_000000.jsonl"),
                    }
                    for entry in entries
                },
            },
        )
        record(
            preparation / f"analysis-{previous}.json",
            analyze_wave.analyze(root, previous),
        )
    return root, candidate, frozen


def test_complete_evidence_release_then_missing_actual_update_gate_refuses(prepared):
    root, candidate, _ = prepared
    actual, receipt, _ = wave6.require_release(root)
    assert actual == candidate
    checked = wave6.read(receipt)
    checked["checks"]["real_data_outer_diagonal_update_parity"] = False
    record(receipt, checked)
    state = root / "run-preparation/wave6-validation-state.json"
    record(state, {"status": "passed", "keeper_restore": 0, "validation_sha256": wave6.sha256(receipt)})
    with pytest.raises(ValueError, match="required gates"):
        wave6.require_release(root)


def test_release_refuses_different_gpu_config_and_stale_predecessor_history(prepared):
    root, _, _ = prepared
    receipt = root / "run-preparation/validation-wave6-gpu.json"
    original = wave6.read(receipt)
    checked = {**original, "config_sha256": {**original["config_sha256"], "outer_leftdiag_fused": "different"}}
    record(receipt, checked)
    state = root / "run-preparation/wave6-validation-state.json"
    record(state, {"status": "passed", "keeper_restore": 0, "validation_sha256": wave6.sha256(receipt)})
    with pytest.raises(ValueError, match="configurations"):
        wave6.require_release(root)
    record(receipt, original)
    record(state, {"status": "passed", "keeper_restore": 0, "validation_sha256": wave6.sha256(receipt)})
    history = root / "runs/wave4-baseline/muont_seed2/metrics_from_000000.jsonl"
    history.write_text("modified after upload")
    with pytest.raises(ValueError, match="scalar verification is stale"):
        wave6.require_release(root)


def test_analysis_refuses_partial_inventory_and_different_executed_source(prepared):
    root, candidate, frozen = prepared
    for arm in wave6.ARMS:
        record(
            root / "runs/wave6" / arm / "run.json",
            {
                "config": wave6.read(candidate / "experiment/configs" / f"{arm}_0p5m_seed0.json"),
                "source_sha256": analyze_wave6.executed_sources(candidate, frozen),
                "data_receipt_sha256": frozen["data_receipt_sha256"],
            },
        )
    report = root / "run-preparation/wave6.json"
    entries = [
        {
            "node": node,
            "arm": arm,
            "seed": 0,
            "batch": "0p5m",
            "pid": 90000 + index,
            "out": str(root / "runs/wave6" / arm),
        }
        for index, (node, arm) in enumerate(zip(wave6.NODES, wave6.ARMS, strict=True))
    ]
    record(report, {"expected_launches": 4, "launches": entries[:-1]})
    with pytest.raises(ValueError, match="partial"):
        analyze_wave6.verify_histories(root)
    record(report, {"expected_launches": 4, "launches": entries})
    analyze_wave6.verify_histories(root)
    run = root / "runs/wave6/outer_leftdiag_fused/run.json"
    changed = wave6.read(run)
    changed["source_sha256"] = {"experiment/train.py": "different-source"}
    record(run, changed)
    with pytest.raises(ValueError, match="source differs"):
        analyze_wave6.verify_histories(root)


@pytest.mark.parametrize("fault", ("empty", "unrelated", "window", "scalar"))
def test_release_refuses_saved_analysis_not_reconstructed_from_verified_history(prepared, fault):
    root, _, _ = prepared
    path = root / "run-preparation/analysis-wave4.json"
    saved = wave6.read(path)
    if fault == "empty":
        saved["analyses"] = []
    elif fault == "unrelated":
        saved["analyses"][0]["baseline"] = "wave0/unrelated"
    elif fault == "window":
        saved["window_tokens"] //= 2
    else:
        pair = saved["analyses"][0]
        next(iter(pair["comparisons"].values()))["training_windows"][0]["mean"] += 1
    record(path, saved)
    with pytest.raises(ValueError, match="analysis is incomplete, unrelated or stale"):
        wave6.require_release(root)


def test_release_rejects_old_analysis_after_history_and_cloud_receipt_change(prepared):
    root, _, _ = prepared
    run = root / "runs/wave4/inner_outer_mlpdiag_lag"
    path = run / "metrics_from_000000.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["train_loss"] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cloud_path = root / "run-preparation/swanlab-upload-wave4.json"
    cloud = wave6.read(cloud_path)
    cloud["runs"][run.name]["metrics_sha256"] = wave6.sha256(path)
    record(cloud_path, cloud)
    with pytest.raises(ValueError, match="analysis is incomplete, unrelated or stale"):
        wave6.require_release(root)
    record(root / "run-preparation/analysis-wave4.json", analyze_wave.analyze(root, "wave4"))
    assert wave6.require_release(root)[0] == root / "candidate"


@pytest.mark.parametrize(
    "extra",
    ("experiment/unexpected.py", "experiment/configs/extra.json", "artifacts/pinned-sources-materialized/extra.py"),
)
def test_binding_refuses_unrecorded_candidate_files(prepared, extra):
    root, candidate, _ = prepared
    (candidate / extra).write_text("unexpected")
    with pytest.raises(ValueError, match="candidate inventory changed"):
        wave6.binding(root)


@pytest.mark.parametrize("alias", ("absolute", "escaping"))
def test_binding_refuses_nonlocal_external_source_alias(prepared, alias):
    root, candidate, _ = prepared
    path = candidate / "artifacts/external-sources"
    path.unlink()
    path.symlink_to(
        candidate / "artifacts/pinned-sources-materialized" if alias == "absolute" else "../../outside",
        target_is_directory=True,
    )
    with pytest.raises(ValueError, match="exact candidate-local relative symlink"):
        wave6.binding(root)


@pytest.fixture
def pending_freeze(prepared, monkeypatch):
    root, candidate, frozen = prepared
    preparation = root / "run-preparation"
    (preparation / "wave6-candidate.json").unlink()
    archive = Path(frozen["archive"])
    archive.unlink()
    task = root / "mutable-task"
    (task / "tracking").mkdir(parents=True)
    (task / "experiment").mkdir()
    (task / "ENDEAVOR.md").write_text("endeavor at freeze")
    (task / "experiment/README.md").write_text("readme at freeze")
    for name in wave6.HELPERS:
        (task / "tracking" / name).write_text(f"helper {name}")
    monkeypatch.setattr(wave6, "TASK", task)
    record(
        candidate / "experiment/cluster.json",
        {
            "nodes": wave6.NODES,
            "wave_name": "wave6",
            "environment": wave6.ENVIRONMENT,
            "wave": [{"arm": arm, "seed": 0, "batch": "0p5m"} for arm in wave6.ARMS],
        },
    )
    record(preparation / "wave5-candidate.json", {"files_sha256": frozen["files_sha256"]})
    record(
        preparation / "validation-wave6-cpu.json",
        {
            "source_sha256": wave6.sources(candidate),
            "evidence": {"cpu_test_exit_code": 0, "model_parameter_count": 265585152},
            "checks": {name: True for name in wave6.GATES[:3]},
            "environment": wave6.ENVIRONMENT,
            "candidate_path": str(candidate),
            "cluster_sha256": wave6.sha256(candidate / "experiment/cluster.json"),
            "upstream_manifest_sha256": wave6.sha256(candidate / "experiment/upstream_manifest.json"),
            "config_sha256": {
                arm: wave6.sha256(candidate / "experiment/configs" / f"{arm}_0p5m_seed0.json") for arm in wave6.ARMS
            },
            "config_paths": {
                arm: str(candidate / "experiment/configs" / f"{arm}_0p5m_seed0.json") for arm in wave6.ARMS
            },
            "data_receipt": str(root / "data/experiment_receipt.json"),
            "data_receipt_sha256": wave6.sha256(root / "data/experiment_receipt.json"),
        },
    )
    return root, candidate, task


def test_freeze_archive_is_self_contained_and_documents_are_informational(pending_freeze, monkeypatch):
    root, candidate, task = pending_freeze
    snapshot = wave6.freeze(root, candidate)
    extracted = root / "extracted"
    with tarfile.open(snapshot["archive"]) as stream:
        stream.extractall(extracted, filter="tar")
    assert (extracted / "artifacts/external-sources/reference.py").read_text() == "reference"
    assert (extracted / "ENDEAVOR.md").read_text() == "endeavor at freeze"
    assert (extracted / "experiment/README.md").read_text() == "readme at freeze"
    assert wave6.candidate_inventory(extracted) == snapshot["files_sha256"]
    assert snapshot["documents_sha256"]["ENDEAVOR.md"] == wave6.sha256(extracted / "ENDEAVOR.md")
    (task / "ENDEAVOR.md").write_text("updated after freeze")
    monkeypatch.setattr(wave6, "__file__", str(Path(snapshot["helpers"]) / "wave6.py"))
    assert wave6.binding(root)[0] == candidate
    with pytest.raises(FileExistsError):
        wave6.freeze(root, candidate)


@pytest.mark.parametrize("fault", ("config", "cluster", "data", "environment", "gate", "upstream", "pinned"))
def test_freeze_refuses_stale_cpu_protocol_and_extra_pinned_inventory_before_copy(pending_freeze, fault):
    root, candidate, _ = pending_freeze
    path = root / "run-preparation/validation-wave6-cpu.json"
    cpu = wave6.read(path)
    if fault == "config":
        cpu["config_sha256"]["outer_leftdiag_fused"] = "stale"
    elif fault == "cluster":
        cpu["cluster_sha256"] = "stale"
    elif fault == "data":
        (root / "data/experiment_receipt.json").write_text("new data receipt")
    elif fault == "environment":
        cpu["environment"] = {}
    elif fault == "gate":
        cpu["checks"]["model_parameter_count"] = False
    elif fault == "upstream":
        cpu["upstream_manifest_sha256"] = "stale"
    else:
        (candidate / "artifacts/pinned-sources-materialized/unrecorded.py").write_text("new pinned code")
    record(path, cpu)
    with pytest.raises(ValueError, match=r"CPU evidence|pinned source inventory"):
        wave6.freeze(root, candidate)
    assert not (root / "controllers/wave6-20261116").exists()
    assert not (root / "run-preparation/wave6-candidate.json").exists()
    assert not (root / "run-preparation/source-wave6-outer-diagonal-20261116.tar").exists()


@pytest.mark.parametrize("persistent", (False, True))
def test_postspawn_ledger_failure_reaps_real_cpu_child(tmp_path, monkeypatch, persistent):
    path = tmp_path / "state.json"
    original_write = Path.write_text
    original_popen = subprocess.Popen
    children = []
    failed = False

    def fail_ledger(file, content, *args, **kwargs):
        nonlocal failed
        if file == path.with_suffix(".tmp") and (persistent or not failed):
            failed = True
            raise OSError("ledger unavailable")
        return original_write(file, content, *args, **kwargs)

    def spawn(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(Path, "write_text", fail_ledger)
    monkeypatch.setattr(subprocess, "Popen", spawn)
    state = {}
    try:
        with pytest.raises(OSError, match="ledger unavailable"):
            wave6.bounded_execution(
                [sys.executable, "-B", "-c", "import signal; signal.pause()"],
                tmp_path,
                tmp_path / "child.log",
                state,
                path,
                5,
            )
        assert children[0].poll() is not None
        assert wave6.tagged_processes(state["child_marker"]) == {}
        if not persistent:
            assert wave6.read(path)["cleanup_exit_code"] == children[0].returncode
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


class FakeProcessIO:
    """Fake only subprocess and /proc boundaries; no keeper, SSH or CUDA command executes."""

    def __init__(self, root, candidate, frozen, monkeypatch):
        self.root = root
        self.candidate = candidate
        self.frozen = frozen
        self.node = wave6.NODES[0]
        self.processes = {}
        self.restored = []
        self.released = []
        self.watcher_attempts = []
        self.launches = []
        self.lost_ack = False
        self.exited = False
        self.watcher_failure = set()
        self.discovery_failure = set()
        self.monitor_failure = False
        self.report_fault = None
        self.manager_starts = []
        self.next_pid = 80000
        original_iterdir = Path.iterdir
        original_text = Path.read_text
        original_bytes = Path.read_bytes
        original_unlink = Path.unlink

        def iterdir(path):
            if path == Path("/proc"):
                return iter(
                    Path("/proc") / str(pid) for pid, fields in self.processes.items() if fields["node"] == self.node
                )
            return original_iterdir(path)

        def text(path, *args, **kwargs):
            if path.parent.parent == Path("/proc") and path.name == "stat":
                pid = int(path.parent.name)
                if pid not in self.processes:
                    raise FileNotFoundError(path)
                fields = self.processes[pid]
                return f"{pid} (python) " + " ".join(
                    [fields["status"], str(fields.get("parent", 1)), str(pid), str(pid), *["0"] * 15, "123"]
                )
            return original_text(path, *args, **kwargs)

        def data(path, *args, **kwargs):
            if path.parent.parent == Path("/proc"):
                pid = int(path.parent.name)
                if pid not in self.processes:
                    raise FileNotFoundError(path)
                if path.name == "cmdline":
                    return b"\0".join(arg.encode() for arg in self.processes[pid]["command"]) + b"\0"
                if path.name == "environ":
                    return b""
            return original_bytes(path, *args, **kwargs)

        monkeypatch.setattr(Path, "iterdir", iterdir)
        monkeypatch.setattr(Path, "read_text", text)
        monkeypatch.setattr(Path, "read_bytes", data)
        monkeypatch.setattr(
            Path,
            "unlink",
            lambda path, **kwargs: (
                None if path == Path("/root/dl/runtime/gpu_matmul.pid") else original_unlink(path, **kwargs)
            ),
        )
        monkeypatch.setattr(subprocess, "run", self.run)
        monkeypatch.setattr(subprocess, "check_output", self.check_output)
        monkeypatch.setattr(subprocess, "Popen", self.popen)

    def entry(self, node):
        arm = wave6.ARMS[wave6.NODES.index(node)]
        out = self.root / "runs/wave6" / arm
        out.parent.mkdir(parents=True, exist_ok=True)
        self.next_pid += 1
        command = [
            wave6.PYTHON,
            "-m",
            "torch.distributed.run",
            "-m",
            "experiment.train",
            "--config",
            str(self.candidate / "experiment/configs" / f"{arm}_0p5m_seed0.json"),
            "--out",
            str(out),
        ]
        entry = {
            "node": node,
            "arm": arm,
            "seed": 0,
            "batch": "0p5m",
            "pid": self.next_pid,
            "out": str(out),
            "command": command,
        }
        if not self.exited:
            self.processes[entry["pid"]] = {"command": command, "status": "S", "node": node}
        return entry

    def check_output(self, command, **kwargs):
        if command == ["hostname", "-I"]:
            return self.node
        if command == ["ps", "-eo", "args="]:
            return "\n".join(" ".join(p["command"]) for p in self.processes.values() if p["node"] == self.node)
        if command[0] == "nvidia-smi":
            return "" if "compute-apps" in command[1] else "A100-SXM4-40GB\n" * 8
        raise AssertionError(f"Unexpected subprocess: {command}")

    def run(self, command, **kwargs):
        if command[0] == "ssh":
            node = command[-2].removeprefix("root@")
            arguments = shlex.split(command[-1])
            phase = arguments[arguments.index("--phase") + 1]
            if phase == "inspect":
                result = {"node": node, "status": "preflight_passed"}
            elif phase == "release":
                self.released.append(node)
                result = {"node": node, "status": "released"}
            elif phase == "restore":
                self.restored.append(node)
                result = {"node": node, "keeper_restore": 0}
            elif phase == "discover":
                if node in self.discovery_failure:
                    raise subprocess.TimeoutExpired(command, 240)
                previous = self.node
                self.node = node
                try:
                    result = wave6.discover(self.root)
                finally:
                    self.node = previous
            else:
                raise AssertionError(phase)
            return subprocess.CompletedProcess(command, 0, json.dumps(result), "")
        if command[:2] == ["bash", watch_wave5.KEEPER]:
            self.manager_starts.append(self.node)
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1:3] == ["-m", "experiment.bootstrap"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"Unexpected subprocess: {command}")

    def popen(self, command, **kwargs):
        self.next_pid += 1
        pid = self.next_pid
        if command[0] == "bash":
            if self.monitor_failure:
                raise OSError("monitor unavailable")
            self.processes[pid] = {"command": command, "status": "S", "node": self.node}
            (self.root / "run-preparation/monitor-wave6/pid").write_text(str(pid))
            return FakeChild(pid, None)
        if command[1:3] == ["-m", "experiment.launch"]:
            return FakeChild(pid, 0, self.launch)
        if Path(command[1]).name == "watch_wave6.py":
            out = Path(command[command.index("--out") + 1])
            self.watcher_attempts.append(out.name)
            if out.name in self.watcher_failure:
                raise OSError("watcher unavailable")
            training_pid = int(command[command.index("--pid") + 1])
            path = Path(command[command.index("--run") + 1])
            self.processes[pid] = {"command": command, "status": "S", "node": self.node}
            record(
                path,
                {
                    "pid": pid,
                    "training_pid": training_pid,
                    "training_start_ticks": (
                        command[command.index("--start-ticks") + 1] if "--start-ticks" in command else None
                    ),
                    "watcher_start_ticks": "123",
                    "root": str(self.root),
                    "out": str(out),
                    "status": "watching" if training_pid in self.processes else "restored",
                    "keeper_restore": 0,
                },
            )
            return FakeChild(pid, None)
        raise AssertionError(f"Unexpected spawn: {command}")

    def launch(self):
        self.launches = [self.entry(node) for node in wave6.NODES]
        report = {"launches": self.launches[:-1] if self.lost_ack else self.launches}
        path = self.root / "run-preparation/wave6.json"
        record(path, report)
        if self.report_fault == "read":
            path.write_text("{not json")
        return 1 if self.lost_ack else 0


class FakeChild:
    def __init__(self, pid, returncode, action=None):
        self.pid = pid
        self.returncode = returncode
        self.action = action

    def wait(self, **kwargs):
        if self.action is not None:
            self.returncode = self.action()
            self.action = None
        return self.returncode

    def poll(self):
        return self.returncode


@pytest.fixture
def process_io(prepared, monkeypatch):
    root, candidate, frozen = prepared
    return FakeProcessIO(root, candidate, frozen, monkeypatch)


def test_lost_ack_and_watcher_failure_keep_identity_and_monitor_all_launches(process_io):
    boundary = process_io
    boundary.lost_ack = True
    boundary.watcher_failure = {wave6.ARMS[-1]}
    with pytest.raises(RuntimeError, match="launch incomplete"):
        wave6.launch(boundary.root)
    report = wave6.read(boundary.root / "run-preparation/wave6.json")
    state = wave6.read(boundary.root / "run-preparation/wave6-launch-state.json")
    assert {entry["node"] for entry in report["launches"]} == set(wave6.NODES)
    assert report["expected_launches"] == 4
    assert wave6.ARMS[-1] in boundary.watcher_attempts
    assert set(boundary.watcher_attempts) == set(wave6.ARMS)
    assert state["node_observations"][wave6.NODES[-1]] == "active"
    assert wave6.NODES[-1] in state["watcher_errors"]
    assert state["monitor_pid"] > 0
    assert boundary.restored == []
    observed = wave6.read(boundary.root / "run-preparation" / f"wave6-observed-{wave6.NODES[-1]}.json")
    assert observed["pid"] == boundary.launches[-1]["pid"]
    assert "keeper_watcher_pid" not in observed


def test_exited_acknowledged_parents_receive_restored_watchers(process_io):
    boundary = process_io
    boundary.exited = True
    state = wave6.launch(boundary.root)
    report = wave6.read(boundary.root / "run-preparation/wave6.json")
    assert state["status"] == "launched"
    assert set(boundary.watcher_attempts) == set(wave6.ARMS)
    assert all(entry["keeper_watcher_status"] == "restored" for entry in report["launches"])
    assert boundary.restored == []


@pytest.mark.parametrize("fault", ("read", "write", "monitor"))
def test_report_and_monitor_failure_do_not_skip_proven_idle_restoration(process_io, monkeypatch, fault):
    boundary = process_io
    original_run = boundary.run
    original_write = Path.write_text
    path = boundary.root / "run-preparation/wave6.json"
    failed_release = wave6.NODES[1]

    def release_failure(command, **kwargs):
        if command[0] == "ssh" and command[-2] == f"root@{failed_release}" and "--phase release" in command[-1]:
            boundary.released.append(failed_release)
            if fault == "read":
                path.write_text("{invalid")
            elif fault == "monitor":
                entry = boundary.entry(wave6.NODES[0])
                record(path, {"launches": [entry]})
            raise RuntimeError("lost release acknowledgment")
        return original_run(command, **kwargs)

    def fail_report(file, content, *args, **kwargs):
        if fault == "write" and file == path.with_suffix(".tmp"):
            raise OSError("report disk unavailable")
        return original_write(file, content, *args, **kwargs)

    boundary.monitor_failure = fault == "monitor"
    monkeypatch.setattr(subprocess, "run", release_failure)
    monkeypatch.setattr(Path, "write_text", fail_report)
    with pytest.raises(RuntimeError, match="lost release acknowledgment"):
        wave6.launch(boundary.root)
    state = wave6.read(boundary.root / "run-preparation/wave6-launch-state.json")
    assert failed_release in boundary.restored
    assert state["restored_nodes"] == boundary.restored
    if fault == "monitor":
        assert wave6.NODES[0] not in boundary.restored
        assert "monitor_error" in state
        assert len(wave6.read(path)["launches"]) == 1
    else:
        assert boundary.restored == list(wave6.NODES[:2])
        assert "report_recovery_error" in state if fault == "read" else state["report_errors"]


def test_unknown_discovery_prevents_restore_despite_lost_release_ack(process_io, monkeypatch):
    boundary = process_io
    node = wave6.NODES[0]
    boundary.discovery_failure = {node}
    original_run = boundary.run

    def lost_ack(command, **kwargs):
        if command[0] == "ssh" and "--phase release" in command[-1]:
            boundary.released.append(node)
            raise RuntimeError("lost release acknowledgment")
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", lost_ack)
    with pytest.raises(RuntimeError, match="lost release acknowledgment"):
        wave6.launch(boundary.root)
    state = wave6.read(boundary.root / "run-preparation/wave6-launch-state.json")
    assert boundary.restored == []
    assert state["unknown_nodes"] == [node]
    assert node in state["discovery_errors"]


def test_restore_ignores_changed_candidate_but_refuses_live_training(process_io):
    boundary = process_io
    shutil.rmtree(boundary.candidate)
    (boundary.root / "data/experiment_receipt.json").write_text("changed data")
    result = wave6.node_operation(boundary.root, "restore")
    assert result["keeper_restore"] == 0
    assert boundary.manager_starts == [boundary.node]
    entry = boundary.entry(boundary.node)
    with pytest.raises(RuntimeError, match="Live training/torchrun prevents keeper restoration"):
        wave6.node_operation(boundary.root, "restore")
    assert boundary.manager_starts == [boundary.node]
    assert entry["pid"] in boundary.processes


@pytest.mark.parametrize("changed", ("helper", "archive"))
def test_restore_refuses_changed_restoration_integrity(process_io, changed):
    boundary = process_io
    path = (
        Path(boundary.frozen["archive"])
        if changed == "archive"
        else Path(boundary.frozen["helpers"]) / "launch_wave5.py"
    )
    path.write_text("changed")
    with pytest.raises(ValueError, match=r"archive changed|controller changed"):
        wave6.node_operation(boundary.root, "restore")
    assert boundary.manager_starts == []


def test_node_mutation_uses_own_lock_not_parent_lock_and_restores_handlers(process_io):
    boundary = process_io
    root = boundary.root
    global_path = root / "run-preparation/.wave6-controller.lock"
    local_path = root / "run-preparation" / f".wave6-node-{boundary.node}.lock"
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    with global_path.open("a") as parent_lock:
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = wave6.node_operation(root, "restore")
        assert result["keeper_restore"] == 0
        with local_path.open("a") as local_lock:
            fcntl.flock(local_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(BlockingIOError):
                wave6.node_operation(root, "restore")
    assert boundary.manager_starts == [boundary.node]
    assert {signum: signal.getsignal(signum) for signum in handlers} == handlers


@pytest.mark.parametrize("signum", (signal.SIGTERM, signal.SIGINT))
def test_signal_during_generation_release_finishes_and_keeps_node_locked(process_io, monkeypatch, signum):
    boundary = process_io
    root = boundary.root
    node = boundary.node
    keeper = 95000
    workers = (keeper + 1, keeper + 2)
    for pid in (keeper, *workers):
        boundary.processes[pid] = {
            "node": node,
            "command": ["python", WORKLOAD.decode()],
            "status": "S",
            "parent": 1 if pid == keeper else keeper,
        }
    original_text = Path.read_text
    original_output = boundary.check_output
    stopped = []
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}

    def read_keeper(path, *args, **kwargs):
        if path == Path("/root/dl/runtime/gpu_matmul.pid"):
            return str(keeper)
        return original_text(path, *args, **kwargs)

    def occupied(command, **kwargs):
        if command[0] == "nvidia-smi" and "compute-apps" in command[1]:
            return "\n".join(str(pid) for pid in workers if pid in boundary.processes)
        return original_output(command, **kwargs)

    def stop_group(group, number):
        assert number == signal.SIGTERM
        assert group in (keeper, *workers)
        with pytest.raises(BlockingIOError):
            wave6.node_operation(root, "restore")
        stopped.append(group)
        boundary.processes.pop(group)
        if len(stopped) == 1:
            os.kill(os.getpid(), signum)

    monkeypatch.setattr(Path, "read_text", read_keeper)
    monkeypatch.setattr(subprocess, "check_output", occupied)
    monkeypatch.setattr(os, "killpg", stop_group)
    state = wave6.node_operation(root, "release")
    assert state["status"] == "released"
    assert state["deferred_signals"] == [signum]
    assert set(stopped) == {keeper, *workers}
    assert boundary.manager_starts == []
    receipt = wave6.read(root / "run-preparation" / f"wave6-release-{node}.json")
    assert receipt["status"] == "released"
    assert {entry[0] for entry in receipt["keeper"]["generation"]} == {keeper, *workers}
    assert {number: signal.getsignal(number) for number in handlers} == handlers
    assert wave6.node_operation(root, "restore")["keeper_restore"] == 0
    assert boundary.manager_starts == [node]


@pytest.mark.parametrize("fault", ("missing", "extra", "materialized_alias", "wrong_path", "duplicate"))
def test_analyzer_requires_full_alias_source_map_and_canonical_inventory(prepared, fault):
    root, candidate, frozen = prepared
    entries = [
        {
            "node": node,
            "arm": arm,
            "seed": 0,
            "batch": "0p5m",
            "pid": 90000 + index,
            "out": str(root / "runs/wave6" / arm),
        }
        for index, (node, arm) in enumerate(zip(wave6.NODES, wave6.ARMS, strict=True))
    ]
    expected = analyze_wave6.executed_sources(candidate, frozen)
    assert any(name.startswith("artifacts/external-sources/") for name in expected)
    for entry in entries:
        record(
            Path(entry["out"]) / "run.json",
            {
                "config": wave6.read(candidate / "experiment/configs" / f"{entry['arm']}_0p5m_seed0.json"),
                "source_sha256": expected,
                "data_receipt_sha256": frozen["data_receipt_sha256"],
            },
        )
    report_path = root / "run-preparation/wave6.json"
    record(report_path, {"expected_launches": 4, "launches": entries})
    analyze_wave6.verify_histories(root)
    run_path = Path(entries[0]["out"]) / "run.json"
    manifest = wave6.read(run_path)
    source_name = next(name for name in expected if name.startswith("artifacts/external-sources/"))
    if fault == "missing":
        manifest["source_sha256"].pop(source_name)
    elif fault == "extra":
        manifest["source_sha256"]["experiment/unexecuted.py"] = "extra"
    elif fault == "materialized_alias":
        manifest["source_sha256"][source_name.replace("external-sources", "pinned-sources-materialized")] = manifest[
            "source_sha256"
        ].pop(source_name)
    elif fault == "wrong_path":
        entries[-1]["out"] = str(root / "wrong-root" / wave6.ARMS[-1])
    else:
        entries[-1] = entries[0]
    record(run_path, manifest)
    record(report_path, {"expected_launches": 4, "launches": entries})
    with pytest.raises(ValueError):
        analyze_wave6.verify_histories(root)
