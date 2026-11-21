# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline receipt acceptance and real CPU coordinator cancellation lifecycle."""

import json
import os
import select
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import related_campaign as campaign_module
import related_execution as execution
import related_sequence as sequence
import torch
from experiment import data_access
from related_campaign import MOUSSE_REFERENCE, PROFILES, TASK, WORLD, read, sha256

CPU_STAGE = """
import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path

mode, directory = sys.argv[1], Path(sys.argv[2])

def event(name, **fields):
    print(json.dumps({"event": name, "pid": os.getpid(), **fields}), flush=True)

if mode == "worker":
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    event("worker_ready")
    received = signal.sigtimedwait({signal.SIGTERM}, 20)
    raise SystemExit(0 if received is not None else 9)

if mode == "wrapper":
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    worker = subprocess.Popen(
        [sys.executable, __file__, "worker", str(directory)],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    gate = os.open(directory / "restore.fifo", os.O_RDONLY | os.O_NONBLOCK)
    try:
        ready = json.loads(worker.stdout.readline())
        assert ready == {"event": "worker_ready", "pid": worker.pid}
        acknowledgment = {"wrapper_pid": os.getpid(), "worker_pid": worker.pid}
        (directory / "started.json").write_text(json.dumps(acknowledgment))
        event("started", **acknowledgment)
        received = signal.sigtimedwait({signal.SIGTERM}, 20)
        assert received is not None, "coordinator did not forward cancellation"
        worker.terminate()
        worker_exit_code = worker.wait(timeout=5)
        event("restore_waiting", worker_pid=worker.pid, worker_exit_code=worker_exit_code)
        assert select.select([gate], [], [], 15)[0], "restore gate was never released"
        assert os.read(gate, 1) == b"1"
        receipt = {
            **acknowledgment,
            "forwarded_signal": received.si_signo,
            "worker_exit_code": worker_exit_code,
        }
        (directory / "restored.json").write_text(json.dumps(receipt))
        event("restored")
    finally:
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)
        worker.stdout.close()
        os.close(gate)
    raise SystemExit(0)

sys.path.insert(0, sys.argv[3])
import related_sequence

try:
    result = related_sequence.run_stage([sys.executable, __file__, "wrapper", str(directory)])
except KeyboardInterrupt:
    acknowledgment = json.loads((directory / "started.json").read_text())
    receipt = {
        "restored": (directory / "restored.json").is_file(),
        "worker_absent": not (Path("/proc") / str(acknowledgment["worker_pid"])).exists(),
        "wrapper_absent": not (Path("/proc") / str(acknowledgment["wrapper_pid"])).exists(),
    }
    (directory / "cancelled.json").write_text(json.dumps(receipt))
    event("cancelled", **receipt)
    raise SystemExit(73)
else:
    event("unexpected_completion", exit_code=result.returncode)
    raise SystemExit(2)
"""


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n")


def checkpoint(directory, step, campaign, config):
    path = directory / "checkpoints" / f"step_{step:06d}"
    save(path / "complete.json", {"step": step, "world": WORLD})
    identity = {
        "schema": 1,
        "source_sha256": campaign["source_sha256"],
        "config_sha256": campaign_module.config_identity(config),
        "data_receipt_sha256": campaign["data_receipt_sha256"],
    }
    for rank in range(WORLD):
        torch.save(
            {
                "step": step,
                "next_global_sequence": step * config["training"]["global_batch_sequences"],
                "identity": identity,
                "rank": rank,
                "world": WORLD,
                "config": config,
                "model": {"weight": torch.tensor([[float(step), float(rank)]])},
            },
            path / f"rank_{rank}.pt",
        )


def manifest(campaign, config, directory, step, *, smoke):
    save(
        directory / "run.json",
        {
            "status": "execution_check_complete" if smoke else "complete",
            "completed_steps": step,
            "execution_check": smoke,
            "config": config,
            "data": campaign["data"],
            "data_receipt_sha256": campaign["data_receipt_sha256"],
            "source_revision": campaign["source_revision"],
            "source_sha256": campaign["source_sha256"],
            "world_size": WORLD,
        },
    )
    checkpoint(directory, step, campaign, config)


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "output"
    for folder, _ in PROFILES.values():
        shutil.copytree(TASK / "experiment/comparisons" / folder, source / "experiment/comparisons" / folder)
    shutil.copyfile(TASK / "experiment/cluster.json", source / "experiment/cluster.json")
    (source / "experiment/train.py").write_text("# offline trainer identity\n")
    revision = read(TASK / "experiment/upstream_manifest.json")["revision"]
    upstream = source / "artifacts/external-sources" / f"Toner-{revision}"
    for name in ("reproduce/marin/train.py", "src/toner/fast/muon.py"):
        path = upstream / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# offline pinned source identity\n")
    components = ("dclm", "starcoder", "proofpile", "paloma/c4_en")
    plan_path = upstream / "reproduce/marin/data_plan.json"
    save(plan_path, {"files": [{"component": name, "file": "tiny.bin", "tokens": 1, "docs": 0} for name in components]})
    names = ("reproduce/marin/train.py", "src/toner/fast/muon.py", "reproduce/marin/data_plan.json")
    save(
        source / "experiment/upstream_manifest.json",
        {"revision": revision, "files": {name: sha256(upstream / name) for name in names}},
    )
    oracle = source / MOUSSE_REFERENCE
    oracle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TASK / MOUSSE_REFERENCE, oracle)
    records = []
    for component in components:
        path = root / "data" / component / "tiny.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x00\x00\x00\x00")
        records.append(
            {"file": f"{component}/tiny.bin", "tokens": 1, "docs": 0, "bytes": 4, "mtime_ns": path.stat().st_mtime_ns}
        )
    save(
        root / "data/experiment_receipt.json",
        {"root": str(root / "data"), "data_plan_sha256": sha256(plan_path), "files": records},
    )
    monkeypatch.setattr(data_access, "PLAN", plan_path)
    monkeypatch.setattr(sequence.allocation, "TASK", source)
    monkeypatch.delenv(sequence.allocation.MARKER, raising=False)
    record = campaign_module.prepare(root, source=source)
    return root, source, record


def validation_receipt(campaign, stage):
    work, receipt_path = Path(stage["work"]), Path(stage["receipt"])
    work.mkdir(parents=True)
    wave = next(wave for wave in campaign["waves"] if wave["wave_name"] == stage["wave_name"])
    sources = {name: digest for name, digest in campaign["source_sha256"].items() if name.startswith("experiment/")}
    commands, probes, smokes = [], {}, {}

    def command_record(command):
        log = work / f"command_{len(commands)}.log"
        log.write_text("offline successful command\n")
        commands.append({"command": command, "exit_code": 0, "log": str(log), "log_sha256": sha256(log)})

    for key, module, checks, world_key in (
        ("gpu_receipt", "experiment.gpu_checks", execution.SYNTHETIC_CHECKS, "world"),
        ("related_gpu_receipt", "experiment.related_gpu_checks", execution.RELATED_CHECKS, "world_size"),
    ):
        path = work / f"{key}.json"
        save(path, {"checks": dict.fromkeys(checks, True), world_key: 2, "source_sha256": sources})
        probes[key] = str(path)
        command_record([*execution.launcher(execution.PYTHON, 2), module, "--receipt", str(path)])
    trials = {trial["name"]: trial for trial in campaign["trials"]}
    for name in wave["trials"]:
        trial = trials[name]
        alias = next(alias for alias in trial["aliases"] if alias["wave"] == wave["wave_name"])
        config = read(Path(alias["config"]))
        complete, split = execution.continuation_steps(config)
        full, resumed = work / f"{name}_continuous", work / f"{name}_resumed"
        common = execution.trial_command(
            Path(alias["config"]), Path(campaign["data"]), Path(campaign["data_receipt"]), full
        )[:-2]
        command_record([*common, "--out", str(full), "--max-steps", str(complete)])
        command_record([*common, "--out", str(resumed), "--max-steps", str(split)])
        manifest(campaign, config, full, complete, smoke=True)
        checkpoint(resumed, split, campaign, config)
        resume_step, pre_refresh = split, None
        if trial["arm"] in execution.RELATED_ARMS and trial["arm"] != "muon_control":
            resume_step = split + 1
            prefix = work / f"{name}_pre_refresh_continuous"
            command_record(
                [
                    *common,
                    "--out",
                    str(resumed),
                    "--max-steps",
                    str(resume_step),
                    "--resume",
                    str(resumed / "checkpoints" / f"step_{split:06d}"),
                ]
            )
            command_record([*common, "--out", str(prefix), "--max-steps", str(resume_step)])
            manifest(campaign, config, prefix, resume_step, smoke=True)
            checkpoint(resumed, resume_step, campaign, config)
            pre_refresh = {"tensor_comparisons": 8, "max_absolute_error": 0.0}
        command_record(
            [
                *common,
                "--out",
                str(resumed),
                "--max-steps",
                str(complete),
                "--resume",
                str(resumed / "checkpoints" / f"step_{resume_step:06d}"),
            ]
        )
        manifest(campaign, config, resumed, complete, smoke=True)
        smokes[name] = {
            "out": str(full),
            "completed_steps": complete,
            "split_steps": split,
            "continuation": {"tensor_comparisons": 8, "max_absolute_error": 0.0},
            "pre_refresh_continuation": pre_refresh,
        }
    save(
        receipt_path,
        {
            "wave_name": wave["wave_name"],
            "source_sha256": sources,
            "config_sha256": stage["config_sha256"],
            "data_receipt_sha256": campaign["data_receipt_sha256"],
            "environment": execution.ENVIRONMENT,
            "checks": dict.fromkeys(execution.REQUIRED_CHECKS, True),
            "evidence": {"cpu_test_exit_code": 0, "gpu_commands": commands, "real_data_smokes": smokes, **probes},
        },
    )


def complete_trials(campaign, stage):
    trial_plan = execution.validate_trials(
        Path(campaign["root"]), stage["trials"], [Path(path) for path in stage["validation_receipts"]]
    )
    children = []
    for trial in trial_plan["trials"]:
        directory, config = Path(trial["out"]), read(Path(trial["config"]))
        steps, batch = config["training"]["steps"], config["training"]["batch_tokens"]
        manifest(campaign, config, directory, steps, smoke=False)
        weights = config["data"]["mixture_weights"]
        losses = {f"heldout/{name}": 3.0 for name in weights}
        selection = sum(weight * losses[f"heldout/{name}"] for name, weight in weights.items()) / sum(weights.values())
        rows = [
            {
                "step": step,
                "tokens": step * batch,
                "train_loss": 4.0,
                "step_seconds": 1.0,
                "elapsed_seconds": step + 10,
                "end_to_end_seconds": step + 11,
            }
            for step in range(1, steps + 1)
        ]
        rows[-1].update(heldout_losses=losses, selection_loss=selection, c4_loss=3.0)
        (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        original = next(item for item in campaign["trials"] if item["name"] == trial["name"])
        children.append(
            {
                "name": trial["name"],
                "status": "complete",
                "completion": campaign_module._trial_status(campaign, original),
            }
        )
    save(
        Path(stage["worker_state"]),
        {"status": "complete", "completed_trials": len(children), "plan": trial_plan, "children": children},
    )


def specifications(root, campaign):
    wave = next(wave for wave in campaign["waves"] if wave["profile"] == "controlled" and wave["lr_index"] == 1)
    return [
        {
            "kind": "validation",
            "wave_name": wave["wave_name"],
            "work": str(root / "validation/new"),
            "receipt": str(root / "preparation/validation.json"),
            "allocation_state": str(root / "preparation/validation-owned.json"),
            "allocation_log": str(root / "preparation/validation-owned.log"),
        },
        {
            "kind": "short-trials",
            "trials": wave["trials"][:2],
            "validation_receipts": [str(root / "preparation/validation.json")],
            "worker_state": str(root / "preparation/worker.json"),
            "allocation_state": str(root / "preparation/trials-owned.json"),
            "allocation_log": str(root / "preparation/trials-owned.log"),
        },
    ]


@pytest.fixture
def planned(campaign):
    root, source, record = campaign
    path = root / "preparation/sequence.json"
    plan = sequence.prepare_plan(root, specifications(root, record), path, node="28.216.17.9")
    return root, source, record, path, plan


def process_boundary(monkeypatch, campaign, plan, *, fault=None):
    events = []
    monkeypatch.setattr(sequence.allocation, "local_node", lambda: "28.216.17.9")

    def run(command):
        index = len(events)
        stage = plan["binding"]["stages"][index]
        if index:
            previous = read(Path(plan["binding"]["stages"][index - 1]["allocation_state"]))
            assert previous["keeper_restore"] == 0
            assert previous["status"] == "complete"
        assert command == sequence.wrapper_command(stage)
        if stage["kind"] == "validation":
            validation_receipt(campaign, stage)
        else:
            complete_trials(campaign, stage)
        log = Path(stage["allocation_log"])
        log.write_text("offline allocation completed\n")
        allocation = {
            "status": "complete",
            "command": stage["command"],
            "log": str(log),
            "log_sha256": sha256(log),
            "source_sha256": {
                name: digest for name, digest in campaign["source_sha256"].items() if name.startswith("experiment/")
            },
            "exit_code": 0,
            "cleanup_exit_code": 0,
            "keeper_restore": 0,
            "keeper_release_complete": True,
            "keeper": {"pid": 1000 + index * 10, "devices": list(range(8))},
            "restored_keeper": {"pid": 1010 + index * 10, "devices": list(range(8))},
        }
        exit_code = 0
        if index == 0:
            if fault == "nonzero":
                exit_code = 7
                allocation.update(status="needs_attention", exit_code=7)
            elif fault == "needs_attention":
                allocation["status"] = "needs_attention"
            elif fault == "restore":
                allocation["keeper_restore"] = False
            elif fault == "allocation_log":
                allocation["log_sha256"] = "0" * 64
            elif fault == "validation_gate":
                receipt = read(Path(stage["receipt"]))
                receipt["checks"]["go_muon_gpu_numerics"] = False
                save(Path(stage["receipt"]), receipt)
            elif fault == "source":
                (Path(campaign["source"]) / "experiment/train.py").write_text("# changed during stage\n")
        save(Path(stage["allocation_state"]), allocation)
        events.append(command)
        return subprocess.CompletedProcess(command, exit_code)

    class Child:
        def __init__(self, command, **kwargs):
            assert kwargs["cwd"] == Path(campaign["source"])
            assert kwargs["stdin"] == subprocess.DEVNULL
            assert kwargs["start_new_session"] is True
            self.pid = 40000 + len(events)
            self.command = command
            self.returncode = None

        def wait(self):
            if self.returncode is None:
                self.returncode = run(self.command).returncode
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(sequence.subprocess, "Popen", Child)
    return events


def cpu_event(process):
    readable, _, _ = select.select([process.stdout], [], [], 15)
    assert readable, "CPU stage did not acknowledge its lifecycle transition"
    line = process.stdout.readline()
    assert line, f"CPU stage exited without acknowledgement: {process.poll()}"
    return json.loads(line)


@pytest.mark.parametrize("cancel_signal", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_run_stage_cancellation_joins_restored_wrapper_and_reaped_worker(tmp_path, cancel_signal):
    program = tmp_path / "cpu_stage.py"
    program.write_text(CPU_STAGE)
    os.mkfifo(tmp_path / "restore.fifo")
    gate = os.open(tmp_path / "restore.fifo", os.O_RDWR | os.O_NONBLOCK)
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop(sequence.allocation.MARKER, None)
    coordinator = None
    handles = {}
    released = False
    try:
        coordinator = subprocess.Popen(
            [sys.executable, "-B", str(program), "coordinator", str(tmp_path), str(Path(sequence.__file__).parent)],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        handles["coordinator"] = sequence.allocation.pidfd(coordinator.pid)
        started = cpu_event(coordinator)
        assert started["event"] == "started"
        assert read(tmp_path / "started.json") == {
            "wrapper_pid": started["wrapper_pid"],
            "worker_pid": started["worker_pid"],
        }
        for name in ("wrapper", "worker"):
            handles[name] = sequence.allocation.pidfd(started[f"{name}_pid"])
        signal.pidfd_send_signal(handles["coordinator"], cancel_signal)
        waiting = cpu_event(coordinator)
        assert waiting["event"] == "restore_waiting"
        assert waiting["worker_pid"] == started["worker_pid"]
        assert waiting["worker_exit_code"] == 0
        assert select.select([handles["worker"]], [], [], 0)[0]
        assert not (Path("/proc") / str(started["worker_pid"])).exists()
        assert not (tmp_path / "restored.json").exists()

        # The restore gate keeps the wrapper alive while cancellation is repeated.
        signal.pidfd_send_signal(handles["coordinator"], signal.SIGINT)
        signal.pidfd_send_signal(handles["coordinator"], signal.SIGTERM)
        assert not select.select([handles["coordinator"], handles["wrapper"]], [], [], 0.5)[0]
        assert coordinator.poll() is None
        assert not (tmp_path / "cancelled.json").exists()
        os.write(gate, b"1")
        released = True
        stdout, stderr = coordinator.communicate(timeout=15)
        assert coordinator.returncode == 73, stderr
        assert read(tmp_path / "restored.json") == {
            "wrapper_pid": started["wrapper_pid"],
            "worker_pid": started["worker_pid"],
            "forwarded_signal": signal.SIGTERM,
            "worker_exit_code": 0,
        }
        assert read(tmp_path / "cancelled.json") == {
            "restored": True,
            "worker_absent": True,
            "wrapper_absent": True,
        }
        assert [json.loads(line)["event"] for line in stdout.splitlines()] == ["restored", "cancelled"]
        assert select.select([handles["wrapper"]], [], [], 0)[0]
    finally:
        if not released:
            os.write(gate, b"1")
        for name in ("worker", "wrapper", "coordinator"):
            if name in handles:
                try:
                    signal.pidfd_send_signal(handles[name], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if coordinator is not None:
            if "coordinator" not in handles and coordinator.poll() is None:
                coordinator.terminate()
            try:
                coordinator.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                coordinator.kill()
                coordinator.communicate(timeout=5)
        for handle in handles.values():
            os.close(handle)
        os.close(gate)


def test_preparation_binds_exact_commands_without_launching_or_creating_stage_outputs(campaign):
    root, _, record = campaign
    path = root / "preparation/sequence.json"
    before = {str(item): sha256(item) for item in root.rglob("*") if item.is_file()}
    plan = sequence.prepare_plan(root, specifications(root, record), path, node="28.216.17.9")
    assert read(path) == plan
    assert sequence.require_plan(path) == plan
    first = plan["binding"]["stages"][0]
    assert first["command"] == [
        execution.PYTHON,
        "-m",
        "experiment.validate",
        "--gpu",
        "--data",
        record["data"],
        "--data-receipt",
        record["data_receipt"],
        "--work",
        first["work"],
        "--receipt",
        first["receipt"],
        "--wave-manifest",
        next(wave["manifest"] for wave in record["waves"] if wave["wave_name"] == first["wave_name"]),
    ]
    assert {str(item): sha256(item) for item in root.rglob("*") if item.is_file() and item != path} == before
    assert not Path(plan["state"]).exists()
    assert all(not Path(value).exists() for stage in plan["binding"]["stages"] for value in stage["outputs"])


def test_validation_then_trials_joins_and_restores_every_stage_before_next(planned, monkeypatch):
    _, _, record, path, plan = planned
    events = process_boundary(monkeypatch, record, plan)
    result = sequence.run_sequence(path)
    assert len(events) == 2
    assert result["status"] == "complete"
    assert result["completed_stages"] == 2
    assert read(Path(plan["state"])) == result
    assert all(stage["allocation"]["keeper_restore"] == 0 for stage in result["stages"])
    assert all(stage["status"] == "complete" for stage in result["stages"])
    with pytest.raises(FileExistsError):
        sequence.run_sequence(path)
    assert len(events) == 2


@pytest.mark.parametrize(
    "fault", ["nonzero", "needs_attention", "restore", "allocation_log", "validation_gate", "source"]
)
def test_failure_halts_without_retry_or_starting_the_next_stage(planned, monkeypatch, fault):
    _, _, record, path, plan = planned
    events = process_boundary(monkeypatch, record, plan, fault=fault)
    with pytest.raises((RuntimeError, ValueError)):
        sequence.run_sequence(path)
    result = read(Path(plan["state"]))
    assert len(events) == 1
    assert result["status"] == "needs_attention"
    assert result["completed_stages"] == 0
    assert Path(plan["binding"]["stages"][0]["allocation_state"]).exists()
    assert Path(plan["binding"]["stages"][0]["receipt"]).exists()
    assert not Path(plan["binding"]["stages"][1]["allocation_state"]).exists()


@pytest.mark.parametrize("damage", ["source", "config", "plan_command", "later_log", "wrong_node"])
def test_stale_binding_or_existing_later_outputs_refuses_before_allocation(planned, monkeypatch, damage):
    _, source, record, path, plan = planned
    if damage == "source":
        (source / "experiment/train.py").write_text("# stale source\n")
    elif damage == "config":
        config = Path(record["trials"][0]["config"])
        payload = read(config)
        payload["optimizer"]["lr"] *= 2
        save(config, payload)
    elif damage == "plan_command":
        payload = read(path)
        payload["binding"]["stages"][0]["command"].append("--unexpected")
        save(path, payload)
    elif damage == "later_log":
        log = Path(plan["binding"]["stages"][1]["allocation_log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("preserve earlier evidence\n")
    events = process_boundary(monkeypatch, record, plan)
    if damage == "wrong_node":
        monkeypatch.setattr(sequence.allocation, "local_node", lambda: "28.216.17.80")
    with pytest.raises((ValueError, FileExistsError, RuntimeError)):
        sequence.run_sequence(path)
    assert events == []
    assert not Path(plan["state"]).exists()


@pytest.mark.parametrize("damage", ["full_stage", "duplicate_trial", "missing_coverage", "overlap", "existing_work"])
def test_unreleased_or_ambiguous_stages_never_publish_a_plan(campaign, damage):
    root, _, record = campaign
    specs = specifications(root, record)
    if damage == "full_stage":
        specs[1]["kind"] = "full-trials"
    elif damage == "duplicate_trial":
        specs[1]["trials"].append(specs[1]["trials"][0])
    elif damage == "missing_coverage":
        specs[1]["trials"][0] = specs[1]["trials"][0].replace("lr1", "lr2")
    elif damage == "overlap":
        specs[1]["allocation_log"] = specs[0]["allocation_log"]
    else:
        Path(specs[0]["work"]).mkdir(parents=True)
    path = root / "preparation/sequence.json"
    with pytest.raises((ValueError, FileExistsError)):
        sequence.prepare_plan(root, specs, path, node="28.216.17.9")
    assert not path.exists()
