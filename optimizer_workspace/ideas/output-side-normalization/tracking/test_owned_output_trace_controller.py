# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only controller regressions with subprocess and process-filesystem fakes."""

import copy
import hashlib
import io
import json
import math
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import owned_output_handoff as handoff
import owned_output_trace_handoff as failed_handoff
import owned_output_validation as contract
import prepare_owned_output_trace as preparation
import pytest
import run_owned_output_trace as controller
import torch
from benchmark_diagonal import source_definitions

EVIDENCE_ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def corrupt_archived_source(path):
    replacement = path.with_suffix(".corrupted.tar")
    changed = False
    with tarfile.open(path) as original, tarfile.open(replacement, "x") as target:
        for member in original.getmembers():
            if not member.isfile():
                target.addfile(member)
                continue
            with original.extractfile(member) as stream:
                data = stream.read()
            if not changed:
                data += b"\narchived source corruption\n"
                member.size = len(data)
                changed = True
            target.addfile(member, io.BytesIO(data))
    replacement.replace(path)
    assert changed


@pytest.fixture
def candidate_cpu(tmp_path, monkeypatch):
    original = controller.read(EVIDENCE_ROOT / preparation.PREVIOUS)
    previous = tmp_path / "previous"
    shutil.copytree(
        Path(original["candidate"]),
        previous,
        symlinks=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    original = copy.deepcopy(original)
    original["candidate"] = str(previous)
    record(tmp_path / preparation.PREVIOUS, original)
    record(tmp_path / "run-preparation/owned-output-handoff-failure-evidence-20261117T1533.json", {})
    monkeypatch.setattr(preparation, "require_original", lambda root: original)
    candidate = tmp_path / "candidate"
    preparation.prepare(tmp_path, candidate)
    invocations = []

    def cpu_run(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert command[:3] == [controller.PYTHON, "-m", "pytest"]
        invocations.append((command, str(kwargs["cwd"])))
        kwargs["stdout"].write("CPU fixture succeeded\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", cpu_run)
    output = tmp_path / "candidate-cpu"
    preparation.cpu(candidate, output)
    return candidate, output / "receipt.json", invocations


def test_cpu_acceptance_keeps_canonical_candidate_pass_and_separate_controller_evidence(candidate_cpu, tmp_path):
    candidate, receipt, invocations = candidate_cpu
    assert controller.require_cpu(receipt)[0] == candidate
    output = tmp_path / "controller-cpu"
    result = controller.cpu(receipt, output)
    assert result["status"] == "owned_output_controller_cpu_verified_gpu_denied"
    assert invocations == [
        (preparation.cpu_command(), str(candidate)),
        (controller.controller_cpu_command(), str(Path(controller.__file__).parent.parent)),
    ]
    assert controller.read(output / "receipt.json")["candidate_cpu_receipt_sha256"] == controller.sha256(receipt)
    assert not controller.locations(tmp_path)[0].exists()


@pytest.mark.parametrize("fault", ("exit_code", "error", "archive_source"))
def test_freeze_refuses_failed_controller_cpu_before_creating_binding(candidate_cpu, tmp_path, fault):
    _, receipt, _ = candidate_cpu
    out = tmp_path / "controller-cpu"
    controller.cpu(receipt, out)
    path = out / "receipt.json"
    saved = controller.read(path)
    if fault == "exit_code":
        saved["exit_code"] = 1
    elif fault == "error":
        saved["error"] = "archival drift after CPU success"
    else:
        corrupt_archived_source(Path(saved["archive"]))
        saved["archive_sha256"] = controller.sha256(Path(saved["archive"]))
    record(path, saved)
    with pytest.raises(ValueError, match="controller CPU"):
        controller.freeze(tmp_path, receipt, controller_cpu_receipt=path)
    assert not controller.locations(tmp_path)[0].exists()
    assert not controller.locations(tmp_path)[1].exists()


@pytest.mark.parametrize(
    "fault",
    (
        "cpu_mask",
        "duplicated_pass",
        "changed_log",
        "changed_preparation",
        "changed_source",
        "cpu_error",
        "archive_source",
    ),
)
def test_cpu_acceptance_rejects_replayed_or_changed_evidence_without_binding(candidate_cpu, tmp_path, fault):
    candidate, receipt, _ = candidate_cpu
    saved = controller.read(receipt)
    if fault == "cpu_mask":
        saved["cuda_visible_devices"] = controller.MASK
    elif fault == "duplicated_pass":
        saved["command"] = [preparation.cpu_command(), preparation.cpu_command()]
    elif fault == "changed_log":
        Path(saved["log"]).write_text("replaced outcome\n")
    elif fault == "changed_preparation":
        initial = controller.read(Path(saved["preparation"]))
        initial["candidate"] = str(tmp_path / "another")
        record(Path(saved["preparation"]), initial)
    elif fault == "changed_source":
        (candidate / "experiment/block_toner.py").write_text("changed numerical source\n")
    elif fault == "cpu_error":
        saved["error"] = "archival drift after CPU success"
    else:
        corrupt_archived_source(Path(saved["archive"]))
        saved["archive_sha256"] = controller.sha256(Path(saved["archive"]))
    record(receipt, saved)
    with pytest.raises(ValueError):
        controller.require_cpu(receipt)
    assert not controller.locations(tmp_path)[0].exists()


def test_controller_cpu_timeout_preserves_failed_evidence_without_freeze(candidate_cpu, tmp_path, monkeypatch):
    _, receipt, _ = candidate_cpu

    def timeout(command, **kwargs):
        kwargs["stdout"].write("controller CPU timed out\n")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    output = tmp_path / "controller-cpu"
    with pytest.raises(RuntimeError, match="CPU failure preserved"):
        controller.cpu(receipt, output)
    saved = controller.read(output / "receipt.json")
    assert saved["status"] == "owned_output_controller_cpu_failed"
    assert saved["exit_code"] == 124 and saved["error"]
    assert controller.sha256(Path(saved["archive"])) == saved["archive_sha256"]
    assert controller.sha256(Path(saved["log"])) == saved["log_sha256"]
    with pytest.raises(ValueError, match="controller CPU"):
        controller.require_controller_cpu(output / "receipt.json", receipt)
    assert not controller.locations(tmp_path)[0].exists()


def test_cpu_hidden_mask_refuses_gpu_transfer_before_binding_or_process_io(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    def forbidden(*args, **kwargs):
        raise AssertionError("No external process or allocation operation is permitted")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)
    with pytest.raises(RuntimeError, match="mask before allocation transfer"):
        controller.validate(tmp_path)
    assert not controller.locations(tmp_path)[2].exists()


def test_old_binding_is_not_a_successor_validation_authority(tmp_path):
    record(controller.locations(tmp_path)[0], {"status": "frozen_pending_upper_lr_and_synthetic_probe"})
    with pytest.raises(ValueError, match="separate original frozen"):
        controller.binding(tmp_path)
    assert not controller.locations(tmp_path)[2].exists()


@pytest.mark.parametrize("fault", (None, "spawn", "missing", "command", "marker", "mask", "session"))
def test_detached_launch_acknowledges_exact_owner_and_never_retries_ambiguous_start(tmp_path, monkeypatch, fault):
    root = tmp_path / "root"
    binding, helpers, state = controller.locations(root)
    helpers.mkdir(parents=True)
    record(binding, {"status": controller.STATUS})
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(handoff, "PROC_ROOT", proc)
    command = [
        controller.PYTHON,
        str(helpers / controller.CONTROLLER_FILES[0]),
        "--root",
        str(root),
        "--phase",
        "validate",
    ]
    launches = []

    def popen(argv, **kwargs):
        launches.append(argv)
        if fault == "spawn":
            raise OSError("external spawn failed")
        assert kwargs["start_new_session"] and kwargs["stdin"] == subprocess.DEVNULL
        directory = proc / "700"
        directory.mkdir()
        fields = ["0"] * 20
        fields[:4] = ["S", "1", "700", "701" if fault == "session" else "700"]
        fields[19] = "12345"
        (directory / "stat").write_text("700 (controller) " + " ".join(fields))
        (directory / "cmdline").write_bytes(("\0".join(["unrelated"] if fault == "command" else argv) + "\0").encode())
        env = dict(kwargs["env"])
        if fault == "marker":
            env.pop(controller.CONTROLLER_MARKER)
        elif fault == "mask":
            env["CUDA_VISIBLE_DEVICES"] = ""
        (directory / "environ").write_bytes(("\0".join(f"{key}={value}" for key, value in env.items()) + "\0").encode())
        if fault == "missing":
            shutil.rmtree(directory)
        return SimpleNamespace(pid=700)

    monkeypatch.setattr(subprocess, "Popen", popen)
    frozen, predecessor = {"candidate": str(root / "candidate")}, {"status": "completed_ceiling"}
    intent_path = helpers.parent / "controller.json"
    if fault is None:
        result = controller.start_controller(root, frozen, predecessor)
        assert result["status"] == "acknowledged"
        assert result["pid"] == 700 and result["start_ticks"] == "12345"
        assert result["command"] == command
    else:
        with pytest.raises((RuntimeError, OSError)):
            controller.start_controller(root, frozen, predecessor)
        assert controller.read(intent_path)["status"] == "unacknowledged"
    saved = intent_path.read_bytes()
    with pytest.raises(FileExistsError):
        controller.start_controller(root, frozen, predecessor)
    assert launches == [command]
    assert intent_path.read_bytes() == saved
    assert not state.exists()


def test_detached_launch_refuses_existing_validation_evidence_before_spawn(tmp_path, monkeypatch):
    binding, helpers, state = controller.locations(tmp_path)
    helpers.mkdir(parents=True)
    record(binding, {"status": controller.STATUS})
    record(state, {"status": "needs_attention"})

    def forbidden(*args, **kwargs):
        raise AssertionError("Existing validation evidence prohibits another controller")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(FileExistsError):
        controller.start_controller(tmp_path, {"candidate": "candidate"}, {})
    assert not (helpers.parent / "controller.json").exists()
    assert controller.read(state)["status"] == "needs_attention"


@pytest.mark.parametrize("fault", (None, "live_controller", "live_child", "child_marker", "foreign_controller_marker"))
def test_failed_validator_handoff_checks_real_process_absence_and_exact_restored_marker_owner(
    tmp_path, monkeypatch, fault
):
    root, proc = tmp_path / "root", tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(handoff, "PROC_ROOT", proc)
    pidfile = tmp_path / "keeper.pid"
    pidfile.write_text("500")
    monkeypatch.setattr(handoff, "KEEPER_PID", pidfile)
    helpers = root / "controllers/failed/tracking"
    helpers.mkdir(parents=True)
    binding_path = root / "run-preparation/owned-output-handoff-binding.json"
    record(binding_path, {"root": str(root), "node": controller.NODE, "helpers": str(helpers)})
    marker, child_marker = "failed-controller", "failed-child"

    def process(pid, parent, argv, environment):
        directory = proc / str(pid)
        directory.mkdir()
        fields = ["0"] * 20
        fields[:4] = ["S", str(parent), str(pid), str(pid)]
        fields[19] = str(pid)
        (directory / "stat").write_text(f"{pid} (fixture) " + " ".join(fields))
        (directory / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
        (directory / "environ").write_bytes(("\0".join(environment) + "\0").encode())

    for pid in range(500, 509):
        argv = (
            [handoff.PYTHON, handoff.TORCHRUN, "--standalone", "--nproc-per-node=8", handoff.WORKLOAD]
            if pid == 500
            else [handoff.PYTHON, "-u", handoff.WORKLOAD]
        )
        process(
            pid,
            1 if pid == 500 else 500,
            [*argv, *handoff.WORKLOAD_ARGS],
            [f"{failed_handoff.CONTROLLER_MARKER}={marker}"],
        )
    for label, pid in (("live_controller", 99), ("live_child", 100), ("foreign_controller_marker", 900)):
        if fault == label:
            process(
                pid,
                1,
                ["unrelated"],
                [f"{failed_handoff.CONTROLLER_MARKER}={marker}"] if pid == 900 else [],
            )
    if fault == "child_marker":
        (proc / "501/environ").write_bytes(f"{failed_handoff.CHILD_MARKER}={child_marker}\0".encode())
    digest = controller.sha256(binding_path)
    record(
        root / "run-preparation/owned-output-handoff-state.json",
        {
            "binding_sha256": digest,
            "controller_pid": 99,
            "controller_start_ticks": "99",
            "child_pid": 100,
            "child_marker": child_marker,
            "status": "needs_attention",
            "exit_code": 1,
            "cleanup_exit_code": 1,
            "keeper_restore": 0,
        },
    )
    record(
        helpers.parent / "controller.json",
        {"binding_sha256": digest, "pid": 99, "start_ticks": "99", "marker": marker, "status": "acknowledged"},
    )

    def check_output(command, **kwargs):
        if command == ["hostname", "-I"]:
            return controller.NODE
        if "--query-gpu=index,uuid,name" in command:
            return "\n".join(f"{index}, GPU-{index}, NVIDIA A100-SXM4-40GB" for index in range(8))
        return "\n".join(f"{501 + index}, GPU-{index}" for index in range(8))

    monkeypatch.setattr(subprocess, "check_output", check_output)
    if fault is None:
        result = failed_handoff.probe(root)
        assert result["status"] == "failed_validator_absent_exact_restored_keeper_verified"
        assert result["inherited_controller_marker_pids"] == list(range(500, 509))
    else:
        with pytest.raises(RuntimeError):
            failed_handoff.probe(root)


def root_engine(step):
    groups = {}
    for name, count, width in (("d", 12, 512), ("f", 4, 2048)):
        groups[name] = {
            "C": torch.tensor(1.0).expand(count, width, width),
            "tol": torch.full((count,), 1e-4),
            "basis_step": [20] * count,
            "warm": True,
            "cached_root_graph_active": True,
            "SB": torch.tensor(1.0),
            "S2": torch.tensor(1.0),
            "pending": [None] * count,
        }
    return {
        "groups": groups,
        "momentum": {},
        "weight": 1 - 0.938**step,
        "steps": step,
        "t": step,
        "fallbacks": 0,
        "extra": 0,
        "refresh_install_age": 1,
    }


def checkpoint(source, frozen, directory, arm, step):
    config = controller.read(source / f"experiment/configs/{arm}_0p5m_seed0.json")
    identity = {
        "schema": 1,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "data_receipt_sha256": frozen["data_receipt_sha256"],
        "source_sha256": controller.executed_sources(source, frozen),
    }
    record(directory / "complete.json", {"step": step, "world": 8})
    for rank in range(8):
        fast = root_engine(step)
        if arm != controller.ARMS[0]:
            fast = {
                "lagged": {},
                "input": fast,
                "output": None,
                "input_diagonal": None,
                "output_diagonal": {
                    "moment": {512: torch.tensor(0.2).expand(160, 512), 2048: torch.tensor(0.3).expand(64, 2048)},
                    "weight": 1 - 0.938**step,
                },
                "momentum": {
                    (512, 512): torch.tensor(0.0).expand(16, 512, 512),
                    (2048, 512): torch.tensor(0.0).expand(8, 2048, 512),
                    (512, 2048): torch.tensor(0.0).expand(4, 512, 2048),
                },
                "steps": step,
            }
        saved = {
            "rank": rank,
            "world": 8,
            "step": step,
            "config": config,
            "identity": identity,
            "next_global_sequence": step * config["training"]["global_batch_sequences"],
            "fast": fast,
        }
        torch.save(saved, directory / f"rank_{rank}.pt")


def rank_records(step, config):
    records = []
    for rank in range(8):
        layers = list(range(rank * 4, rank * 4 + 4))
        entry = {
            "rank": rank,
            "step": step,
            "layers": layers,
            "keys": [[layer, kind] for layer in layers for kind in ("q", "k", "v", "o", "gate", "up", "down")],
            "checks": dict.fromkeys(contract.OWNED_RANK_CHECKS, True),
            "pooled_state_keys": ["moment", "weight"],
            "pooled_moment_shapes": {"512": [160, 512], "2048": [64, 2048]},
            "ema_weight": 1 - 0.938**step,
            "max_moment_error": 2e-7,
            "max_factor_error": 3e-7,
        }
        entry.update(local_rank=rank, gpu_uuid=f"GPU-fake-{rank}", gpu="NVIDIA A100", capability=[8, 0])
        entry["owned_shapes"] = [
            {"shape": [512, 512], "count": 16},
            {"shape": [2048, 512], "count": 8},
            {"shape": [512, 2048], "count": 4},
        ]
        entry["checks"] = dict.fromkeys(contract.OWNED_SYNTHETIC_CHECKS, True)
        entry["update_stack_records"] = []
        for arm in controller.ARMS[2:]:
            for shape, kinds in (
                ([16, 512, 512], ("q", "k", "v", "o")),
                ([8, 2048, 512], ("gate", "up")),
                ([4, 512, 2048], ("down",)),
            ):
                entry["update_stack_records"].append(
                    {
                        "step": 24,
                        "arm": arm,
                        "shape": shape,
                        "keys": [[layer, kind] for layer in layers for kind in kinds],
                        "ns_mode": config["optimizer"]["ns_mode"],
                        "lr": config["optimizer"]["lr"],
                        "weight_decay": config["optimizer"]["weight_decay"],
                        "checks": (
                            dict.fromkeys(contract.OWNED_STACK_CHECKS, True)
                            | {"null_update_exact": arm == controller.ARMS[3]}
                        ),
                        "max_errors": {"same_moment_update": 0.0, "restored_update": 0.0, "output_equation": 1e-7},
                    }
                )
        records.append(entry)
    return records


def captured_records(step, arm):
    weight = 0.0
    for _ in range(step):
        weight = 0.938 * weight + 1 - 0.938
    upstream = {
        "upstream_product_parity_passed": True,
        "upstream_scheduled_scalars_passed": True,
        "upstream_pinned_input_exact": True,
        "upstream_output_equation_passed": True,
        "upstream_null_exact": arm == controller.ARMS[3],
        "upstream_owned_stacks_checked": 3,
        "upstream_output_equation_max_error": 1e-7,
    }
    shapes = [{"width": 512, "moment_shape": [160, 512]}, {"width": 2048, "moment_shape": [64, 2048]}]
    errors = {"moment_torch": 1e-7, "moment_fp64": 2e-7, "factor_same_state": 0.0, "factor_fp64": 3e-7, "weight": 0.0}
    records = [
        {
            "schema": 1,
            "rank": rank,
            "world": 8,
            "step": step,
            "model": {"layers": 32, "hidden": 512, "mlp": 2048},
            "backend": "triton",
            "device_type": "cuda",
            "state_schema": ["moment", "weight"],
            "weight": weight,
            "global_count": 224,
            "global_keys": [[layer, kind] for layer in range(32) for kind in ("q", "k", "v", "o", "gate", "up", "down")],
            "global_shapes": copy.deepcopy(shapes),
            "owned_keys": [
                [layer, kind]
                for layer in range(rank * 4, rank * 4 + 4)
                for kind in ("q", "k", "v", "o", "gate", "up", "down")
            ],
            "owned_count": 28,
            "owned_shapes": [
                {"shape": [512, 512], "count": 16},
                {"shape": [2048, 512], "count": 8},
                {"shape": [512, 2048], "count": 4},
            ],
            "checks": dict.fromkeys(contract.OWNED_CAPTURE_CHECKS, True),
            "max_errors": errors.copy(),
            "upstream": upstream.copy(),
        }
        for rank in range(8)
    ]
    return {
        **upstream,
        "owned_output_parity_passed": True,
        "owned_output_world": 8,
        "owned_output_step": step,
        "owned_output_global_count": 224,
        "owned_output_global_shapes": shapes,
        "owned_output_max_errors": errors | {"upstream_output_equation": 1e-7},
        "owned_output_records": records,
    }


@pytest.fixture
def validation_evidence(tmp_path):
    source = tmp_path / "candidate"
    experiment = source / "experiment"
    experiment.mkdir(parents=True)
    helpers = Path(controller.__file__).parent
    for name in ("owned_output_validation.py", "upstream_validation.py"):
        shutil.copy2(helpers / name, experiment / name)
    predecessor = EVIDENCE_ROOT / "candidates/owned-output-observed-formatted-20261117/experiment"
    for name in ("records.py", "validate.py"):
        shutil.copy2(predecessor / name, experiment / name)
    (experiment / "block_toner.py").write_text(
        'BLOCK_KINDS = ("q","k","v","o","gate","up","down")\n'
        'HIDDEN_OUTPUTS = ("q","k","v","o","down")\nMLP_OUTPUTS = ("gate","up")\n'
    )
    record(experiment / "upstream_manifest.json", {"revision": "fixture", "files": {}})
    for arm in controller.ARMS:
        record(
            experiment / f"configs/{arm}_0p5m_seed0.json",
            {
                "model": {"layers": 32, "hidden": 512, "mlp": 2048},
                "training": {"world_size": 8, "global_batch_sequences": 128},
                "statistics": {"momentum": 0.938},
                "experiment": {"arm": arm},
                "optimizer": {"ns_mode": "tpu", "lr": 0.02262741696, "weight_decay": 0.1},
            },
        )
    files = {str(path.relative_to(source)): controller.sha256(path) for path in experiment.rglob("*") if path.is_file()}
    sources = {name: digest for name, digest in files.items() if name.endswith(".py")}
    work = tmp_path / "validation/owned_output_trace"
    work.mkdir(parents=True)
    commands = controller.validation_commands(controller.PYTHON, source, tmp_path, work)
    frozen = {
        "candidate": str(source),
        "files_sha256": files,
        "source_sha256": sources,
        "data_receipt_sha256": "bound-data",
        "commands": commands,
    }
    owner = {"controller_pid": 1234, "controller_start_ticks": "5678"}
    record(controller.locations(tmp_path)[2], owner)
    executions = []
    for index, command in enumerate(commands):
        log = work / f"command-{index}.log"
        log.write_text("fake successful command\n")
        executions.append({"command": command, "exit_code": 0, "log": str(log), "log_sha256": controller.sha256(log)})
    record(
        work / "gpu_checks.json",
        {
            "world": 2,
            "source_sha256": sources,
            "checks": dict.fromkeys(controller.SYNTHETIC_GATES, True),
        },
    )
    config = controller.read(experiment / f"configs/{controller.ARMS[2]}_0p5m_seed0.json")
    owned = {
        "kind": "owned_output_synthetic_gpu_correctness",
        "world": 8,
        "updates": 24,
        "owner": owner,
        "source_sha256": sources,
        "config_sha256": contract.owned_config_sha256(config),
        "rtol": 1e-5,
        "atol": 1e-6,
        "resume": {"checkpoint_step": 22, "final_step": 24},
        "environment": controller.VALIDATION_ENVIRONMENT,
        "checks": dict.fromkeys(contract.OWNED_SYNTHETIC_CHECKS, True),
        "rank_records": rank_records(24, config),
    }
    owned_path = work / "owned_output_gpu_checks.json"
    record(owned_path, owned)
    histories = source_definitions(
        experiment / "validate.py",
        ("require_cached_root_checkpoint", "require_fixed_age_history"),
        {"torch": torch, "json": json},
    )
    smokes = {}
    for arm in controller.ARMS:
        full, resumed = work / f"{arm}_continuous", work / f"{arm}_resumed"
        for directory in (full, resumed):
            directory.mkdir()
            for step in (22, 24):
                checkpoint(source, frozen, directory / f"checkpoints/step_{step:06d}", arm, step)
            rows = [
                {
                    "step": step,
                    "input_refresh_installations": (
                        [{"installation_step": step - 1, "source_step": step - 2}] if step > 1 else []
                    ),
                    "diagonal_capture_parity_passed": True,
                    "outer_diagonal_update_parity_passed": True,
                    "upstream_product_parity_passed": True,
                    "upstream_scheduled_scalars_passed": True,
                    "upstream_pinned_input_exact": True,
                    "upstream_output_equation_passed": True,
                    "upstream_null_exact": arm == controller.ARMS[3],
                    "upstream_owned_stacks_checked": 3,
                    "upstream_output_equation_max_error": 1e-7,
                    **captured_records(step, arm),
                }
                for step in range(1, 25)
            ]
            (directory / "metrics_from_000000.jsonl").write_text("\n".join(json.dumps(row) for row in rows[:22]) + "\n")
            (directory / "metrics_from_000022.jsonl").write_text("\n".join(json.dumps(row) for row in rows[22:]) + "\n")
            record(
                directory / "run.json",
                {
                    "status": "execution_check_complete",
                    "completed_steps": 24,
                    "config": controller.read(experiment / f"configs/{arm}_0p5m_seed0.json"),
                    "data_receipt_sha256": "bound-data",
                    "source_sha256": sources,
                },
            )
        smokes[arm] = {
            "out": str(full),
            "cached_root_checkpoint": histories["require_cached_root_checkpoint"](
                resumed / "checkpoints/step_000022", 8
            ),
            "boundary_continuation": {
                "tensor_comparisons": 64 if arm == controller.ARMS[0] else 104,
                "max_absolute_error": 0.0,
            },
            "continuation": {"tensor_comparisons": 64 if arm == controller.ARMS[0] else 104, "max_absolute_error": 0.0},
            "fixed_age_history": {
                label: histories["require_fixed_age_history"](directory)
                for label, directory in (("continuous", full), ("resumed", resumed))
            },
            "owned_output_observations": (
                {
                    label: contract.require_owned_capture_history(directory, arm)
                    for label, directory in (("continuous", full), ("resumed", resumed))
                }
                if arm in controller.ARMS[2:]
                else None
            ),
        }
    receipt = {
        "owner": owner,
        "source_sha256": sources,
        "environment": controller.VALIDATION_ENVIRONMENT,
        "data_receipt_sha256": "bound-data",
        "config_sha256": {
            arm: controller.sha256(experiment / f"configs/{arm}_0p5m_seed0.json") for arm in controller.ARMS
        },
        "checks": dict.fromkeys(controller.GATES, True),
        "evidence": {
            "gpu_commands": executions,
            "real_data_smokes": smokes,
            "owned_gpu_receipt": str(owned_path),
            "owned_gpu_receipt_sha256": controller.sha256(owned_path),
            "owned_gpu_summary": contract.require_owned_synthetic_receipt(owned, sources, config),
        },
    }
    path = tmp_path / "run-preparation/validation-owned-output-trace-gpu.json"
    record(path, receipt)
    return tmp_path, frozen, path


def test_validation_reconstructs_all_four_trajectories_and_owned_receipts(validation_evidence):
    root, frozen, path = validation_evidence
    accepted = controller.require_validation(root, frozen)
    assert accepted == controller.read(path)
    assert accepted["evidence"]["owned_gpu_summary"]["pooled_keys_per_rank"] == 224
    assert (
        accepted["evidence"]["real_data_smokes"][controller.ARMS[3]]["owned_output_observations"]["resumed"][
            "rank_updates"
        ]
        == 192
    )


@pytest.mark.parametrize(
    "fault", ("rank7_full_moment", "shared_missing_moments", "weight", "steps", "schema", "owned_momentum", "root_only")
)
def test_validation_rejects_rank_seven_or_shared_checkpoint_corruption(validation_evidence, fault):
    root, frozen, _ = validation_evidence
    arm = controller.ARMS[2]
    for label in ("continuous", "resumed"):
        directory = root / "validation/owned_output_trace" / f"{arm}_{label}" / "checkpoints/step_000024"
        ranks = range(8) if fault == "shared_missing_moments" else (7,)
        for rank in ranks:
            path = directory / f"rank_{rank}.pt"
            saved = torch.load(path, weights_only=False)
            fast = saved["fast"]
            if fault == "rank7_full_moment":
                fast["output_diagonal"]["moment"][512] = fast["output_diagonal"]["moment"][512].clone()
                fast["output_diagonal"]["moment"][512][0, 0] += 0.001
            elif fault == "shared_missing_moments":
                fast["output_diagonal"]["moment"][512] = torch.zeros(20, 512)
                fast["output_diagonal"]["moment"][2048] = torch.zeros(8, 2048)
            elif fault == "weight":
                fast["output_diagonal"]["weight"] = 0.0
            elif fault == "steps":
                fast["steps"] = 22
            elif fault == "schema":
                fast["output_diagonal"]["factor_cache"] = {}
            elif fault == "owned_momentum":
                fast["momentum"][(512, 2048)] = torch.zeros(3, 512, 2048)
            else:
                fast["input"]["momentum"] = {"unexpected": torch.tensor(1.0)}
            torch.save(saved, path)
    with pytest.raises(ValueError, match="Owned-output"):
        controller.require_validation(root, frozen)


@pytest.mark.parametrize(
    "fault", ("generic_synthetic", "duplicate_command", "changed_log", "missing_original_gate", "rank7_history")
)
def test_validation_rejects_generic_synthetic_or_false_success_receipts(validation_evidence, fault):
    root, frozen, path = validation_evidence
    saved = controller.read(path)
    if fault == "generic_synthetic":
        owned_path = Path(saved["evidence"]["owned_gpu_receipt"])
        record(owned_path, controller.read(root / "validation/owned_output_trace/gpu_checks.json"))
    elif fault == "duplicate_command":
        saved["evidence"]["gpu_commands"][-1]["command"] = saved["evidence"]["gpu_commands"][-2]["command"]
    elif fault == "changed_log":
        Path(saved["evidence"]["gpu_commands"][-1]["log"]).write_text("changed command outcome\n")
    elif fault == "missing_original_gate":
        saved["checks"]["gpu_upstream_scalar_step"] = False
    else:
        history_path = root / "validation/owned_output_trace" / f"{controller.ARMS[3]}_resumed/metrics_from_000022.jsonl"
        rows = [json.loads(line) for line in history_path.read_text().splitlines()]
        rows[-1]["owned_output_records"][-1]["checks"]["inner_factors_absent"] = False
        history_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    record(path, saved)
    with pytest.raises(ValueError):
        controller.require_validation(root, frozen)


def test_spawn_failure_preserves_intent_and_explicit_child_mask(tmp_path, monkeypatch):
    seen = {}

    def spawn(command, **kwargs):
        seen.update(kwargs["env"])
        raise OSError("spawn unavailable")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(subprocess, "Popen", spawn)
    path = tmp_path / "state.json"
    with pytest.raises(OSError, match="spawn unavailable"):
        controller.bounded_validator(["validator"], tmp_path, tmp_path / "validator.log", {}, path)
    state = controller.read(path)
    assert state["spawn_status"] == "intent" and "child_pid" not in state
    assert seen["CUDA_VISIBLE_DEVICES"] == controller.MASK
    assert seen[controller.PROCESS_MARKER] == state["child_marker"]


def test_bounded_validator_acknowledges_fake_process_then_cleans_its_generation(tmp_path, monkeypatch):
    processes = {}
    original_read, original_iterdir = Path.read_text, Path.iterdir

    def proc_read(path, *args, **kwargs):
        if path == Path("/proc/9001/stat"):
            state = processes.get(9001)
            if state is None:
                raise FileNotFoundError(path)
            fields = [state["status"], "1", *("0" for _ in range(17)), "777", "0"]
            return "9001 (validator) " + " ".join(fields)
        return original_read(path, *args, **kwargs)

    original_bytes = Path.read_bytes

    def proc_bytes(path, *args, **kwargs):
        if path == Path("/proc/9001/cmdline"):
            return b"validator\0"
        if path == Path("/proc/9001/environ"):
            return f"{controller.PROCESS_MARKER}={processes[9001]['marker']}\0".encode()
        return original_bytes(path, *args, **kwargs)

    class Child:
        pid = 9001
        returncode = None

        def wait(self, timeout=None):
            persisted = controller.read(tmp_path / "state.json")
            assert persisted["spawn_status"] == "acknowledged"
            assert persisted["child_start_ticks"] == "777"
            processes[self.pid]["status"] = "Z"
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

    def spawn(command, **kwargs):
        processes[9001] = {"status": "S", "marker": kwargs["env"][controller.PROCESS_MARKER]}
        return Child()

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(Path, "read_text", proc_read)
    monkeypatch.setattr(Path, "read_bytes", proc_bytes)
    monkeypatch.setattr(
        Path, "iterdir", lambda path: iter([Path("/proc/9001")]) if path == Path("/proc") else original_iterdir(path)
    )
    path = tmp_path / "state.json"
    assert controller.bounded_validator(["validator"], tmp_path, tmp_path / "validator.log", {}, path) == 0
    state = controller.read(path)
    assert state["cleanup_exit_code"] == 0
    assert state["child_generation"] == {"9001": "777"}
    assert "cleanup_error" not in state


@pytest.fixture
def ceiling_evidence(tmp_path, monkeypatch):
    root = tmp_path
    directory = root / controller.CEILING
    helpers = directory / "execution-helpers"
    helpers.mkdir(parents=True)
    for name in ("run_fixed_age_tuning.py", "analyze_lr_ceiling.py", "prepare_lr_ceiling.py"):
        (helpers / name).write_text("# Frozen original observation helper fixture\n")
    archive = directory / "execution-source.tar"
    archive.write_bytes(b"original recovery archive")
    record(directory / "validation-binding.json", {"identity": "original validation"})
    release_helpers = root / "controllers/wave9-launch-20261117/tracking"
    release_helpers.mkdir(parents=True)
    (release_helpers / "upstream_production.py").write_text("# Original Wave-9 observation fixture\n")
    release_archive = root / "run-preparation/wave9-source.tar"
    release_archive.parent.mkdir(parents=True)
    release_archive.write_bytes(b"Wave-9 source fixture")
    release_path = root / "run-preparation/wave9-launch-binding.json"
    record(
        release_path,
        {
            "status": "frozen_validated_launch_release",
            "root": str(root),
            "helpers": str(release_helpers),
            "helpers_sha256": {"upstream_production.py": controller.sha256(release_helpers / "upstream_production.py")},
            "archive": str(release_archive),
            "archive_sha256": controller.sha256(release_archive),
        },
    )
    campaign_path = directory / "campaign.json"
    trials, plan, rows_by_trial = [], [], {}
    for family, node in zip(controller.CEILING_FAMILIES, controller.NODES, strict=True):
        commands = []
        for label, lr in (("4x", 0.04), ("4sqrt2x", 0.04 * 2**0.5)):
            name = f"{family}_lr_ceiling_{label}"
            config_path = directory / "configs" / f"{name}.json"
            config = {
                "optimizer": {"lr": lr},
                "training": {"steps": 320, "world_size": 8, "batch_tokens": 524288, "tokens": 167772160},
            }
            record(config_path, config)
            trials.append(
                {
                    "name": name,
                    "family": family,
                    "config": str(config_path),
                    "config_sha256": controller.sha256(config_path),
                    "matrix_lr": lr,
                    "tokens": 167772160,
                    "steps": 320,
                }
            )
            run = directory / "runs" / name
            record(run / "run.json", {"config": config, "status": "complete", "completed_steps": 320})
            rows = []
            for step in range(1, 321):
                row = {"step": step, "tokens": step * 524288, "train_loss": 3.0 - step / 1000}
                if step in (1, 41, 81, 121, 161, 201, 241, 281, 320):
                    row.update(selection_loss=3.1 - step / 1000, c4_loss=3.2 - step / 1000)
                rows.append(row)
            (run / "metrics_from_000000.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            rows_by_trial[name] = rows
            checkpoint_path = run / "checkpoints/step_000320"
            record(checkpoint_path / "complete.json", {"step": 320, "world": 8})
            for rank in range(8):
                (checkpoint_path / f"rank_{rank}.pt").write_bytes(b"completed rank fixture")
            commands.append(
                {
                    "name": name,
                    "config_sha256": controller.sha256(config_path),
                    "tokens": 167772160,
                    "out": str(run),
                    "command": ["torchrun-fixture", "--config", str(config_path)],
                }
            )
        plan.append({"family": family, "node": node, "trials": commands})
    campaign = {
        "root": str(root),
        "predecessor_wave": "wave9",
        "recovery": {"original": True},
        "predecessor_release_sha256": controller.sha256(release_path),
        "selection": {"training_window_tokens": 33554432},
        "trials": trials,
    }
    record(campaign_path, campaign)
    execution_path = directory / "execution-binding.json"
    execution = {
        "campaign": str(campaign_path),
        "campaign_sha256": controller.sha256(campaign_path),
        "helpers": str(helpers),
        "helpers_sha256": {path.name: controller.sha256(path) for path in helpers.iterdir()},
        "archive": str(archive),
        "archive_sha256": controller.sha256(archive),
        "validation_binding_sha256": controller.sha256(directory / "validation-binding.json"),
        "environment": controller.VALIDATION_ENVIRONMENT,
        "plan": plan,
    }
    record(execution_path, execution)
    observed = {
        "status": "complete",
        "all_workers_terminal": True,
        "workers": [],
        "campaign_sha256": controller.sha256(campaign_path),
        "binding_sha256": controller.sha256(execution_path),
    }
    for assignment in plan:
        receipts = []
        for planned in assignment["trials"]:
            name = planned["name"]
            log = directory / "workers" / f"{name}.log"
            log.parent.mkdir(exist_ok=True)
            log.write_text("completed original trial fixture\n")
            child = directory / "workers" / f"{name}.execution.json"
            record(
                child, {"command": planned["command"], "cleanup_exit_code": 0, "child_pid": 1111, "child_marker": name}
            )
            receipts.append(
                {
                    "name": name,
                    "command": planned["command"],
                    "status": "complete",
                    "exit_code": 0,
                    "log": str(log),
                    "log_sha256": controller.sha256(log),
                    "execution_sha256": controller.sha256(child),
                }
            )
        observed["workers"].append(
            {
                "status": "complete",
                "family": assignment["family"],
                "receipt": {
                    "status": "complete",
                    "family": assignment["family"],
                    "node": assignment["node"],
                    "binding_sha256": controller.sha256(execution_path),
                    "keeper_restore": 0,
                    "trials": receipts,
                },
            }
        )
    failed_state = root / "run-preparation/owned-output-handoff-state.json"
    record(failed_state, {"status": "needs_attention"})
    record(root / "run-preparation/owned-output-handoff-binding.json", {"status": "failed_source"})
    record(
        root / "run-preparation/owned-output-handoff-failure-evidence-20261117T1533.json",
        {"status": "preserved_failure"},
    )
    actual = {
        "status": "complete",
        "trials": [
            {
                "name": trial["name"],
                "complete": True,
                "observed_tokens": 167772160,
                "matrix_lr": trial["matrix_lr"],
                "training_windows": [
                    {
                        "end_tokens": (index + 1) * 33554432,
                        "mean": (
                            sum(row["train_loss"] for row in rows_by_trial[trial["name"]][index * 64 : (index + 1) * 64])
                            / 64
                        ),
                    }
                    for index in range(5)
                ],
                "late_evaluation_tokens": {
                    metric: [step * 524288 for step in (161, 201, 241, 281, 320)]
                    for metric in ("selection_loss", "c4_loss")
                },
            }
            for trial in trials
        ],
    }
    # Use the exact independently evaluated fsum, rather than rounded accumulated addition.
    for trial in actual["trials"]:
        for index, window in enumerate(trial["training_windows"]):
            window["mean"] = (
                math.fsum(row["train_loss"] for row in rows_by_trial[trial["name"]][index * 64 : (index + 1) * 64]) / 64
            )
    record(directory / "analysis.json", actual)
    cloud = {
        "status": "verified",
        "visibility": "PRIVATE",
        "project": "fixture/output-side-normalization-marin",
        "campaign_sha256": controller.sha256(campaign_path),
        "execution_binding_sha256": controller.sha256(execution_path),
        "runs": {},
    }
    for trial in trials:
        run = directory / "runs" / trial["name"]
        checkpoint_path = run / "checkpoints/step_000320"
        cloud["runs"][trial["name"]] = {
            "status": "verified",
            "manifest_sha256": controller.sha256(run / "run.json"),
            "metrics_sha256": controller.sha256(run / "metrics_from_000000.jsonl"),
            "checkpoint_receipt_sha256": controller.sha256(checkpoint_path / "complete.json"),
            "checkpoint_rank_bytes": [(checkpoint_path / f"rank_{rank}.pt").stat().st_size for rank in range(8)],
            "campaign_sha256": cloud["campaign_sha256"],
            "execution_binding_sha256": cloud["execution_binding_sha256"],
            "trial_config_sha256": trial["config_sha256"],
            "verification": {
                "state": "FINISHED",
                "point_counts": {
                    "tokens": 320,
                    "train/loss": 320,
                    "eval/selection_loss": 9,
                    "eval/c4_loss": 9,
                },
            },
        }
    record(directory / "swanlab-upload.json", cloud)
    io = {"ssh_error": False, "absent": True, "observed": observed, "analysis": actual, "calls": []}

    def run(command, **kwargs):
        io["calls"].append(command)
        assert kwargs["capture_output"] and kwargs["text"] and kwargs["check"]
        if command[0] == "ssh":
            if io["ssh_error"]:
                raise subprocess.CalledProcessError(1, command, stderr="SSH unavailable")
            if "owned_output_trace_handoff.py" in command[-1]:
                payload = {
                    "status": "failed_validator_absent_exact_restored_keeper_verified",
                    "node": controller.NODE,
                    **controller.failed_predecessor_identity(root),
                    "original_handles_absent": io["absent"],
                    "training_child_markers_absent": True,
                    "keeper": {"generation": {str(pid): {} for pid in range(9)}, "occupied": list(range(8))},
                }
            else:
                family = next(family for family in controller.CEILING_FAMILIES if command[-1].endswith(family))
                payload = {
                    "status": "original_training_absent_exact_keeper_verified",
                    "family": family,
                    "node": controller.NODES[controller.CEILING_FAMILIES.index(family)],
                    "original_handles_absent": io["absent"],
                    "training_child_markers_absent": True,
                    "keeper": {"generation": {str(pid): {} for pid in range(9)}, "occupied": list(range(8))},
                }
        elif "--phase" in command and command[-1] == "check":
            payload = {"status": "frozen"}
        elif command[:2] == [controller.PYTHON, "-c"] and "snapshot" in command[2]:
            payload = io["observed"]
        elif command[:2] == [controller.PYTHON, "-c"] and "analyze_lr_ceiling" in command[2]:
            payload = io["analysis"]
        else:
            raise AssertionError("Unexpected process outside observation boundary")
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(subprocess, "run", run)
    prerequisite = controller.require_ceiling_binding(root)
    return root, {"ceiling": prerequisite, "failed_predecessor": controller.failed_predecessor_identity(root)}, io


def test_completed_ceiling_requires_fresh_all_node_absence_and_matching_final_provenance(ceiling_evidence):
    root, frozen, io = ceiling_evidence
    result = controller.require_completed_ceiling(root, frozen)
    assert result["status"] == "completed_ceiling_workers_restored_absent_cloud_verified"
    assert {command[5] for command in io["calls"] if command[0] == "ssh"} == {
        f"root@{node}" for node in controller.NODES
    }
    assert not controller.locations(root)[2].exists()


@pytest.mark.parametrize(
    "fault",
    (
        "active_worker",
        "keeper_bool",
        "cleanup",
        "ssh",
        "live_original",
        "partial_window",
        "lr",
        "late_eval",
        "cloud_points",
        "cloud_checkpoint",
    ),
)
def test_ceiling_refuses_incomplete_or_changed_recovery_without_allocation_transfer(ceiling_evidence, fault):
    root, frozen, io = ceiling_evidence
    directory = root / controller.CEILING
    if fault == "active_worker":
        io["observed"]["workers"][-1]["status"] = "running"
    elif fault == "keeper_bool":
        io["observed"]["workers"][-1]["receipt"]["keeper_restore"] = False
    elif fault == "cleanup":
        trial = io["observed"]["workers"][-1]["receipt"]["trials"][0]
        child = directory / "workers" / f"{trial['name']}.execution.json"
        saved = controller.read(child)
        saved["cleanup_exit_code"] = False
        record(child, saved)
        trial["execution_sha256"] = controller.sha256(child)
    elif fault == "ssh":
        io["ssh_error"] = True
    elif fault == "live_original":
        io["absent"] = False
    elif fault in ("partial_window", "lr", "late_eval"):
        row = io["analysis"]["trials"][-1]
        if fault == "partial_window":
            row["training_windows"][-1]["end_tokens"] -= 524288
        elif fault == "lr":
            row["matrix_lr"] /= 2
        else:
            row["late_evaluation_tokens"]["selection_loss"].pop(0)
        record(directory / "analysis.json", io["analysis"])
    else:
        cloud_path = directory / "swanlab-upload.json"
        cloud = controller.read(cloud_path)
        entry = next(iter(cloud["runs"].values()))
        if fault == "cloud_points":
            entry["verification"]["point_counts"]["train/loss"] = 319
        else:
            entry["checkpoint_receipt_sha256"] = "replaced receipt"
        record(cloud_path, cloud)
    with pytest.raises((ValueError, RuntimeError, subprocess.CalledProcessError)):
        controller.require_completed_ceiling(root, frozen)
    assert not controller.locations(root)[2].exists()
    assert not (root / "validation/owned_output_trace").exists()
