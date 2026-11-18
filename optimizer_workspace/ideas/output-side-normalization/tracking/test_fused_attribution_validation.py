# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU protocol and exact restored-keeper whitelist regressions."""

import copy
import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from analyze_tuning import executed_sources
from fused_attribution_validation import (
    HANDOFF_STATUS,
    INTERRUPTED_MANIFEST,
    checkpoint_evidence,
    failed_profiler_evidence,
    loss_continuation_evidence,
    model_shapes,
    plan,
    require_capture_history,
    require_failed_profiler_handoff,
    save_interrupted_manifest,
    trajectory_evidence,
)
from fused_attribution_validation_contract import (
    BATCH_TOKENS,
    FROZEN_STATUS,
    PREPARATION,
    PREPARED_STATUS,
    SCHEDULE,
    WORLD,
    config_name,
    locations,
    predecessor_locations,
    trial_specs,
)
from fused_attribution_validation_test_support import record, write_failed_profiler
from launch_wave5 import PYTHON, read, sha256
from wave7_release import candidate_inventory

PARENT = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/"
    "candidates/fused-attribution-full-schedule-20261118T0257"
)


def config(spec):
    return {
        "model": {"sequence_length": 4096},
        "training": {
            "global_batch_sequences": 128,
            "steps": 5120,
            "tokens": 2684354560,
            "batch_tokens": BATCH_TOKENS,
            "world_size": 8,
            "seed": 0,
        },
        "experiment": {"arm": spec["family"], "refresh_install_age": 1, "diagonal_backend": "triton"},
        "optimizer": {"lr": spec["matrix_lr"], "ns_mode": "tpu", "ns_steps": 10},
        "runtime": {"compile": "default", "attention": "FA2", "head": "compiled"},
        "root": {"graphs": True},
        "schedule": SCHEDULE,
    }


def write_configs(candidate):
    for spec in trial_specs():
        record(candidate / spec["config_name"], config(spec))


def test_nine_actual_config_plan_is_independent_and_no_authorization(tmp_path):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    write_configs(candidate)
    before = {path: path.read_bytes() for path in candidate.rglob("*.json")}
    result = plan(candidate, root, locations(root)[3])
    assert result["planned_validation_tokens"] == 226492416
    assert len(result["trials"]) == 9
    commands = [command for trial in result["trials"] for command in trial["commands"]]
    assert len(commands) == 27
    assert len({tuple(command) for command in commands}) == 27
    assert len({trial["continuous"] for trial in result["trials"]}) == 9
    for trial in result["trials"]:
        assert [command[command.index("--max-steps") + 1] for command in trial["commands"]] == ["24", "22", "24"]
        assert all(command[command.index("--execution-boundary") + 1] == "22" for command in trial["commands"])
        assert all(
            command[command.index("--launch-release") + 1] == str(locations(root)[0]) for command in trial["commands"]
        )
        assert "--resume" not in trial["commands"][0] and "--resume" not in trial["commands"][1]
        assert trial["commands"][2][-1] == str(Path(trial["resumed"]) / "checkpoints/step_000022")
        assert all(
            ("--verify-outer-diagonal-updates" in command) == (trial["family"] != "muont")
            for command in trial["commands"]
        )
        assert not any(trial["checks"].values())
    assert result["gpu_commands"] == []
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not root.exists()


@pytest.mark.parametrize("fault", ("lr", "schedule", "ns", "graphs", "short", "alias"))
def test_actual_config_plan_refuses_grid_or_runtime_drift(tmp_path, fault):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    write_configs(candidate)
    path = candidate / config_name("null_leftdiag_fused", "upper")
    value = json.loads(path.read_text())
    if fault == "lr":
        value["optimizer"]["lr"] *= 1.01
    elif fault == "schedule":
        value["schedule"]["stable_steps"] = 1023
    elif fault == "ns":
        value["optimizer"]["ns_mode"] = "f32"
    elif fault == "graphs":
        value["root"]["graphs"] = False
    elif fault == "short":
        value["training"]["steps"] = 24
    else:
        target = tmp_path / "alias.json"
        path.rename(target)
        path.symlink_to(target)
    if fault != "alias":
        record(path, value)
    with pytest.raises(ValueError):
        plan(candidate, root, locations(root)[3])


@pytest.mark.parametrize(
    "fault", ("none", "extra_marker", "reused_keeper", "child_marker", "live_controller", "devices", "rank_gpu_swap")
)
def test_failed_profiler_handoff_allows_only_exact_inherited_keeper_controller_markers(tmp_path, fault):
    root = tmp_path / "root"
    write_failed_profiler(root)
    expected = failed_profiler_evidence(root)
    current = copy.deepcopy(expected["restored_keeper"])
    marked = {int(pid): process for pid, process in current["generation"].items()}
    child, handles = {}, {201: None, 202: None}
    if fault == "extra_marker":
        marked[999] = current["generation"]["101"] | {"pid": 999}
    elif fault == "reused_keeper":
        current["generation"]["101"]["start_ticks"] = "reused"
    elif fault == "child_marker":
        child = {101: marked[101]}
    elif fault == "live_controller":
        handles[201] = marked[101]
    elif fault == "devices":
        current["devices"][0][1] = "GPU-replaced"
        current["occupied"][0][1] = "GPU-replaced"
    elif fault == "rank_gpu_swap":
        current["occupied"][0][1], current["occupied"][1][1] = current["occupied"][1][1], current["occupied"][0][1]
    if fault == "none":
        require_failed_profiler_handoff(expected, current, marked, child, handles)
        assert expected["status"] == HANDOFF_STATUS
    else:
        with pytest.raises((RuntimeError, ValueError)):
            require_failed_profiler_handoff(expected, current, marked, child, handles)


def test_terminal_serialization_rejects_rehashed_unrestored_profiler(tmp_path):
    root = tmp_path / "root"
    write_failed_profiler(root)
    path = predecessor_locations(root)[2]
    state = json.loads(path.read_text())
    state["keeper_restore"] = 1
    record(path, state)
    with pytest.raises(ValueError):
        failed_profiler_evidence(root)


@pytest.mark.parametrize("missing", (None, "output_work_retained_passed", "null_identity_outer_passed"))
def test_null_history_requires_strict_work_and_identity_gates(tmp_path, missing):
    rows = [
        {
            "step": step,
            "tokens": step * BATCH_TOKENS,
            "diagonal_capture_parity_passed": True,
            "outer_diagonal_update_parity_passed": True,
            "output_work_retained_passed": True,
            "null_identity_outer_passed": True,
        }
        for step in range(1, 25)
    ]
    if missing:
        del rows[12][missing]
    (tmp_path / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if missing:
        with pytest.raises(ValueError):
            require_capture_history(tmp_path, "null_leftdiag_fused")
    else:
        assert require_capture_history(tmp_path, "null_leftdiag_fused")["updates"] == 24


@pytest.fixture
def independent_trajectories(tmp_path):
    candidate = tmp_path / "candidate"
    shutil.copytree(PARENT, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for current in trial_specs():
        path = candidate / current["config_name"]
        value = read(path)
        value["model"].update(layers=8, hidden=4, mlp=8, vocab_size=16)
        record(path, value)
    spec = next(spec for spec in trial_specs() if spec["family"] == "null_leftdiag_fused")
    configuration = json.loads((candidate / spec["config_name"]).read_text())
    root = tmp_path / "root"
    record(root / "data/experiment_receipt.json", {"data": "fixture"})
    frozen = {
        "status": PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "files_sha256": candidate_inventory(candidate),
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "parent_metadata_sha256": {},
    }
    record(candidate / PREPARATION, frozen)
    identity = {
        "schema": 1,
        "config_sha256": hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest(),
        "source_sha256": executed_sources(candidate, frozen),
        "data_receipt_sha256": frozen["data_receipt_sha256"],
    }
    rows = [
        {
            "step": step,
            "tokens": step * BATCH_TOKENS,
            "train_loss": 4.0 - step / 1000,
            "input_refresh_installations": (
                [{"installation_step": step - 1, "source_step": step - 2}] if step >= 2 else []
            ),
            "diagonal_capture_parity_passed": True,
            "outer_diagonal_update_parity_passed": True,
            "output_work_retained_passed": True,
            "null_identity_outer_passed": True,
        }
        for step in range(1, 25)
    ]
    for row in (rows[0], rows[21], rows[-1]):
        row.update(
            c4_loss=3.5,
            selection_loss=3.4,
            heldout_losses={f"heldout/{name}": 3.4 for name in configuration["data"]["mixture_weights"]},
        )
    rng = torch.get_rng_state()

    def numerical_runtime(step):
        hidden, mlp, layers = (configuration["model"][key] for key in ("hidden", "mlp", "layers"))
        local = layers // WORLD
        groups = {}
        for name, count, width in (("d", 3 * local, hidden), ("f", local, mlp)):
            groups[name] = {
                "C": torch.eye(width).expand(count, width, width).clone(),
                "tol": torch.full((count, 3), 1e-4),
                "basis_step": [20] * count,
                "warm": True,
                "cached_root_graph_active": True,
                "SB": torch.ones(count, 2 * width, width, dtype=torch.float16),
                "S2": torch.ones(count, width, 2 * width, dtype=torch.float16),
                "pending": [
                    {"step": step - 1, "basis": torch.eye(width), "tol": torch.full((3,), 1e-4)},
                    *([None] * (count - 1)),
                ],
            }
        engine = {
            "groups": groups,
            "momentum": {},
            "weight": 1 - configuration["statistics"]["momentum"] ** step,
            "steps": step,
            "t": step,
            "fallbacks": 0,
            "extra": 0,
            "refresh_install_age": 1,
        }
        momentum = {
            (hidden, hidden): torch.zeros(4 * local, hidden, hidden),
            (mlp, hidden): torch.zeros(2 * local, mlp, hidden),
            (hidden, mlp): torch.zeros(local, hidden, mlp),
        }
        return {
            "lagged": {},
            "input": engine,
            "output": None,
            "input_diagonal": None,
            "output_diagonal": {
                "moment": {hidden: torch.ones(5 * layers, hidden), mlp: torch.ones(2 * layers, mlp)},
                "weight": engine["weight"],
            },
            "momentum": momentum,
            "steps": step,
        }

    def auxiliary_state(step):
        shapes = model_shapes(configuration)
        names = [["head.weight"], ["body.emb.weight"]]
        names[1].extend(f"body.blocks.{layer}.{kind}.weight" for layer in range(8) for kind in ("ln1", "ln2"))
        names[1].append("body.norm.weight")
        settings = configuration["optimizer"]
        saved, groups, index = {}, [], 0
        for group_index, parameters in enumerate(names):
            indices = []
            for name in parameters:
                indices.append(index)
                saved[index] = {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.zeros(shapes[name]),
                    "exp_avg_sq": torch.ones(shapes[name]),
                }
                index += 1
            groups.append(
                {
                    "params": indices,
                    "lr": settings["auxiliary_lr"],
                    "base_lr": settings["auxiliary_lr"],
                    "betas": tuple(settings["betas"]),
                    "eps": settings["eps"],
                    "weight_decay": settings["head_weight_decay"] if group_index == 0 else 0.0,
                    "fused": True,
                    "amsgrad": False,
                }
            )
        return {"state": saved, "param_groups": groups}

    def output(directory, endpoint):
        directory.mkdir()
        record(directory / "config.json", configuration)
        record(
            directory / "run.json",
            {
                "status": "execution_check_complete",
                "execution_check": True,
                "config": configuration,
                "source_sha256": identity["source_sha256"],
                "data_receipt_sha256": frozen["data_receipt_sha256"],
                "completed_steps": endpoint,
                "resume": None,
            },
        )
        (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:endpoint]))
        for step in (22, 24):
            if step > endpoint:
                continue
            checkpoint = directory / f"checkpoints/step_{step:06d}"
            record(checkpoint / "complete.json", {"step": step, "world": WORLD})
            for rank in range(WORLD):
                state = {
                    "step": step,
                    "next_global_sequence": step * configuration["training"]["global_batch_sequences"],
                    "config": configuration,
                    "rank": rank,
                    "world": WORLD,
                    "identity": identity,
                    "optimizers": [None, auxiliary_state(step)],
                    "fast": numerical_runtime(step),
                    "output": None,
                    "torch_rng": rng,
                    "cuda_rng": torch.zeros(16, dtype=torch.uint8),
                    "python_rng": random.Random(rank + step).getstate(),
                }
                if rank == 0:
                    state["model"] = {name: torch.zeros(shape) for name, shape in model_shapes(configuration).items()}
                torch.save(state, checkpoint / f"rank_{rank}.pt")

    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    output(continuous, 24)
    output(resumed, 22)
    save_interrupted_manifest(resumed)
    interruption = trajectory_evidence(resumed, configuration, candidate, frozen, mode="interrupted")
    manifest = json.loads((resumed / "run.json").read_text())
    manifest.update(completed_steps=24, resume=str(resumed / "checkpoints/step_000022"))
    record(resumed / "run.json", manifest)
    (resumed / "metrics_from_000022.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[22:]))
    shutil.copytree(continuous / "checkpoints/step_000024", resumed / "checkpoints/step_000024")
    item = spec | {
        "config": str(candidate / spec["config_name"]),
        "continuous": str(continuous),
        "resumed": str(resumed),
        "checks": {
            "cached_root_checkpoint": False,
            "boundary_continuation": False,
            "post_cached_root_continuation": False,
            "loss_continuation": False,
            "output_work_retained": False,
            "null_identity_outer": False,
        },
    }
    return candidate, frozen, configuration, continuous, resumed, interruption, item


def test_independent_interruption_manifest_survives_resume_without_prefix_copy(independent_trajectories):
    candidate, frozen, configuration, continuous, resumed, interruption, _ = independent_trajectories
    assert trajectory_evidence(resumed, configuration, candidate, frozen, mode="interrupted") == interruption
    assert json.loads((resumed / INTERRUPTED_MANIFEST).read_text())["completed_steps"] == 22
    result = trajectory_evidence(resumed, configuration, candidate, frozen, mode="resumed")
    assert result["observed_updates"] == 24
    assert set(result["checkpoints"]) == {"22", "24"}
    assert (continuous / "checkpoints/step_000022/rank_7.pt").stat().st_ino != (
        resumed / "checkpoints/step_000022/rank_7.pt"
    ).stat().st_ino
    assert loss_continuation_evidence(continuous, resumed)["scalar_comparisons"] == 39


@pytest.mark.parametrize(
    "fault",
    (
        "missing_state_both",
        "missing_momentum_both",
        "missing_cuda_rng_both",
        "invalid_cuda_rng_both",
        "invalid_torch_rng_both",
        "invalid_python_rng_both",
        "future_pending_both",
        "rank7_source",
        "prefix_alias",
        "interrupted_manifest",
    ),
)
def test_trajectory_rejects_common_state_loss_or_replaced_prefix(independent_trajectories, fault):
    candidate, frozen, configuration, continuous, resumed, _, _ = independent_trajectories
    if fault.endswith("_both") or fault == "rank7_source":
        for directory in (continuous, resumed):
            path = directory / "checkpoints/step_000022/rank_7.pt"
            state = torch.load(path, weights_only=False)
            if fault == "missing_state_both":
                del state["python_rng"]
            elif fault == "missing_momentum_both":
                del state["fast"]["momentum"][4, 4]
            elif fault == "missing_cuda_rng_both":
                state["cuda_rng"] = None
            elif fault == "invalid_cuda_rng_both":
                state["cuda_rng"][8:] = torch.tensor(list((1).to_bytes(8, byteorder="little")), dtype=torch.uint8)
            elif fault == "invalid_torch_rng_both":
                state["torch_rng"] = torch.zeros_like(state["torch_rng"])
            elif fault == "invalid_python_rng_both":
                state["python_rng"] = (3, (7, 22), None)
            elif fault == "future_pending_both":
                state["fast"]["input"]["groups"]["d"]["pending"][0]["step"] = 22
            else:
                state["identity"]["source_sha256"] = {}
            torch.save(state, path)
        with pytest.raises((ValueError, RuntimeError)):
            checkpoint_evidence(continuous, configuration, candidate, frozen, 22)
    elif fault == "prefix_alias":
        path = resumed / "metrics_from_000000.jsonl"
        path.unlink()
        path.symlink_to(continuous / "metrics_from_000000.jsonl")
        with pytest.raises(ValueError):
            trajectory_evidence(resumed, configuration, candidate, frozen, mode="resumed")
    else:
        value = json.loads((resumed / INTERRUPTED_MANIFEST).read_text())
        value["completed_steps"] = 24
        record(resumed / INTERRUPTED_MANIFEST, value)
        with pytest.raises(ValueError):
            trajectory_evidence(resumed, configuration, candidate, frozen, mode="interrupted")


def test_foreign_cached_experiment_is_rejected_before_import(tmp_path):
    foreign, candidate = tmp_path / "foreign", tmp_path / "candidate"
    package = foreign / "experiment"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    marker = tmp_path / "unauthenticated-import"
    (package / "validate.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    result = subprocess.run(
        [
            PYTHON,
            "-P",
            "-c",
            "import sys;from pathlib import Path;from fused_attribution_validation import numerical_modules;"
            "sys.path.insert(0,sys.argv[1]);import experiment;numerical_modules(Path(sys.argv[2]))",
            str(foreign),
            str(candidate),
        ],
        cwd=tmp_path,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(Path(__file__).parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert not marker.exists()


@pytest.mark.parametrize("fault", ("none", "rank7_continuation", "null_work", "heldout_loss"))
def test_trial_reconstruction_uses_real_checkpoint_readers_and_rejects_missing_gates(
    independent_trajectories, tmp_path, fault
):
    candidate, frozen, _, continuous, resumed, _, item = independent_trajectories
    if fault == "rank7_continuation":
        path = resumed / "checkpoints/step_000024/rank_7.pt"
        state = torch.load(path, weights_only=False)
        state["fast"]["momentum"][4, 4] += 0.2
        torch.save(state, path)
    elif fault in ("null_work", "heldout_loss"):
        path = resumed / "metrics_from_000022.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "null_work":
            rows[-1]["null_identity_outer_passed"] = False
        else:
            rows[-1]["heldout_losses"]["heldout/dclm"] += 0.01
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    inputs = tmp_path / "inputs.json"
    record(inputs, {"item": item, "frozen": frozen})
    result = subprocess.run(
        [
            PYTHON,
            "-P",
            "-c",
            "import json,sys;from pathlib import Path;"
            "from fused_attribution_validation import trial_evidence;"
            "v=json.loads(Path(sys.argv[2]).read_text());"
            "print(json.dumps(trial_evidence(v['item'],Path(sys.argv[1]),v['frozen'])))",
            str(candidate),
            str(inputs),
        ],
        cwd=tmp_path,
        env=os.environ
        | {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(Path(__file__).parent),
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    if fault == "none":
        assert result.returncode == 0, result.stderr
        evidence = json.loads(result.stdout)
        assert evidence["evidence"]["boundary_continuation"]["max_absolute_error"] == 0
        assert evidence["evidence"]["post_cached_root_continuation"]["max_absolute_error"] == 0
        assert evidence["evidence"]["cached_root_checkpoint"]["resumed"]["ranks"] == 8
        assert all(evidence["checks"].values())
    else:
        assert result.returncode != 0
        assert (resumed / "metrics_from_000000.jsonl").read_bytes() == b"".join(
            (continuous / "metrics_from_000000.jsonl").read_bytes().splitlines(keepends=True)[:22]
        )


@pytest.mark.parametrize("fault", ("none", "missing_command", "changed_log", "wrong_owner", "false_trial"))
def test_full_receipt_reconstruction_rejects_incomplete_or_rebound_execution(independent_trajectories, tmp_path, fault):
    candidate, _, _, continuous, resumed, _, _ = independent_trajectories
    root = tmp_path / "root"
    record(root / "data/experiment_receipt.json", {"data": "fixture"})
    prepared = {
        "status": PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "files_sha256": candidate_inventory(candidate),
        "parent_metadata_sha256": {},
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
    }
    record(candidate / PREPARATION, prepared)
    binding_path, _, state_path, work, receipt_path = locations(root)
    result = plan(candidate, root, work)
    binding = {
        "status": FROZEN_STATUS,
        "candidate": str(candidate),
        "files_sha256": prepared["files_sha256"],
        "preparation_sha256": sha256(candidate / PREPARATION),
        "commands": [command for item in result["trials"] for command in item["commands"]],
    }
    record(binding_path, binding)
    owner = {"controller_pid": 201, "controller_start_ticks": "301"}
    record(state_path, owner | {"binding_sha256": sha256(binding_path)})
    for item in result["trials"]:
        configuration = read(Path(item["config"]))
        for name, original in (("continuous", continuous), ("resumed", resumed)):
            output = Path(item[name])
            shutil.copytree(original, output)
            record(output / "config.json", configuration)
            for filename in ("run.json", INTERRUPTED_MANIFEST):
                path = output / filename
                if not path.exists():
                    continue
                manifest = read(path)
                manifest["config"] = configuration
                manifest["data_receipt_sha256"] = prepared["data_receipt_sha256"]
                if filename == "run.json" and name == "resumed":
                    manifest["resume"] = str(output / "checkpoints/step_000022")
                record(path, manifest)
            for step in (22, 24):
                for rank in range(WORLD):
                    path = output / f"checkpoints/step_{step:06d}/rank_{rank}.pt"
                    state = torch.load(path, weights_only=False)
                    state["config"] = configuration
                    state["identity"]["config_sha256"] = hashlib.sha256(
                        json.dumps(configuration, sort_keys=True).encode()
                    ).hexdigest()
                    state["identity"]["data_receipt_sha256"] = prepared["data_receipt_sha256"]
                    if item["family"] == "muont":
                        state["fast"] = state["fast"]["input"] | {
                            "momentum": {shape: {"m": value} for shape, value in state["fast"]["momentum"].items()}
                        }
                    torch.save(state, path)
        for mode, command in zip(("continuous", "interrupted", "resumed"), item["commands"], strict=True):
            log = work / f"{item['name']}_{mode}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("original command output\n")
            result["gpu_commands"].append(
                {
                    "trial": item["name"],
                    "mode": mode,
                    "command": command,
                    "exit_code": 0,
                    "log": str(log),
                    "log_sha256": sha256(log),
                }
            )
    from_status = "passed_fused_attribution_actual_config_continuation_and_work_gates"
    result.update(
        status=from_status,
        owner=owner,
        execution_binding_sha256=sha256(binding_path),
        candidate_preparation_sha256=sha256(candidate / PREPARATION),
    )
    record(receipt_path, result)
    environment = os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(Path(__file__).parent)}
    populate = subprocess.run(
        [
            PYTHON,
            "-P",
            "-c",
            "import json,sys;from pathlib import Path;import fused_attribution_validation as v;"
            "root,candidate,receipt=map(Path,sys.argv[1:]);"
            "prepared=v.require_source(candidate,root);result=v.read(receipt);"
            "result['trials']=[v.trial_evidence(item,candidate,prepared) for item in result['trials']];"
            "evidence={item['name']:item['evidence'] for item in result['trials']};"
            "[(execution.update(trajectory=evidence[execution['trial']]['trajectory'][execution['mode']]))"
            " for execution in result['gpu_commands']];v.write(receipt,result)",
            str(root),
            str(candidate),
            str(receipt_path),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert populate.returncode == 0, populate.stderr
    result = read(receipt_path)
    if fault == "missing_command":
        result["gpu_commands"].pop()
    elif fault == "changed_log":
        log = Path(result["gpu_commands"][-1]["log"])
        log.write_text("replaced output\n")
    elif fault == "wrong_owner":
        result["owner"] = owner | {"controller_start_ticks": "reused"}
    elif fault == "false_trial":
        result["trials"][-1]["evidence"]["trajectory"]["resumed"]["observed_updates"] = 23
    record(receipt_path, result)
    checked = subprocess.run(
        [
            PYTHON,
            "-P",
            str(Path(__file__).parent / "fused_attribution_validation.py"),
            "--root",
            str(root),
            "--candidate",
            str(candidate),
            "--work",
            str(work),
            "--receipt",
            str(receipt_path),
            "--reconstruct",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if fault == "none":
        assert checked.returncode == 0, checked.stderr
        assert json.loads(checked.stdout)["trials"] == 9
        assert len(result["gpu_commands"]) == 27
    else:
        assert checked.returncode != 0
    assert read(receipt_path) == result
