# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual NS configs, complete state trees, rejections and bounded cleanup on CPU."""

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import ns_full_model_contract as contract
import ns_full_model_validation as validation
import owned_output_handoff as handoff
import pytest
import torch
from analyze_tuning import executed_sources
from fused_attribution_validation import model_shapes, save_interrupted_manifest
from fused_attribution_validation_test_support import record
from selected_lr_release_owner import candidate_files, read, sha256

PRESERVED = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/"
    "candidates/ns-budget-full-model-alias-corrected-20261119T0104"
)


def write(path, value):
    record(path, value)


def fixture_source() -> Path:
    archived = Path(__file__).resolve().parents[1]
    return archived if (archived / contract.specs()[0]["config_name"]).is_file() else PRESERVED


def build_candidate(tmp_path):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    write(root / "data/experiment_receipt.json", {"data": "CPU fixture"})
    for directory in ("experiment", "artifacts/pinned-sources-materialized"):
        shutil.copytree(
            fixture_source() / directory,
            candidate / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    for spec in contract.specs():
        paths = (
            candidate / f"experiment/configs/{spec['family']}_full_lr_lower_0p5m_seed0.json",
            candidate / spec["config_name"],
        )
        for path in paths:
            config = read(path)
            config["model"].update(layers=8, hidden=4, mlp=8, vocab_size=16)
            write(path, config)
    frozen = {
        "status": contract.FROZEN_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "files_sha256": candidate_files(candidate),
        "data_receipt_sha256": sha256(root / "data/experiment_receipt.json"),
        "plan": validation.plan(root, candidate),
    }
    frozen["commands"] = frozen["plan"]["commands"]
    write(contract.locations(root)[0], frozen)
    return root, candidate, frozen


@pytest.fixture
def candidate(tmp_path):
    return build_candidate(tmp_path)


def test_complete_actual_config_matrix_preserves_schedule_and_denies_real_unowned_execution(candidate):
    root, path, frozen = candidate
    result = validation.plan(root, path)
    assert validation.source(root, path) == frozen
    assert len(result["trials"]) == 18 and len(result["commands"]) == 54
    assert len({tuple(command) for command in result["commands"]}) == 54
    assert result["planned_tokens"] == 452984832
    for item in result["trials"]:
        assert [command[command.index("--max-steps") + 1] for command in item["commands"]] == ["24", "22", "24"]
        assert all(command[command.index("--execution-boundary") + 1] == "22" for command in item["commands"])
        assert all(
            ("--verify-outer-diagonal-updates" in command) == (item["family"] != "muont") for command in item["commands"]
        )
        config = read(path / item["config_name"])
        assert config["training"]["steps"] == 5120 and config["root"]["graphs"] is True
        assert config["optimizer"]["ns_steps"] == item["budget"]
        assert config["optimizer"]["ns_mode"] == item["operand"]
    denial = subprocess.run(
        [
            contract.PYTHON,
            "-P",
            str(Path(validation.__file__)),
            "--root",
            str(root),
            "--candidate",
            str(path),
            "--phase",
            "execute",
        ],
        cwd=path,
        env=os.environ
        | {
            "PYTHONPATH": f"{Path(validation.__file__).parent}:{path}",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert denial.returncode != 0 and "require_gpu_owner" in denial.stderr
    assert not contract.locations(root)[3].exists() and not contract.locations(root)[4].exists()


@pytest.mark.parametrize("fault", ("budget", "operand", "schedule", "lr", "graphs", "auxiliary", "alias"))
def test_actual_config_protocol_rejects_rehashed_unrelated_or_invalid_arm_settings(candidate, fault):
    _, path, _ = candidate
    spec = contract.specs()[-1]
    config_path = path / spec["config_name"]
    config = read(config_path)
    if fault == "budget":
        config["optimizer"]["ns_steps"] = 3
    elif fault == "operand":
        config["optimizer"]["ns_mode"] = "fp64"
    elif fault == "schedule":
        config["schedule"]["stable_steps"] = 2048
    elif fault == "lr":
        config["optimizer"]["lr"] *= 2
    elif fault == "graphs":
        config["root"]["graphs"] = False
    elif fault == "auxiliary":
        config["optimizer"]["auxiliary_lr"] *= 2
    else:
        target = config_path.with_suffix(".aliased")
        config_path.rename(target)
        config_path.symlink_to(target)
    if fault != "alias":
        write(config_path, config)
    with pytest.raises(ValueError):
        contract.require_config(path, spec)


def fast_state(config, step, rank):
    hidden, mlp, layers = (config["model"][key] for key in ("hidden", "mlp", "layers"))
    groups = {}
    for name, count, width in (("d", 3, hidden), ("f", 1, mlp)):
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
    momentum = {
        (hidden, hidden): torch.full((4, hidden, hidden), float(rank)),
        (mlp, hidden): torch.zeros(2, mlp, hidden),
        (hidden, mlp): torch.zeros(1, hidden, mlp),
    }
    engine = {
        "groups": groups,
        "momentum": {},
        "weight": 1 - config["statistics"]["momentum"] ** step,
        "steps": step,
        "t": step,
        "fallbacks": 0,
        "extra": 0,
        "refresh_install_age": 1,
    }
    if config["experiment"]["arm"] == "muont":
        engine["momentum"] = {shape: {"m": value} for shape, value in momentum.items()}
        return engine
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


def auxiliary_state(config, step):
    shapes = model_shapes(config)
    names = [["head.weight"], ["body.emb.weight"]]
    names[1].extend(
        f"body.blocks.{layer}.{kind}.weight" for layer in range(config["model"]["layers"]) for kind in ("ln1", "ln2")
    )
    names[1].append("body.norm.weight")
    settings, states, groups, index = config["optimizer"], {}, [], 0
    for group_index, parameters in enumerate(names):
        indices = []
        for name in parameters:
            indices.append(index)
            states[index] = {
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
    return {"state": states, "param_groups": groups}


def saved_trial(root, candidate, frozen, spec):
    config = contract.require_config(candidate, spec)
    item = next(item for item in validation.plan(root, candidate)["trials"] if item["name"] == spec["name"])
    identity = {
        "schema": 1,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "source_sha256": executed_sources(candidate, frozen),
        "data_receipt_sha256": frozen["data_receipt_sha256"],
    }
    rows = []
    for step in range(1, 25):
        row = {
            "step": step,
            "tokens": step * contract.BATCH_TOKENS,
            "train_loss": 4.0 - step / 1000,
            "input_refresh_installations": (
                [{"installation_step": step - 1, "source_step": step - 2}] if step >= 2 else []
            ),
            "diagonal_capture_parity_passed": True,
            "outer_diagonal_update_parity_passed": True,
            "output_work_retained_passed": True,
            "null_identity_outer_passed": True,
        }
        if step in (1, 22, 24):
            row.update(
                c4_loss=3.5,
                selection_loss=3.4,
                heldout_losses={f"heldout/{name}": 3.4 for name in config["data"]["mixture_weights"]},
            )
        rows.append(row)
    rng = torch.get_rng_state()

    def output(directory, endpoint):
        directory.mkdir(parents=True)
        write(directory / "config.json", config)
        write(
            directory / "run.json",
            {
                "status": "execution_check_complete",
                "execution_check": True,
                "config": config,
                "source_sha256": identity["source_sha256"],
                "data_receipt_sha256": frozen["data_receipt_sha256"],
                "completed_steps": endpoint,
                "resume": None,
            },
        )
        (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[:endpoint]))
        for step in contract.CHECKPOINTS:
            if step > endpoint:
                continue
            folder = directory / f"checkpoints/step_{step:06d}"
            write(folder / "complete.json", {"step": step, "world": 8})
            for rank in range(8):
                state = {
                    "step": step,
                    "next_global_sequence": step * config["training"]["global_batch_sequences"],
                    "config": config,
                    "rank": rank,
                    "world": 8,
                    "identity": identity,
                    "optimizers": [None, auxiliary_state(config, step)],
                    "fast": fast_state(config, step, rank),
                    "output": None,
                    "torch_rng": rng.clone(),
                    "cuda_rng": torch.zeros(16, dtype=torch.uint8),
                    "python_rng": random.Random(rank + step).getstate(),
                }
                if rank == 0:
                    state["model"] = {name: torch.zeros(shape) for name, shape in model_shapes(config).items()}
                torch.save(state, folder / f"rank_{rank}.pt")

    continuous, resumed = Path(item["continuous"]), Path(item["resumed"])
    output(continuous, 24)
    output(resumed, 22)
    save_interrupted_manifest(resumed)
    prefix = validation.trajectory(resumed, config, candidate, frozen, "interrupted")
    manifest = read(resumed / "run.json")
    manifest.update(completed_steps=24, resume=str(resumed / "checkpoints/step_000022"))
    write(resumed / "run.json", manifest)
    (resumed / "metrics_from_000022.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[22:]))
    for step in (23, 24):
        shutil.copytree(continuous / f"checkpoints/step_{step:06d}", resumed / f"checkpoints/step_{step:06d}")
    return item, config, prefix


@pytest.mark.parametrize("family", contract.FAMILIES)
def test_actual_full_state_readers_reconstruct_all_families_and_preserve_independent_prefix(candidate, family):
    root, path, frozen = candidate
    spec = next(spec for spec in contract.specs() if spec["family"] == family and spec["operand"] == "f32")
    item, config, prefix = saved_trial(root, path, frozen, spec)
    result = validation.trial(item, path, frozen)
    assert result["accepted"] is True and all(result["gates"].values())
    assert result["trajectories"]["interrupted"] == prefix
    assert set(result["continuation"]) == {"22", "23", "24"}
    assert result["loss"]["comparisons"] == 39
    assert all(len(pair["ranks"]) == 8 for pair in result["continuation"].values())
    assert result["first_resumed_diagnostic_only"] is True
    assert validation.trajectory(Path(item["resumed"]), config, path, frozen, "resumed")["updates"] == 24


@pytest.mark.parametrize(
    "fault",
    (
        "common_missing_rng",
        "common_bad_philox",
        "common_missing_momentum",
        "common_pending_shape",
        "missing_rank",
        "replaced_prefix",
        "missing_capture",
        "rank7_difference",
        "loss_difference",
        "first_graph_rewarming",
    ),
)
def test_full_model_common_loss_and_numerical_rejections_cannot_be_hidden(candidate, fault):
    root, path, frozen = candidate
    spec = contract.specs()[-1]
    item, _, _ = saved_trial(root, path, frozen, spec)
    directories = [Path(item["continuous"]), Path(item["resumed"])]
    if fault.startswith("common_") or fault in ("rank7_difference", "first_graph_rewarming", "missing_rank"):
        if fault == "missing_rank":
            (directories[1] / "checkpoints/step_000024/rank_7.pt").unlink()
        else:
            for directory in directories if fault.startswith("common_") else directories[1:]:
                step = 23 if fault == "first_graph_rewarming" else 24
                checkpoint = directory / f"checkpoints/step_{step:06d}/rank_7.pt"
                state = torch.load(checkpoint, weights_only=False)
                if fault == "common_missing_rng":
                    state["cuda_rng"] = None
                elif fault == "common_bad_philox":
                    state["cuda_rng"][8] = 1
                elif fault == "common_missing_momentum":
                    state["fast"]["momentum"].pop((4, 4))
                elif fault == "common_pending_shape":
                    state["fast"]["input"]["groups"]["d"]["pending"][0]["basis"] = torch.zeros(3, 3)
                elif fault == "first_graph_rewarming":
                    state["fast"]["input"]["groups"]["d"]["warm"] = False
                    state["fast"]["input"]["groups"]["d"]["cached_root_graph_active"] = False
                else:
                    state["fast"]["momentum"][(4, 4)][0, 0, 0] += 0.1
                torch.save(state, checkpoint)
    else:
        segment = directories[1] / "metrics_from_000022.jsonl"
        rows = [json.loads(line) for line in segment.read_text().splitlines()]
        if fault == "missing_capture":
            rows[-1].pop("output_work_retained_passed")
        elif fault == "loss_difference":
            rows[-1]["selection_loss"] += 0.01
        else:
            segment = directories[1] / "metrics_from_000000.jsonl"
            rows = [json.loads(line) for line in segment.read_text().splitlines()]
            rows[0]["step"] = 2
        segment.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if fault in ("rank7_difference", "loss_difference", "first_graph_rewarming"):
        result = validation.trial(item, path, frozen)
        if fault == "first_graph_rewarming":
            assert result["accepted"] is True and result["continuation"]["23"]["within_tolerance"] is False
        else:
            assert result["accepted"] is False
            assert not result["gates"][
                "post_cached_root_all_rank_continuation" if fault == "rank7_difference" else "scalar_loss_continuation"
            ]
    else:
        with pytest.raises((ValueError, FileNotFoundError, KeyError)):
            validation.trial(item, path, frozen)


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "timeout", "spawn_failure"))
def test_real_bounded_command_cleanup_preserves_failures_without_gpu(tmp_path, monkeypatch, outcome):
    _, _, state, work, _ = contract.locations(tmp_path)
    write(state, {"controller_pid": os.getpid()})
    work.mkdir(parents=True)
    helper = tmp_path / "child.py"
    helper.write_text(
        "import os,signal,subprocess,sys\nfrom pathlib import Path\n"
        "if sys.argv[1]=='timeout':signal.pause()\n"
        "if sys.argv[1]=='detached':\n"
        " r,w=os.pipe()\n"
        " child=subprocess.Popen([sys.executable,'-c',"
        "'import os,signal,sys;os.write(int(sys.argv[1]),b\"R\");signal.pause()',str(w)],"
        "pass_fds=(w,),start_new_session=True)\n"
        " os.close(w);assert os.read(r,1)==b'R';os.close(r)\n"
        " Path(sys.argv[2]).write_text(str(child.pid))\n"
        "sys.exit(0 if sys.argv[1]=='zero' else 7)\n"
    )
    command = (
        [str(tmp_path / "missing")]
        if outcome == "spawn_failure"
        else [sys.executable, str(helper), outcome, str(tmp_path / "descendant.pid")]
    )
    monkeypatch.setattr(contract, "ENVIRONMENT", contract.ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": ""})
    monkeypatch.setattr(validation, "COMMAND_TIMEOUT", 0.25 if outcome == "timeout" else 10)
    log = work / "command.log"
    if outcome in ("timeout", "spawn_failure"):
        with pytest.raises(subprocess.TimeoutExpired if outcome == "timeout" else FileNotFoundError):
            validation.run_command(tmp_path, tmp_path, command, log)
    else:
        entry = validation.run_command(tmp_path, tmp_path, command, log)
        if outcome == "zero":
            validation.require_entry(entry, command, log)
        else:
            with pytest.raises(ValueError):
                validation.require_entry(entry, command, log)
    saved = read(log.with_suffix(".execution.json"))
    assert saved["log_sha256"] == sha256(log)
    if outcome != "spawn_failure":
        assert saved["marked_descendants_absent"] is True
        assert not handoff.marker_processes(contract.COMMAND_MARKER, saved["marker"])
        assert saved["generation"]["parent"] == os.getpid()
    assert not contract.locations(tmp_path)[4].exists()


@pytest.fixture(scope="module")
def complete_receipt(tmp_path_factory):
    root, path, frozen = build_candidate(tmp_path_factory.mktemp("ns-full-protocol"))
    planned = validation.plan(root, path)
    validator = 302
    owner = {
        "controller_pid": 301,
        "controller_start_ticks": "controller-fixture",
        "validator_pid": validator,
        "validator_start_ticks": "validator-fixture",
    }
    entries, measured = [], []
    work = contract.locations(root)[3]
    for index, spec in enumerate(contract.specs()):
        item, config, _ = saved_trial(root, path, frozen, spec)
        for offset, (mode, command) in enumerate(zip(contract.MODES, item["commands"], strict=True)):
            log = work / f"{item['name']}_{mode}.log"
            log.write_text("CPU external command receipt fixture\n")
            pid = 1000 + index * 3 + offset
            entry = {
                "command": command,
                "marker": f"command-{pid}",
                "spawn_status": "acknowledged",
                "generation": {
                    "pid": pid,
                    "command": command,
                    "parent": validator,
                    "start_ticks": str(pid + 900),
                    "group": pid,
                    "session": pid,
                },
                "exit_code": 0,
                "cleanup_exit_code": 0,
                "marked_descendants_absent": True,
                "generations": {},
                "log": str(log),
                "log_sha256": sha256(log),
            }
            write(log.with_suffix(".execution.json"), entry)
            directory = Path(item["continuous"] if mode == "continuous" else item["resumed"])
            entries.append(
                entry
                | {
                    "trial": item["name"],
                    "mode": mode,
                    "trajectory": validation.trajectory(directory, config, path, frozen, mode),
                }
            )
        measured.append(validation.trial(item, path, frozen))
    write(
        contract.locations(root)[2],
        {
            "controller_pid": owner["controller_pid"],
            "controller_start_ticks": owner["controller_start_ticks"],
            "child_pid": validator,
            "child_start_ticks": owner["validator_start_ticks"],
            "binding_sha256": sha256(contract.locations(root)[0]),
        },
    )
    result = planned | {
        "status": validation.STATUS,
        "owner": owner,
        "binding_sha256": sha256(contract.locations(root)[0]),
        "gpu_commands": entries,
        "results": measured,
        "accepted_cases": [item["name"] for item in measured],
        "rejected_cases": [],
        "all_cases_accepted": True,
    }
    write(contract.locations(root)[4], result)
    return root, path, result


def test_complete_54_command_fixture_has_all_exact_cases_and_private_command_records(complete_receipt):
    root, _, result = complete_receipt
    assert len(result["results"]) == 18 and len(result["gpu_commands"]) == 54
    assert result["accepted_cases"] == [spec["name"] for spec in contract.specs()]
    assert result["rejected_cases"] == []
    for entry in result["gpu_commands"]:
        validation.require_entry(
            entry, entry["command"], contract.locations(root)[3] / f"{entry['trial']}_{entry['mode']}.log"
        )


def test_complete_54_command_retained_reconstruction_rejects_incomplete_or_rebound_evidence(complete_receipt):
    root, path, result = complete_receipt
    receipt = contract.locations(root)[4]
    assert validation.reconstruct(root, path) == result
    for fault in ("missing_command", "missing_case", "bool_exit", "ancestry", "owner", "rejection_summary"):
        changed = json.loads(json.dumps(result))
        if fault == "missing_command":
            changed["gpu_commands"].pop()
        elif fault == "missing_case":
            changed["results"].pop()
        elif fault == "bool_exit":
            changed["gpu_commands"][0]["exit_code"] = False
        elif fault == "ancestry":
            changed["gpu_commands"][0]["generation"]["parent"] = 999
        elif fault == "owner":
            changed["owner"]["validator_pid"] = 999
        else:
            changed["rejected_cases"] = [changed["results"][0]["name"]]
        write(receipt, changed)
        with pytest.raises(ValueError):
            validation.reconstruct(root, path)
    write(receipt, result)
