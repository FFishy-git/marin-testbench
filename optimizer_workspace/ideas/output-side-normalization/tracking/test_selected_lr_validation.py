# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check actual full schedules and the exact execution-only continuation protocol."""

import json
from pathlib import Path

import pytest
from selected_lr_validation import ARMS, locations, plan


def full_config(arm):
    return {
        "experiment": {"arm": arm, "refresh_install_age": 1},
        "training": {"steps": 5120, "tokens": 2684354560, "batch_tokens": 524288, "world_size": 8, "seed": 0},
        "runtime": {"compile": "default", "attention": "FA2", "head": "compiled"},
        "optimizer": {"lr": 0.064 if arm in ("muont", "outer_leftdiag_fused") else 0.09050966784},
    }


def test_selected_lr_plan_retains_actual_full_budget_and_resume_endpoint(tmp_path):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    configs = candidate / "experiment/configs"
    configs.mkdir(parents=True)
    before = {}
    for arm in ARMS:
        config = full_config(arm)
        path = configs / f"{arm}_0p5m_seed0.json"
        path.write_text(json.dumps(config) + "\n")
        before[path] = path.read_bytes()
    work = locations(root)[3]
    result = plan(candidate, root, work)
    assert result["planned_validation_tokens"] == 100663296
    assert result["continuation_tolerance"] == {"rtol": 1e-5, "atol": 1e-6}
    assert result["gpu_commands"] == []
    for trial in result["trials"]:
        commands = trial["commands"]
        assert [command[command.index("--max-steps") + 1] for command in commands] == ["24", "22", "24"]
        assert all(command[command.index("--execution-boundary") + 1] == "22" for command in commands)
        assert Path(commands[2][commands[2].index("--resume") + 1]) == (
            Path(trial["resumed"]) / "checkpoints/step_000022"
        )
        assert all(command[command.index("--config") + 1] == trial["config"] for command in commands)
        if trial["family"] == "outer_leftdiag_fused":
            assert all("--verify-diagonal-capture" in command for command in commands)
            assert all("--verify-outer-diagonal-updates" in command for command in commands)
        assert not any(trial["checks"].values())
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert not work.exists()


def test_selected_lr_plan_refuses_short_campaign_configuration(tmp_path):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    config = full_config("muont")
    config["training"].update(steps=320, tokens=320 * 524288)
    config["experiment"]["refresh_install_age"] = 1
    path = candidate / "experiment/configs/muont_0p5m_seed0.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        plan(candidate, root, locations(root)[3])
    assert not locations(root)[3].exists()
