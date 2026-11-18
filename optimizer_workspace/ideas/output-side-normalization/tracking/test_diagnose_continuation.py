# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect real CPU checkpoint files with exact boundary state and later root divergence."""

import json

import torch
from diagnose_continuation import inspect_pair


def checkpoint(step):
    return {
        "identity": {"source": "bound", "data": "fixed"},
        "config": {"training": {"seed": 0}},
        "rank": 0,
        "world": 8,
        "step": step,
        "next_global_sequence": step * 128,
        "model": {"down.weight": torch.tensor([[1.0, 2.0]])},
        "optimizers": [{"momentum": torch.ones(2)}],
        "fast": {
            "groups": {
                "f": {
                    "C": torch.eye(2),
                    "S2": torch.eye(2),
                    "basis_step": [step - 1],
                    "pending": [{"step": step - 1, "basis": torch.eye(2)}],
                }
            }
        },
        "torch_rng": torch.tensor([1, 2], dtype=torch.uint8),
    }


def test_saved_boundary_is_exact_then_model_and_refresh_change_are_reported(tmp_path):
    expected_path = tmp_path / "continuous.pt"
    actual_path = tmp_path / "resumed.pt"
    for step in (22, 24):
        expected = checkpoint(step)
        actual = checkpoint(step)
        if step == 24:
            actual["model"]["down.weight"][0, 0] += 0.01
            actual["fast"]["groups"]["f"]["S2"][0, 0] += 0.1
            actual["fast"]["groups"]["f"]["basis_step"] = [22]
            actual["fast"]["groups"]["f"]["pending"][0]["step"] = 22
        torch.save(expected, expected_path)
        torch.save(actual, actual_path)
        report = inspect_pair(actual_path, expected_path, 0, step)
        assert report["tensor_comparisons"] == 6
        assert report["root_groups"]["f"]["moment_exact"]
        if step == 22:
            assert report["changed_tensors"] == report["scalar_or_structure_differences"] == []
            assert len(report["actual_checkpoint_sha256"]) == len(report["expected_checkpoint_sha256"]) == 64
        else:
            assert report["failed_tensor_count"] == 2
            assert {row["name"] for row in report["changed_tensors"]} == {
                "rank_0.model.down.weight",
                "rank_0.fast.groups.f.S2",
            }
            assert report["root_groups"]["f"]["basis_step_actual"] == [22]
            assert report["root_groups"]["f"]["basis_step_expected"] == [23]
            assert len(report["scalar_or_structure_differences"]) == 2


def test_missing_state_and_changed_rng_remain_visible_in_saved_diagnostic(tmp_path):
    expected = checkpoint(24)
    actual = checkpoint(24)
    actual.pop("optimizers")
    actual["torch_rng"][1] = 3
    expected["fast"]["groups"]["f"]["pending"] = [None]
    expected_path, actual_path = tmp_path / "continuous.pt", tmp_path / "resumed.pt"
    torch.save(expected, expected_path)
    torch.save(actual, actual_path)
    report = inspect_pair(actual_path, expected_path, 0, 24)
    assert report["scalar_or_structure_differences"] == [
        {"name": "optimizers", "kind": "missing_section"},
        {"name": "rank_0.fast.groups.f.pending.0", "kind": "value_structure"},
    ]
    assert report["changed_tensors"][0]["name"] == "rank_0.torch_rng"
    assert report["changed_tensors"][0]["changed_elements"] == 1
    assert report["failed_tensor_count"] == 1
    assert json.loads(json.dumps(report, allow_nan=False))["scalar_or_structure_differences"] == (
        report["scalar_or_structure_differences"]
    )
