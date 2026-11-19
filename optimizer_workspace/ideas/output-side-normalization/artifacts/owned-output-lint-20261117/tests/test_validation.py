# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep cached-root and trajectory gates independent of the unissued GPU release."""

import json

import pytest
import torch

from experiment.validate import require_cached_root_checkpoint, require_fixed_age_history, require_outer_capture_history


def save_root_checkpoint(directory, rank, *, cached, steps=22):
    group = {
        "C": torch.zeros(2, 3, 3),
        "S2": torch.eye(3).repeat(2, 1, 1),
        "cached_root_graph_active": cached,
    }
    torch.save(
        {"fast": {"groups": {"hidden": group}, "steps": steps, "refresh_install_age": 1}},
        directory / f"rank_{rank}.pt",
    )


def test_cached_root_gate_checks_owned_groups_on_every_checkpoint_rank(tmp_path):
    save_root_checkpoint(tmp_path, 0, cached=True)
    save_root_checkpoint(tmp_path, 1, cached=True)
    result = require_cached_root_checkpoint(tmp_path, 2)
    assert result["owned_graph_groups"] == 2
    save_root_checkpoint(tmp_path, 1, cached=False)
    with pytest.raises(ValueError, match="Rank 1"):
        require_cached_root_checkpoint(tmp_path, 2)


def test_cached_root_gate_refuses_pretransition_checkpoint_despite_graph_claim(tmp_path):
    save_root_checkpoint(tmp_path, 0, cached=True, steps=3)
    with pytest.raises(ValueError, match="transition"):
        require_cached_root_checkpoint(tmp_path, 1)


def test_outer_capture_gate_requires_pre_and_post_resume_update_receipts(tmp_path):
    rows = [
        {"step": step, "diagonal_capture_parity_passed": True, "outer_diagonal_update_parity_passed": True}
        for step in range(1, 25)
    ]
    first = tmp_path / "metrics_from_000000.jsonl"
    tail = tmp_path / "metrics_from_000022.jsonl"
    first.write_text("\n".join(json.dumps(row) for row in rows[:22]))
    tail.write_text("\n".join(json.dumps(row) for row in rows[22:]))
    assert require_outer_capture_history(tmp_path)["updates"] == 24
    rows[23]["outer_diagonal_update_parity_passed"] = False
    tail.write_text("\n".join(json.dumps(row) for row in rows[22:]))
    with pytest.raises(ValueError, match="missing"):
        require_outer_capture_history(tmp_path)
    tail.unlink()
    with pytest.raises(ValueError, match="24 consecutive"):
        require_outer_capture_history(tmp_path)


def test_fixed_age_history_refuses_early_installation_after_resume(tmp_path):
    rows = [
        {
            "step": step,
            "input_refresh_installations": (
                [{"source_step": step - 2, "installation_step": step - 1}] if step > 21 else []
            ),
        }
        for step in range(1, 25)
    ]
    first = tmp_path / "metrics_from_000000.jsonl"
    tail = tmp_path / "metrics_from_000022.jsonl"
    first.write_text("\n".join(json.dumps(row) for row in rows[:22]))
    tail.write_text("\n".join(json.dumps(row) for row in rows[22:]))
    assert require_fixed_age_history(tmp_path)["observed_installations"] == 3
    rows[23]["input_refresh_installations"][0]["source_step"] = 23
    tail.write_text("\n".join(json.dumps(row) for row in rows[22:]))
    with pytest.raises(ValueError, match="fixed update age"):
        require_fixed_age_history(tmp_path)
