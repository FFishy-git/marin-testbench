# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check nested saved block state against real CPU checkpoint files."""

import json

import torch
from diagnose_continuation import digest
from diagnose_fused_attribution_continuation import diagnose, inspect_pair
from fused_attribution_validation_contract import locations


def checkpoint(step):
    return {
        "identity": {"source": "paired", "data": "fixed"},
        "config": {"experiment": {"arm": "outer_leftdiag_fused"}},
        "rank": 0,
        "world": 8,
        "step": step,
        "next_global_sequence": step * 128,
        "model": {"weight": torch.ones(2, 2)},
        "fast": {
            "input": {
                "groups": {
                    "d": {
                        "C": torch.eye(2),
                        "basis_step": [step - 1],
                        "pending": [{"step": step - 1, "basis": torch.eye(2)}],
                    }
                }
            },
            "output": None,
            "output_diagonal": {"moment": torch.ones(2)},
        },
    }


def test_nested_checkpoint_reports_exact_boundary_and_each_later_section(tmp_path):
    actual_path, expected_path = tmp_path / "actual.pt", tmp_path / "expected.pt"
    for step in (22, 24):
        actual, expected = checkpoint(step), checkpoint(step)
        if step == 24:
            actual["model"]["weight"][0, 0] += 0.01
            actual["fast"]["input"]["groups"]["d"]["C"][0, 0] += 0.1
            actual["fast"]["output_diagonal"]["moment"][0] += 0.1
        torch.save(actual, actual_path)
        torch.save(expected, expected_path)
        result = inspect_pair(actual_path, expected_path, 0, step)
        assert result["tensor_comparisons"] == 4
        assert result["scalar_or_structure_differences"] == []
        if step == 22:
            assert result["changed_tensors"] == []
            assert result["actual_checkpoint_sha256"] == digest(actual_path)
            assert result["expected_checkpoint_sha256"] == digest(expected_path)
        else:
            assert result["failures_by_section"] == {"model": 1, "fast": 2}
            assert result["failed_tensor_count"] == 3
            assert not result["root_groups"]["input.d"]["moment_exact"]
            assert result["root_groups"]["input.d"]["pending_steps_actual"] == [23]


def test_failed_terminal_diagnostic_does_not_require_success_only_receipt_fields(tmp_path):
    root = tmp_path / "root"
    binding, helpers, state_path, work, gpu_path = locations(root)
    helpers.mkdir(parents=True)
    binding.parent.mkdir(parents=True)
    candidate = root / "candidate"
    candidate.mkdir()
    source = candidate / "source.py"
    source.write_text("# preserved source\n")
    archive = root / "source.tar"
    archive.write_bytes(b"preserved archive identity")
    log = helpers.parent / "validator.log"
    log.write_text("original continuation failure\n")
    binding.write_text(
        json.dumps(
            {
                "candidate": str(candidate),
                "archive": str(archive),
                "archive_sha256": digest(archive),
                "files_sha256": {"source.py": digest(source)},
                "helpers_sha256": {},
            }
        )
    )
    owner = {"controller_pid": 11, "controller_start_ticks": "original"}
    state_path.write_text(
        json.dumps(
            owner
            | {
                "status": "needs_attention",
                "keeper_restore": 0,
                "binding_sha256": digest(binding),
                "log": str(log),
                "log_sha256": digest(log),
            }
        )
    )
    gpu_path.write_text(
        json.dumps(
            {
                "status": "needs_attention",
                "owner": owner,
                "execution_binding_sha256": digest(binding),
                "error": "AssertionError: rank_0.model.body.emb.weight",
                "gpu_commands": [{"log": str(log), "log_sha256": digest(log)} for _ in range(6)],
            }
        )
    )
    for mode in ("resumed", "continuous"):
        directory = work / f"outer_leftdiag_fused_full_lr_lower_{mode}"
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps({"completed_steps": 24, "status": "execution_check_complete"}))
        for step in (22, 24):
            checkpoint_directory = directory / f"checkpoints/step_{step:06d}"
            checkpoint_directory.mkdir(parents=True)
            for rank in range(8):
                saved = checkpoint(step)
                saved["rank"] = rank
                if mode == "resumed" and step == 24:
                    saved["model"]["weight"][0, 0] += 0.01
                torch.save(saved, checkpoint_directory / f"rank_{rank}.pt")
    result = diagnose(root)
    assert len(result["pairs"]) == 16
    assert [item["failed_tensor_count"] for item in result["pairs"]] == [0] * 8 + [1] * 8
    assert result["evidence_sha256"][str(state_path)] == digest(state_path)
