# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise saved all-rank and loss comparisons without GPU or production acceptance."""

import json

import fused_continuation_validation as validation
import pytest
import torch


@pytest.mark.parametrize("fault", ("none", "manifest", "history", "complete", "rank", "config"))
def test_original_comparison_requires_retained_manifest_history_and_checkpoint_bytes(tmp_path, fault):
    candidate, directory = tmp_path / "candidate", tmp_path / "original"
    config = candidate / validation.CONFIG
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"fixture": "original full schedule"}) + "\n")
    directory.mkdir()
    (directory / "config.json").write_bytes(config.read_bytes())
    manifest = directory / "run.json"
    manifest.write_text(json.dumps({"source": "original failed attribution"}) + "\n")
    history = directory / "metrics_from_000000.jsonl"
    history.write_text(json.dumps({"train_loss": 6.23}) + "\n")
    checkpoints = {}
    for step in (22, 24):
        checkpoint = directory / f"checkpoints/step_{step:06d}"
        checkpoint.mkdir(parents=True)
        receipt = checkpoint / "complete.json"
        receipt.write_text(json.dumps({"step": step, "world": 8}) + "\n")
        ranks = {}
        for rank in range(8):
            path = checkpoint / f"rank_{rank}.pt"
            torch.save({"rank": rank, "state": torch.ones(2) * step}, path)
            ranks[path.name] = validation.sha256(path)
        checkpoints[str(step)] = {"receipt_sha256": validation.sha256(receipt), "rank_sha256": ranks}
    execution = {
        "exit_code": 0,
        "trajectory": {
            "manifest_sha256": validation.sha256(manifest),
            "history_sha256": {history.name: validation.sha256(history)},
            "checkpoints": checkpoints,
        },
    }
    paths = {
        "manifest": manifest,
        "history": history,
        "complete": receipt,
        "rank": path,
        "config": directory / "config.json",
    }
    if fault != "none":
        paths[fault].write_text("{}\n")
        with pytest.raises(ValueError):
            validation.require_original_trajectory(directory, execution, candidate)
    else:
        before = {source: validation.sha256(source) for source in paths.values()}
        validation.require_original_trajectory(directory, execution, candidate)
        assert {source: validation.sha256(source) for source in paths.values()} == before


@pytest.mark.parametrize("fault", ("none", "tensor", "counter", "config"))
def test_saved_all_rank_comparison_reports_first_update_failures_and_preserves_bytes(tmp_path, fault):
    actual, expected = tmp_path / "resumed", tmp_path / "continuous"
    paths = []
    for directory in (actual, expected):
        checkpoint = directory / "checkpoints/step_000023"
        checkpoint.mkdir(parents=True)
        for rank in range(8):
            state = {
                "rank": rank,
                "world": 8,
                "step": 23,
                "config": {"fixture": "same-source"},
                "next_global_sequence": 23 * 128,
                "optimizers": [{"state": torch.tensor([0.3, 0.7])}],
                "fast": {"momentum": torch.tensor([[0.2, 0.8]]), "steps": 23},
                "output": None,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.zeros(16, dtype=torch.uint8),
                "python_rng": (3, (1, 2), None),
            }
            if rank == 0:
                state["model"] = {"weight": torch.ones(2)}
            if directory == actual and rank == 3:
                if fault == "tensor":
                    state["fast"]["momentum"][0, 1] += 0.05
                elif fault == "counter":
                    state["fast"]["steps"] = 22
                elif fault == "config":
                    state["config"] = {"fixture": "wrong-source"}
            path = checkpoint / f"rank_{rank}.pt"
            torch.save(state, path)
            paths.append(path)
    before = {path: validation.sha256(path) for path in paths}
    if fault == "config":
        with pytest.raises(ValueError):
            validation.checkpoint_pair(actual, expected, 23)
    else:
        result = validation.checkpoint_pair(actual, expected, 23)
        assert len(result["ranks"]) == 8
        assert result["within_tolerance"] == (fault == "none")
        rank = result["ranks"][3]
        if fault == "tensor":
            assert rank["failed_tensor_count"] == 1
            assert rank["changed_tensors"][0]["name"] == "rank_3.fast.momentum"
        elif fault == "counter":
            assert rank["scalar_or_structure_differences"][0]["name"] == "rank_3.fast.steps"
        else:
            assert sum(item["failed_tensor_count"] for item in result["ranks"]) == 0
    assert {path: validation.sha256(path) for path in paths} == before


@pytest.mark.parametrize("fault", ("none", "loss", "coverage"))
def test_loss_comparison_checks_complete_histories_and_retains_out_of_tolerance_values(tmp_path, fault):
    directories = [tmp_path / name for name in ("actual", "expected")]
    for directory in directories:
        directory.mkdir()
        rows = [{"step": step, "tokens": step * 524288, "train_loss": 6.0 + step / 100} for step in range(1, 25)]
        for row in rows:
            if row["step"] in (1, 22, 24):
                row.update(c4_loss=7.0, selection_loss=7.2, heldout_losses={"heldout/example": 7.2})
        if directory == directories[0]:
            if fault == "loss":
                rows[22]["train_loss"] += 0.02
            elif fault == "coverage":
                rows[-1].pop("c4_loss")
        config = {"training": {"steps": 5120, "batch_tokens": 524288, "tokens": 2684354560}}
        (directory / "config.json").write_text(json.dumps(config) + "\n")
        (directory / "run.json").write_text(
            json.dumps({"completed_steps": 24, "config": config, "status": "execution_check_complete"}) + "\n"
        )
        (directory / "metrics_from_000000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if fault == "coverage":
        with pytest.raises(ValueError):
            validation.loss_pair(*directories)
    else:
        result = validation.loss_pair(*directories)
        assert result["comparisons"] == 33
        assert result["within_tolerance"] == (fault == "none")
        if fault == "loss":
            assert len(result["failures"]) == 1
            failure = result["failures"][0]
            assert failure["step"] == 23 and failure["metric"] == "train_loss"
            assert failure["actual"] == pytest.approx(6.25) and failure["expected"] == pytest.approx(6.23)
