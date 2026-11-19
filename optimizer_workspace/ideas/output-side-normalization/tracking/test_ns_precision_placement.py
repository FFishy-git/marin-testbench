# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check precision-placement numerics against independent per-matrix arithmetic."""

import json
import subprocess
from pathlib import Path

import ns_operand_precision as precision
import ns_precision_placement as placement
import prepare_ns_precision_placement as preparation
import pytest
import torch

PINNED = "artifacts/pinned-sources-materialized/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/src/toner/fast/muon.py"
PARENT = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/run-preparation/ns-compiler-localization-execution-cpu/20261119T1225"
)


@pytest.fixture
def source_path():
    staged = Path(__file__).resolve().parents[1] / PINNED
    return staged if staged.is_file() else PARENT / PINNED


def source_batch(shape):
    size = shape[0] * shape[1] * shape[2]
    return torch.sin(torch.arange(size, dtype=torch.float32).reshape(shape) * 0.73) + 0.1


@pytest.mark.parametrize("shape", ((2, 3, 5), (2, 5, 3), (2, 4, 4)))
@pytest.mark.parametrize("mode", ("f32", "bf16_1"))
def test_placement_homogeneous_schedule_matches_pinned_map_exactly(source_path, shape, mode):
    source = source_batch(shape)
    before, rng = source.clone(), torch.get_rng_state().clone()
    actual = placement.numerical(source_path, (mode,) * 7, eps=1e-5)(source)
    expected = precision.pinned_numerical(source_path, mode, steps=7, eps=1e-5)(source)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


def independent_placement(source, schedule):
    outputs = []
    for matrix in source:
        x = matrix / (matrix.norm() + 1e-5)
        transposed = x.shape[0] > x.shape[1]
        if transposed:
            x = x.T
        for mode in schedule:

            def product(left, right, mode=mode):
                if mode == "bf16_1":
                    left, right = left.to(torch.bfloat16).float(), right.to(torch.bfloat16).float()
                return left @ right

            gram = product(x, x.T)
            correction = -4.7750 * gram + 2.0315 * product(gram, gram)
            x = 3.4445 * x + product(correction, x)
        outputs.append(x.T if transposed else x)
    return torch.stack(outputs)


def independent_batched_placement(source, schedule):
    x = source / (torch.linalg.matrix_norm(source, dim=(-2, -1), keepdim=True) + 1e-5)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    for mode in schedule:

        def product(left, right, mode=mode):
            if mode == "bf16_1":
                left, right = left.to(torch.bfloat16).float(), right.to(torch.bfloat16).float()
            return torch.bmm(left, right)

        gram = product(x, x.transpose(-2, -1))
        correction = -4.7750 * gram + 2.0315 * product(gram, gram)
        x = 3.4445 * x + product(correction, x)
    return x.transpose(-2, -1) if transposed else x


@pytest.mark.parametrize("shape", ((2, 3, 5), (2, 5, 3)))
@pytest.mark.parametrize("schedule", (("f32", "bf16_1", "bf16_1", "bf16_1", "bf16_1"), ("bf16_1",) * 4 + ("f32",)))
def test_placement_mixed_iterations_match_independent_products_without_state_leak(source_path, shape, schedule):
    source = source_batch(shape)
    before = source.clone()
    evaluate = placement.numerical(source_path, schedule, eps=1e-5)
    expected = independent_batched_placement(source, schedule)
    first, second = evaluate(source), evaluate(source)
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    opposite = placement.numerical(source_path, tuple(reversed(schedule)), eps=1e-5)(source)
    assert not torch.equal(first, opposite)


@pytest.mark.parametrize("shape", ((2, 3, 5), (2, 5, 3)))
def test_placement_preserves_per_matrix_reduction_difference_as_negative_evidence(source_path, shape):
    source = source_batch(shape)
    schedule = ("f32",) + ("bf16_1",) * 4
    expected = independent_placement(source, schedule)
    actual = placement.numerical(source_path, schedule, eps=1e-5)(source)
    comparison = placement.replay.compare_outputs(actual, expected)
    assert comparison["within_tolerance"] is False
    assert comparison["changed_elements"] > 0
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()


def test_placement_retained_summary_exposes_a_changed_repeat(source_path):
    source = source_batch((2, 3, 5))
    reference = precision.reference_fp64(source, (3.4445, -4.7750, 2.0315), steps=5, eps=1e-5)
    groups = {name: {"source": source, "output": source} for name in ("continuous", "resumed")}
    references = {name: {"5": reference, "partial_polar": reference} for name in groups}
    evaluate = placement.numerical(source_path, ("f32",) * 5, eps=1e-5)
    outputs = tuple(evaluate(source).clone() for _ in range(4))
    first = placement.summaries(outputs, groups, references, 5)
    assert first["same_input_repeats"]["resumed"]["exact"]
    outputs[3][0, 0, 0] += 0.125
    changed = placement.summaries(outputs, groups, references, 5)
    assert changed["same_input_repeats"]["resumed"]["changed_elements"] == 1
    assert changed["same_input_repeats"]["resumed"]["within_tolerance"] is False
    assert changed["output_pair"]["exact"]


def test_placement_run_refuses_rebound_helper_before_output_creation(tmp_path, monkeypatch):
    helpers = Path(placement.__file__).resolve().parent
    receipt = tmp_path / "cpu.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "cpu_precision_placement_sources_tests_verified_no_gpu_authority",
                "helpers": str(helpers),
                "helpers_sha256": {"ns_operand_precision.py": "unused"},
            }
        )
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(precision, "__file__", str(tmp_path / "mutable_precision.py"))
    out = tmp_path / "measurement"
    with pytest.raises(RuntimeError, match="mutable helper"):
        placement.run(out, receipt)
    assert not out.exists()


def test_placement_verify_refuses_mutable_entry_before_retained_tensor_reads(tmp_path, monkeypatch):
    receipt = tmp_path / "cpu.json"
    receipt.write_text(json.dumps({"helpers": str(tmp_path / "archived"), "helpers_sha256": {}}))
    result = {
        "status": "completed_cpu_precision_placement_not_cuda_or_optimizer_acceptance",
        "trials": [{"name": name} for name in placement.schedules()],
        "cpu_receipt": str(receipt),
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError, match="archived entry"):
        placement.verify(tmp_path)
    assert not (tmp_path / "retained-verification.json").exists()


def test_placement_retained_replay_reconstructs_all_saved_comparisons_and_exposes_mutation(source_path):
    source = source_batch((2, 3, 5))
    groups = {name: {"source": source, "output": source * 0.5} for name in ("continuous", "resumed")}
    trial = placement.replay.replay_pair(groups, placement.numerical(source_path, ("f32",) * 5, eps=1e-5))
    reconstructed = placement.retained_replay(trial.outputs, groups)
    assert reconstructed == trial.report
    trial.outputs[3][0, 0, 0] += 0.125
    changed = placement.retained_replay(trial.outputs, groups)
    assert changed != reconstructed
    assert (
        changed["saved_comparisons"][3]["own_saved_output"] != reconstructed["saved_comparisons"][3]["own_saved_output"]
    )


def test_placement_preparation_timeout_preserves_failure_receipt_and_source_archive(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(preparation.subprocess, "run", timeout)
    out = tmp_path / "failed-cpu"
    with pytest.raises(OSError):
        preparation.cpu(placement.replay.ROOT, out)
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "needs_attention"
    assert receipt["exit_code"] == 124
    assert receipt["test_count"] == 0
    assert receipt["error"]
    assert (out / "source.tar").is_file()
    assert (out / "cpu.log").is_file()
