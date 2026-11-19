# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent reference and retained-output regressions for the unissued CUDA start trial."""

import copy
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
import root_start_accuracy as accuracy
import root_start_rational as rational
import root_start_trial as trial
import torch
from benchmark_diagonal import source_definitions


@pytest.fixture
def pinned():
    task = Path(__file__).resolve().parents[1]
    manifest = json.loads((task / "experiment/upstream_manifest.json").read_bytes())
    source = task / f"artifacts/external-sources/Toner-{manifest['revision']}/src/toner/fast/roots.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["files"]["src/toner/fast/roots.py"]
    return source_definitions(
        source,
        ("_start",),
        {"torch": torch, "EPS": 1e-3, "QUARTER": 0.25, "U32": 2.0**-24, "LOWP": torch.float16},
    )["_start"]


@pytest.mark.parametrize("scale", (1e-6, 1e-3, 1.0, 64.0, 2048.0))
def test_log_reference_matches_decimal_for_equal_adjacent_near_and_distant_pairs(scale):
    base = torch.tensor(scale, dtype=torch.float64)
    values = torch.stack((base, torch.nextafter(base, torch.full_like(base, float("inf"))), base * 1.000002, base * 8))
    with localcontext() as context:
        context.prec = 100
        expected = []
        for left in values:
            x = Decimal.from_float(float(left))
            row = []
            for right in values:
                y = Decimal.from_float(float(right))
                coefficient = (
                    -Decimal("0.25") / (x * x.sqrt().sqrt())
                    if x == y
                    else (1 / x.sqrt().sqrt() - 1 / y.sqrt().sqrt()) / (x - y)
                )
                row.append(float(coefficient))
            expected.append(row)
    observed = accuracy.fp64_divided_difference(values)
    torch.testing.assert_close(observed, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=0)
    torch.testing.assert_close(observed, observed.mT, rtol=1e-12, atol=0)


def test_log_reference_handles_mixed_scales_without_negative_log1p_cancellation():
    values = torch.tensor([1e-6, 1e-3, 1.0, 2048.0, 1e20], dtype=torch.float64)
    with localcontext() as context:
        context.prec = 100
        expected = []
        for left in values:
            x = Decimal.from_float(float(left))
            row = []
            for right in values:
                y = Decimal.from_float(float(right))
                coefficient = (
                    -Decimal("0.25") / (x * x.sqrt().sqrt())
                    if x == y
                    else (1 / x.sqrt().sqrt() - 1 / y.sqrt().sqrt()) / (x - y)
                )
                row.append(float(coefficient))
            expected.append(row)
    observed = accuracy.fp64_divided_difference(values)
    torch.testing.assert_close(observed, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=0)
    assert torch.equal(observed, observed.mT)
    assert torch.isfinite(observed).all()


def fixture(pinned, index):
    diagonal = torch.tensor([[1.0, 1.000002, 2.0]], dtype=torch.float32)
    off = torch.tensor([[0.0, 0.01, -0.02], [0.01, 0.0, 0.03], [-0.02, 0.03, 0.0]])
    B = torch.diag_embed(diagonal) + off
    sources = {group: {mode: B.clone() for mode in accuracy.MODES} for group in accuracy.GROUPS}
    retained = []
    for coordinate in accuracy.coordinates(index):
        start = pinned if coordinate["variant"] == "pinned" else rational.rational_start
        output = start(sources[coordinate["group"]][coordinate["mode"]], 0.0)
        retained.append({**coordinate, "output": tuple(value.clone() for value in output)})
    return sources, retained


@pytest.mark.parametrize("index", (0, 1))
def test_complete_retained_analysis_exposes_accuracy_gain_without_inventing_crossed_improvement(pinned, index):
    sources, retained = fixture(pinned, index)
    original = copy.deepcopy(sources)
    rng = torch.get_rng_state().clone()
    result = accuracy.summarize(retained, sources, pinned, 0.0, index)
    assert result["all_repeat_outputs_exact_finite"]
    assert result["all_unaffected_outputs_exact_finite"]
    assert result["all_casts_match_independent_rounding"]
    assert result["all_rational_O_within_fp64_tolerance"]
    assert any(not item["within_tolerance"] for item in result["O_fp64_accuracy"] if item["variant"] == "pinned")
    assert all(item["exact"] for item in result["crossed_outputs"])
    assert len(result["O_fp64_accuracy"]) == 32
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(torch.equal(sources[group][mode], original[group][mode]) for group in sources for mode in sources[group])
    json.dumps(result, allow_nan=False)


def test_repeat_intermediate_drift_is_retained_even_when_O_is_unchanged(pinned):
    sources, retained = fixture(pinned, 0)
    target = next(item for item in retained if item["variant"] == "rational" and item["phase"] == "replay_repeat")
    target["output"][3][0, 0] += 0.01
    result = accuracy.summarize(retained, sources, pinned, 0.0, 0)
    assert not result["all_repeat_outputs_exact_finite"]
    assert not result["all_unaffected_outputs_exact_finite"]
    assert result["all_rational_O_within_fp64_tolerance"]
    failed = [item for item in result["repeat_checks"] if not item["exact"]]
    assert len(failed) == 1
    assert failed[0]["phase"] == "replay_repeat"
    assert failed[0]["changed_tensors"][0]["mismatched_elements"] == 1


def test_wrong_fp16_operand_is_detected_without_discarding_good_FP32_accuracy(pinned):
    sources, retained = fixture(pinned, 0)
    target = next(item for item in retained if item["variant"] == "rational")
    target["output"][5][0, 0, 1] += 0.01
    result = accuracy.summarize(retained, sources, pinned, 0.0, 0)
    assert not result["all_casts_match_independent_rounding"]
    assert not result["all_repeat_outputs_exact_finite"]
    assert result["all_rational_O_within_fp64_tolerance"]
    assert any(item["operand"] == "Ob" and not item["exact"] for item in result["cast_checks"])


def test_nonfinite_O_cannot_become_an_accuracy_pass_or_a_repeat_pass(pinned):
    sources, retained = fixture(pinned, 0)
    target = next(item for item in retained if item["variant"] == "rational")
    target["output"][4].fill_(float("inf"))
    result = accuracy.summarize(retained, sources, pinned, 0.0, 0)
    assert not result["all_repeat_outputs_exact_finite"]
    assert not result["all_rational_O_within_fp64_tolerance"]
    assert any(item["max_absolute_error"] is None for item in result["O_fp64_accuracy"])
    json.dumps(result, allow_nan=False)


def test_missing_final_phase_and_partial_output_cannot_be_analyzed_as_complete(pinned):
    sources, retained = fixture(pinned, 0)
    with pytest.raises(ValueError, match="coordinates"):
        accuracy.summarize(retained[:-1], sources, pinned, 0.0, 0)
    retained[0]["output"] = retained[0]["output"][:-1]
    with pytest.raises(ValueError, match="all ten"):
        accuracy.summarize(retained, sources, pinned, 0.0, 0)


def test_graph_component_rejects_cpu_input_before_cuda_or_numerical_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unowned CPU graph trial touched CUDA or executed numerics")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", forbidden)
    with pytest.raises(RuntimeError, match="caller-owned resident CUDA"):
        trial.retain_kernel(torch.eye(3).unsqueeze(0), 0.0, forbidden)
