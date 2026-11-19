# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent divided-difference and pinned start-output tests for the CPU hypothesis."""

import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

import pytest
import root_start_rational as rational
import torch
from benchmark_diagonal import source_definitions


def decimal_coefficient(left: float, right: float) -> float:
    with localcontext() as context:
        context.prec = 100
        x, y = Decimal.from_float(left), Decimal.from_float(right)
        if x == y:
            return float(-Decimal("0.25") * (1 / x.sqrt().sqrt()) / x)
        return float((1 / x.sqrt().sqrt() - 1 / y.sqrt().sqrt()) / (x - y))


@pytest.fixture
def pinned_start():
    task = Path(__file__).resolve().parents[1]
    manifest = json.loads((task / "experiment/upstream_manifest.json").read_text())
    source = task / f"artifacts/external-sources/Toner-{manifest['revision']}/src/toner/fast/roots.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["files"]["src/toner/fast/roots.py"]
    return source_definitions(
        source,
        ("_start",),
        {"torch": torch, "EPS": 1e-3, "QUARTER": 0.25, "U32": 2.0**-24, "LOWP": torch.float16},
    )["_start"]


@pytest.mark.parametrize("scale", (1e-6, 1e-3, 1.0, 64.0, 2048.0))
def test_quarter_coefficients_match_independent_decimal_across_near_and_distant_pairs(scale):
    values = torch.tensor([scale, scale * (1 + 2e-6), scale * 1.125, scale * 8], dtype=torch.float32)
    values = torch.cat((values, torch.nextafter(values[:1], torch.full_like(values[:1], float("inf")))))
    expected = torch.tensor(
        [[decimal_coefficient(float(left), float(right)) for right in values] for left in values],
        dtype=torch.float64,
    )
    actual = rational.quarter_divided_difference(values)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, actual.mT)


@pytest.mark.parametrize("freeze", (0.0, 2.0))
def test_rational_start_preserves_unaffected_outputs_and_matches_decimal_off_diagonal(pinned_start, freeze):
    diagonal = torch.tensor([[1.0, 1.000002, 2.0], [0.25, 0.250001, 0.75]], dtype=torch.float32)
    off = torch.tensor([[0.0, 0.01, -0.02], [0.01, 0.0, 0.03], [-0.02, 0.03, 0.0]])
    B = torch.diag_embed(diagonal) + off
    before, rng = B.clone(), torch.get_rng_state().clone()
    original, actual = pinned_start(B, freeze), rational.rational_start(B, freeze)
    for index in (0, 1, 2, 3, 6, 7, 8):
        torch.testing.assert_close(actual[index], original[index], rtol=0, atol=0)
    expected = (
        torch.stack(
            [
                torch.tensor(
                    [[decimal_coefficient(float(left), float(right)) for right in row] for left in row],
                    dtype=torch.float64,
                )
                for row in original[0]
            ]
        )
        * original[1].double()
    )
    torch.testing.assert_close(actual[4].double(), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[5], actual[4].half(), rtol=0, atol=0)
    torch.testing.assert_close(actual[9], actual[4].square().sum((-2, -1)).sqrt(), rtol=0, atol=0)
    assert torch.equal(B, before)
    assert torch.equal(torch.get_rng_state(), rng)
    assert (original[4].double() - expected).abs().max() > (actual[4].double() - expected).abs().max()


def test_rational_start_retains_pinned_clamped_pairs(pinned_start):
    B = torch.tensor([[[-0.0009998, 0.00001], [0.00001, 1.0]]], dtype=torch.float32)
    original, actual = pinned_start(B, 0.0), rational.rational_start(B, 0.0)
    torch.testing.assert_close(actual[4], original[4], rtol=0, atol=0)
    torch.testing.assert_close(actual[5], original[5], rtol=0, atol=0)
