# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute source-authenticated compiled start variants against independent CPU arithmetic."""

import json
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import rational_root_start as installation
import torch
from benchmark_diagonal import source_definitions


@pytest.fixture
def pinned_roots():
    task = Path(__file__).resolve().parents[2 if Path(__file__).parent.name == "tests" else 1]
    manifest = json.loads((task / "experiment/upstream_manifest.json").read_bytes())
    source = task / f"artifacts/external-sources/Toner-{manifest['revision']}/src/toner/fast/roots.py"
    assert installation.digest(source) == manifest["files"]["src/toner/fast/roots.py"]
    start = source_definitions(
        source, ("_start",), {"torch": torch, "EPS": 1e-3, "QUARTER": 0.25, "U32": 2.0**-24, "LOWP": torch.float16}
    )["_start"]
    return SimpleNamespace(__file__=str(source), _start=start, start=torch.compile(start, dynamic=False))


def decimal_output(diagonal, off_diagonal):
    with localcontext() as context:
        context.prec = 100
        coefficients = []
        for row in diagonal:
            values = [Decimal.from_float(float(value)) for value in row]
            coefficients.append(
                [
                    [
                        float(
                            -Decimal("0.25") / (x * x.sqrt().sqrt())
                            if x == y
                            else (1 / x.sqrt().sqrt() - 1 / y.sqrt().sqrt()) / (x - y)
                        )
                        for y in values
                    ]
                    for x in values
                ]
            )
    return torch.tensor(coefficients, dtype=torch.float64) * off_diagonal.double()


@pytest.mark.parametrize("variant", ("pinned", "rational"))
def test_installed_compiled_start_preserves_inputs_and_exposes_independent_accuracy(pinned_roots, variant):
    B = torch.diag_embed(torch.tensor([[1.0, 1.000002, 2.0], [0.25, 0.250001, 0.75]]))
    B += torch.tensor([[0.0, 0.01, -0.02], [0.01, 0.0, 0.03], [-0.02, 0.03, 0.0]])
    original_callable = pinned_roots.start
    original = pinned_roots._start(B, 0.0)
    before, rng = B.clone(), torch.get_rng_state().clone()
    receipt = installation.install_start(pinned_roots, variant)
    actual = pinned_roots.start(B, 0.0)
    repeated = pinned_roots.start(B, 0.0)
    expected = decimal_output(original[0], original[1])
    for value, again in zip(actual, repeated, strict=True):
        torch.testing.assert_close(value, again, rtol=0, atol=0)
        assert torch.isfinite(value).all()
    for index in (0, 1, 2, 3, 6, 7, 8):
        torch.testing.assert_close(actual[index], original[index], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual[5], actual[4].half(), rtol=0, atol=0)
    torch.testing.assert_close(actual[9], actual[4].square().sum((-2, -1)).sqrt(), rtol=1e-5, atol=1e-6)
    assert torch.equal(B, before) and torch.equal(torch.get_rng_state(), rng)
    assert receipt["variant"] == variant and receipt["compilation_preserved"] is True
    if variant == "rational":
        torch.testing.assert_close(actual[4].double(), expected, rtol=1e-5, atol=1e-6)
        assert (actual[4].double() - expected).abs().max() < (original[4].double() - expected).abs().max()
    else:
        assert pinned_roots.start is original_callable
        torch.testing.assert_close(actual[4], original[4], rtol=1e-5, atol=1e-6)


def test_installer_refuses_already_replaced_callable_without_mutation(pinned_roots):
    installation.install_start(pinned_roots, "rational")
    installed = pinned_roots.start
    with pytest.raises(ValueError, match="original compiled/source identity"):
        installation.install_start(pinned_roots, "pinned")
    assert pinned_roots.start is installed
