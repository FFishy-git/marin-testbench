# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Candidate-only corruption checks for retained full moments and owned-factor coverage."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from experiment.block_toner import BLOCK_KINDS, DiagonalMoments, OwnedOutputMoments
from experiment.owned_output_observer import OwnedOutputVerifier


@pytest.mark.parametrize("corruption", [None, "unowned_moment", "coverage"])
def test_owned_output_observer_checks_full_moments_and_exact_projection_coverage(corruption, monkeypatch):
    # Use saved pooled moments while the original distributed tests exercise the
    # all-reduce. The eight-rank constructor still derives its real ownership.
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(dist, "new_group", lambda backend: None)
    rows = {
        (layer, kind): torch.full((6, 4), float(layer + 1), dtype=torch.bfloat16)
        for layer in range(8)
        for kind in BLOCK_KINDS
    }
    settings = {
        "decay": 0.938,
        "damping": 0.001,
        "inner_power": 0.5,
        "outer_power": 0.125,
        "rows_total": 6,
        "device": "cpu",
    }
    reference = DiagonalMoments(rows, world=1, **settings)
    reference.update()
    actual = OwnedOutputMoments(rows, rank=1, layers=8, world=8, **settings)
    actual.load_state_dict(reference.state_dict())
    verifier = OwnedOutputVerifier.__new__(OwnedOutputVerifier)
    verifier.actual, verifier.reference = actual, reference
    verifier.fp64_moment = {width: moment.double().clone() for width, moment in reference.moment.items()}
    verifier.factors = reference.factors()
    owned_keys = [(layer, kind) for layer, kind in rows if layer == 1]
    verifier.fast = SimpleNamespace(
        rank=1,
        shape_groups=[((4, 4), [[], [(layer, kind, None) for layer, kind in owned_keys]])],
    )
    if corruption == "unowned_moment":
        actual.moment[4][-1, 0] += 1
        with pytest.raises(AssertionError):
            verifier.check_factors()
    elif corruption == "coverage":
        verifier.fast.shape_groups[0][1][1].pop()
        with pytest.raises(ValueError, match="exactly the locally updated projections"):
            verifier.check_factors()
    else:
        verifier.check_factors()
        assert set(actual.factors()) == set(owned_keys)
