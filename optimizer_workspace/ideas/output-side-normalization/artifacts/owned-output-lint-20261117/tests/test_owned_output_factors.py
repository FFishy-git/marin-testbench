# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Candidate-only parity of rank-owned outer factors against the original full factor path."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from experiment.block_toner import (
    BLOCK_KINDS,
    DiagonalMoments,
    OwnedOutputMoments,
    block_update_stack,
)
from toner.fast.engine import layer_counts

SETTINGS = {
    "decay": 0.938,
    "damping": 0.001,
    "inner_power": 0.5,
    "outer_power": 0.125,
    "device": "cpu",
}
ARGS = SimpleNamespace(muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=10, muon_eps=1e-5, muon_ns_mode="f32")


def rows_for(layers):
    generator = torch.Generator().manual_seed(293)
    rows = {}
    for layer in range(layers):
        for kind in BLOCK_KINDS:
            width = 8 if kind in ("gate", "up") else 4
            value = torch.randn(6, width, generator=generator).to(torch.bfloat16)
            if layer == 0:
                value.zero_()
            rows[layer, kind] = value
    return rows


def pair(rows, rank, layers, world=1):
    settings = SETTINGS | {"rows_total": 6, "world": world}
    reference = DiagonalMoments(rows, **settings)
    owned = OwnedOutputMoments(rows, rank=rank, layers=layers, **settings)
    return reference, owned


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("layers", [8, 32])
def test_owned_outer_factors_equal_original_and_cover_only_the_rank_partition(rank, layers, monkeypatch):
    # This test consumes saved pooled moments; distributed EMA behavior remains in
    # the original two-rank tests. Avoid opening unused process groups here.
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(dist, "new_group", lambda backend: None)
    rows = rows_for(layers)
    reference, owned = pair(rows, rank, layers, world=8)
    pooled = DiagonalMoments(rows, **(SETTINGS | {"rows_total": 6, "world": 1}))
    counts = layer_counts(layers, 8)
    first, last = sum(counts[:rank]), sum(counts[: rank + 1])
    expected_keys = {key for key in rows if first <= key[0] < last}
    for update in range(3):
        if update:
            for value in rows.values():
                value.add_(update)
        pooled.update()
        saved = pooled.state_dict()
        reference.load_state_dict(saved)
        owned.load_state_dict(saved)
        expected, actual = reference.factors(), owned.factors()
        assert set(actual) == expected_keys
        for key, (inner, outer) in actual.items():
            assert inner is None
            torch.testing.assert_close(outer, expected[key][1], rtol=0, atol=0)
        assert owned.weight == reference.weight
        for width, moment in owned.moment.items():
            torch.testing.assert_close(moment, reference.moment[width], rtol=0, atol=0)


@pytest.mark.parametrize("apply_left", [True, False])
def test_owned_outer_factors_preserve_original_parameter_and_momentum_updates(apply_left):
    rows = rows_for(32)
    reference, owned = pair(rows, 0, 4)
    reference.update()
    owned.update()
    original = reference.factors()
    selected = owned.factors()
    generator = torch.Generator().manual_seed(359)
    for key, (_, outer) in selected.items():
        width = outer.numel()
        weights, gradients, momenta = [torch.randn(1, width, 4, generator=generator) for _ in range(3)]
        right = (torch.eye(4)[None], torch.eye(4)[None])
        expected = block_update_stack(
            weights,
            gradients,
            momenta,
            (None, original[key][1][None] if apply_left else None),
            right,
            args=ARGS,
            lr=0.02,
            weight_decay=0.1,
        )
        actual = block_update_stack(
            weights,
            gradients,
            momenta,
            (None, outer[None] if apply_left else None),
            right,
            args=ARGS,
            lr=0.02,
            weight_decay=0.1,
        )
        for value, baseline in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, baseline, rtol=0, atol=0)


def test_owned_outer_checkpoint_retains_all_moments_and_exact_next_factors():
    rows = rows_for(4)
    reference, continuous = pair(rows, 0, 4)
    for _ in range(22):
        reference.update()
        continuous.update()
    saved = copy.deepcopy(continuous.state_dict())
    restored_rows = {key: value.clone() for key, value in rows.items()}
    _, restored = pair(restored_rows, 0, 4)
    restored.load_state_dict(saved)
    assert set(saved) == set(reference.state_dict()) == {"moment", "weight"}
    for update in (23, 24):
        for key, value in rows.items():
            value.add_(update / 100)
            restored_rows[key].copy_(value)
        reference.update()
        continuous.update()
        restored.update()
        assert restored.weight == continuous.weight == reference.weight
        for width, moment in reference.moment.items():
            assert saved["moment"][width].shape == moment.shape
            torch.testing.assert_close(restored.moment[width], moment, rtol=0, atol=0)
        baseline = reference.factors()
        for key, (_, outer) in restored.factors().items():
            torch.testing.assert_close(outer, continuous.factors()[key][1], rtol=0, atol=0)
            torch.testing.assert_close(outer, baseline[key][1], rtol=0, atol=0)
