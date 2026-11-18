# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU regression for full-shaped trace and exact owned roots at production widths."""

import pytest
import torch
import torch.distributed as dist
from experiment.block_toner import BLOCK_KINDS, DiagonalMoments, OwnedOutputMoments
from torch.utils._python_dispatch import TorchDispatchMode


class TraceShapes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten.sum.dim_IntList:
            self.shapes.append(tuple(args[0].shape))
        return function(*args, **({} if kwargs is None else kwargs))


@pytest.mark.parametrize("rank", range(8))
def test_owned_root_uses_full_trace_shape_and_retains_exact_production_width_factors(rank, monkeypatch):
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(dist, "new_group", lambda backend: None)
    rows = {
        (layer, kind): torch.zeros(1, 2048 if kind in ("gate", "up") else 512, dtype=torch.bfloat16)
        for layer in range(32)
        for kind in BLOCK_KINDS
    }
    settings = {
        "decay": 0.938,
        "damping": 0.001,
        "inner_power": 0.5,
        "outer_power": 0.125,
        "rows_total": 128,
        "world": 8,
        "device": "cpu",
    }
    full = DiagonalMoments(rows, **settings)
    owned = OwnedOutputMoments(rows, rank=rank, layers=32, **settings)
    generator = torch.Generator().manual_seed(891)
    for moment in full.moment.values():
        moment.copy_(torch.rand(moment.shape, generator=generator))
        moment[::9] = 0
    owned.load_state_dict(full.state_dict())
    expected = full.factors()
    with TraceShapes() as trace:
        actual = owned.factors()
    assert trace.shapes == [(160, 512), (64, 2048)]
    assert set(actual) == {key for key in rows if rank * 4 <= key[0] < (rank + 1) * 4}
    for key, (inner, outer) in actual.items():
        assert inner is None
        torch.testing.assert_close(outer, expected[key][1], rtol=0, atol=0)
    for width, moment in full.moment.items():
        torch.testing.assert_close(owned.moment[width], moment, rtol=0, atol=0)
