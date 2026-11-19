# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU evidence for refresh age and saved pending-decomposition behavior."""

import concurrent.futures
from types import SimpleNamespace

import pytest
import torch
from experiment.refresh_toner import install_refreshes, join_refreshes
from experiment.toner_state import engine_state, restore_engine
from toner.fast import roots as PR


def group():
    return SimpleNamespace(
        per=1,
        stage=torch.zeros(1, 2, 2),
        C=torch.eye(2)[None],
        tol=torch.ones(1, 3),
        basis_step=[20],
        warm=True,
        graph=object(),
        basis=PR.Basis(torch.eye(2)[None]),
        pending=[None],
        host_out=torch.eye(2)[None],
        host_tol=torch.ones(1, 3),
    )


def engine(statistic):
    return SimpleNamespace(
        groups={"f": statistic},
        side=SimpleNamespace(synchronize=lambda: None),
        state={},
        weight=0.5,
        steps=22,
        t=22,
        fallbacks=0,
        extra=0,
        refresh_install_age=1,
        device="cpu",
    )


class StoredBasis:
    """In-memory basis storage for CPU scheduling tests, not compiled-kernel evidence."""

    def __init__(self, matrix):
        self.SB = matrix.clone()
        self.S2 = matrix.clone()

    def install_many(self, matrix, mask):
        self.SB.copy_(torch.where(mask[:, None, None], matrix, self.SB))
        self.S2.copy_(torch.where(mask[:, None, None], matrix, self.S2))


@pytest.mark.parametrize("completed_early", [False, True])
def test_refresh_joins_at_eligible_update_without_installing_early(completed_early):
    statistic = group()
    future = concurrent.futures.Future()
    if completed_early:
        future.set_result(None)
    statistic.pending[0] = (future, 21)
    assert join_refreshes(statistic, step=21, age=1) == []
    if not completed_early:
        # A real worker completes only after the eligible boundary asks to join.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as workers:
            joined = workers.submit(join_refreshes, statistic, step=22, age=1)
            future.set_result(None)
            assert joined.result(timeout=5) == [(0, 21)]
    else:
        assert join_refreshes(statistic, step=22, age=1) == [(0, 21)]


def test_checkpoint_completion_does_not_advance_pending_installation_age():
    statistic = group()
    future = concurrent.futures.Future()
    future.set_result(None)
    statistic.pending[0] = (future, 22)
    original = engine(statistic)
    saved = engine_state(original)
    restored = engine(group())
    restore_engine(restored, saved)
    assert join_refreshes(restored.groups["f"], step=22, age=1) == []
    assert join_refreshes(restored.groups["f"], step=23, age=1) == [(0, 22)]
    torch.testing.assert_close(restored.groups["f"].host_out, statistic.host_out, rtol=0, atol=0)
    torch.testing.assert_close(restored.groups["f"].host_tol, statistic.host_tol, rtol=0, atol=0)
    assert restored.groups["f"].basis_step == statistic.basis_step


def test_failed_host_decomposition_propagates_at_installation_boundary():
    statistic = group()
    future = concurrent.futures.Future()
    future.set_exception(RuntimeError("host decomposition failed"))
    statistic.pending[0] = (future, 22)
    assert join_refreshes(statistic, step=22, age=1) == []
    with pytest.raises(RuntimeError, match="host decomposition failed"):
        join_refreshes(statistic, step=23, age=1)


@pytest.mark.parametrize("fallback_step", [20, 23])
def test_installation_preserves_newer_fallback_and_records_pending_age(fallback_step):
    statistic = group()
    statistic.basis = StoredBasis(torch.eye(2)[None])
    statistic.basis_step = [fallback_step]
    statistic.host_out.copy_(torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]]))
    statistic.host_tol.fill_(0.25)
    future = concurrent.futures.Future()
    future.set_result(None)
    statistic.pending[0] = (future, 22)
    before = engine_state(engine(statistic))
    assert install_refreshes(statistic, step=22, age=1, device="cpu") == []
    records = install_refreshes(statistic, step=23, age=1, device="cpu")
    after = engine_state(engine(statistic))
    assert records == [{"index": 0, "source_step": 22, "installation_step": 23, "installed": fallback_step <= 22}]
    assert after["groups"]["f"]["pending"] == [None]
    if fallback_step <= 22:
        expected = StoredBasis(statistic.host_out)
        torch.testing.assert_close(after["groups"]["f"]["SB"], expected.SB, rtol=0, atol=0)
        torch.testing.assert_close(after["groups"]["f"]["S2"], expected.S2, rtol=0, atol=0)
        torch.testing.assert_close(after["groups"]["f"]["tol"], statistic.host_tol, rtol=0, atol=0)
        assert after["groups"]["f"]["basis_step"] == [22]
    else:
        for key in ("SB", "S2", "tol"):
            torch.testing.assert_close(after["groups"]["f"][key], before["groups"]["f"][key], rtol=0, atol=0)
        assert after["groups"]["f"]["basis_step"] == [23]
