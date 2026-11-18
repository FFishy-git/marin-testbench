# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU lifecycle and storage tests; CUDA graph numerical parity remains unexecuted."""

import ast
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import prepare_restored_roots as preparation
import pytest
import torch

TASK = Path(__file__).resolve().parents[1]


@pytest.fixture
def engine():
    groups = {}
    for name, width in (("d", 4), ("f", 6)):
        matrix = torch.arange(2 * width * width, dtype=torch.float32).reshape(2, width, width)
        covariance = matrix @ matrix.mT + torch.eye(width)
        basis = torch.linalg.qr(matrix + torch.eye(width))[0]
        high = basis.half()
        low = ((basis - high.float()) * 4096).half()
        groups[name] = SimpleNamespace(
            per=2,
            n=width,
            C=covariance,
            tol=torch.tensor([1e-4, 2e-4]),
            basis=SimpleNamespace(SB=torch.cat([low, high], -2), S2=torch.cat([high, low], -1)),
            basis_step=[19, 20],
            pending=[{"step": 21, "basis": basis[0].clone()}, None],
            graph=None,
            warm=False,
        )
    item = SimpleNamespace(
        args=SimpleNamespace(toner_graphs=True, toner_inline=20),
        groups=groups,
        device="cpu",
        steps=22,
        begun=False,
        state={"momentum": torch.arange(12, dtype=torch.float32)},
        weight=0.5,
        t=8,
        fallbacks=3,
        extra=7,
        cand={"previous": torch.tensor([1.0])},
    )

    def numerical(group):
        values, vectors = torch.linalg.eigh(group.C)
        return (vectors * values.rsqrt().unsqueeze(-2)) @ vectors.mT

    item._roots = numerical
    return item


def numerical_snapshot(engine):
    return copy.deepcopy(
        {
            "groups": {
                name: {key: value for key, value in vars(group).items() if key not in ("graph", "warm")}
                for name, group in engine.groups.items()
            },
            "state": engine.state,
            "steps": engine.steps,
            "weight": engine.weight,
            "t": engine.t,
            "fallbacks": engine.fallbacks,
            "extra": engine.extra,
            "cand": engine.cand,
            "begun": engine.begun,
        }
    )


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_same(left, right)
    elif isinstance(expected, SimpleNamespace):
        assert_same(vars(actual), vars(expected))
    else:
        assert actual == expected


def test_preparation_retains_numerical_state_rng_and_independent_root_inputs(engine):
    before = numerical_snapshot(engine)
    rng = torch.random.get_rng_state().clone()
    root = engine._roots
    staged = []

    def observable_roots(group):
        original = next(value for value in engine.groups.values() if value.n == group.C.shape[-1])
        for actual, expected in (
            (group.C, original.C),
            (group.tol, original.tol),
            (group.basis.SB, original.basis.SB),
            (group.basis.S2, original.basis.S2),
        ):
            assert actual.untyped_storage().data_ptr() != expected.untyped_storage().data_ptr()
            assert actual.stride() == expected.stride()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        output = root(group)
        staged.append(output)
        return output

    engine._roots = observable_roots
    assert preparation.prepare_restored_roots(engine) == ["d", "f"]
    assert all(group.warm and group.graph is None for group in engine.groups.values())
    assert_same(numerical_snapshot(engine), before)
    assert torch.equal(torch.random.get_rng_state(), rng)
    staged[0].zero_()
    assert torch.count_nonzero(engine.groups["d"].C) > 0


def test_later_group_failure_keeps_all_readiness_false_and_restored_inputs_unchanged(engine):
    before = numerical_snapshot(engine)
    numerical = engine._roots

    def fail_after_scratch_mutation(group):
        if group.C.shape[-1] == 6:
            group.C.zero_()
            group.tol.fill_(1)
            group.basis.SB.zero_()
            group.basis.S2.zero_()
            raise RuntimeError("synthetic root failure")
        return numerical(group)

    engine._roots = fail_after_scratch_mutation
    for group in engine.groups.values():
        group.warm = True
    with pytest.raises(RuntimeError, match="synthetic root failure"):
        preparation.prepare_restored_roots(engine)
    assert all(not group.warm and group.graph is None for group in engine.groups.values())
    assert_same(numerical_snapshot(engine), before)


@pytest.mark.parametrize("failure", ("none", "synchronize"))
def test_side_stream_preparation_orders_restoration_and_commits_readiness_after_success(engine, monkeypatch, failure):
    events = []
    main = object()
    before = numerical_snapshot(engine)
    numerical = engine._roots
    engine.device = "cuda:0"

    class Side:
        def wait_stream(self, stream):
            assert stream is main
            events.append("ordered")

        def synchronize(self):
            assert all(not group.warm for group in engine.groups.values())
            events.append("synchronized")
            if failure == "synchronize":
                raise RuntimeError("synthetic side-stream failure")

    engine.side = Side()

    @contextmanager
    def stream_context(side):
        assert side is engine.side
        assert events == ["ordered"]
        yield

    def root(group):
        assert events[0] == "ordered"
        assert all(not value.warm for value in engine.groups.values())
        return numerical(group)

    engine._roots = root
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: main)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    if failure == "synchronize":
        with pytest.raises(RuntimeError, match="synthetic side-stream failure"):
            preparation.prepare_restored_roots(engine)
        assert all(not group.warm for group in engine.groups.values())
    else:
        assert preparation.prepare_restored_roots(engine) == ["d", "f"]
        assert events[-1] == "synchronized"
        assert all(group.warm for group in engine.groups.values())
    assert_same(numerical_snapshot(engine), before)


def test_scratch_preserves_transposed_root_inputs_and_rejects_offset_normalization(engine):
    group = engine.groups["d"]
    group.C = group.C.transpose(-2, -1)
    group.basis.SB = group.basis.SB.transpose(-2, -1)
    scratch = preparation.scratch_group(group)
    for actual, expected in ((scratch.C, group.C), (scratch.basis.SB, group.basis.SB)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.stride() == expected.stride()
        assert actual.storage_offset() == expected.storage_offset()
    group.C = torch.ones(3, 4, 4)[1:]
    with pytest.raises(ValueError, match="raw layout"):
        preparation.prepare_restored_roots(engine)
    assert all(not value.warm for value in engine.groups.values())


@pytest.mark.parametrize("active", ("update", "graph"))
def test_active_update_or_graph_is_preserved_when_preparation_is_refused(engine, active):
    if active == "update":
        engine.begun = True
    else:
        engine.groups["d"].graph = object()
    before = numerical_snapshot(engine)
    graph = engine.groups["d"].graph
    with pytest.raises(RuntimeError):
        preparation.prepare_restored_roots(engine)
    assert_same(numerical_snapshot(engine), before)
    assert engine.groups["d"].graph is graph
    assert all(not group.warm for group in engine.groups.values())


@pytest.mark.parametrize("regime", ("disabled", "inline", "absent_basis", "zero_ownership"))
def test_non_cached_root_regimes_do_not_prepare_or_change_lifecycle(engine, regime):
    if regime == "disabled":
        engine.args.toner_graphs = False
    elif regime == "inline":
        engine.steps = 19
    elif regime == "absent_basis":
        for group in engine.groups.values():
            group.basis = None
    else:
        for group in engine.groups.values():
            group.per = 0
    before = numerical_snapshot(engine)

    def forbidden(group):
        raise AssertionError("Noncached-root path must not prepare root kernels")

    engine._roots = forbidden
    assert preparation.prepare_restored_roots(engine) == []
    assert_same(numerical_snapshot(engine), before)
    assert all(not group.warm and group.graph is None for group in engine.groups.values())


@pytest.fixture
def pinned_graph_method():
    manifest = json.loads((TASK / "experiment/upstream_manifest.json").read_text())
    source = TASK / "artifacts/external-sources" / f"Toner-{manifest['revision']}/src/toner/fast/engine.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["files"]["src/toner/fast/engine.py"]
    tree = ast.parse(source.read_bytes())
    core = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FastTonerCore")
    method = next(node for node in core.body if isinstance(node, ast.FunctionDef) and node.name == "_roots_graphed")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_roots_graphed"]


def test_first_actual_call_captures_then_replays_without_consuming_preparation_outputs(
    engine, pinned_graph_method, monkeypatch
):
    captured, replayed = [], []
    retained = torch.empty_like(engine.groups["d"].C)
    eigen_root = engine._roots
    current = {"group": None}

    class Graph:
        def replay(self):
            replayed.append(True)
            retained.copy_(eigen_root(current["group"]))

    @contextmanager
    def graph_context(graph, **kwargs):
        captured.append(graph)
        yield

    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", graph_context)
    preparation.prepare_restored_roots(engine)
    engine.side, engine.graph_pool = object(), object()

    def actual_roots(group):
        current["group"] = group
        retained.copy_(eigen_root(group))
        return retained, None, torch.zeros(group.per, dtype=torch.bool), None

    engine._roots = actual_roots
    group = engine.groups["d"]
    first = pinned_graph_method(engine, group)
    assert len(captured) == len(replayed) == 1
    torch.testing.assert_close(first[0], eigen_root(group), rtol=0, atol=0)
    group.C.mul_(2)
    second = pinned_graph_method(engine, group)
    assert len(captured) == 1 and len(replayed) == 2
    assert first[0] is second[0]
    torch.testing.assert_close(second[0], eigen_root(group), rtol=0, atol=0)
