# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce before-call capture flags with the executed pinned engine's state machine."""

import ast
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import root_preparation_reconstruction as reconstruction
from selected_lr_release_owner import sha256


@pytest.mark.parametrize("regime", ("unprepared", "prepared"))
@pytest.mark.parametrize("mode", ("continuous", "resumed"))
def test_before_call_graph_lifecycle_matches_pinned_warm_capture_replay(regime, mode):
    root = reconstruction.controller.locations(
        Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")
    )[0]
    binding = json.loads(root.read_bytes())
    candidate = Path(binding["candidate"])
    manifest = json.loads((candidate / "experiment/upstream_manifest.json").read_bytes())
    source = candidate / f"artifacts/pinned-sources-materialized/Toner-{manifest['revision']}/src/toner/fast/engine.py"
    assert sha256(source) == binding["files_sha256"][str(source.relative_to(candidate))]
    archived = (
        Path(__file__).parents[1] / f"artifacts/external-sources/Toner-{manifest['revision']}/src/toner/fast/engine.py"
    )
    if archived.is_file():
        assert sha256(archived) == sha256(source)
        source = archived
    method = next(
        function
        for node in ast.parse(source.read_bytes()).body
        if isinstance(node, ast.ClassDef) and node.name == "FastTonerCore"
        for function in node.body
        if isinstance(function, ast.FunctionDef) and function.name == "_roots_graphed"
    )

    class Graph:
        def replay(self):
            return None

    @contextmanager
    def capture(*args, **kwargs):
        yield

    namespace = {"torch": SimpleNamespace(cuda=SimpleNamespace(CUDAGraph=Graph, graph=capture))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)

    def numerical(group):
        return ("retained-numerical-output",)

    engine = SimpleNamespace(args=SimpleNamespace(toner_graphs=True), graph_pool=None, side=None, _roots=numerical)
    config = json.loads((candidate / reconstruction.validation.CONFIG).read_bytes())
    assert config["root"]["inline_steps"] == 20
    group = SimpleNamespace(warm=False, graph=None)
    namespace["_roots_graphed"](engine, group)
    observed = {"22": {"warm": group.warm, "graph_active": group.graph is not None}}
    namespace["_roots_graphed"](engine, group)
    if mode == "resumed":
        group = SimpleNamespace(warm=regime == "prepared", graph=None)
    for step in (23, 24):
        observed[str(step)] = {"warm": group.warm, "graph_active": group.graph is not None}
        output = namespace["_roots_graphed"](engine, group)
        assert output == ("retained-numerical-output",)
    lifecycle = {
        f"{rank}:{step}": {name: observed[str(step)].copy() for name in ("d", "f")}
        for rank in range(8)
        for step in (22, 23, 24)
    }
    reconstruction.require_trace_lifecycle(lifecycle, regime, mode)
    assert observed["22"] == {"warm": True, "graph_active": False}
    if regime == "unprepared" and mode == "resumed":
        assert observed["23"] == {"warm": False, "graph_active": False}
        assert observed["24"] == {"warm": True, "graph_active": False}
        lifecycle["0:24"]["d"]["graph_active"] = True
        with pytest.raises(ValueError, match="before-call lifecycle"):
            reconstruction.require_trace_lifecycle(lifecycle, regime, mode)
    else:
        assert observed["24"] == {"warm": True, "graph_active": True}
