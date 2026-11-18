# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Save rank-local optimizer intermediates outside captured root graphs."""

import hashlib
import json
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import torch

FIRST, LAST = 22, 24


def snapshot(value):
    """Copy tensor storage without retaining live graph outputs or autograd state."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(item) for item in value)
    return value


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def group_state(group) -> dict:
    basis = group.basis
    return {
        "C": group.C,
        "tol": group.tol,
        "basis_step": list(group.basis_step),
        "pending_steps": [None if item is None else item[1] for item in group.pending],
        "SB": None if basis is None else basis.SB,
        "S2": None if basis is None else basis.S2,
        "warm": group.warm,
        "graph_active": group.graph is not None,
    }


class ContinuationTrace:
    """Record a bounded trajectory without mutating numerical inputs or outputs.

    Copies add work and can change scheduling. Trace-off/on GPU parity is a
    required diagnostic gate; CPU parity alone cannot establish transparency.
    """

    def __init__(self, output: Path, *, rank: int, provenance: dict):
        self.output, self.rank, self.provenance = output, rank, provenance
        self.current = None
        self.records = []
        self.fast = None

    @contextmanager
    def copying(self):
        verifier = None if self.fast is None else self.fast.update_verifier
        observer = None if verifier is None else verifier.observation
        if observer is None:
            yield
            return
        enabled = observer.enabled
        observer.enabled = False
        try:
            yield
        finally:
            observer.enabled = enabled

    def record(self, stage: str, values) -> None:
        if self.current is None:
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Continuation diagnostic copies cannot enter a captured CUDA graph")
        with self.copying():
            copied = snapshot(values)
        self.records.append({"stage": stage, "values": copied})

    def begin_update(self, update: int) -> None:
        if not FIRST <= update <= LAST:
            return
        if self.current is not None:
            raise RuntimeError("Continuation trace has an unfinished update")
        target = self.output / f"update_{update:06d}.pt"
        if target.exists():
            raise FileExistsError("Preserve the original continuation trace update")
        self.current, self.records = update, []

    def end_update(self, update: int) -> None:
        if self.current is None:
            return
        if update != self.current:
            raise ValueError("Continuation trace end differs from its active update")
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f"update_{update:06d}.pt"
        receipt_path = path.with_suffix(".json")
        if path.exists() or receipt_path.exists():
            raise FileExistsError("Continuation trace evidence already exists")
        result = {
            "schema": 1,
            "rank": self.rank,
            "update": update,
            "provenance": self.provenance,
            "records": cpu_tree(self.records),
        }
        with path.open("xb") as stream:
            torch.save(result, stream)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = {
            "status": "saved_continuation_intermediates_not_numerical_acceptance",
            "rank": self.rank,
            "update": update,
            "provenance": self.provenance,
            "records": len(self.records),
            "stages": [record["stage"] for record in self.records],
            "trace_sha256": digest,
        }
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        self.current, self.records = None, []

    def wrap(self, function, stage: str):
        @wraps(function)
        def traced(*args, **kwargs):
            if self.current is not None:
                self.record(f"{stage}/input", {"args": args, "kwargs": kwargs})
            result = function(*args, **kwargs)
            if self.current is not None:
                self.record(f"{stage}/output", result)
            return result

        return traced

    def install(self, fast, block_module, root_module) -> None:
        self.fast = fast
        engine = fast.input
        if engine is None or fast.output is not None:
            raise ValueError("Continuation trace requires the original fused input-root/output-diagonal arm")
        stats = engine._stats
        roots = engine._roots_graphed
        finish = engine._finish_roots

        @wraps(stats)
        def traced_stats():
            result = stats()
            self.record("input/statistics", {name: group_state(group) for name, group in engine.groups.items()})
            return result

        @wraps(roots)
        def traced_roots(group):
            name = next(name for name, value in engine.groups.items() if value is group)
            self.record(f"input/{name}/before_root", group_state(group))
            result = roots(group)
            self.record(f"input/{name}/candidate", result)
            return result

        @wraps(finish)
        def traced_finish():
            with torch.cuda.stream(engine.side):
                self.record(
                    "input/before_finish",
                    {
                        "steps": engine.steps,
                        "fallbacks": engine.fallbacks,
                        "extra": engine.extra,
                        "groups": {name: group_state(group) for name, group in engine.groups.items()},
                    },
                )
            result = finish()
            self.record(
                "input/finished",
                {
                    "roots": result,
                    "fallbacks": engine.fallbacks,
                    "extra": engine.extra,
                    "groups": {name: group_state(group) for name, group in engine.groups.items()},
                },
            )
            return result

        engine._stats, engine._roots_graphed, engine._finish_roots = traced_stats, traced_roots, traced_finish
        engine.froot.froot_more = self.wrap(engine.froot.froot_more, "input/refinement")
        root_module.exact_eigh_powers = self.wrap(root_module.exact_eigh_powers, "input/exact_fallback")
        update = block_module.block_update_stack

        @wraps(update)
        def traced_update(weights, gradients, momenta, left, right, **kwargs):
            self.record(
                "matrix/before_update",
                {
                    "weights": weights,
                    "gradients": gradients,
                    "momentum": momenta,
                    "left": left,
                    "right": right,
                    "lr": kwargs["lr"],
                    "weight_decay": kwargs["weight_decay"],
                },
            )
            result = update(weights, gradients, momenta, left, right, **kwargs)
            self.record(
                "matrix/after_update",
                {"momentum": result[0], "delta": result[1], "trace": kwargs.get("trace")},
            )
            return result

        block_module.block_update_stack = traced_update
