# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture bounded actual-training stage traces without changing optimizer arithmetic."""

import hashlib
import json
from contextlib import nullcontext
from functools import wraps
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

STAGE_METHODS = (
    ("input", "begin", "input_statistics_launch"),
    ("input", "_finish_roots", "input_roots_finish"),
    ("output", "begin", "output_statistics_launch"),
    ("output", "_finish_roots", "output_roots_finish"),
    ("input_diagonal", "update", "input_diagonal_reduce"),
    ("input_diagonal", "factors", "input_diagonal_factors"),
    ("output_diagonal", "update", "output_diagonal_reduce"),
    ("output_diagonal", "factors", "output_diagonal_factors"),
)
BLOCK_FUNCTIONS = (
    ("stack_factors", "factor_stack"),
    ("gather_uneven", "update_gather_launch"),
    ("block_update_stack", "legacy_block_update"),
    ("combined_outer_update_stack", "combined_block_update"),
    ("upstream_outer_update_stack", "upstream_block_update"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StageProfiler:
    """Trace an inclusive update interval on every rank, with immutable per-rank outputs."""

    def __init__(self, output: Path, *, first: int, last: int, rank: int, provenance: dict, activities=None):
        if first < 1 or last < first:
            raise ValueError("Profiling requires a positive inclusive update interval")
        self.output = output
        self.first, self.last, self.rank = first, last, rank
        self.provenance = provenance
        self.activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if activities is None else list(activities)
        self.profiler = None
        self.active = False
        self.current = None
        self.observed = []
        self.exported = False

    def scope(self, name: str):
        return record_function(f"osn/{name}") if self.active else nullcontext()

    def wrap(self, function, name: str):
        """Keep numerical arguments/results unchanged while naming an outer launch region."""

        @wraps(function)
        def scoped(*args, **kwargs):
            with self.scope(name):
                return function(*args, **kwargs)

        return scoped

    def install(self, fast, block_module) -> None:
        """Attach host launch scopes before compilation; compiled kernels retain their own events."""
        if fast is None:
            return
        for attribute, method, label in STAGE_METHODS:
            engine = getattr(fast, attribute, None)
            if engine is not None:
                setattr(engine, method, self.wrap(getattr(engine, method), label))
        if isinstance(fast, block_module.BlockToner):
            block_module.profile_scope = self.scope
        else:
            fast.begin = self.wrap(fast.begin, "input_statistics_launch")
            fast._finish_roots = self.wrap(fast._finish_roots, "input_roots_finish")
        fast.step = self.wrap(fast.step, "optimizer_step")
        for name, label in BLOCK_FUNCTIONS:
            setattr(block_module, name, self.wrap(getattr(block_module, name), label))

    def begin_update(self, update: int) -> None:
        if self.first <= update <= self.last:
            if self.exported or self.current is not None:
                raise RuntimeError("Profiling update was repeated or an earlier scope remains open")
            if self.profiler is None:
                if self.output.exists():
                    raise FileExistsError("Preserve the original rank profile; do not retry its output identity")
                self.profiler = profile(
                    activities=self.activities,
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                )
                self.profiler.start()
                self.profiler.add_metadata_json("osn_provenance", json.dumps(self.provenance, allow_nan=False))
                self.active = True
            expected = self.first + len(self.observed)
            if update != expected:
                raise ValueError("Profiling updates are not contiguous")
            self.current = self.scope("training_update")
            self.current.__enter__()
            self.observed.append(update)

    def end_update(self, update: int) -> None:
        if self.current is None:
            return
        if self.observed[-1] != update:
            raise ValueError("Profiling update end differs from its start")
        self.current.__exit__(None, None, None)
        self.current = None
        if update == self.last:
            self.finish(status="captured")

    def finish(self, *, status: str = "incomplete") -> None:
        if self.profiler is None or self.exported:
            return
        if self.current is not None:
            self.current.__exit__(None, None, None)
            self.current = None
        self.profiler.stop()
        self.exported = True
        self.active = False
        self.output.mkdir(parents=True, exist_ok=False)
        trace = self.output / "trace.json"
        self.profiler.export_chrome_trace(str(trace))
        regions = [
            {
                "name": event.key,
                "calls": event.count,
                "cpu_total_microseconds": event.cpu_time_total,
                "cpu_self_microseconds": event.self_cpu_time_total,
                "device_total_microseconds": event.device_time_total,
                "device_self_microseconds": event.self_device_time_total,
            }
            for event in self.profiler.key_averages()
            if event.key.startswith("osn/")
        ]
        complete = self.observed == list(range(self.first, self.last + 1))
        result = {
            "status": status if complete else "incomplete",
            "rank": self.rank,
            "torch_version": torch.__version__,
            "activities": [activity.name for activity in self.activities],
            "first_update": self.first,
            "last_update": self.last,
            "observed_updates": self.observed,
            "provenance": self.provenance,
            "trace_sha256": digest(trace),
            "regions": regions,
            "limitation": (
                "Instrumented diagnostic timing. CPU regions include launch and wait time; nested totals "
                "must not be added. Device kernels, overlapping streams and CUDA-graph replay require trace "
                "inspection. Cotangent collection inside compiled backward is not an isolated host scope. "
                "No uninstrumented throughput, cross-rank critical path or final quality claim follows."
            ),
        }
        (self.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
