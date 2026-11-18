# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain root-kernel intermediates without copying tensors inside CUDA graphs."""

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from types import ModuleType

import torch
from root_preparation_probe import comparison, cpu_tree

OUTPUT_FIELDS = {
    "basis_transform": ("B", "mu"),
    "start": ("d", "E", "Eb", "lam", "O", "Ob", "sq", "small", "rho", "dn0"),
    "iter_mid": ("P", "Pb"),
    "_bmm32": ("product",),
    "mm_sym": ("product",),
    "iter_w": ("W",),
    "iter_post": ("lam", "O", "Ob", "update_norm"),
    "square": ("lam", "O"),
    "sqrt_start": ("s", "Q", "Qb", "S0", "dn0"),
    "sqrt_post": ("s", "Q", "Qb", "update_norm"),
    "_formation": ("root",),
}
PRODUCTS = frozenset(("_bmm32", "mm_sym"))
LIMITATION = (
    "Retaining intermediate references changes tensor lifetimes and can affect allocation or graph pools. "
    "No tensor copies, reductions or new numerical kernels are inserted during root computation. "
    "Independent uninstrumented/instrumented CUDA parity is required before interpreting captured stages. "
    "CPU tests are not CUDA transparency, historical-output reproduction or continuation acceptance."
)


@dataclass
class RootStageTrace:
    """Hold live intermediate references until the caller completes and synchronizes its root call."""

    records: list[dict]

    def snapshot(self, output: object) -> dict:
        """Copy completed outputs and intermediates to independent CPU storage.

        The CUDA caller must synchronize its producing stream before calling this
        method. Snapshotting during CUDA capture is rejected.
        """
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Root stage snapshots cannot enter a captured CUDA graph")
        return {
            "schema": 1,
            "records": cpu_tree(self.records),
            "output": cpu_tree(output),
            "limitation": LIMITATION,
        }


def _values(name: str, output: object) -> dict:
    fields = OUTPUT_FIELDS[name]
    values = (output,) if len(fields) == 1 else output
    if not isinstance(values, tuple) or len(values) != len(fields):
        raise ValueError(f"Root kernel {name} returned a different output structure")
    return dict(zip(fields, values, strict=True))


@contextmanager
def observe_root_stages(roots: ModuleType) -> Iterator[RootStageTrace]:
    """Temporarily record calls to the existing root kernels without replacing their numerics.

    The caller owns numerical execution, source binding and any CUDA allocation.
    Use this only in a separately issued diagnostic, never an existing trainer.
    All original functions are restored even if a root call fails.
    """
    originals = {name: getattr(roots, name) for name in OUTPUT_FIELDS}
    if any(
        not callable(function) or getattr(function, "_root_stage_observer", False) for function in originals.values()
    ):
        raise RuntimeError("Root stage observation requires unwrapped callable kernels")
    trace, counts = RootStageTrace([]), Counter()

    def wrapper(name: str, numerical: Callable) -> Callable:
        @wraps(numerical)
        def observed(*args, **kwargs):
            output = numerical(*args, **kwargs)
            values = _values(name, output)
            if name in PRODUCTS:
                values = {"left": args[0], "right": args[1], **values}
            trace.records.append({"stage": name, "occurrence": counts[name], "values": values})
            counts[name] += 1
            return output

        observed._root_stage_observer = True
        return observed

    try:
        for name, numerical in originals.items():
            setattr(roots, name, wrapper(name, numerical))
        yield trace
    finally:
        for name, numerical in originals.items():
            setattr(roots, name, numerical)


def compare_root_stages(actual: dict, expected: dict) -> dict:
    """Compare complete ordered root-stage snapshots and retain every changed value."""
    summaries = []
    for label, value in (("actual", actual), ("expected", expected)):
        if value.get("schema") != 1 or not isinstance(value.get("records"), list):
            raise ValueError(f"{label} root stage snapshot lacks its schema or records")
        counts = Counter()
        for record in value["records"]:
            if (
                set(record) != {"stage", "occurrence", "values"}
                or record["stage"] not in OUTPUT_FIELDS
                or type(record["occurrence"]) is not int
                or record["occurrence"] != counts[record["stage"]]
            ):
                raise ValueError("Root stage order or occurrence identity differs")
            fields = set(OUTPUT_FIELDS[record["stage"]])
            if record["stage"] in PRODUCTS:
                fields |= {"left", "right"}
            if not isinstance(record["values"], dict) or set(record["values"]) != fields:
                raise ValueError("Root stage values lack their complete named kernel outputs")
            counts[record["stage"]] += 1
    coordinates = {
        side: [[record["stage"], record["occurrence"]] for record in value["records"]]
        for side, value in (("actual", actual), ("expected", expected))
    }
    if coordinates["actual"] != coordinates["expected"]:
        return {
            "complete_stage_coverage": False,
            "stage_coordinates": coordinates,
            "stages": [],
            "output": comparison(actual["output"], expected["output"], "root.output"),
            "limitation": "Different stage orders are not prefix-paired.",
        }
    for observed, reference in zip(actual["records"], expected["records"], strict=True):
        name, occurrence = reference["stage"], reference["occurrence"]
        summaries.append(
            {
                "stage": name,
                "occurrence": occurrence,
                **comparison(observed["values"], reference["values"], f"root.{name}[{occurrence}]"),
            }
        )
    return {
        "complete_stage_coverage": bool(summaries),
        "stage_coordinates": coordinates,
        "tensor_comparisons": sum(item["tensor_comparisons"] for item in summaries),
        "stages": summaries,
        "output": comparison(actual["output"], expected["output"], "root.output"),
        "limitation": LIMITATION,
    }
