# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct retained start-kernel accuracy; CUDA ownership and execution stay separate."""

from collections.abc import Callable

import root_stage_trace as tracing
import torch
from diagnose_fused_diagonal import differences

FIELDS = tracing.OUTPUT_FIELDS["start"]
UNCHANGED = ("d", "E", "Eb", "lam", "sq", "small", "rho")
VARIANTS = ("pinned", "rational")
GROUPS = ("d", "f")
MODES = ("continuous", "resumed")
PHASES = ("eager", "capture", "replay", "replay_repeat")
TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
LIMITATION = (
    "Retained CPU-output arithmetic only. This consumer does not authenticate a CUDA owner, source archive, "
    "execution, graph, input/RNG preservation or historical-output reproduction. Its FP64 log1p/expm1 reference "
    "uses a different coefficient algorithm from the proposed rational form. Accuracy does not establish "
    "full-root accuracy, checkpoint continuation, optimizer quality, runtime or production acceptance."
)


def coordinates(index: int) -> list[dict]:
    """Describe opposite variant orders in two fresh processes."""
    if index not in (0, 1):
        raise ValueError("Start trial requires one of its two fresh-process indices")
    variants = VARIANTS if index == 0 else tuple(reversed(VARIANTS))
    return [
        {"variant": variant, "group": group, "mode": mode, "phase": phase}
        for variant in variants
        for group in GROUPS
        for mode in MODES
        for phase in PHASES
    ]


def fp64_divided_difference(d: torch.Tensor) -> torch.Tensor:
    """Evaluate inverse-quarter differences with log1p/expm1 instead of subtracted powers."""
    if (
        d.device.type != "cpu"
        or d.dtype != torch.float64
        or not d.numel()
        or not torch.isfinite(d).all()
        or not (d > 0).all()
    ):
        raise ValueError("FP64 reference requires finite positive CPU diagonal values")
    left, right = d[..., :, None], d[..., None, :]
    smaller, larger = torch.minimum(left, right), torch.maximum(left, right)
    delta = larger - smaller
    denominator = torch.where(delta == 0, 1.0, delta)
    difference = smaller.pow(-0.25) * torch.expm1(-0.25 * torch.log1p(delta / smaller))
    return torch.where(delta == 0, -0.25 * smaller.pow(-1.25), difference / denominator)


def named_output(output: tuple, shape: torch.Size) -> dict[str, torch.Tensor]:
    """Validate all ten start outputs, including their original shapes and dtypes."""
    if not isinstance(output, tuple) or len(output) != len(FIELDS):
        raise ValueError("Start trial must retain all ten start outputs")
    vector = shape[:-1]
    matrices = {"E", "Eb", "O", "Ob"}
    vectors = {"d", "lam", "sq", "small"}
    lowp = {"Eb", "Ob"}
    values = dict(zip(FIELDS, output, strict=True))
    for name, value in values.items():
        expected_shape = shape if name in matrices else vector if name in vectors else shape[:-2]
        expected_dtype = torch.bool if name == "small" else torch.float16 if name in lowp else torch.float32
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.shape != expected_shape
            or value.dtype != expected_dtype
        ):
            raise ValueError(f"Start trial retained {name} has a different shape, dtype or device")
    return values


def exact_finite(value: dict) -> bool:
    return value["tensor_comparisons"] > 0 and value["finite"] and value["exact"]


def summarize(retained: list[dict], sources: dict, pinned: Callable, freeze: float, index: int) -> dict:
    """Recompute repeat, unchanged-output, FP16-cast and per-matrix FP64 accuracy checks.

    Args:
        retained: CPU copies of complete ten-output tuples at every declared
            variant/group/mode/phase coordinate. Execution and graph claims need
            an independent source-bound terminal verifier.
        sources: The full CPU B batches for each d/f continuous/resumed pair.
        pinned: Caller-authenticated raw pinned start for the independent FP64
            diagonal and off-diagonal construction.
        freeze: The caller-authenticated original noise-pair setting.
        index: Original fresh-process order, zero or one.

    Returns:
        Arithmetic checks with every matrix preserved. These checks cannot
        authorize CUDA, accept a training candidate or infer an efficiency win.
    """
    expected = coordinates(index)
    if [{key: item[key] for key in ("variant", "group", "mode", "phase")} for item in retained] != expected:
        raise ValueError("Start trial lacks complete ordered retained coordinates")
    if set(sources) != set(GROUPS) or any(set(pair) != set(MODES) for pair in sources.values()):
        raise ValueError("Start trial requires both full input groups and modes")
    references = {}
    before = {}
    for group, pair in sources.items():
        if pair["continuous"].shape != pair["resumed"].shape:
            raise ValueError("Start trial source pair shapes differ")
        for mode, B in pair.items():
            if (
                B.device.type != "cpu"
                or B.dtype != torch.float32
                or B.ndim != 3
                or B.shape[-1] != B.shape[-2]
                or not B.numel()
                or not torch.isfinite(B).all()
            ):
                raise ValueError("Start trial requires full finite FP32 CPU square batches")
            before[(group, mode)] = B.clone()
            reference = pinned(B.double(), freeze)
            if len(reference) != len(FIELDS) or not (reference[0] >= 1e-6).all():
                raise ValueError("Start trial FP64 coefficient attribution requires unclamped positive pairs")
            references[(group, mode)] = fp64_divided_difference(reference[0]) * reference[1]
    indexed, named = {}, {}
    for item in retained:
        key = tuple(item[name] for name in ("variant", "group", "mode", "phase"))
        B = sources[item["group"]][item["mode"]]
        named[key] = named_output(item["output"], B.shape)
        indexed[key] = item["output"]
    repeated, preserved, casts, accuracy = [], [], [], []
    for group in GROUPS:
        for mode in MODES:
            for variant in VARIANTS:
                reference = indexed[(variant, group, mode, "eager")]
                for phase in PHASES[1:]:
                    repeated.append(
                        {
                            "variant": variant,
                            "group": group,
                            "mode": mode,
                            "phase": phase,
                            **tracing.comparison(
                                indexed[(variant, group, mode, phase)], reference, f"{variant}.{group}.{mode}.{phase}"
                            ),
                        }
                    )
            for phase in PHASES:
                original, proposed = (named[(variant, group, mode, phase)] for variant in VARIANTS)
                preserved.append(
                    {
                        "group": group,
                        "mode": mode,
                        "phase": phase,
                        **tracing.comparison(
                            {name: proposed[name] for name in UNCHANGED},
                            {name: original[name] for name in UNCHANGED},
                            "unchanged.start",
                        ),
                    }
                )
                for variant in VARIANTS:
                    values = named[(variant, group, mode, phase)]
                    for producer, operand in (("E", "Eb"), ("O", "Ob")):
                        rounded = values[producer].half()
                        casts.append(
                            {
                                "variant": variant,
                                "group": group,
                                "mode": mode,
                                "phase": phase,
                                "producer": producer,
                                "operand": operand,
                                **tracing.comparison(values[operand], rounded, f"{producer}.{operand}.cpu.rounding"),
                            }
                        )
                    for matrix, (actual, reference) in enumerate(
                        zip(values["O"], references[(group, mode)], strict=True)
                    ):
                        accuracy.append(
                            {
                                "variant": variant,
                                "group": group,
                                "mode": mode,
                                "phase": phase,
                                "matrix": matrix,
                                **differences(actual.double(), reference, TOLERANCE),
                            }
                        )
    if any(not torch.equal(sources[group][mode], value) for (group, mode), value in before.items()):
        raise ValueError("Start accuracy reference mutated a source input")
    crossed = [
        {
            "variant": variant,
            "group": group,
            "phase": phase,
            **tracing.comparison(
                indexed[(variant, group, "resumed", phase)],
                indexed[(variant, group, "continuous", phase)],
                f"{variant}.{group}.{phase}.crossed",
            ),
        }
        for variant in VARIANTS
        for group in GROUPS
        for phase in PHASES
    ]
    return {
        "worker_index": index,
        "coordinates": expected,
        "shapes": {group: list(sources[group]["continuous"].shape) for group in GROUPS},
        "repeat_checks": repeated,
        "unchanged_output_checks": preserved,
        "cast_checks": casts,
        "O_fp64_accuracy": accuracy,
        "crossed_outputs": crossed,
        "all_repeat_outputs_exact_finite": all(exact_finite(item) for item in repeated),
        "all_unaffected_outputs_exact_finite": all(exact_finite(item) for item in preserved),
        "all_casts_match_independent_rounding": all(exact_finite(item) for item in casts),
        "all_rational_O_within_fp64_tolerance": all(
            item["within_tolerance"] for item in accuracy if item["variant"] == "rational"
        ),
        "tolerance": TOLERANCE,
        "limitation": LIMITATION,
    }
