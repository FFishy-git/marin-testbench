# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check pinned FP32 maps and explicitly emulated CPU precision schedules without CUDA."""

import hashlib
from pathlib import Path

import ns_precision_placement_maps as placement
import pytest
import torch
from benchmark_diagonal import source_definitions

PINNED = "artifacts/pinned-sources-materialized/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/src/toner/fast/muon.py"
PARENT = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/run-preparation/ns-compiler-localization-execution-cpu/20261119T1225"
)
REFERENCE_SCHEDULES = (
    ("q5_tpu", ("tpu",) * 5),
    ("q5_f32", ("f32",) * 5),
    ("q10_tpu", ("tpu",) * 10),
    ("q5_tpu2_f32suffix3", ("tpu", "tpu", "f32", "f32", "f32")),
    ("q5_f32prefix3_tpu2", ("f32", "f32", "f32", "tpu", "tpu")),
)


@pytest.fixture
def source_path():
    staged = Path(__file__).resolve().parents[1] / PINNED
    return staged if staged.is_file() else PARENT / PINNED


def uncompiled(numerical):
    return numerical


def dynamo_eager(numerical):
    return torch.compile(numerical, backend="eager", fullgraph=True, dynamic=False)


def source_batch(shape):
    size = shape[0] * shape[1] * shape[2]
    return torch.sin(torch.arange(size, dtype=torch.float32).reshape(shape) * 0.73) + 0.1


def cpu_emulated_product(left, right, mode):
    """Test-only CPU BF16 rounding with FP32 bmm; this does not reproduce a CUDA kernel."""
    if left.device.type != "cpu" or right.device.type != "cpu":
        raise RuntimeError("CPU product emulation requires CPU operands")
    if mode == "tpu":
        return torch.bmm(left.to(torch.bfloat16).float(), right.to(torch.bfloat16).float())
    if mode == "f32":
        return torch.bmm(left, right)
    raise ValueError("Unexpected product mode in CPU emulation")


def independent_cpu_recurrence(source, schedule, eps):
    x = source.float()
    x = x / (torch.linalg.matrix_norm(x, dim=(-2, -1), keepdim=True) + eps)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    for mode in schedule:
        gram_input = x.to(torch.bfloat16).float() if mode == "tpu" else x
        gram = torch.bmm(gram_input, gram_input.transpose(-2, -1))
        square_input = gram.to(torch.bfloat16).float() if mode == "tpu" else gram
        gram_square = torch.bmm(square_input, square_input)
        correction = -4.7750 * gram + 2.0315 * gram_square
        correction_input = correction.to(torch.bfloat16).float() if mode == "tpu" else correction
        iterate_input = x.to(torch.bfloat16).float() if mode == "tpu" else x
        x = 3.4445 * x + torch.bmm(correction_input, iterate_input)
    return (x.transpose(-2, -1) if transposed else x).float()


@pytest.mark.parametrize("shape", ((2, 3, 5), (2, 5, 3), (2, 4, 4)))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_precision_placement_f32_maps_match_unmodified_pinned_routine(source_path, shape, dtype):
    maps = placement.maps(source_path, "q5_f32", uncompiled)
    original = source_definitions(source_path, ("ABC", "_bmm", "newton_schulz_batched"), {"torch": torch})
    source = source_batch(shape).to(dtype)
    before, rng = source.clone(), torch.get_rng_state().clone()
    expected = original["newton_schulz_batched"](source, 5, 1e-5, mode="f32")
    for numerical in (maps.eager, maps.compiled):
        torch.testing.assert_close(numerical(source), expected, rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert maps.schedule_name == "q5_f32" and maps.schedule == ("f32",) * 5
    assert maps.source_sha256 == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert maps.canonical_filename == str(source_path)
    assert maps.product_implementation == "pinned_bmm"
    assert maps.eager.func.__code__ == original["newton_schulz_batched"].__code__


@pytest.mark.parametrize("shape", ((2, 3, 5), (2, 5, 3)))
@pytest.mark.parametrize("name,schedule", REFERENCE_SCHEDULES)
def test_precision_placement_cpu_emulation_matches_independent_schedule(source_path, shape, name, schedule):
    maps = placement.maps(
        source_path,
        name,
        uncompiled,
        cpu_test_product=cpu_emulated_product,
    )
    source = source_batch(shape)
    before, rng = source.clone(), torch.get_rng_state().clone()
    expected = independent_cpu_recurrence(source, schedule, 1e-5)
    for numerical in (maps.eager, maps.compiled):
        first, second = numerical(source), numerical(source)
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        torch.testing.assert_close(second, first, rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert maps.schedule_name == name and maps.schedule == schedule
    assert maps.product_implementation == "cpu_test_product"


@pytest.mark.parametrize("name,schedule", REFERENCE_SCHEDULES)
def test_precision_placement_dynamo_eager_retains_exact_schedule_outputs(source_path, name, schedule):
    maps = placement.maps(
        source_path,
        name,
        dynamo_eager,
        cpu_test_product=None if name == "q5_f32" else cpu_emulated_product,
    )
    source = source_batch((2, 5, 3))
    before, rng = source.clone(), torch.get_rng_state().clone()
    expected = independent_cpu_recurrence(source, schedule, 1e-5)
    for numerical in (maps.eager, maps.compiled):
        first, second = numerical(source), numerical(source)
        torch.testing.assert_close(first, expected, rtol=0, atol=0)
        torch.testing.assert_close(second, first, rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state())


def test_precision_placement_cpu_product_maps_refuse_non_cpu_input(source_path):
    maps = placement.maps(
        source_path,
        "q5_tpu",
        dynamo_eager,
        cpu_test_product=cpu_emulated_product,
    )
    source = torch.empty((1, 2, 3), device="meta")
    for numerical in (maps.eager, maps.compiled):
        with pytest.raises(RuntimeError):
            numerical(source)
