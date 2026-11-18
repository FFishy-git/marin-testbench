# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU behavioral coverage only; no issued saved-input GPU experiment."""

import hashlib
import io
import json
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import pytest
import replay_saved_ns as replay
import torch
from benchmark_diagonal import source_definitions
from saved_ns_test_support import saved_evidence


@pytest.fixture
def pinned_ns():
    task = Path(__file__).resolve().parents[1]
    manifest = json.loads((task / "experiment/upstream_manifest.json").read_text())
    path = task / "artifacts/external-sources" / f"Toner-{manifest['revision']}/src/toner/fast/muon.py"
    with replay.analyzer._bound_file(path, replay.MUON_SHA256) as stream:
        definitions = source_definitions(
            Path(f"/proc/self/fd/{stream.fileno()}"),
            ("ABC", "_bmm", "newton_schulz_batched"),
            {"torch": torch},
        )
    return partial(definitions["newton_schulz_batched"], steps=10, eps=1e-5, mode="f32")


@pytest.fixture
def groups(pinned_ns):
    source = torch.tensor([[[1.0, 0.2], [-0.1, 0.3], [0.2, 0.4]], [[0.3, -0.2], [0.4, 1.0], [0.5, -0.6]]])
    resumed = source.clone()
    resumed[1, 2, 1] += 0.125
    return {
        "continuous": {"source": source, "output": pinned_ns(source)},
        "resumed": {"source": resumed, "output": pinned_ns(resumed)},
    }


def test_pinned_f32_replay_retains_entire_batch_exact_repeats_and_saved_cross_errors(groups, pinned_ns):
    before = {mode: {name: value.clone() for name, value in group.items()} for mode, group in groups.items()}
    result = replay.replay_pair(groups, pinned_ns)
    assert len(result.outputs) == 4
    for output, mode in zip(result.outputs, ("continuous", "resumed", "continuous", "resumed"), strict=True):
        torch.testing.assert_close(output, groups[mode]["output"], rtol=0, atol=0)
        assert output.shape == (2, 3, 2)
        assert not output.requires_grad
    assert all(row["exact"] and row["changed_elements"] == 0 for row in result.report["repeats"].values())
    assert all(row["own_saved_output"]["exact"] for row in result.report["saved_comparisons"])
    assert all(not row["other_saved_output"]["exact"] for row in result.report["saved_comparisons"])
    assert result.report["cross_outputs"][0]["max_absolute_error"] > 0
    assert result.report["owner_context"] is None
    assert result.report["gpu_status"] == "unissued_owner_adapter_absent"
    for mode, group in groups.items():
        for name, value in group.items():
            torch.testing.assert_close(value, before[mode][name], rtol=0, atol=0)
    assert json.loads(json.dumps(result.report, allow_nan=False))["status"] == "cpu_numerical_replay_not_acceptance"


def test_replay_clones_reused_output_storage_for_later_cross_process_comparison(groups, pinned_ns):
    buffer = torch.empty_like(groups["continuous"]["source"])

    def reuse_storage(source):
        buffer.copy_(pinned_ns(source))
        return buffer

    first = replay.replay_pair(groups, reuse_storage)
    second = replay.replay_pair(groups, pinned_ns)
    buffer.zero_()
    for actual, expected in zip(first.outputs, second.outputs, strict=True):
        assert replay.compare_outputs(actual, expected)["exact"]
    first.outputs[0].zero_()
    assert not replay.compare_outputs(first.outputs[0], second.outputs[0])["exact"]
    assert replay.compare_outputs(first.outputs[2], second.outputs[2])["exact"]
    assert replay.compare_outputs(groups["continuous"]["output"], second.outputs[0])["exact"]


@pytest.mark.parametrize("field", ("source", "output"))
def test_replay_refuses_callable_mutation_of_either_saved_pair(groups, field):
    def mutate(source):
        groups["resumed"][field][1, 2, 1] += 1
        return source.clone()

    with pytest.raises(ValueError, match="mutated"):
        replay.replay_pair(groups, mutate)


def test_replay_saved_reference_error_preserves_original_pointwise_and_norm_metrics(groups, pinned_ns):
    groups["continuous"]["output"][0, 0, 0] += 0.25
    result = replay.replay_pair(groups, pinned_ns)
    metrics = result.report["saved_comparisons"][0]["own_saved_output"]
    expected = groups["continuous"]["output"].double()
    assert not metrics["exact"]
    assert not metrics["within_tolerance"]
    assert metrics["changed_elements"] == metrics["mismatched_elements"] == 1
    assert metrics["max_absolute_error"] == pytest.approx(0.25)
    assert metrics["mean_absolute_error"] == pytest.approx(0.25 / 12)
    assert metrics["reference_norm"] == pytest.approx(expected.norm().item())
    assert metrics["relative_frobenius_error"] == pytest.approx(0.25 / expected.norm().item())
    assert result.report["saved_comparisons"][1]["own_saved_output"]["exact"]


@pytest.mark.parametrize("field", ("source", "output"))
def test_replay_refuses_nonfinite_saved_inputs(groups, pinned_ns, field):
    groups["resumed"][field][0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        replay.replay_pair(groups, pinned_ns)


def test_replay_nonfinite_outputs_remain_strict_json_failures_even_when_repeats_are_exact(groups):
    def nonfinite(source):
        return torch.full_like(source, float("inf"))

    result = replay.replay_pair(groups, nonfinite)
    report = json.loads(json.dumps(result.report, allow_nan=False))
    assert report["repeats"]["continuous"]["exact"]
    assert not report["repeats"]["continuous"]["within_tolerance"]
    metrics = report["saved_comparisons"][0]["own_saved_output"]
    assert not metrics["finite"]
    assert metrics["actual_nonfinite_elements"] == 12
    assert metrics["max_absolute_error"] is None
    assert not metrics["within_tolerance"]


def test_cuda_worker_request_is_denied_without_opening_evidence(monkeypatch):
    def refuse_evidence(*args, **kwargs):
        raise AssertionError("CUDA denial must precede filesystem I/O")

    monkeypatch.setattr(Path, "read_bytes", refuse_evidence)
    with pytest.raises(RuntimeError, match="owner adapter is absent"):
        replay.replay_saved_ns(device="cuda:0")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bound_saved_pair(tmp_path, monkeypatch, request):
    layout = getattr(request, "param", "contiguous")
    root = tmp_path / "root"
    receipt_path, paths = saved_evidence(root, layout)
    monkeypatch.setattr(replay, "ROOT", root)
    monkeypatch.setattr(replay, "RECEIPT_SHA256", sha256(receipt_path))
    monkeypatch.setattr(replay, "BINDING_SHA256", sha256(paths[0]))
    monkeypatch.setattr(replay, "MUON_SHA256", sha256(paths[1]))
    return receipt_path, paths


def test_bound_loader_authenticates_full_saved_batch_without_mutating_files(bound_saved_pair):
    receipt, paths = bound_saved_pair
    before = {path: sha256(path) for path in (receipt, *paths)}
    with ExitStack() as stack:
        groups, provenance = replay.load_saved_group(stack)
        for mode, number in (("continuous", 1.0), ("resumed", 2.0)):
            for value in groups[mode].values():
                assert value.shape == (16, 512, 512)
                assert torch.count_nonzero(value != number) == 0
        groups["continuous"]["source"].zero_()
    assert {path: sha256(path) for path in before} == before
    assert provenance["saved_gate_dependency"]["independently_reconstructed"] is False
    assert (provenance["rank"], provenance["update"], provenance["occurrence"]) == (0, 23, 0)


def test_bound_loader_rejects_replaced_trace_bytes(bound_saved_pair):
    _, paths = bound_saved_pair
    paths[-4].write_bytes(b"replaced trace")
    with ExitStack() as stack, pytest.raises(ValueError, match="digest differs"):
        replay.load_saved_group(stack)


@pytest.mark.parametrize("bound_saved_pair", ("transposed",), indirect=True)
def test_bound_loader_preserves_full_noncontiguous_raw_source_and_output_layouts(bound_saved_pair):
    _, paths = bound_saved_pair
    with ExitStack() as stack:
        groups, provenance = replay.load_saved_group(stack)
        for mode, path in (("continuous", paths[-4]), ("resumed", paths[-2])):
            raw = torch.load(path, map_location="cpu", weights_only=True)["records"][0]["values"]["trace"]
            for name, trace_name in (("source", "ns_source"), ("output", "ns_output")):
                value = groups[mode][name]
                torch.testing.assert_close(value, raw[trace_name], rtol=0, atol=0)
                assert tuple(value.shape) == (16, 512, 512)
                assert value.stride() == (262144, 1, 512)
                assert not value.is_contiguous()
                assert value[-1, -2, -1] == 7
                assert provenance["raw_layouts"][mode][name] == {
                    "shape": list(raw[trace_name].shape),
                    "stride": list(raw[trace_name].stride()),
                    "storage_offset": raw[trace_name].storage_offset(),
                }
                assert value.storage_offset() == raw[trace_name].storage_offset() == 0


@pytest.mark.parametrize(
    "bound_saved_pair, message",
    (("offset", "raw trace layout"), ("wrong_shape", "full")),
    indirect=("bound_saved_pair",),
)
def test_bound_loader_rejects_layout_normalization_and_nonfull_batches(bound_saved_pair, message):
    with ExitStack() as stack, pytest.raises(ValueError, match=message):
        replay.load_saved_group(stack)


def test_retained_outputs_round_trip_and_expose_cross_process_difference(groups, pinned_ns):
    first = replay.replay_pair(groups, pinned_ns)
    stream = io.BytesIO()
    torch.save(first.outputs, stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=True)
    second = replay.replay_pair(groups, pinned_ns)
    assert all(replay.compare_outputs(a, b)["exact"] for a, b in zip(restored, second.outputs, strict=True))
    restored[3][1, 2, 1] += 0.125
    difference = replay.compare_outputs(restored[3], second.outputs[3])
    assert not difference["exact"]
    assert difference["changed_elements"] == 1
    assert not difference["within_tolerance"]
