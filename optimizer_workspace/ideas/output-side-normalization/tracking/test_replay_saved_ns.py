# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU behavioral coverage only; no issued saved-input GPU experiment."""

import hashlib
import io
import json
from contextlib import ExitStack
from functools import partial
from pathlib import Path

import ns_budget_execution as execution
import ns_budget_trial as trial
import ns_operand_precision as precision
import ns_step_budget as budget
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


@pytest.mark.parametrize("shape", ((2, 3, 5, 4), (2, 5, 3, 2)))
def test_compensated_operands_improve_fp64_product_accuracy_and_preserve_noncontiguous_inputs(shape):
    batch, rows, inner, columns = shape
    left = torch.linspace(-1.13, 0.91, batch * rows * inner).reshape(batch, inner, rows).transpose(-2, -1)
    right = torch.linspace(0.79, -0.97, batch * inner * columns).reshape(batch, columns, inner).transpose(-2, -1)
    snapshots = (left.clone(), right.clone())
    reference = torch.bmm(left.double(), right.double())
    outputs = {mode: precision.operand_bmm(left, right, mode) for mode in precision.MODES}
    errors = {mode: (output.double() - reference).abs().max().item() for mode, output in outputs.items()}
    assert errors["bf16_3"] < errors["bf16_1"] * 0.02
    assert errors["bf16_6"] < errors["bf16_3"] * 0.2
    torch.testing.assert_close(outputs["bf16_6"].double(), reference, rtol=1e-5, atol=1e-6)
    assert torch.equal(left, snapshots[0]) and torch.equal(right, snapshots[1])
    assert not left.is_contiguous() and not right.is_contiguous()


def test_compensated_product_retains_first_and_second_residual_levels_under_cancellation():
    left = torch.tensor([[[1.00390625, 1.0]]])
    right = torch.tensor([[[1.00390625], [-1.0]]])
    reference = torch.bmm(left.double(), right.double())
    single = precision.operand_bmm(left, right, "bf16_1")
    three = precision.operand_bmm(left, right, "bf16_3")
    six = precision.operand_bmm(left, right, "bf16_6")
    assert single.item() == 0.0
    assert (three.double() - reference).abs().item() > 0
    torch.testing.assert_close(six.double(), reference, rtol=0, atol=0)


@pytest.mark.parametrize("transposed", (False, True))
@pytest.mark.parametrize("regime", ("full_rank", "near_rank_deficient"))
def test_precision_polynomial_preserves_pinned_f32_and_exposes_fp64_accuracy_limits(regime, transposed, pinned_ns):
    task = Path(__file__).resolve().parents[1]
    manifest = json.loads((task / "experiment/upstream_manifest.json").read_text())
    path = task / "artifacts/external-sources" / f"Toner-{manifest['revision']}/src/toner/fast/muon.py"
    if regime == "full_rank":
        source = torch.tensor(
            [
                [[0.3, 0.1, -0.05, 0.2, 0.0], [0.0, 0.5, 0.1, -0.1, 0.1], [0.1, 0.0, 0.7, 0.05, -0.1]],
                [[0.4, -0.1, 0.1, 0.0, 0.2], [0.1, 0.6, 0.0, 0.1, -0.2], [-0.1, 0.1, 0.8, 0.1, 0.0]],
            ]
        )
        if transposed:
            source = source.transpose(-2, -1)
    else:
        shape = (2, 5, 3) if transposed else (2, 3, 5)
        source = torch.linspace(-0.97, 1.03, torch.tensor(shape).prod().item()).reshape(shape)
    original = source.clone()
    reference = precision.reference_fp64(source, (3.4445, -4.7750, 2.0315), steps=10, eps=1e-5)
    f32 = precision.pinned_numerical(path, "f32", steps=10, eps=1e-5)(source)
    compensated = precision.pinned_numerical(path, "bf16_6", steps=10, eps=1e-5)(source)
    torch.testing.assert_close(f32, pinned_ns(source), rtol=0, atol=0)
    if regime == "full_rank":
        torch.testing.assert_close(f32.double(), reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(compensated.double(), reference, rtol=1e-5, atol=1e-6)
    else:
        # Both FP32 recurrence variants amplify errors in nearly null singular directions.
        assert not replay.compare_outputs(f32, reference.float())["within_tolerance"]
        assert not replay.compare_outputs(compensated, reference.float())["within_tolerance"]
        assert torch.isfinite(f32).all() and torch.isfinite(compensated).all()
    assert torch.equal(source, original)


@pytest.mark.parametrize("shape", ((2, 3, 5), (2, 5, 3)))
def test_ns_step_svd_reference_matches_independent_matrix_polynomial_and_keeps_input(shape):
    source = torch.linspace(-0.91, 1.07, torch.tensor(shape).prod().item()).reshape(shape)
    before = source.clone()
    references, spectra = budget.spectral_references(source, (3.4445, -4.7750, 2.0315), 1e-5)
    for step in budget.STEPS:
        matrix_reference = precision.reference_fp64(source, (3.4445, -4.7750, 2.0315), steps=step, eps=1e-5)
        torch.testing.assert_close(references[str(step)], matrix_reference, rtol=1e-5, atol=1e-6)
    assert len(spectra["matrices"]) == source.shape[0]
    assert torch.equal(source, before)


def test_ns_budget_partial_polar_has_zero_null_directions_and_reports_rank():
    source = torch.diag_embed(torch.tensor([[3.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
    references, spectra = budget.spectral_references(source, (3.4445, -4.7750, 2.0315), 1e-5)
    torch.testing.assert_close(
        references["partial_polar"],
        torch.diag_embed(torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])).double(),
        rtol=0,
        atol=0,
    )
    assert [item["numerical_rank"] for item in spectra["matrices"]] == [2, 0]
    assert all(torch.count_nonzero(references[str(step)][1]) == 0 for step in budget.STEPS)
    metrics = budget.geometry(source, source, references["partial_polar"])["per_matrix"]
    assert metrics[1]["cosine_to_partial_polar"] is None
    assert metrics[1]["equal_norm_descent_ratio"] is None


def test_ns_budget_geometry_exposes_length_and_direction_distortion_without_grafting():
    source = torch.diag_embed(torch.tensor([[3.0, 1.0]]))
    polar = torch.eye(2).unsqueeze(0).double()
    same_direction = 2 * polar
    same = budget.geometry(same_direction, source, polar)["per_matrix"][0]
    distorted = budget.geometry(source, source, polar)["per_matrix"][0]
    assert same["cosine_to_partial_polar"] == pytest.approx(1.0)
    assert same["equal_norm_relative_polar_error"] == pytest.approx(0.0)
    assert same["equal_norm_descent_ratio"] == pytest.approx(1.0)
    assert distorted["cosine_to_partial_polar"] < 1.0
    assert distorted["equal_norm_relative_polar_error"] > 0.0
    torch.testing.assert_close(same_direction, 2 * polar, rtol=0, atol=0)


def test_ns_graph_retained_summary_exposes_one_phase_drift_and_exact_timing_statistics():
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    retained = [{"phase": phase, "output": source.clone()} for phase in trial.PHASES]
    timings = list(range(1, trial.TIMING_REPLAYS + 1))
    summary = trial.summarize_kernel(retained, timings)
    assert summary["all_phase_outputs_exact_finite"]
    assert summary["median_milliseconds"] == 25.5
    assert summary["p95_milliseconds"] == 48
    retained[-1]["output"][1, 2, 3] += 0.25
    changed = trial.summarize_kernel(retained, timings)
    assert not changed["all_phase_outputs_exact_finite"]
    assert changed["repeat_comparisons"][-1]["changed_elements"] == 1
    assert changed["repeat_comparisons"][-1]["max_absolute_error"] == 0.25


def test_ns_graph_component_refuses_cpu_execution_before_calling_numerical():
    with pytest.raises(RuntimeError, match="caller-owned resident CUDA"):
        trial.retain_kernel(torch.ones(2, 3, 4), None)


def test_ns_graph_summary_retains_nonfinite_final_output_as_failed_evidence():
    source = torch.ones(2, 3, 4)
    retained = [{"phase": phase, "output": source.clone()} for phase in trial.PHASES]
    retained[-1]["output"][1, 2, 3] = float("inf")
    summary = trial.summarize_kernel(retained, [0.5] * trial.TIMING_REPLAYS)
    assert not summary["all_phase_outputs_exact_finite"]
    failed = summary["repeat_comparisons"][-1]
    assert failed["phase"] == "timed_final"
    assert not failed["finite"] and not failed["within_tolerance"]
    assert failed["actual_nonfinite_elements"] == 1


def budget_cpu_retainer(mode, steps, source, coordinate):
    output = precision.reference_fp64(source, (3.4445, -4.7750, 2.0315), steps=steps, eps=1e-5).float()
    phases = [{"phase": phase, "output": output} for phase in execution.PHASES]
    evidence = trial.summarize_kernel(phases[1:], [0.5] * trial.TIMING_REPLAYS)
    evidence.update(
        source_shape=list(source.shape),
        source_stride=list(source.stride()),
        source_storage_offset=source.storage_offset(),
        rng_exact=True,
        compilation_preserved=True,
        graphs_preserved=True,
        graph_replays=3 + trial.TIMING_REPLAYS,
        input_exact_finite=execution.tracing.comparison(source, source, "input"),
    )
    return phases, evidence


@pytest.mark.parametrize("index", (0, 1))
def test_ns_budget_protocol_preserves_opposite_order_independent_snapshots_and_all_comparisons(groups, index):
    retained, lifecycle = execution.retain_protocol(groups, index, budget_cpu_retainer)
    before = groups["continuous"]["source"].clone()
    analysis = execution.summarize(retained, groups, (3.4445, -4.7750, 2.0315), index)
    assert [{key: item[key] for key in execution.coordinates(index)[0]} for item in retained] == (
        execution.coordinates(index)
    )
    assert len(retained) == 144 and len(lifecycle) == 24
    assert analysis["all_graph_phases_match_compiled_call"]
    assert analysis["all_compiled_phases_match_pinned_eager"]
    assert analysis["all_same_input_repeats_exact_finite"]
    assert all(
        item["timing_replays"] == 50 for item in execution.reconstruct_lifecycle(retained, lifecycle, groups, index)
    )
    retained[0]["output"].add_(1)
    assert not torch.equal(retained[0]["output"], retained[1]["output"])
    assert torch.equal(groups["continuous"]["source"], before)


@pytest.mark.parametrize("fault", ("input", "storage", "missing_phase", "partial_output"))
def test_ns_budget_retention_rejects_mutation_and_incomplete_evidence(groups, fault):
    def retain(mode, steps, source, coordinate):
        phases, evidence = budget_cpu_retainer(mode, steps, source, coordinate)
        if fault == "input":
            groups["resumed"]["source"].add_(0.01)
        elif fault == "storage":
            groups["resumed"]["output"] = groups["resumed"]["output"].clone()
        elif fault == "missing_phase":
            phases.pop()
        else:
            phases[0]["output"] = phases[0]["output"][:1]
        return phases, evidence

    with pytest.raises(ValueError):
        execution.retain_protocol(groups, 0, retain)


def test_ns_budget_summary_exposes_historical_compilation_timing_drift_and_changed_arm_outputs(groups):
    retained, lifecycle = execution.retain_protocol(groups, 0, budget_cpu_retainer)
    analysis = execution.summarize(retained, groups, (3.4445, -4.7750, 2.0315), 0)
    assert not analysis["all_ten_step_tpu_phases_reproduce_saved_outputs"]
    q5 = next(arm for arm in analysis["arms"] if arm["mode"] == "tpu" and arm["steps"] == 5)
    assert not q5["calls"][0]["phases"][0]["vs_same_mode_ten_steps"]["exact"]
    retained[1]["output"][1, 2, 1] += 0.125
    changed = execution.summarize(retained, groups, (3.4445, -4.7750, 2.0315), 0)
    assert not changed["all_compiled_phases_match_pinned_eager"]
    with pytest.raises(ValueError, match="phase or timing"):
        execution.reconstruct_lifecycle(retained, lifecycle, groups, 0)


def test_ns_budget_opposite_order_comparison_pairs_by_source_and_exposes_one_late_changed_phase(groups):
    first, _ = execution.retain_protocol(groups, 0, budget_cpu_retainer)
    second, _ = execution.retain_protocol(groups, 1, budget_cpu_retainer)
    comparison = execution.compare_processes(first, second, groups)
    assert comparison["all_cross_process_exact_finite"] and len(comparison["comparisons"]) == 144
    second[0]["output"][1, 2, 1] += 0.125
    changed = execution.compare_processes(first, second, groups)
    failed = [item for item in changed["comparisons"] if not item["exact"]]
    assert len(failed) == 1
    assert (failed[0]["mode"], failed[0]["steps"], failed[0]["source"], failed[0]["occurrence"], failed[0]["phase"]) == (
        "f32",
        5,
        "resumed",
        0,
        "pinned_eager",
    )
    assert failed[0]["changed_elements"] == 1


def test_ns_budget_unissued_owner_denial_precedes_cuda_or_output_creation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("NS budget execution initialized CUDA without an owner")

    def denied():
        raise RuntimeError("No separately accepted NS budget GPU owner")

    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    with pytest.raises(RuntimeError, match="accepted NS budget GPU owner"):
        execution.run(tmp_path, {}, 0, tmp_path / "process-0.pt", tmp_path / "process-0.json", denied)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fault", ("none", "output", "analysis", "timing", "lifecycle", "device", "order"))
def test_ns_budget_reconstruction_rederives_real_retained_files_and_rejects_rehashed_substitution(
    groups, tmp_path, monkeypatch, fault
):
    task = Path(__file__).resolve().parents[1]
    helpers = Path(execution.__file__).resolve().parent
    hashes = {name: sha256(helpers / name) for name in execution.REQUIRED_HELPERS}
    manifest = json.loads((task / "experiment/upstream_manifest.json").read_text())
    muon = task / "artifacts/external-sources" / f"Toner-{manifest['revision']}/src/toner/fast/muon.py"
    config_path, receipt_path = tmp_path / "config.json", tmp_path / "receipt.json"
    config_path.write_text(json.dumps({"runtime": {"deterministic": True}}))
    receipt_path.write_text(json.dumps({"config": str(config_path)}))
    provenance = {
        "pinned_muon": str(muon),
        "pinned_muon_sha256": replay.MUON_SHA256,
        "config_sha256": sha256(config_path),
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    monkeypatch.setattr(replay.analyzer, "RECEIPT_RELATIVE", receipt_path.name)
    monkeypatch.setattr(replay, "RECEIPT_SHA256", sha256(receipt_path))
    monkeypatch.setattr(replay, "load_saved_group", lambda stack: (groups, provenance))
    retained, lifecycle = execution.retain_protocol(groups, 0, budget_cpu_retainer)
    owner = {"physical_uuid": "GPU-00000000-0000-0000-0000-000000000001"}
    report_path, output_path = tmp_path / "process-0.json", tmp_path / "process-0.pt"
    report = {
        "status": execution.STATUS,
        "worker_index": 0,
        "owner": owner,
        "report_path": str(report_path),
        "helper_sources_sha256": hashes,
        "coordinates": execution.coordinates(0),
        "rng_exact": True,
        "compilation_preserved": True,
        "graphs_preserved": True,
        "compile_options": {"dynamic": False},
        "physical_uuid": owner["physical_uuid"],
        "actual_uuid": owner["physical_uuid"],
        "outputs": str(output_path),
        "provenance": provenance,
        "runtime": {"deterministic": True},
        "lifecycle": lifecycle,
        "analysis": execution.summarize(retained, groups, (3.4445, -4.7750, 2.0315), 0),
        "timings": execution.reconstruct_lifecycle(retained, lifecycle, groups, 0),
    }
    if fault == "output":
        retained[-1]["output"][1, 2, 1] += 0.125
    elif fault == "analysis":
        report["analysis"]["all_same_input_repeats_exact_finite"] = False
    elif fault == "timing":
        lifecycle[0]["execution"]["cuda_event_milliseconds"][-1] = 100
    elif fault == "lifecycle":
        lifecycle.pop()
    elif fault == "device":
        report["actual_uuid"] = "GPU-00000000-0000-0000-0000-000000000002"
    elif fault == "order":
        report["coordinates"] = execution.coordinates(1)
    torch.save(retained, output_path)
    report["outputs_sha256"] = sha256(output_path)
    report_path.write_text(json.dumps(report))
    if fault != "none":
        with pytest.raises(ValueError):
            execution.reconstruct(report_path, sha256(report_path), owner, hashes, 0)
        return
    result, independent = execution.reconstruct(report_path, sha256(report_path), owner, hashes, 0)
    assert result["analysis"] == report["analysis"] and result["timings"] == report["timings"]
    assert len(independent) == 144 and result["analysis"]["all_same_input_repeats_exact_finite"]
    independent[0]["output"].zero_()
    assert not torch.equal(independent[0]["output"], retained[0]["output"])
