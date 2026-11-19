# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU protocol regressions for a separately owned, source-bound CUDA start component."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import root_start_accuracy as accuracy
import root_start_execution as execution
import root_start_rational as rational
import torch
from benchmark_diagonal import source_definitions


@pytest.fixture
def pinned():
    task = Path(__file__).resolve().parents[1]
    manifest = json.loads((task / "experiment/upstream_manifest.json").read_bytes())
    source = task / f"artifacts/external-sources/Toner-{manifest['revision']}/src/toner/fast/roots.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["files"]["src/toner/fast/roots.py"]
    return source_definitions(
        source,
        ("_start",),
        {"torch": torch, "EPS": 1e-3, "QUARTER": 0.25, "U32": 2.0**-24, "LOWP": torch.float16},
    )["_start"]


def sources():
    diagonal = torch.tensor([[1.0, 1.000002, 2.0]], dtype=torch.float32)
    off = torch.tensor([[0.0, 0.01, -0.02], [0.01, 0.0, 0.03], [-0.02, 0.03, 0.0]])
    B = torch.diag_embed(diagonal) + off
    return {group: {mode: B.clone() for mode in accuracy.MODES} for group in accuracy.GROUPS}


@pytest.mark.parametrize("index", (0, 1))
def test_cpu_protocol_preserves_complete_opposite_order_and_independent_phase_snapshots(pinned, index):
    inputs = sources()
    original = copy.deepcopy(inputs)

    def retain(variant, value, coordinate):
        start = pinned if variant == "pinned" else rational.rational_start
        output = start(value, 0.0)
        return [{"phase": phase, "output": output} for phase in accuracy.PHASES], {"cpu_control": True}

    retained, records = execution.retain_protocol(inputs, index, retain)
    analyzed = accuracy.summarize(retained, inputs, pinned, 0.0, index)
    assert [{key: item[key] for key in ("variant", "group", "mode", "phase")} for item in retained] == (
        accuracy.coordinates(index)
    )
    assert len(records) == 8
    assert analyzed["all_repeat_outputs_exact_finite"]
    assert analyzed["all_unaffected_outputs_exact_finite"]
    assert analyzed["all_rational_O_within_fp64_tolerance"]
    assert all(torch.equal(inputs[group][mode], original[group][mode]) for group in inputs for mode in inputs[group])
    retained[0]["output"][4].fill_(0.0)
    assert not torch.equal(retained[0]["output"][4], retained[1]["output"][4])
    assert torch.equal(inputs["d"]["continuous"], original["d"]["continuous"])


@pytest.mark.parametrize("failure", ("mutate_values", "replace_storage", "missing_phase", "partial_output"))
def test_bad_retention_cannot_publish_complete_protocol(pinned, failure):
    inputs = sources()

    def retain(variant, value, coordinate):
        output = pinned(value, 0.0)
        phases = [{"phase": phase, "output": output} for phase in accuracy.PHASES]
        if failure == "mutate_values":
            inputs["f"]["resumed"].add_(0.01)
        elif failure == "replace_storage":
            inputs["f"]["resumed"] = inputs["f"]["resumed"].clone()
        elif failure == "missing_phase":
            phases.pop()
        else:
            phases[0]["output"] = output[:-1]
        return phases, {}

    with pytest.raises(ValueError):
        execution.retain_protocol(inputs, 0, retain)


def test_unissued_owner_denial_precedes_cuda_initialization_and_output_creation(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unissued start component initialized CUDA")

    def denied():
        raise RuntimeError("No separately accepted GPU owner")

    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(torch.cuda, "get_device_properties", forbidden)
    with pytest.raises(RuntimeError, match="accepted GPU owner"):
        execution.run(tmp_path, {}, 0, tmp_path / "outputs.pt", tmp_path / "report.json", denied)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fault", ("none", "retained_output", "analysis", "missing_lifecycle", "device"))
def test_reconstruction_rederives_authenticated_arithmetic_and_rejects_changed_evidence(
    pinned, tmp_path, monkeypatch, fault
):
    inputs = sources()
    task = Path(__file__).resolve().parents[1]
    manifest = json.loads((task / "experiment/upstream_manifest.json").read_bytes())
    roots = (task / f"artifacts/external-sources/Toner-{manifest['revision']}/src/toner/fast/roots.py").resolve()
    config = json.loads((task / "experiment/configs/inner_outer_leftdiag_0p5m_seed0.json").read_bytes())
    provenance = {group: {"pinned_roots": str(roots), "config": config} for group in accuracy.GROUPS}

    def retain(variant, value, coordinate):
        start = pinned if variant == "pinned" else rational.rational_start
        result = start(value, 0.0)
        return [{"phase": phase, "output": result} for phase in accuracy.PHASES], {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "storage_offset": value.storage_offset(),
            "source_shape": list(value.shape),
            "source_stride": list(value.stride()),
            "source_storage_offset": value.storage_offset(),
            "rng_exact": True,
            "compilation_preserved": True,
            "graphs_preserved": True,
            "captured_graph_replays": 3,
            "input_exact_finite": execution.tracing.comparison(value, value, "input"),
        }

    retained, lifecycle = execution.retain_protocol(inputs, 0, retain)
    owner = {"physical_uuid": "GPU-00000000-0000-0000-0000-000000000001"}
    report_path, output_path = tmp_path / "process-0.json", tmp_path / "process-0.pt"
    report = {
        "status": execution.STATUS,
        "worker_index": 0,
        "owner": owner,
        "report_path": str(report_path),
        "helper_sources_sha256": {},
        "coordinates": accuracy.coordinates(0),
        "rng_exact": True,
        "compilation_preserved": True,
        "graphs_preserved": True,
        "compile_options": {"dynamic": False},
        "physical_uuid": owner["physical_uuid"],
        "actual_uuid": owner["physical_uuid"],
        "outputs": str(output_path),
        "provenance": provenance,
        "runtime": config["runtime"],
        "freeze": 0,
        "lifecycle": lifecycle,
        "analysis": accuracy.summarize(retained, inputs, pinned, 0.0, 0),
    }
    if fault == "retained_output":
        retained[0]["output"][4].add_(0.01)
    elif fault == "analysis":
        report["analysis"]["all_rational_O_within_fp64_tolerance"] = False
    elif fault == "missing_lifecycle":
        report["lifecycle"].pop()
    elif fault == "device":
        report["actual_uuid"] = "GPU-00000000-0000-0000-0000-000000000002"
    torch.save(retained, output_path)
    report["outputs_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))
    report_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()

    def load_saved(stack, group):
        return {mode: {"saved_output": (None, value)} for mode, value in inputs[group].items()}, provenance[group]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(execution.replay, "load_saved_roots", load_saved)
    if fault != "none":
        with pytest.raises(ValueError):
            execution.reconstruct(report_path, report_hash, owner, {}, 0)
        return
    result, independent = execution.reconstruct(report_path, report_hash, owner, {}, 0)
    assert result["analysis"] == report["analysis"]
    assert result["analysis"]["all_repeat_outputs_exact_finite"]
    assert result["analysis"]["all_rational_O_within_fp64_tolerance"]
    assert len(independent) == 32
    independent[0]["output"][4].zero_()
    assert not torch.equal(independent[0]["output"][4], retained[0]["output"][4])
