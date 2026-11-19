# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise streamed numerical evidence and authenticated reconstruction without CUDA."""

import copy
import json
from pathlib import Path

import ns_compiler_localization as localization
import ns_compiler_localization_execution as execution
import pytest
import replay_saved_ns as replay
import torch
from selected_lr_release_owner import sha256

PINNED = "artifacts/pinned-sources-materialized/Toner-dab7f3777c75e740536df822f50ffa1d929a340c/src/toner/fast/muon.py"
PARENT = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
    "/run-preparation/ns-compiler-localization-component-cpu/20261119T1200"
)


@pytest.fixture
def evidence_fixture():
    local = Path(__file__).resolve().parents[1] / PINNED
    source_path = local if local.is_file() else PARENT / PINNED
    maps = localization.compiler_maps(source_path, lambda fn: fn, mode="f32", steps=10, eps=1e-5)
    source = torch.sin(torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) * 0.73) + 0.1
    groups = {
        "continuous": {"source": source, "output": maps.controls["eager"](source)},
        "resumed": {"source": source + 0.02, "output": maps.controls["eager"](source + 0.02)},
    }

    def retain(call, source):
        trace = maps.traces["eager"](source)
        # Synthetic casts exercise the persisted BF16 operand schema, not CUDA numerics.
        if call["mode"] == "tpu":
            for record in trace["records"]:
                if record["stage"] in localization.PRODUCT_STAGES:
                    for name in ("left_operand", "right_operand"):
                        record["values"][name] = record["values"][name].to(torch.bfloat16)
        for phase in localization.PHASES:
            yield {"phase": phase, "control": trace["output"].clone(), "trace": trace}

    return groups, retain


@pytest.mark.parametrize("index", (0, 1))
def test_localization_files_reconstruct_complete_opposite_orders_and_full_comparisons(evidence_fixture, tmp_path, index):
    groups, retain = evidence_fixture
    directory = tmp_path / "phases"
    manifest = execution.retain_files(groups, index, directory, retain)
    report = execution.summarize(manifest, directory, groups, index)
    assert len(manifest) == len(report["comparisons_to_eager"]) == 128
    assert len(report["source_pairs"]) == len(report["same_input_repeats"]) == 64
    assert report["all_traces_transparent_exact_finite"]
    assert report["all_control_phases_exact_finite"] and report["all_same_input_repeats_exact_finite"]
    assert all(row["stage_comparison"]["first_changed_stage"] is None for row in report["comparisons_to_eager"])
    assert all(not row["controls"]["exact"] for row in report["source_pairs"])
    assert report["all_tpu_eager_phases_reproduce_saved_outputs"]
    files = execution.PhaseFiles(manifest, directory, groups, index)
    first = files.load(execution.coordinates(index)[0])
    second = files.load(execution.coordinates(index)[1])
    first["control"].zero_()
    assert torch.count_nonzero(second["control"]) > 0
    assert torch.count_nonzero(files.load(execution.coordinates(index)[0])["control"]) > 0


def test_localization_files_snapshot_each_yield_before_reused_output_changes(evidence_fixture, tmp_path):
    groups, original = evidence_fixture

    def retain(call, source):
        buffer = None
        for phase in original(call, source):
            buffer = phase["control"] if buffer is None else buffer
            phase["control"] = buffer
            yield phase
            phase["control"].add_(0.125)

    directory = tmp_path / "phases"
    manifest = execution.retain_files(groups, 0, directory, retain)
    report = execution.summarize(manifest, directory, groups, 0)
    assert not report["all_control_phases_exact_finite"]
    assert report["all_same_input_repeats_exact_finite"]
    files = execution.PhaseFiles(manifest, directory, groups, 0)
    actual = files.load(execution.coordinates(0)[0])
    torch.testing.assert_close(actual["control"], groups["continuous"]["output"], rtol=0, atol=0)
    assert report["phases_vs_call"][1]["control"]["changed_elements"] == 6


@pytest.mark.parametrize("fault", ("source", "storage", "phase", "partial"))
def test_localization_files_refuse_mutation_and_incomplete_persisted_protocol(evidence_fixture, tmp_path, fault):
    groups, original = evidence_fixture

    def retain(call, source):
        phases = list(original(call, source))
        if fault == "source":
            groups["resumed"]["source"].add_(1)
        elif fault == "storage":
            groups["resumed"]["output"] = groups["resumed"]["output"].clone()
        elif fault == "phase":
            phases.pop()
        else:
            phases[0]["control"] = phases[0]["control"][:, :1]
        yield from phases

    with pytest.raises(ValueError):
        execution.retain_files(groups, 0, tmp_path / "phases", retain)
    assert list(tmp_path.iterdir())


@pytest.mark.parametrize("fault", ("missing", "duplicate", "order", "aliased", "changed"))
def test_localization_files_reject_incomplete_or_substituted_inventory(evidence_fixture, tmp_path, fault):
    groups, retain = evidence_fixture
    directory = tmp_path / "phases"
    manifest = execution.retain_files(groups, 0, directory, retain)
    if fault == "missing":
        manifest.pop()
    elif fault == "duplicate":
        manifest[-1] = copy.deepcopy(manifest[0])
    elif fault == "order":
        manifest.reverse()
    elif fault == "aliased":
        alias = tmp_path / "alias.pt"
        alias.symlink_to(manifest[0]["path"])
        manifest[0]["path"] = str(alias)
    else:
        path = Path(manifest[0]["path"])
        row = torch.load(path, weights_only=True)
        row["control"].add_(0.125)
        torch.save(row, path)
    with pytest.raises(ValueError):
        execution.summarize(manifest, directory, groups, 0)


@pytest.mark.parametrize("fault", ("output", "nonfinite", "intermediate"))
def test_localization_files_preserve_failed_transparency_or_changed_stage(evidence_fixture, tmp_path, fault):
    groups, original = evidence_fixture

    def retain(call, source):
        for phase in original(call, source):
            phase = copy.deepcopy(phase)
            if call["boundary"] == "full_compiled":
                if fault == "output":
                    phase["control"][0, 0, 0] += 0.125
                elif fault == "nonfinite":
                    phase["trace"]["records"][3]["values"]["product"][0, 0, 0] = float("nan")
                else:
                    phase["trace"]["records"][3]["values"]["product"][0, 0, 0] += 0.125
            yield phase

    directory = tmp_path / "phases"
    manifest = execution.retain_files(groups, 0, directory, retain)
    report = execution.summarize(manifest, directory, groups, 0)
    changed = [row for row in report["comparisons_to_eager"] if row["boundary"] == "full_compiled"]
    if fault in ("output", "nonfinite"):
        assert not report["all_traces_transparent_exact_finite"]
        assert all(row["stage_comparison"] is None for row in changed)
    else:
        assert report["all_traces_transparent_exact_finite"]
        assert all(row["controls"]["exact"] for row in changed)
        assert all(row["stage_comparison"]["first_changed_stage"]["stage"] == "gram" for row in changed)
    json.dumps(report, allow_nan=False)


def test_localization_files_cross_process_pairing_exposes_internal_trace_drift(evidence_fixture, tmp_path):
    groups, retain = evidence_fixture
    directories = (tmp_path / "process-0", tmp_path / "process-1")
    manifests = [execution.retain_files(groups, index, directory, retain) for index, directory in enumerate(directories)]
    report = execution.compare_processes(*manifests, directories, groups)
    assert report["all_cross_process_exact_finite"] and len(report["comparisons"]) == 128
    path = Path(manifests[1][0]["path"])
    row = torch.load(path, weights_only=True)
    row["trace"]["records"][3]["values"]["product"][0, 0, 0] += 0.125
    torch.save(row, path)
    manifests[1][0]["sha256"] = sha256(path)
    changed = execution.compare_processes(*manifests, directories, groups)
    assert not changed["all_cross_process_exact_finite"]
    failures = [row for row in changed["comparisons"] if not row["trace"]["exact"]]
    assert len(failures) == 1
    assert failures[0]["source"] == "resumed" and failures[0]["occurrence"] == 0 and failures[0]["phase"] == "call"
    assert failures[0]["controls"]["exact"] and failures[0]["stage_interpretation_permitted"]
    assert failures[0]["stage_comparison"]["first_changed_stage"]["stage"] == "gram"


def test_localization_unissued_owner_denial_precedes_cuda_initialization_or_files(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Localization initialized CUDA without a separately accepted owner")

    def denied():
        raise RuntimeError("No separately accepted localization GPU owner")

    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    with pytest.raises(RuntimeError, match="accepted localization GPU owner"):
        execution.run(tmp_path, {}, 0, tmp_path / "process-0.json", denied)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fault", ("none", "output", "analysis", "lifecycle", "device", "order"))
def test_localization_reconstruction_rederives_files_and_rejects_rehashed_substitution(
    evidence_fixture,
    tmp_path,
    monkeypatch,
    fault,
):
    groups, retain = evidence_fixture
    helpers = Path(execution.__file__).resolve().parent
    hashes = {name: sha256(helpers / name) for name in execution.REQUIRED_HELPERS}
    config_path, receipt_path = tmp_path / "config.json", tmp_path / "receipt.json"
    config_path.write_text(json.dumps({"runtime": {"deterministic": True}}))
    receipt_path.write_text(json.dumps({"config": str(config_path)}))
    provenance = {"config_sha256": sha256(config_path)}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(replay, "ROOT", tmp_path)
    monkeypatch.setattr(replay.analyzer, "RECEIPT_RELATIVE", receipt_path.name)
    monkeypatch.setattr(replay, "RECEIPT_SHA256", sha256(receipt_path))
    monkeypatch.setattr(replay, "load_saved_group", lambda stack: (groups, provenance))
    report_path = tmp_path / "process-0.json"
    directory = report_path.with_suffix(".phases")
    manifest = execution.retain_files(groups, 0, directory, retain)
    lifecycle = [
        call
        | {
            "execution": {
                "source_shape": list(groups[call["source"]]["source"].shape),
                "source_stride": list(groups[call["source"]]["source"].stride()),
                "source_storage_offset": groups[call["source"]]["source"].storage_offset(),
                "input_exact_finite": execution.tracing.comparison(
                    groups[call["source"]]["source"], groups[call["source"]]["source"], "input"
                ),
                "rng_exact": True,
                "graphs_preserved": True,
                "graph_replays": 3,
            }
        }
        for call in localization.call_coordinates(0)
    ]
    owner = {"physical_uuid": "GPU-00000000-0000-0000-0000-000000000001"}
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
        "phase_directory": str(directory),
        "phase_files": manifest,
        "provenance": provenance,
        "runtime": {"deterministic": True},
        "lifecycle": lifecycle,
        "analysis": execution.summarize(manifest, directory, groups, 0),
        "scope": execution.SCOPE,
    }
    if fault == "output":
        path = Path(manifest[-1]["path"])
        row = torch.load(path, weights_only=True)
        row["control"].add_(0.125)
        torch.save(row, path)
        manifest[-1]["sha256"] = sha256(path)
    elif fault == "analysis":
        report["analysis"]["all_same_input_repeats_exact_finite"] = False
    elif fault == "lifecycle":
        lifecycle.pop()
    elif fault == "device":
        report["actual_uuid"] = "GPU-00000000-0000-0000-0000-000000000002"
    elif fault == "order":
        report["coordinates"] = execution.coordinates(1)
    report_path.write_text(json.dumps(report, allow_nan=False))
    if fault != "none":
        with pytest.raises(ValueError):
            execution.reconstruct(report_path, sha256(report_path), owner, hashes, 0)
        return
    result = execution.reconstruct(report_path, sha256(report_path), owner, hashes, 0)
    assert result["analysis"] == report["analysis"] and result["phase_files"] == manifest
    assert result["analysis"]["all_traces_transparent_exact_finite"]
