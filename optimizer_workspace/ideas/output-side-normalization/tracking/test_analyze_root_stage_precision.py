# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent rounding and retained-file regressions for saved root-stage localization."""

import copy
import json
from collections import Counter

import analyze_root_stage_precision as precision
import pytest
import root_stage_gpu_worker as worker
import root_stage_trace as tracing
import torch
from selected_lr_release_owner import sha256


def snapshot():
    matrix = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) / 128 + 0.5
    counts, records = Counter(), []
    for stage in worker.stage_sequence(3):
        values = {field: matrix.clone() for field in tracing.OUTPUT_FIELDS[stage]}
        if stage in tracing.PRODUCTS:
            values.update(left=matrix.clone(), right=matrix.clone())
        for source, cast in precision.CAST_PAIRS.get(stage, ()):
            values[cast] = values[source].to(torch.float16)
        records.append({"stage": stage, "occurrence": counts[stage], "values": values})
        counts[stage] += 1
    output = (
        (matrix.clone(), matrix.clone()),
        matrix.clone(),
        torch.zeros(2, dtype=torch.bool),
        (matrix[:, 0].clone(), matrix.clone(), torch.ones(2)),
    )
    return {"schema": 1, "records": records, "output": output}


def test_exact_retained_pairs_pass_rounding_without_changing_input_storage():
    original = snapshot()
    before = copy.deepcopy(original)
    result = precision.paired_stages(original, copy.deepcopy(original))
    assert result["all_casts_match_independent_rounding"]
    assert result["first_exact_difference"] is None
    assert result["first_tolerance_failure"] is None
    assert len(result["actual_casts"]) == 9
    assert all(item["independent_rounding_crossings"] == 0 for item in result["crossed_casts"])
    assert tracing.comparison(original, before, "source")["exact"]
    assert json.loads(json.dumps(result, allow_nan=False))["all_casts_match_independent_rounding"]


@pytest.mark.parametrize(
    "stage,source,cast",
    [
        ("start", "E", "Eb"),
        ("start", "O", "Ob"),
        ("iter_mid", "P", "Pb"),
        ("iter_post", "O", "Ob"),
        ("sqrt_start", "Q", "Qb"),
        ("sqrt_post", "Q", "Qb"),
    ],
)
def test_changed_low_precision_value_is_localized_to_its_matrix_stage_and_cast(stage, source, cast):
    expected, actual = snapshot(), snapshot()
    target = next(record for record in actual["records"] if record["stage"] == stage)
    target["values"][cast][-1, -1, -1] += 0.125
    result = precision.paired_stages(actual, expected)
    changed = [item for item in result["actual_casts"] if not item["matches_independent_rounding"]]
    assert len(changed) == 1
    assert (changed[0]["stage"], changed[0]["source"], changed[0]["cast"]) == (stage, source, cast)
    assert changed[0]["mismatches_per_matrix"] == [0, 1]
    assert result["first_tolerance_failure"]["stage"] == stage
    assert result["first_tolerance_failure"]["occurrence"] == 0
    assert result["stage_comparison"]["output"]["exact"]
    assert not result["all_casts_match_independent_rounding"]


def test_small_source_difference_crossing_half_threshold_is_not_a_cast_inconsistency():
    expected, actual = snapshot(), snapshot()
    midpoint = 1.0 + 2**-11
    for value, offset in ((expected, -(2**-23)), (actual, 2**-23)):
        stage = next(record for record in value["records"] if record["stage"] == "start")
        stage["values"]["E"].fill_(midpoint + offset)
        stage["values"]["Eb"] = stage["values"]["E"].to(torch.float16)
    result = precision.paired_stages(actual, expected)
    crossing = next(item for item in result["crossed_casts"] if item["source"] == "E")
    assert crossing["source_comparison"]["within_tolerance"]
    assert crossing["independent_rounding_crossings"] == 18
    assert crossing["crossings_per_matrix"] == [9, 9]
    assert crossing["source_within_tolerance_but_rounded_values_differ"]
    assert result["all_casts_match_independent_rounding"]


def test_nonfinite_cast_pair_cannot_be_reported_as_consistent_rounding():
    value = snapshot()
    stage = next(record for record in value["records"] if record["stage"] == "start")
    stage["values"]["E"][0, 0, 0] = float("inf")
    stage["values"]["Eb"][0, 0, 0] = float("inf")
    result = precision.cast_records(value)
    cast = next(item for item in result if item["source"] == "E")
    assert cast["mismatched_elements"] == 0
    assert not cast["matches_independent_rounding"]
    assert not cast["comparison"]["finite"]


def test_unequal_root_stage_sequences_are_not_prefix_localized():
    expected, actual = snapshot(), snapshot()
    actual["records"].pop()
    with pytest.raises(ValueError, match="prefix-pair"):
        precision.paired_stages(actual, expected)


def retained(index):
    values = []
    for coordinate in worker.coordinates(index):
        stages = snapshot()
        values.append(
            {
                **coordinate,
                "output": stages["output"],
                "stages": stages if coordinate["protocol"] == "trace" else None,
            }
        )
    return values


@pytest.fixture
def saved_terminal(tmp_path, monkeypatch):
    root = tmp_path
    preparation = root / "run-preparation"
    work = root / "diagnostics/root-stage-replay-20261118"
    preparation.mkdir()
    work.mkdir(parents=True)
    helpers = root / "issued"
    helpers.mkdir()
    source = helpers / "root_stage_gpu_controller.py"
    source.write_text("# independently accepted controller fixture\n")
    cpu = root / precision.CPU
    cpu.parent.mkdir(parents=True, exist_ok=True)
    cpu.write_text("{}")
    cpu_hash = sha256(cpu)
    binding = {
        "status": "frozen_root_stage_replay_diagnostic_only",
        "cpu_receipt": str(cpu),
        "cpu_receipt_sha256": cpu_hash,
        "helpers": str(helpers),
        "helpers_sha256": {source.name: sha256(source)},
    }
    binding_path = preparation / "root-stage-replay-binding-20261118.json"
    binding_path.write_text(json.dumps(binding))
    reports, evidence = [], {str(binding_path.relative_to(root)): sha256(binding_path)}
    for index in (0, 1):
        output, report_path = work / f"process-{index}.pt", work / f"process-{index}.json"
        torch.save(retained(index), output)
        report = {
            "worker_index": index,
            "outputs": str(output),
            "outputs_sha256": sha256(output),
            "report_path": str(report_path),
            "analysis": {
                **{
                    key: True
                    for key in (
                        "trace_transparency_exact",
                        "capture_repeat_safety_exact",
                        "capture_repeat_stages_exact",
                        "trace_stages_finite",
                        "intermediates_interpretable",
                    )
                },
                "historical_outputs_within_tolerance": False,
            },
        }
        report_path.write_text(json.dumps(report))
        evidence.update({str(path.relative_to(root)): sha256(path) for path in (output, report_path)})
        reports.append(report)
    terminal = {
        "status": precision.TERMINAL_STATUS,
        "binding_sha256": sha256(binding_path),
        "verifier_sha256": sha256(source),
        "intermediates_interpretable": True,
        "evidence_sha256": evidence,
        "comparison": {
            "intermediates_interpretable": True,
            "all_cross_process_outputs_exact": True,
            "all_cross_process_stages_exact": True,
            "reports": reports,
        },
    }
    path = preparation / "terminal.json"
    path.write_text(json.dumps(terminal))
    monkeypatch.setattr(precision.replay, "ROOT", root)
    monkeypatch.setattr(precision, "CPU_SHA256", cpu_hash)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return root, path, terminal


def test_file_bound_precision_analysis_preserves_failed_historical_attribution(saved_terminal):
    root, path, _ = saved_terminal
    out = root / "run-preparation/precision.json"
    result = precision.analyze(root, path, sha256(path), out)
    assert json.loads(out.read_text())["status"] == precision.STATUS
    assert len(result["processes"]) == 2
    assert all(len(process["pairs"]) == 6 for process in result["processes"])
    assert all(process["all_recorded_casts_match_independent_rounding"] for process in result["processes"])
    assert not result["historical_interpretation_allowed"]
    assert not result["historical_outputs_reproduced_within_original_tolerance"]


@pytest.mark.parametrize("fault", ("transparency", "terminal_hash", "output_hash", "source_hash"))
def test_substituted_saved_evidence_refuses_before_publishing_precision_analysis(saved_terminal, fault):
    root, path, terminal = saved_terminal
    expected_hash = sha256(path)
    if fault == "transparency":
        terminal["comparison"]["reports"][0]["analysis"]["trace_transparency_exact"] = False
        path.write_text(json.dumps(terminal))
        expected_hash = sha256(path)
    elif fault == "terminal_hash":
        path.write_text(json.dumps(terminal) + " ")
    elif fault == "output_hash":
        target = root / "diagnostics/root-stage-replay-20261118/process-1.pt"
        with target.open("ab") as stream:
            stream.write(b"changed")
    else:
        (root / "issued/root_stage_gpu_controller.py").write_text("# changed accepted source\n")
    out = root / "run-preparation/denied.json"
    with pytest.raises((ValueError, RuntimeError)):
        precision.analyze(root, path, expected_hash, out)
    assert not out.exists()
