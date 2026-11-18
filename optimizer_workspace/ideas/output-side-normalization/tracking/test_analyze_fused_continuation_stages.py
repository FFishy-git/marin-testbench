# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic CPU regressions, not evidence from an executed optimizer trajectory."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import analyze_fused_continuation_stages as analyzer
import pytest
import torch


def record(stage, value):
    return {"stage": stage, "values": {"tensor": torch.tensor(value, dtype=torch.float32)}}


def stable_records():
    return [
        record("input/statistics", [1.0, 2.0]),
        record("input/d/before_root", [3.0]),
        record("input/d/candidate", [4.0]),
        record("model/gradients", [5.0]),
        record("input/before_finish", [6.0]),
        record("input/finished", [7.0]),
        record("matrix/before_update", [8.0]),
        record("matrix/after_update", [9.0]),
        record("matrix/before_update", [10.0]),
        record("matrix/after_update", [11.0]),
        record("model/after_update", [12.0]),
    ]


def test_unequal_branch_lengths_preserve_stable_tensor_comparisons():
    expected = stable_records()
    actual = copy.deepcopy(expected)
    actual[5:5] = [record("input/refinement/input", [20.0]), record("input/refinement/output", [21.0])]
    actual[10]["values"]["tensor"][0] += 0.25
    result = analyzer.compare_stages(actual, expected)
    assert result["complete_stable_coverage"]
    assert result["tensor_comparisons"] == len(expected)
    assert not result["stable_stages_within_tolerance"]
    changed = [pair for pair in result["matched_stages"] if pair["changed_tensors"]]
    assert [(pair["stage"], pair["occurrence"]) for pair in changed] == [("matrix/before_update", 1)]
    assert changed[0]["changed_tensors"][0]["max_absolute_error"] == 0.25
    assert [item["record_index"] for item in result["branch_records_not_compared"]["actual"]] == [5, 6]
    assert result["branch_records_not_compared"]["expected"] == []


def test_equal_length_different_labels_are_retained_not_positionally_compared():
    expected = stable_records()
    actual = copy.deepcopy(expected)
    actual[1] = record("input/f/before_root", [99.0])
    result = analyzer.compare_stages(actual, expected)
    assert result["tensor_comparisons"] == len(expected) - 1
    assert not result["complete_stable_coverage"]
    assert result["unmatched_records"]["actual"][0]["stage"] == "input/f/before_root"
    assert result["unmatched_records"]["expected"][0]["stage"] == "input/d/before_root"
    assert all(not pair["changed_tensors"] for pair in result["matched_stages"])


def test_repeated_stable_stages_pair_by_label_and_deterministic_occurrence():
    expected = stable_records()
    actual = copy.deepcopy(expected)
    actual[8:8] = [record("input/exact_fallback/input", [90.0]), record("input/exact_fallback/output", [91.0])]
    actual[11]["values"]["tensor"] += 0.5
    result = analyzer.compare_stages(actual, expected)
    pairs = [pair for pair in result["matched_stages"] if pair["stage"] == "matrix/after_update"]
    assert [(pair["occurrence"], pair["actual_index"], pair["expected_index"]) for pair in pairs] == [
        (0, 7, 7),
        (1, 11, 9),
    ]
    assert pairs[0]["changed_tensors"] == []
    assert pairs[1]["changed_tensors"][0]["changed_elements"] == 1
    assert pairs[1]["changed_tensors"][0]["max_absolute_error"] == 0.5


@pytest.mark.parametrize("side", ("actual", "both"))
def test_missing_required_stage_remains_visible_even_if_both_traces_omit_it(side):
    actual, expected = stable_records(), stable_records()
    actual = [item for item in actual if item["stage"] != "input/statistics"]
    if side == "both":
        expected = [item for item in expected if item["stage"] != "input/statistics"]
    result = analyzer.compare_stages(actual, expected)
    assert result["missing_required_stages"]["actual"] == ["input/statistics"]
    assert result["missing_required_stages"]["expected"] == (["input/statistics"] if side == "both" else [])
    assert not result["complete_stable_coverage"]
    assert not result["stable_stages_within_tolerance"]


def test_unequal_matrix_counts_do_not_shift_or_prefix_pair_matrix_identity():
    actual, expected = stable_records(), stable_records()
    del actual[6:8]
    result = analyzer.compare_stages(actual, expected)
    assert result["ambiguous_matrix_stages"] == list(analyzer.MATRIX_STAGES)
    assert not any(pair["stage"].startswith("matrix/") for pair in result["matched_stages"])
    assert len(result["unmatched_records"]["actual"]) == 2
    assert len(result["unmatched_records"]["expected"]) == 4
    assert not result["complete_stable_coverage"]


def test_equal_branch_counts_do_not_claim_semantic_branch_tensor_parity():
    actual, expected = stable_records(), stable_records()
    for target, offset in ((actual, 100.0), (expected, 200.0)):
        target[5:5] = [
            record("input/refinement/input", [offset]),
            record("input/refinement/output", [offset + 1]),
            record("input/refinement/input", [offset + 2]),
            record("input/refinement/output", [offset + 3]),
        ]
    result = analyzer.compare_stages(actual, expected)
    assert result["stable_stages_within_tolerance"]
    assert result["tensor_comparisons"] == len(stable_records())
    assert all(not pair["stage"].startswith("input/refinement/") for pair in result["matched_stages"])
    for side in ("actual", "expected"):
        assert [(item["stage"], item["occurrence"]) for item in result["branch_records_not_compared"][side]] == [
            ("input/refinement/input", 0),
            ("input/refinement/output", 0),
            ("input/refinement/input", 1),
            ("input/refinement/output", 1),
        ]


def test_unknown_stage_and_changed_scalar_or_tensor_structure_are_not_silently_accepted():
    actual, expected = stable_records(), stable_records()
    actual.append(record("future/unbound_stage", [13.0]))
    actual[0]["values"]["tensor"] = torch.ones(3)
    actual[5]["values"]["counter"] = 2
    expected[5]["values"]["counter"] = 1
    result = analyzer.compare_stages(actual, expected)
    assert result["unmatched_records"]["actual"][0]["stage"] == "future/unbound_stage"
    pairs = {pair["stage"]: pair for pair in result["matched_stages"]}
    assert pairs["input/statistics"]["scalar_or_structure_differences"][0]["kind"] == "tensor_structure"
    assert pairs["input/finished"]["scalar_or_structure_differences"][0]["actual"] == 2
    assert not result["stable_stages_within_tolerance"]


@pytest.mark.parametrize("value", (float("inf"), float("-inf")))
def test_matching_nonfinite_nested_tensors_are_reported_even_when_exactly_equal(value):
    expected = stable_records()
    expected[0]["values"] = {"nested": [{"tensor": torch.tensor([value, 1.0, value])}, (torch.tensor([value]),)]}
    result = analyzer.compare_stages(copy.deepcopy(expected), expected)
    pair = result["matched_stages"][0]
    assert pair["changed_tensors"] == []
    assert pair["tensor_comparisons"] == 2
    assert pair["nonfinite_element_count"] == result["nonfinite_element_count"] == 6
    for side in ("actual", "expected"):
        assert pair["nonfinite_tensors"][side] == [
            {
                "name": "trace.input/statistics[0].nested.0.tensor",
                "shape": [3],
                "dtype": "torch.float32",
                "nonfinite_elements": 2,
            },
            {
                "name": "trace.input/statistics[0].nested.1.0",
                "shape": [1],
                "dtype": "torch.float32",
                "nonfinite_elements": 1,
            },
        ]
    assert result["sufficient_tensor_coverage"]
    assert not pair["within_tolerance"]
    assert not result["stable_stages_within_tolerance"]
    assert json.loads(json.dumps(result, allow_nan=False))["nonfinite_element_count"] == 6


@pytest.mark.parametrize("empty_all", (False, True))
def test_matching_empty_payloads_report_insufficient_per_stage_tensor_coverage(empty_all):
    expected = stable_records()
    empty_records = expected if empty_all else expected[:1]
    for item in empty_records:
        item["values"] = {}
    result = analyzer.compare_stages(copy.deepcopy(expected), expected)
    assert result["complete_stable_coverage"]
    assert result["tensor_comparisons"] == (0 if empty_all else len(expected) - 1)
    assert not result["sufficient_tensor_coverage"]
    assert [item["stage"] for item in result["insufficient_tensor_coverage"]] == [
        item["stage"] for item in empty_records
    ]
    assert [
        item["occurrence"] for item in result["insufficient_tensor_coverage"] if item["stage"] == "matrix/before_update"
    ] == ([0, 1] if empty_all else [])
    assert not result["matched_stages"][0]["within_tolerance"]
    assert not result["stable_stages_within_tolerance"]


def test_zero_owned_root_tensors_count_as_coverage_despite_empty_shapes():
    expected = stable_records()
    for index in (1, 2):
        expected[index]["values"] = {"roots": [torch.empty((0, 3, 3)), torch.empty((0,))]}
    result = analyzer.compare_stages(copy.deepcopy(expected), expected)
    assert result["tensor_comparisons"] == len(expected) + 2
    assert result["nonfinite_element_count"] == 0
    assert result["insufficient_tensor_coverage"] == []
    assert result["sufficient_tensor_coverage"]
    assert result["stable_stages_within_tolerance"]
    assert all(pair["tensor_coverage_sufficient"] for pair in result["matched_stages"])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture
def saved_diagnostic(tmp_path, request):
    selected_update = getattr(request, "param", 23)
    root = tmp_path / "synthetic-output-root"
    root.mkdir()
    inventories = {}
    selected = []
    for mode in analyzer.MODES:
        inventory = {}
        for rank in range(8):
            for update in (22, 23, 24):
                path = (
                    root / analyzer.WORK_RELATIVE / f"trace_{mode}/continuation-trace/rank_{rank}/update_{update:06d}.pt"
                )
                inventory[str(path)] = inventory[str(path.with_suffix(".json"))] = "0" * 64
        path = root / analyzer.WORK_RELATIVE / f"trace_{mode}/continuation-trace/rank_3/update_{selected_update:06d}.pt"
        path.parent.mkdir(parents=True)
        records = stable_records()
        if mode == "resumed":
            records[5:5] = [record("input/refinement/input", [100.0]), record("input/refinement/output", [101.0])]
            records[-1]["values"]["tensor"] += 0.5
        provenance = {
            "identity": {"source": "synthetic-only", "config": "same", "data": "fixed"},
            "start_step": 22 if mode == "resumed" and selected_update > 22 else 0,
        }
        torch.save(
            {"schema": 1, "rank": 3, "update": selected_update, "provenance": provenance, "records": records}, path
        )
        sidecar = path.with_suffix(".json")
        write_json(
            sidecar,
            {
                "status": "saved_continuation_intermediates_not_numerical_acceptance",
                "rank": 3,
                "update": selected_update,
                "provenance": provenance,
                "records": len(records),
                "stages": [item["stage"] for item in records],
                "trace_sha256": sha256(path),
            },
        )
        inventory[str(path)], inventory[str(sidecar)] = sha256(path), sha256(sidecar)
        inventories[mode] = {"files_sha256": inventory}
        selected.extend((path, sidecar))
    executions = [
        {
            "protocol": protocol,
            "mode": mode,
            "exit_code": 0,
            "command": ["synthetic-command", protocol, mode],
            "trajectory": {},
        }
        for protocol in ("control", "trace")
        for mode in ("continuous", "interrupted", "resumed")
    ]
    receipt = root / analyzer.RECEIPT_RELATIVE
    write_json(
        receipt,
        {
            "status": analyzer.COMPLETE_STATUS,
            "root": str(root),
            "world": 8,
            "boundary": 22,
            "first_resumed": 23,
            "endpoint": 24,
            "tolerance": {"rtol": 1e-5, "atol": 1e-6},
            "commands": [item["command"] for item in executions],
            "gpu_commands": executions,
            "analysis": {"trace_transparent": True, "extra_checkpoint_transparent": True, "traces": inventories},
        },
    )
    return receipt, selected


def test_gated_offline_pair_authenticates_selected_bytes_and_preserves_all_inputs(saved_diagnostic):
    receipt, traces = saved_diagnostic
    before = {path: path.read_bytes() for path in [receipt, *traces]}
    result = analyzer.analyze_pair(receipt, sha256(receipt), rank=3, update=23)
    assert result["comparison"]["tensor_comparisons"] == len(stable_records())
    assert result["comparison"]["complete_stable_coverage"]
    assert not result["comparison"]["stable_stages_within_tolerance"]
    assert result["saved_gate_dependency"]["receipt_sha256"] == sha256(receipt)
    assert result["saved_gate_dependency"]["independently_reconstructed"] is False
    for mode, value in result["trace_pair"].items():
        assert value["sha256"] == sha256(Path(value["path"]))
        assert value["sidecar_sha256"] == sha256(Path(value["sidecar"]))
        assert mode in ("continuous", "resumed")
    assert {path: path.read_bytes() for path in before} == before
    assert json.loads(json.dumps(result, allow_nan=False))["update"] == 23


@pytest.mark.parametrize("saved_diagnostic", (22, 24), indirect=True)
def test_boundary_and_endpoint_pairs_authenticate_expected_start_steps_without_mutation(saved_diagnostic):
    receipt, traces = saved_diagnostic
    update = json.loads(traces[1].read_text())["update"]
    before = {path: path.read_bytes() for path in [receipt, *traces]}
    result = analyzer.analyze_pair(receipt, sha256(receipt), rank=3, update=update)
    assert result["update"] == update
    assert result["comparison"]["tensor_comparisons"] == len(stable_records())
    assert result["comparison"]["sufficient_tensor_coverage"]
    assert not result["comparison"]["stable_stages_within_tolerance"]
    assert [json.loads(traces[index].read_text())["provenance"]["start_step"] for index in (1, 3)] == (
        [0, 0] if update == 22 else [0, 22]
    )
    assert all(
        result["trace_pair"][mode]["sha256"] == sha256(traces[index])
        for mode, index in (("continuous", 0), ("resumed", 2))
    )
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("saved_diagnostic", (22, 24), indirect=True)
@pytest.mark.parametrize("mode", ("continuous", "resumed"))
def test_boundary_and_endpoint_reject_authenticated_wrong_start_step(saved_diagnostic, mode):
    receipt, traces = saved_diagnostic
    offset = 0 if mode == "continuous" else 2
    path, sidecar_path = traces[offset : offset + 2]
    trace = torch.load(path, weights_only=True)
    update = trace["update"]
    trace["provenance"]["start_step"] = 0 if trace["provenance"]["start_step"] == 22 else 22
    torch.save(trace, path)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["provenance"] = trace["provenance"]
    sidecar["trace_sha256"] = sha256(path)
    write_json(sidecar_path, sidecar)
    value = json.loads(receipt.read_text())
    value["analysis"]["traces"][mode]["files_sha256"].update(
        {str(path): sha256(path), str(sidecar_path): sha256(sidecar_path)}
    )
    write_json(receipt, value)
    with pytest.raises(ValueError):
        analyzer.analyze_pair(receipt, sha256(receipt), rank=3, update=update)


@pytest.mark.parametrize(
    "fault",
    ("trace_false", "checkpoint_false", "missing_gate", "incomplete", "incomplete_commands", "changed_tolerance"),
)
def test_false_or_incomplete_saved_gates_refuse_before_opening_traces(saved_diagnostic, monkeypatch, fault):
    receipt, traces = saved_diagnostic
    value = json.loads(receipt.read_text())
    if fault == "trace_false":
        value["analysis"]["trace_transparent"] = False
    elif fault == "checkpoint_false":
        value["analysis"]["extra_checkpoint_transparent"] = False
    elif fault == "missing_gate":
        value["analysis"].pop("trace_transparent")
    elif fault == "incomplete":
        value["status"] = "diagnosing"
    elif fault == "changed_tolerance":
        value["tolerance"]["atol"] = 1e-3
    else:
        value["gpu_commands"].pop()
    write_json(receipt, value)
    open_file = os.open

    def refuse_trace_open(path, *args, **kwargs):
        if Path(path) in traces:
            raise AssertionError("Ungated intermediate file was opened")
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", refuse_trace_open)
    with pytest.raises(ValueError):
        analyzer.analyze_pair(receipt, sha256(receipt), rank=3, update=23)
    assert all(path.exists() for path in traces)


@pytest.mark.parametrize("fault", ("trace_digest", "sidecar_digest", "unbound_path", "provenance", "receipt_digest"))
def test_authenticated_pair_refuses_replaced_bytes_paths_or_provenance(saved_diagnostic, fault):
    receipt, traces = saved_diagnostic
    value = json.loads(receipt.read_text())
    pinned = sha256(receipt)
    if fault == "trace_digest":
        torch.save({"wrong": torch.ones(2)}, traces[0])
    elif fault == "sidecar_digest":
        write_json(traces[1], {"wrong": "sidecar"})
    elif fault == "unbound_path":
        inventory = value["analysis"]["traces"]["continuous"]["files_sha256"]
        inventory[str(traces[0].parent / "other.pt")] = inventory.pop(str(traces[0]))
        write_json(receipt, value)
        pinned = sha256(receipt)
    elif fault == "provenance":
        path = traces[2]
        trace = torch.load(path, weights_only=True)
        trace["provenance"]["identity"]["data"] = "different"
        torch.save(trace, path)
        sidecar = json.loads(traces[3].read_text())
        sidecar["provenance"] = trace["provenance"]
        sidecar["trace_sha256"] = sha256(path)
        write_json(traces[3], sidecar)
        value["analysis"]["traces"]["resumed"]["files_sha256"].update(
            {str(path): sha256(path), str(traces[3]): sha256(traces[3])}
        )
        write_json(receipt, value)
        pinned = sha256(receipt)
    else:
        write_json(receipt, value | {"changed": True})
    with pytest.raises(ValueError):
        analyzer.analyze_pair(receipt, pinned, rank=3, update=23)


@pytest.mark.parametrize("target", ("receipt", "trace"))
def test_input_change_during_loading_refuses_report_instead_of_accepting_stale_binding(
    saved_diagnostic, monkeypatch, target
):
    receipt, traces = saved_diagnostic
    load = torch.load
    changed = False

    def load_then_mutate(*args, **kwargs):
        nonlocal changed
        result = load(*args, **kwargs)
        if not changed:
            if target == "receipt":
                value = json.loads(receipt.read_text())
                write_json(receipt, value | {"concurrent_change": True})
            else:
                replacement = traces[0].with_suffix(".replacement")
                replacement.write_bytes(traces[0].read_bytes())
                os.replace(replacement, traces[0])
            changed = True
        return result

    monkeypatch.setattr(torch, "load", load_then_mutate)
    with pytest.raises(ValueError):
        analyzer.analyze_pair(receipt, sha256(receipt), rank=3, update=23)
    assert changed


def test_cli_emits_saved_gate_dependent_json_without_writing_evidence(saved_diagnostic):
    receipt, traces = saved_diagnostic
    before = {path: sha256(path) for path in [receipt, *traces]}
    command = [
        sys.executable,
        str(Path(analyzer.__file__)),
        "--receipt",
        str(receipt),
        "--receipt-sha256",
        sha256(receipt),
        "--rank",
        "3",
        "--update",
        "23",
    ]
    result = subprocess.run(
        command,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    output = json.loads(result.stdout)
    assert output["status"] == "saved_gate_dependent_stage_analysis_not_causal_proof"
    assert output["saved_gate_dependency"]["independently_reconstructed"] is False
    assert output["comparison"]["tensor_comparisons"] == len(stable_records())
    assert {path: sha256(path) for path in before} == before
