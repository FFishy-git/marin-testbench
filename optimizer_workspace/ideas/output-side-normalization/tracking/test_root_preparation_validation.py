# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise retained root outputs and all-rank intervention evidence without CUDA."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import root_preparation_probe as probe
import root_preparation_validation as validation
import torch
from fused_continuation_test_support import ISSUED_CANDIDATE
from selected_lr_release_owner import sha256


@pytest.mark.parametrize("fault", ("none", "tensor", "nonfinite", "structure"))
def test_root_output_comparison_reports_numerical_failures_without_mutating_inputs(fault):
    expected = ({"root": torch.ones(2, 3)}, torch.tensor([False, True]), [torch.arange(3.0)])
    actual = probe.cpu_tree(expected)
    if fault == "tensor":
        actual[0]["root"][0, 1] += 0.02
    elif fault == "nonfinite":
        actual[0]["root"][0, 1] = float("nan")
    elif fault == "structure":
        actual[0]["root"] = torch.ones(2, 4)
    result = probe.comparison(actual, expected, "root")
    assert result["within_tolerance"] == (fault == "none")
    assert result["exact"] == (fault == "none")
    assert result["finite"] == (fault != "nonfinite")
    assert torch.equal(expected[0]["root"], torch.ones(2, 3))


@pytest.mark.parametrize(
    "regime,mode,step,active",
    (
        ("unprepared", "resumed", 23, False),
        ("prepared", "resumed", 23, True),
        ("prepared", "resumed", 24, True),
        ("unprepared", "continuous", 23, True),
    ),
)
def test_checkpoint_lifecycle_distinguishes_preparation_from_eager_resume(regime, mode, step, active):
    saved = {
        "fast": {
            "input": {"groups": {name: {"warm": True, "cached_root_graph_active": active} for name in probe.GROUPS}}
        }
    }
    result = validation.graph_lifecycle(saved, mode, regime, step)
    assert all(item == {"warm": True, "graph_active": active} for item in result.values())
    saved["fast"]["input"]["groups"]["f"]["cached_root_graph_active"] = not active
    with pytest.raises(ValueError):
        validation.graph_lifecycle(saved, mode, regime, step)


def test_source_bound_saved_root_inputs_cover_both_full_shapes_and_original_evidence():
    root = ISSUED_CANDIDATE.parents[1]
    saved, hashes = probe.saved_inputs(root, {"parent_candidate": str(ISSUED_CANDIDATE)})
    assert set(saved) == {"continuous", "resumed"}
    assert len(hashes) == 5 and all(sha256(root / name) == value for name, value in hashes.items())
    for mode in saved.values():
        assert mode["d"]["inputs"]["C"].shape == (12, 512, 512)
        assert mode["f"]["inputs"]["C"].shape == (4, 2048, 2048)
        assert all(probe.finite_tree(item) for item in mode.values())


@pytest.mark.parametrize("fault", ("none", "missing_rank", "rng", "wrong_rank", "leaked"))
def test_preparation_receipts_require_all_owned_ranks_and_exact_state_rng(tmp_path, monkeypatch, fault):
    candidate, out = tmp_path / "candidate", tmp_path / "out"
    config = candidate / validation.CONFIG
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    out.mkdir()
    identity = {"source": "fixture"}
    records = SimpleNamespace(checkpoint_identity=lambda config, data: identity)
    monkeypatch.setattr(validation, "numerical_modules", lambda candidate: (None, records, None))
    for rank in range(8):
        result = {
            "status": "restored_root_scratch_preparation_state_rng_exact_not_continuation_acceptance",
            "rank": rank,
            "checkpoint_step": 22,
            "groups": ["d", "f"],
            "numerical_state_exact": True,
            "torch_rng_exact": True,
            "cuda_rng_exact": True,
            "python_rng_exact": True,
            "fresh_graphs": True,
            "scratch_outputs_consumed": False,
            "identity": identity,
        }
        if rank == 3 and fault == "rng":
            result["cuda_rng_exact"] = False
        if rank == 3 and fault == "wrong_rank":
            result["rank"] = 4
        if rank == 3 and fault == "missing_rank":
            continue
        (out / f"root_preparation_rank_{rank}.json").write_text(json.dumps(result))
    before = {path: sha256(path) for path in out.iterdir()}
    regime = "unprepared" if fault == "leaked" else "prepared"
    if fault == "none":
        result = validation.preparation_receipts(out, candidate, {"root": str(tmp_path)}, regime, "resumed")
        assert (
            len(result) == 8 and result["root_preparation_rank_3.json"] == before[out / "root_preparation_rank_3.json"]
        )
    else:
        with pytest.raises(ValueError):
            validation.preparation_receipts(out, candidate, {"root": str(tmp_path)}, regime, "resumed")
    assert {path: sha256(path) for path in out.iterdir()} == before


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "tensor",
        "historical",
        "order",
        "both_order",
        "missing",
        "report",
        "bool_exit",
        "checks",
        "lifecycle",
        "owner",
        "identity",
        "state",
        "rng",
        "readiness",
        "log",
    ),
)
def test_cold_probe_reconstructs_retained_tensors_and_denies_report_only_pass(tmp_path, monkeypatch, fault):
    root, candidate = tmp_path / "root", tmp_path / "candidate"
    work = validation.locations(root)[3] / "cold-root"
    work.mkdir(parents=True)
    binding = validation.locations(root)[0]
    binding.parent.mkdir(parents=True)
    binding.write_text("{}")
    owner = {"controller_pid": 400, "controller_start_ticks": "123"}
    validation.locations(root)[2].write_text(json.dumps(owner))
    config = candidate / validation.CONFIG
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    identity = {"fixture": "candidate"}
    monkeypatch.setattr(
        validation,
        "numerical_modules",
        lambda candidate: (None, SimpleNamespace(checkpoint_identity=lambda config, data: identity), None),
    )
    saved = {
        mode: {
            name: {
                "saved_output": (torch.ones(2), torch.tensor([False])),
                "inputs": {key: torch.ones(2) for key in ("C", "tol", "SB", "S2")},
            }
            for name in probe.GROUPS
        }
        for mode in ("continuous", "resumed")
    }
    evidence = {"fixture": "saved-source"}
    monkeypatch.setattr(validation, "saved_inputs", lambda root, prepared: (saved, evidence))
    prepared = {"allocation_predecessor": {"restored_keeper": {"devices": [["0", "uuid", "A100"]]}}}
    executions = []
    for index in range(2):
        values = [
            item | {"output": probe.cpu_tree(saved[item["mode"]][item["group"]]["saved_output"])}
            for item in probe.retained_order()
        ]
        if fault == "historical":
            for value in values:
                value["output"][0][0] += 0.02
        if index == 1 and fault == "tensor":
            values[-1]["output"][0][0] += 0.02
        historical, safety = probe.output_checks(values, saved)
        if fault == "both_order" or (index == 1 and fault == "order"):
            values.reverse()
        if index == 1 and fault == "missing":
            values.pop()
        preparation = {}
        final_inputs = {}
        for mode in saved:
            inputs = {name: saved[mode][name]["inputs"] for name in probe.GROUPS}
            rng = {
                "torch_rng": torch.zeros(4, dtype=torch.uint8),
                "cuda_rng": torch.zeros(16, dtype=torch.uint8),
                "python_rng": (3, (1, 2), None),
            }
            preparation[mode] = {
                "inputs_before": probe.cpu_tree(inputs),
                "inputs_after": probe.cpu_tree(inputs),
                "rng_before": probe.cpu_tree(rng),
                "rng_after": probe.cpu_tree(rng),
                "prepared_groups": list(probe.GROUPS),
                "readiness": {name: {"warm": True, "graph_active": False} for name in probe.GROUPS},
            }
            final_inputs[mode] = probe.cpu_tree(inputs)
        if fault == "state" and index == 1:
            final_inputs["resumed"]["f"]["S2"][0] += 0.02
        if fault == "rng" and index == 1:
            preparation["resumed"]["rng_after"]["cuda_rng"][0] = 1
        if fault == "readiness" and index == 1:
            preparation["resumed"]["readiness"]["f"]["graph_active"] = True
        tensors = work / f"process-{index}.pt"
        torch.save({"retained": values, "preparation": preparation, "final_inputs": final_inputs}, tensors)
        report = {
            "status": probe.STATUS,
            "index": index,
            "owner": owner,
            "identity": identity,
            "binding_sha256": sha256(binding),
            "saved_inputs_sha256": evidence,
            "order": list(probe.ORDER),
            "groups": list(probe.GROUPS),
            "lifecycle": probe.expected_lifecycle(),
            "checks": historical + safety,
            "state_rng_exact": True,
            "scratch_outputs_consumed": False,
            "outputs": str(tensors),
            "outputs_sha256": sha256(tensors),
            "physical_uuid": "uuid",
            "all_within_tolerance": all(item["within_tolerance"] for item in historical + safety),
            "safety_within_tolerance": all(item["within_tolerance"] for item in safety),
            "historical_saved_within_tolerance": all(item["within_tolerance"] for item in historical),
        }
        if index == 1 and fault == "report":
            report["outputs_sha256"] = "0" * 64
        if index == 1 and fault == "checks":
            report["checks"][-1]["within_tolerance"] = True
            report["checks"][-1]["tensor_comparisons"] = 0
        if index == 1 and fault == "lifecycle":
            report["lifecycle"][0]["fresh"] = False
        if index == 1 and fault == "owner":
            report["owner"] = owner | {"controller_start_ticks": "different"}
        if index == 1 and fault == "identity":
            report["identity"] = {}
        (work / f"process-{index}.json").write_text(json.dumps(report))
        log = validation.locations(root)[3] / f"cold-root-{index}.log"
        log.write_text("complete\n")
        executions.append(
            {
                "index": index,
                "command": probe.command(root, candidate, index),
                "exit_code": False if fault == "bool_exit" and index == 1 else 0,
                "log": str(log if fault != "log" or index != 1 else tensors),
                "log_sha256": sha256(log),
            }
        )
    if fault not in ("none", "tensor", "historical"):
        with pytest.raises(ValueError):
            validation.cold_probe_evidence(root, candidate, prepared, executions)
    else:
        result = validation.cold_probe_evidence(root, candidate, prepared, executions)
        assert len(result["cross_process"]) == 12
        assert result["within_tolerance"] == (fault != "tensor")
        assert result["historical_saved_within_tolerance"] == (fault == "none")
        assert len(result["reconstructed_checks"]) == 2
        if fault == "tensor":
            assert any(not item["within_tolerance"] for item in result["cross_process"])


@pytest.mark.parametrize(
    "fault",
    (
        "none",
        "tensor",
        "loss",
        "missing_step",
        "missing_rank",
        "duplicate_rank",
        "empty",
        "counter",
        "rng",
        "bool",
        "empty_loss",
        "wrong_step",
        "rehashed_pass",
    ),
)
def test_pair_acceptance_requires_every_checkpoint_rank_counter_rng_and_loss(fault):
    result = {
        "checkpoints": {
            str(step): {
                "step": step,
                "within_tolerance": True,
                "ranks": [
                    {
                        "rank": rank,
                        "tensor_comparisons": 12,
                        "failed_tensor_count": 0,
                        "scalar_or_structure_differences": [],
                        "changed_tensors": [],
                        "checkpoint_sha256": {"actual": "a" * 64, "expected": "b" * 64},
                    }
                    for rank in range(8)
                ],
            }
            for step in (22, 23, 24)
        },
        "loss": {"within_tolerance": True, "comparisons": 39, "failures": []},
    }
    rank = result["checkpoints"]["23"]["ranks"][3]
    if fault == "tensor":
        result["checkpoints"]["23"]["within_tolerance"] = False
    elif fault == "loss":
        result["loss"]["within_tolerance"] = False
    elif fault == "missing_step":
        result["checkpoints"].pop("23")
    elif fault == "missing_rank":
        result["checkpoints"]["23"]["ranks"].pop()
    elif fault == "duplicate_rank":
        rank["rank"] = 4
    elif fault == "empty":
        rank["tensor_comparisons"] = 0
    elif fault == "counter":
        rank["scalar_or_structure_differences"].append({"name": "rank_3.fast.steps"})
    elif fault == "rng":
        rank["changed_tensors"].append({"name": "rank_3.cuda_rng", "within_tolerance": True})
    elif fault == "bool":
        rank["tensor_comparisons"] = True
    elif fault == "empty_loss":
        result["loss"]["comparisons"] = 0
    elif fault == "wrong_step":
        result["checkpoints"]["23"]["step"] = 24
    elif fault == "rehashed_pass":
        rank["changed_tensors"].append({"name": "rank_3.fast.C", "within_tolerance": False})
    assert validation.pair_set_pass(result) == (fault == "none")


def test_input_snapshot_retains_only_numerical_fields_without_opaque_graphs():
    engine = SimpleNamespace(
        groups={
            name: SimpleNamespace(
                C=torch.ones(2),
                tol=torch.ones(2),
                graph=object(),
                basis=SimpleNamespace(SB=torch.ones(2), S2=torch.ones(2)),
            )
            for name in probe.GROUPS
        }
    )
    saved = probe.input_state(engine)
    assert set(saved) == set(probe.GROUPS)
    assert set(saved["d"]) == {"C", "tol", "SB", "S2"}
    engine.groups["d"].C[0] = 2
    assert saved["d"]["C"][0] == 1


def test_unowned_real_probe_denies_before_cuda_or_output_creation(tmp_path):
    root, candidate = tmp_path / "unused", ISSUED_CANDIDATE
    root.mkdir()
    result = subprocess.run(
        [
            probe.PYTHON,
            "-P",
            str(Path(probe.__file__)),
            "--root",
            str(root),
            "--candidate",
            str(candidate),
            "--index",
            "0",
        ],
        env=os.environ
        | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(Path(probe.__file__).parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "binding" in result.stderr or "candidate" in result.stderr
    assert list(root.iterdir()) == []
