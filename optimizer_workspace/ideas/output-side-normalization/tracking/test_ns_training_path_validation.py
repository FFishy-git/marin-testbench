# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU persisted-evidence regressions for the distinct original-training-path diagnostic."""

import copy
import json
import os
import subprocess
from pathlib import Path

import ns_training_path_contract as contract
import ns_training_path_test_support as support
import ns_training_path_validation as validation
import owned_output_handoff as handoff
import pytest
import torch
from fused_attribution_validation import INTERRUPTED_MANIFEST
from selected_lr_release_owner import candidate_files, read, sha256


@pytest.fixture
def candidate(tmp_path):
    return support.candidate(tmp_path)


def test_complete_plan_pins_original_schedule_and_refuses_old_owner_before_outputs(candidate):
    root, path, frozen = candidate
    planned = validation.plan(root, path)
    assert validation.source(root, path) == frozen
    assert len(planned["trials"]) == 18 and len(planned["commands"]) == 54
    assert planned["planned_tokens"] == 452984832
    for item in planned["trials"]:
        config = contract.require_config(path, {key: item[key] for key in contract.specs()[0]})
        assert config["training"]["steps"] == 5120
        assert config["optimizer"]["lr"] == item["matrix_lr"]
        assert config["root"]["graphs"] is True
        assert [command[command.index("--max-steps") + 1] for command in item["commands"]] == ["24", "22", "24"]
        for command in item["commands"]:
            assert command[command.index("--launch-release") + 1] == str(contract.locations(root)[0])
            assert command[command.index("--execution-boundary") + 1] == "22"
            assert "--nproc_per_node=8" in command
            assert ("--verify-diagonal-capture" in command) == (item["family"] != "muont")
            assert ("--verify-outer-diagonal-updates" in command) == (item["family"] != "muont")
        assert item["commands"][-1][item["commands"][-1].index("--resume") + 1] == str(
            Path(item["resumed"]) / "checkpoints/step_000022"
        )
    denied = subprocess.run(
        [
            contract.PYTHON,
            "-P",
            str(Path(validation.__file__)),
            "--root",
            str(root),
            "--candidate",
            str(path),
            "--phase",
            "execute",
        ],
        cwd=path,
        env=os.environ
        | {
            "PYTHONPATH": f"{Path(validation.__file__).parent}:{path}",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert denied.returncode != 0 and "require_ns_training_path_gpu_owner" in denied.stderr
    assert not contract.locations(root)[3].exists()
    assert not contract.locations(root)[4].exists()
    assert not contract.locations(root)[2].exists()


@pytest.mark.parametrize("name", tuple(contract.TRAINING_FILES))
def test_rehashed_training_source_changes_cannot_redefine_original_numerical_boundary(candidate, name):
    root, path, frozen = candidate
    changed = path / name
    changed.write_bytes(changed.read_bytes() + b"\n# altered original numerical boundary\n")
    frozen["files_sha256"] = candidate_files(path)
    support.record(contract.locations(root)[0], frozen)
    with pytest.raises(ValueError):
        validation.source(root, path)
    assert not contract.locations(root)[3].exists()
    assert not contract.locations(root)[4].exists()


@pytest.mark.parametrize("fault", ("data", "foreign_binding"))
def test_source_refuses_rehashed_data_or_prior_experiment_binding(candidate, fault):
    root, _, frozen = candidate
    if fault == "data":
        support.record(root / "data/experiment_receipt.json", {"data": "changed"})
    else:
        frozen["status"] = "frozen_ns_full_model_validation_not_production"
        support.record(contract.locations(root)[0], frozen)
    with pytest.raises(ValueError):
        validation.source(root, Path(frozen["candidate"]))


@pytest.mark.parametrize("family", ("muont", "outer_leftdiag_fused", "null_leftdiag_fused"))
def test_unchanged_readers_reconstruct_full_rank_scalar_capture_gates_and_private_prefix(candidate, family):
    root, path, frozen = candidate
    item = next(item for item in validation.plan(root, path)["trials"] if item["family"] == family)
    support.saved_trial(path, frozen, item)
    resumed = Path(item["resumed"])
    prefix = resumed / INTERRUPTED_MANIFEST
    before = prefix.read_bytes()
    result = validation.trial(item, path, frozen)
    assert result["accepted"] is True and all(result["gates"].values())
    assert set(result["continuation"]) == {"22", "23", "24"}
    assert all(len(pair["ranks"]) == 8 for pair in result["continuation"].values())
    assert result["loss"]["comparisons"] == 39
    assert result["trajectories"]["interrupted"]["updates"] == 22
    assert result["trajectories"]["resumed"]["updates"] == 24
    assert prefix.read_bytes() == before
    if family != "muont":
        assert result["capture"]["resumed"]["updates"] == 24
        assert ("null_identity_outer_passed" in result["capture"]["resumed"]["checks"]) == (
            family == "null_leftdiag_fused"
        )


@pytest.mark.parametrize(
    "fault",
    (
        "common_missing_rng",
        "common_bad_philox",
        "common_missing_momentum",
        "common_pending_shape",
        "missing_rank",
        "missing_heldout",
        "missing_capture",
        "fixed_age",
        "rank7_difference",
        "loss_difference",
        "first_graph_rewarming",
    ),
)
def test_original_readers_preserve_common_invalid_state_and_negative_continuation_results(candidate, fault):
    root, path, frozen = candidate
    item = validation.plan(root, path)["trials"][-1]
    support.saved_trial(path, frozen, item)
    continuous, resumed = Path(item["continuous"]), Path(item["resumed"])
    if fault.startswith("common_") or fault in ("rank7_difference", "first_graph_rewarming", "missing_rank"):
        if fault == "missing_rank":
            (resumed / "checkpoints/step_000024/rank_7.pt").unlink()
        else:
            for directory in (continuous, resumed) if fault.startswith("common_") else (resumed,):
                step = 23 if fault == "first_graph_rewarming" else 24
                checkpoint_path = directory / f"checkpoints/step_{step:06d}/rank_7.pt"
                value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if fault == "common_missing_rng":
                    value["cuda_rng"] = None
                elif fault == "common_bad_philox":
                    value["cuda_rng"][8] = 1
                elif fault == "common_missing_momentum":
                    value["fast"]["momentum"].pop((4, 4))
                elif fault == "common_pending_shape":
                    value["fast"]["input"]["groups"]["d"]["pending"][0]["basis"] = torch.zeros(3, 3)
                elif fault == "first_graph_rewarming":
                    value["fast"]["input"]["groups"]["d"].update(warm=False, cached_root_graph_active=False)
                else:
                    value["fast"]["momentum"][(4, 4)][0, 0, 0] += 0.1
                torch.save(value, checkpoint_path)
    else:
        segment = resumed / "metrics_from_000022.jsonl"
        rows = [json.loads(line) for line in segment.read_text().splitlines()]
        if fault == "missing_capture":
            rows[-1].pop("null_identity_outer_passed")
        elif fault == "missing_heldout":
            rows[-1]["heldout_losses"].pop(next(iter(rows[-1]["heldout_losses"])))
        elif fault == "fixed_age":
            rows[-1]["input_refresh_installations"][0]["source_step"] -= 1
        else:
            rows[-1]["selection_loss"] += 0.01
        segment.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if fault in ("rank7_difference", "loss_difference", "first_graph_rewarming"):
        result = validation.trial(item, path, frozen)
        if fault == "first_graph_rewarming":
            assert result["accepted"] is True
            assert result["continuation"]["23"]["within_tolerance"] is False
            assert result["first_resumed_diagnostic_only"] is True
        else:
            assert result["accepted"] is False
            assert not result["gates"][
                "post_cached_root_all_rank_continuation" if fault == "rank7_difference" else "scalar_loss_continuation"
            ]
    else:
        with pytest.raises((ValueError, FileNotFoundError, KeyError)):
            validation.trial(item, path, frozen)


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "timeout", "spawn_failure"))
def test_cpu_command_receipts_preserve_cleanup_failure_and_new_namespace(tmp_path, outcome):
    _, _, state, work, _ = contract.locations(tmp_path)
    support.record(state, {"controller_pid": os.getpid()})
    work.mkdir(parents=True)
    helper = tmp_path / "cpu_child.py"
    # This subprocess never imports an accelerator library or invokes a training command.
    helper.write_text(
        "import signal,sys\n" "if sys.argv[1]=='timeout':signal.pause()\n" "sys.exit(0 if sys.argv[1]=='zero' else 7)\n"
    )
    command = [str(tmp_path / "missing")] if outcome == "spawn_failure" else [contract.PYTHON, str(helper), outcome]
    log = work / "cpu-command.log"
    if outcome in ("timeout", "spawn_failure"):
        with pytest.raises(subprocess.TimeoutExpired if outcome == "timeout" else FileNotFoundError):
            validation.run_command(tmp_path, tmp_path, command, log, timeout=0.25 if outcome == "timeout" else 10)
    else:
        entry = validation.run_command(tmp_path, tmp_path, command, log)
        if outcome == "zero":
            validation.require_entry(entry, command, log)
        else:
            with pytest.raises(ValueError):
                validation.require_entry(entry, command, log)
    saved = read(log.with_suffix(".execution.json"))
    assert saved["log_sha256"] == sha256(log)
    assert read(state)["active_command"] == command
    if outcome != "spawn_failure":
        assert saved["marked_descendants_absent"] is True
        assert not handoff.marker_processes(contract.COMMAND_MARKER, saved["marker"])
        assert saved["generation"]["parent"] == os.getpid()
    if outcome in ("timeout", "spawn_failure"):
        assert "error" in saved
    assert not contract.locations(tmp_path)[4].exists()


@pytest.fixture(scope="module")
def complete_receipt(tmp_path_factory):
    return support.complete_receipt(tmp_path_factory.mktemp("ns-training-path-complete"))


def test_all_54_saved_commands_reconstruct_rejected_case_without_changing_evidence(complete_receipt):
    root, path, result = complete_receipt
    receipt = contract.locations(root)[4]
    before = receipt.read_bytes()
    reconstructed = validation.reconstruct(root, path)
    assert reconstructed == result
    assert len(reconstructed["gpu_commands"]) == 54
    assert len(reconstructed["results"]) == 18
    assert reconstructed["rejected_cases"] == [contract.specs()[0]["name"]]
    assert len(reconstructed["accepted_cases"]) == 17
    assert reconstructed["all_cases_accepted"] is False
    assert reconstructed["results"][0]["continuation"]["24"]["within_tolerance"] is False
    assert receipt.read_bytes() == before


@pytest.mark.parametrize(
    "fault", ("missing_command", "missing_case", "bool_exit", "ancestry", "owner", "hide_rejection")
)
def test_retained_complete_receipt_refuses_missing_rebound_or_erased_negative_evidence(complete_receipt, fault):
    root, path, result = complete_receipt
    receipt = contract.locations(root)[4]
    before = receipt.read_bytes()
    changed = copy.deepcopy(result)
    if fault == "missing_command":
        changed["gpu_commands"].pop()
    elif fault == "missing_case":
        changed["results"].pop()
    elif fault == "bool_exit":
        changed["gpu_commands"][0]["exit_code"] = False
    elif fault == "ancestry":
        changed["gpu_commands"][0]["generation"]["parent"] = 999
    elif fault == "owner":
        changed["owner"]["validator_pid"] = 999
    else:
        changed["results"][0]["accepted"] = True
        changed["accepted_cases"] = [item["name"] for item in changed["results"]]
        changed["rejected_cases"] = []
        changed["all_cases_accepted"] = True
    try:
        support.record(receipt, changed)
        with pytest.raises(ValueError):
            validation.reconstruct(root, path)
    finally:
        receipt.write_bytes(before)
