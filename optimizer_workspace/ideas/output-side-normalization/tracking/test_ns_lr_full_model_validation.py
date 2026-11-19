# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate actual-LR checkpoint decisions and bounded subprocess evidence with hidden CUDA."""

import copy
import json
import os
import subprocess
from pathlib import Path

import ns_lr_full_model_contract as contract
import ns_lr_full_model_test_support as support
import ns_lr_full_model_validation as validation
import owned_output_handoff as handoff
import pytest
import torch
from selected_lr_release_owner import read


@pytest.fixture
def candidate(tmp_path):
    return support.candidate(tmp_path)


def test_actual_lr_complete_protocol_keeps_equal_grids_and_refuses_lower_lr_execution_adapter(candidate):
    root, path, frozen = candidate
    planned = validation.plan(root, path)
    assert validation.source(root, path) == frozen
    assert len(planned["trials"]) == 54 and len(planned["commands"]) == 162
    assert len({tuple(command) for command in planned["commands"]}) == 162
    assert planned["planned_tokens"] == 1358954496
    for item in planned["trials"]:
        assert [command[command.index("--max-steps") + 1] for command in item["commands"]] == ["24", "22", "24"]
        assert all(
            command[command.index("--config") + 1] == str(path / item["config_name"]) for command in item["commands"]
        )
        assert item["matrix_lr"] == read(path / item["config_name"])["optimizer"]["lr"]
    before = (root / "run-preparation/ns-actual-lr-binding-20261119.json").read_bytes()
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
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{Path(validation.__file__).parent}:{path}",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert denied.returncode != 0 and "require_ns_actual_lr_gpu_owner" in denied.stderr
    assert not contract.locations(root)[3].exists() and not contract.locations(root)[4].exists()
    assert contract.locations(root)[0].read_bytes() == before


@pytest.mark.parametrize("label", ("center", "upper"))
@pytest.mark.parametrize("family", contract.FAMILIES)
def test_actual_lr_real_all_rank_readers_accept_center_upper_exact_trajectories(candidate, family, label):
    root, path, frozen = candidate
    item = next(
        item
        for item in validation.plan(root, path)["trials"]
        if item["family"] == family and item["operand"] == "f32" and item["budget"] == 5 and item["lr_label"] == label
    )
    support.saved_trial(root, path, frozen, item)
    result = validation.trial(item, path, frozen)
    assert result["accepted"] is True and all(result["gates"].values())
    assert (
        result["matrix_lr"]
        == contract.require_config(path, {key: item[key] for key in contract.specs()[0]})["optimizer"]["lr"]
    )
    assert result["trajectories"]["interrupted"]["updates"] == 22
    assert result["trajectories"]["resumed"]["updates"] == 24
    assert (
        result["trajectories"]["interrupted"]["manifest_sha256"] != result["trajectories"]["resumed"]["manifest_sha256"]
    )
    assert len(result["continuation"]["24"]["ranks"]) == 8
    assert result["first_resumed_diagnostic_only"] is True
    assert result["loss"]["comparisons"] == 39


@pytest.mark.parametrize("fault", ("endpoint_tensor", "scalar_loss", "common_missing_rng", "capture"))
def test_actual_lr_retains_numerical_rejection_and_refuses_incomplete_common_evidence(candidate, fault):
    root, path, frozen = candidate
    item = next(
        item
        for item in validation.plan(root, path)["trials"]
        if item["family"] == "outer_leftdiag_fused" and item["lr_label"] == "upper"
    )
    support.saved_trial(root, path, frozen, item)
    resumed, continuous = Path(item["resumed"]), Path(item["continuous"])
    if fault == "endpoint_tensor":
        checkpoint = resumed / "checkpoints/step_000024/rank_7.pt"
        value = torch.load(checkpoint, map_location="cpu", weights_only=False)
        value["fast"]["momentum"][(4, 4)][0, 0, 0] += 0.1
        torch.save(value, checkpoint)
    elif fault == "common_missing_rng":
        for directory in (continuous, resumed):
            checkpoint = directory / "checkpoints/step_000024/rank_0.pt"
            value = torch.load(checkpoint, map_location="cpu", weights_only=False)
            value.pop("cuda_rng")
            torch.save(value, checkpoint)
    else:
        metrics = resumed / "metrics_from_000022.jsonl"
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        if fault == "scalar_loss":
            rows[-1]["train_loss"] += 0.1
        else:
            rows[-1]["output_work_retained_passed"] = False
        metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if fault in ("common_missing_rng", "capture"):
        with pytest.raises(ValueError):
            validation.trial(item, path, frozen)
    else:
        result = validation.trial(item, path, frozen)
        assert result["accepted"] is False
        if fault == "endpoint_tensor":
            assert result["continuation"]["24"]["ranks"][7]["failed_tensor_count"] > 0
            assert not result["gates"]["post_cached_root_all_rank_continuation"]
        else:
            assert result["loss"]["failures"] and not result["gates"]["scalar_loss_continuation"]


@pytest.mark.parametrize("fault", ("actual_lr", "binding", "config_gate"))
def test_actual_lr_refuses_config_or_rebound_protocol_even_when_source_hash_is_rewritten(candidate, fault):
    root, path, frozen = candidate
    item = validation.plan(root, path)["trials"][-1]
    if fault == "config_gate":
        changed = copy.deepcopy(frozen)
        changed["plan"]["tolerance"]["rtol"] *= 2
        support.record(contract.locations(root)[0], changed)
    else:
        config_path = path / item["config_name"]
        config = read(config_path)
        config["optimizer"]["lr"] *= 2
        support.record(config_path, config)
        if fault == "binding":
            changed = copy.deepcopy(frozen)
            changed["files_sha256"] = support.candidate_files(path)
            support.record(contract.locations(root)[0], changed)
    with pytest.raises(ValueError):
        validation.source(root, path)


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "timeout", "spawn_failure"))
def test_actual_lr_real_bounded_child_retains_failures_and_cleans_detached_descendants(tmp_path, monkeypatch, outcome):
    root, path = tmp_path / "root", tmp_path / "candidate"
    path.mkdir()
    support.record(contract.locations(root)[2], {"status": "CPU bounded fixture"})
    monkeypatch.setattr(
        contract,
        "ENVIRONMENT",
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONFAULTHANDLER": "1",
        },
    )
    helper = tmp_path / "child.py"
    helper.write_text(
        "import os,signal,subprocess,sys\nfrom pathlib import Path\n"
        "if sys.argv[1]=='timeout':signal.pause()\n"
        "if sys.argv[1]=='detached':\n"
        " r,w=os.pipe()\n"
        " child=subprocess.Popen([sys.executable,'-c',"
        "'import os,signal,sys;os.write(int(sys.argv[1]),b\"R\");signal.pause()',str(w)],"
        "pass_fds=(w,),start_new_session=True)\n"
        " os.close(w);assert os.read(r,1)==b'R';os.close(r)\n"
        " Path(sys.argv[2]).write_text(str(child.pid))\n"
        "sys.exit(0 if sys.argv[1]=='zero' else 7)\n"
    )
    command = (
        [str(tmp_path / "missing")]
        if outcome == "spawn_failure"
        else [contract.PYTHON, str(helper), outcome, str(tmp_path / "descendant.pid")]
    )
    monkeypatch.setattr(validation, "COMMAND_TIMEOUT", 0.25 if outcome == "timeout" else 10)
    log = root / "child.log"
    if outcome in ("timeout", "spawn_failure"):
        with pytest.raises(subprocess.TimeoutExpired if outcome == "timeout" else FileNotFoundError):
            validation.run_command(root, path, command, log)
    else:
        entry = validation.run_command(root, path, command, log)
        assert entry["exit_code"] == entry["cleanup_exit_code"] == (0 if outcome == "zero" else 7)
        if outcome == "zero":
            validation.require_entry(entry, command, log)
        else:
            with pytest.raises(ValueError):
                validation.require_entry(entry, command, log)
    saved = read(log.with_suffix(".execution.json"))
    assert saved["log_sha256"] == support.sha256(log)
    if outcome != "spawn_failure":
        assert saved["generation"]["parent"] == os.getpid()
        assert saved["marked_descendants_absent"] is True
        assert not handoff.marker_processes(contract.COMMAND_MARKER, saved["marker"])
    if outcome == "timeout":
        assert saved["error"].startswith("TimeoutExpired:")
        assert saved["cleanup_exit_code"] == -15
    assert not contract.locations(root)[4].exists()


@pytest.fixture(scope="module")
def complete_receipt(tmp_path_factory):
    return support.complete_receipt(tmp_path_factory.mktemp("ns-actual-lr-protocol"))


def test_actual_lr_complete_162_command_reconstruction_preserves_rejection_and_refuses_rebound_evidence(
    complete_receipt,
):
    root, path, result = complete_receipt
    receipt = contract.locations(root)[4]
    rebuilt = validation.reconstruct(root, path)
    assert rebuilt == result
    assert len(rebuilt["results"]) == 54 and len(rebuilt["gpu_commands"]) == 162
    assert len(rebuilt["accepted_cases"]) == 53 and rebuilt["rejected_cases"] == [contract.specs()[0]["name"]]
    assert rebuilt["all_cases_accepted"] is False
    for fault in ("missing_command", "missing_case", "bool_exit", "ancestry", "owner", "rejection_summary"):
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
            changed["rejected_cases"] = []
        support.record(receipt, changed)
        with pytest.raises(ValueError):
            validation.reconstruct(root, path)
    support.record(receipt, result)


def test_actual_lr_reconstruction_rederives_consistently_forged_numerical_decisions(complete_receipt):
    root, path, result = complete_receipt
    receipt = contract.locations(root)[4]
    changed = copy.deepcopy(result)
    rejected = changed["results"][0]
    rejected["accepted"] = True
    rejected["gates"]["post_cached_root_all_rank_continuation"] = True
    changed["accepted_cases"] = [item["name"] for item in changed["results"]]
    changed["rejected_cases"] = []
    changed["all_cases_accepted"] = True
    support.record(receipt, changed)
    try:
        with pytest.raises(ValueError):
            validation.reconstruct(root, path)
    finally:
        support.record(receipt, result)
