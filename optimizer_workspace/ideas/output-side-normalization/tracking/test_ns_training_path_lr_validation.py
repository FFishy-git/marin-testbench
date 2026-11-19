# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the actual-LR source boundary, retained decisions and one-shot child evidence."""

import copy
import json
import os
import subprocess
from pathlib import Path

import ns_training_path_contract as original
import ns_training_path_lr_contract as contract
import ns_training_path_lr_test_support as support
import ns_training_path_lr_validation as validation
import owned_output_handoff as handoff
import pytest
import torch
from fused_attribution_validation import INTERRUPTED_MANIFEST
from selected_lr_release_owner import candidate_files, read, sha256


def test_actual_lr_protocol_refuses_current_training_path_adapter_before_outputs(tmp_path):
    root, path, frozen = support.candidate(tmp_path, full=True)
    assert validation.source(root, path) == frozen
    planned = frozen["plan"]
    assert len(planned["trials"]) == 54 and len({tuple(command) for command in planned["commands"]}) == 162
    assert planned["planned_tokens"] == 1358954496
    for item in planned["trials"]:
        config = contract.require_config(path, {key: item[key] for key in contract.specs()[0]})
        assert config["training"]["steps"] == 5120 and config["root"]["graphs"] is True
        assert [command[command.index("--max-steps") + 1] for command in item["commands"]] == ["24", "22", "24"]
        assert all(
            command[command.index("--launch-release") + 1] == str(contract.locations(root)[0])
            and command[command.index("--config") + 1] == str(path / item["config_name"])
            and ("--verify-diagonal-capture" in command) == (item["family"] != "muont")
            and ("--verify-outer-diagonal-updates" in command) == (item["family"] != "muont")
            for command in item["commands"]
        )
    before = contract.locations(root)[0].read_bytes()
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
    assert denied.returncode != 0 and "require_ns_training_path_lr_gpu_owner" in denied.stderr
    assert contract.locations(root)[0].read_bytes() == before
    assert all(not target.exists() for target in contract.locations(root)[2:])
    assert all(not target.exists() for target in original.locations(root))


@pytest.mark.parametrize("name", tuple(contract.TRAINING_FILES))
def test_rehashed_source_cannot_replace_original_training_boundary(tmp_path, name):
    root, path, frozen = support.candidate(tmp_path)
    changed = path / name
    changed.write_bytes(changed.read_bytes() + b"\n# changed numerical boundary\n")
    frozen["files_sha256"] = candidate_files(path)
    support.record(contract.locations(root)[0], frozen)
    with pytest.raises(ValueError):
        validation.source(root, path)
    assert not contract.locations(root)[3].exists()


@pytest.mark.parametrize("family", contract.study.contract.FAMILIES)
@pytest.mark.parametrize("label", ("center", "upper"))
def test_per_lr_readers_accept_complete_all_rank_continuation_and_capture(tmp_path, family, label):
    root, path, frozen = support.candidate(tmp_path)
    item = next(
        trial
        for trial in frozen["plan"]["trials"]
        if trial["family"] == family and trial["lr_label"] == label and trial["operand"] == "f32"
    )
    support.saved_trial(root, path, frozen, item)
    before = (Path(item["resumed"]) / INTERRUPTED_MANIFEST).read_bytes()
    result = validation.trial(item, path, frozen)
    assert result["accepted"] is True
    assert len(result["continuation"]["24"]["ranks"]) == 8 and result["loss"]["comparisons"] == 39
    assert result["trajectories"]["interrupted"]["updates"] == 22
    assert result["trajectories"]["resumed"]["updates"] == 24
    assert (Path(item["resumed"]) / INTERRUPTED_MANIFEST).read_bytes() == before
    if family != "muont":
        assert result["capture"]["resumed"]["updates"] == 24
        assert ("null_identity_outer_passed" in result["capture"]["resumed"]["checks"]) == (
            family == "null_leftdiag_fused"
        )


@pytest.mark.parametrize("fault", ("rank7", "loss", "capture", "common_rng", "graph_rewarm"))
def test_actual_lr_retains_rejections_and_refuses_invalid_common_states(tmp_path, fault):
    root, path, frozen = support.candidate(tmp_path)
    item = next(
        trial
        for trial in frozen["plan"]["trials"]
        if trial["family"] == "null_leftdiag_fused" and trial["lr_label"] == "upper"
    )
    support.saved_trial(root, path, frozen, item)
    continuous, resumed = Path(item["continuous"]), Path(item["resumed"])
    if fault in ("rank7", "common_rng", "graph_rewarm"):
        for directory in (continuous, resumed) if fault == "common_rng" else (resumed,):
            step = 23 if fault == "graph_rewarm" else 24
            checkpoint = directory / f"checkpoints/step_{step:06d}/rank_7.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if fault == "common_rng":
                state.pop("cuda_rng")
            elif fault == "graph_rewarm":
                state["fast"]["input"]["groups"]["d"].update(warm=False, cached_root_graph_active=False)
            else:
                state["fast"]["momentum"][(4, 4)][0, 0, 0] += 0.1
            torch.save(state, checkpoint)
    else:
        metrics = resumed / "metrics_from_000022.jsonl"
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        if fault == "loss":
            rows[-1]["train_loss"] += 0.1
        else:
            rows[-1]["null_identity_outer_passed"] = False
        metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    if fault in ("capture", "common_rng"):
        with pytest.raises(ValueError):
            validation.trial(item, path, frozen)
    else:
        result = validation.trial(item, path, frozen)
        if fault == "graph_rewarm":
            assert result["accepted"] is True and result["continuation"]["23"]["within_tolerance"] is False
            assert result["first_resumed_diagnostic_only"] is True
        else:
            assert result["accepted"] is False
            assert not result["gates"][
                "post_cached_root_all_rank_continuation" if fault == "rank7" else "scalar_loss_continuation"
            ]


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "timeout", "spawn_failure"))
def test_actual_lr_cpu_child_preserves_sidecar_and_cleans_descendants(tmp_path, monkeypatch, outcome):
    support.record(contract.locations(tmp_path)[2], {"status": "CPU child fixture"})
    monkeypatch.setattr(
        contract, "ENVIRONMENT", {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONFAULTHANDLER": "1"}
    )
    helper = tmp_path / "cpu_child.py"
    helper.write_text(
        "import os,signal,subprocess,sys\n"
        "if sys.argv[1]=='timeout':signal.pause()\n"
        "if sys.argv[1]=='detached':\n"
        " r,w=os.pipe()\n"
        " child=subprocess.Popen([sys.executable,'-c',"
        "'import os,signal,sys;os.write(int(sys.argv[1]),b\"R\");signal.pause()',str(w)],"
        "pass_fds=(w,),start_new_session=True)\n"
        " os.close(w);assert os.read(r,1)==b'R';os.close(r)\n"
        "sys.exit(0 if sys.argv[1]=='zero' else 7)\n"
    )
    command = [str(tmp_path / "missing")] if outcome == "spawn_failure" else [contract.PYTHON, str(helper), outcome]
    log = tmp_path / "child.log"
    if outcome in ("timeout", "spawn_failure"):
        with pytest.raises(subprocess.TimeoutExpired if outcome == "timeout" else FileNotFoundError):
            validation.run_command(tmp_path, tmp_path, command, log, timeout=0.25 if outcome == "timeout" else 10)
    else:
        entry = validation.run_command(tmp_path, tmp_path, command, log, timeout=10)
        if outcome == "zero":
            validation.require_entry(entry, command, log)
        else:
            with pytest.raises(ValueError):
                validation.require_entry(entry, command, log)
    saved = read(log.with_suffix(".execution.json"))
    assert saved["log_sha256"] == sha256(log)
    if outcome != "spawn_failure":
        assert saved["generation"]["parent"] == os.getpid()
        assert saved["marked_descendants_absent"] is True
        assert not handoff.marker_processes(contract.COMMAND_MARKER, saved["marker"])
    if outcome == "timeout":
        assert saved["error"].startswith("TimeoutExpired:") and saved["cleanup_exit_code"] == -15
    assert not contract.locations(tmp_path)[4].exists()


@pytest.fixture(scope="module")
def complete_receipt(tmp_path_factory):
    return support.complete_receipt(tmp_path_factory.mktemp("training-path-actual-lr-protocol"))


@pytest.mark.timeout(180)
def test_all_162_command_reconstruction_preserves_negative_case_and_exact_original_source(complete_receipt):
    root, path, result = complete_receipt
    receipt = contract.locations(root)[4]
    before = receipt.read_bytes()
    rebuilt = validation.reconstruct(root, path)
    assert rebuilt == result and receipt.read_bytes() == before
    assert len(rebuilt["gpu_commands"]) == 162 and len(rebuilt["results"]) == 54
    assert rebuilt["rejected_cases"] == [contract.specs()[0]["name"]] and len(rebuilt["accepted_cases"]) == 53
    assert rebuilt["all_cases_accepted"] is False
    assert all(sha256(path / name) == digest for name, digest in contract.TRAINING_FILES.items())


@pytest.mark.parametrize("fault", ("missing_command", "missing_case", "bool_exit", "ancestry", "hidden_rejection"))
def test_complete_evidence_refuses_missing_rebound_or_erased_rejection(complete_receipt, fault):
    root, path, result = complete_receipt
    receipt = contract.locations(root)[4]
    before, changed = receipt.read_bytes(), copy.deepcopy(result)
    if fault == "missing_command":
        changed["gpu_commands"].pop()
    elif fault == "missing_case":
        changed["results"].pop()
    elif fault == "bool_exit":
        changed["gpu_commands"][0]["exit_code"] = False
    elif fault == "ancestry":
        changed["gpu_commands"][0]["generation"]["parent"] = 999
    else:
        changed["rejected_cases"] = []
    try:
        support.record(receipt, changed)
        with pytest.raises(ValueError):
            validation.reconstruct(root, path)
    finally:
        receipt.write_bytes(before)
