# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU lifecycle acceptance without original payload reconstruction or GPU issuance."""

import fcntl
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import ns_training_path_lr_contract as contract
import ns_training_path_lr_gpu_controller as controller
import ns_training_path_lr_gpu_owner as owner
import ns_training_path_lr_gpu_test_support as support
import owned_output_handoff as handoff
import pytest
from fused_attribution_validation_test_support import keeper, record
from selected_lr_release_owner import read, sha256


@pytest.mark.parametrize("outcome", ("zero", "nonzero", "detached", "timeout", "spawn_failure"))
def test_validator_retains_outcome_and_cleans_real_cpu_descendants(tmp_path, monkeypatch, outcome):
    _, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    state = {"controller_pid": os.getpid(), "controller_start_ticks": "CPU-controller"}
    record(state_path, state)
    detached = tmp_path / "detached.pid"
    argv = [sys.executable, str(support.validator_program(helpers)), str(state_path), outcome, str(detached)]
    monkeypatch.setattr(owner, "validator_command", lambda root, candidate: argv)
    monkeypatch.setattr(owner, "TIMEOUT", 0.25 if outcome == "timeout" else 10)
    support.cpu_validator_io(monkeypatch)
    if outcome == "spawn_failure":
        monkeypatch.setattr(controller.subprocess, "Popen", support.fail_spawn)
        with pytest.raises(OSError):
            controller.bounded_validator(tmp_path, {"candidate": str(tmp_path)}, state)
    elif outcome == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            controller.bounded_validator(tmp_path, {"candidate": str(tmp_path)}, state)
    else:
        assert controller.bounded_validator(tmp_path, {"candidate": str(tmp_path)}, state) == (
            0 if outcome == "zero" else 7
        )
    saved = read(state_path)
    assert saved["log_sha256"] == sha256(helpers.parent / "validator.log")
    assert not handoff.marker_processes(contract.CHILD_MARKER, saved["child_marker"])
    if outcome == "spawn_failure":
        assert saved["spawn_status"] == "starting" and "child_pid" not in saved
    else:
        assert saved["marked_descendants_absent"] is True and handoff.observed(saved["child_pid"]) is None
    if outcome == "detached":
        observed = handoff.observed(int(detached.read_text()))
        assert observed is None or observed["status"] == "Z"


@pytest.mark.parametrize("outcome", ("spawn_failure", "nonzero", "timeout"))
def test_controller_restores_locked_allocation_after_validator_failure(tmp_path, monkeypatch, outcome):
    restored, events, node_path = support.failed_execution(tmp_path, monkeypatch, outcome)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises((OSError, RuntimeError, subprocess.TimeoutExpired)):
        controller.execute(tmp_path)
    saved = read(contract.locations(tmp_path)[2])
    assert saved["status"] == "needs_attention" and saved["keeper_restore"] == 0
    assert events == ["released", "restored"] and saved["restored_keeper"] == restored
    with node_path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize(
    "fault", ("none", "unowned_pid", "wrong_uuid", "unreleased", "keeper_pointer", "competing", "lock")
)
def test_physical_allocation_requires_owned_ranks_exact_uuids_transfer_and_real_locks(tmp_path, monkeypatch, fault):
    root, candidate, state, frozen, occupied, marked = support.physical_owner(tmp_path, monkeypatch)
    node = root / "run-preparation" / f".wave7-node-{owner.NODE}.lock"
    manager = root / "manager.lock"
    monkeypatch.setattr(owner, "MANAGER_LOCK", manager)
    with support.locked(node) as node_fd, support.locked(manager) as manager_fd:
        state["lock_descriptors"] = {"node": node_fd, "manager": manager_fd}
        if fault == "unowned_pid":
            occupied[0][0] = "840000"
        elif fault == "wrong_uuid":
            occupied[0][1] = frozen["devices"][4][1]
        elif fault == "unreleased":
            state["keeper_release_complete"] = False
        elif fault == "keeper_pointer":
            owner.handoff.KEEPER_PID.write_text("100\n")
        elif fault == "competing":
            marked[840000] = support.generation(840000, state["child_pid"], state["active_command"])
        elif fault == "lock":
            fcntl.flock(manager_fd, fcntl.LOCK_UN)
        record(contract.locations(root)[2], state)
        if fault == "none":
            saved, binding = owner.require_controller(root, candidate)
            assert saved == state and binding == frozen
        else:
            with pytest.raises(RuntimeError):
                owner.require_controller(root, candidate)


@pytest.mark.parametrize("fault", ("none", "permuted", "missing", "driver_error"))
def test_empty_occupancy_requires_identity_checked_cuda_ordinals_before_work(tmp_path, monkeypatch, fault):
    root, candidate, state, frozen, occupied, marked = support.physical_owner(tmp_path, monkeypatch)
    occupied.clear()
    marked.clear()
    identifiers, driver = support.cuda_driver(monkeypatch, frozen["devices"])
    if fault == "permuted":
        identifiers[0], identifiers[1] = identifiers[1], identifiers[0]
    elif fault == "missing":
        identifiers.pop()
    elif fault == "driver_error":
        driver.cuDeviceGetUuid_v2 = lambda *args: 999
    node = root / "run-preparation" / f".wave7-node-{owner.NODE}.lock"
    manager = root / "manager.lock"
    monkeypatch.setattr(owner, "MANAGER_LOCK", manager)
    with support.locked(node) as node_fd, support.locked(manager) as manager_fd:
        state["lock_descriptors"] = {"node": node_fd, "manager": manager_fd}
        state.pop("active_command")
        record(contract.locations(root)[2], state)
        before = contract.locations(root)[2].read_bytes()
        if fault == "none":
            saved, binding = owner.require_controller(root, candidate)
            assert saved == state and binding == frozen
        else:
            with pytest.raises(RuntimeError):
                owner.require_controller(root, candidate)
        assert contract.locations(root)[2].read_bytes() == before
        assert not contract.locations(root)[3].exists() and not contract.locations(root)[4].exists()


@pytest.mark.parametrize("fault", ("none", "launcher_parent", "rank_session", "rank_topology", "command", "marker"))
def test_rank_requires_exact_command_original_launcher_and_session_topology(tmp_path, monkeypatch, fault):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    argv = [
        contract.PYTHON,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=8",
        "-m",
        "experiment.train",
        "--config",
        str(candidate / "config.json"),
        "--launch-release",
        str(contract.locations(tmp_path)[0]),
        "--max-steps",
        "24",
    ]
    state = {"active_command": argv, "child_pid": 830000, "child_marker": "validator-marker"}
    own = support.generation(os.getpid(), 830001, [argv[0], "-u", *argv[5:]])
    launcher = support.generation(830001, 830000, argv)
    processes = {own["pid"]: own, launcher["pid"]: launcher}
    monkeypatch.setattr(owner, "require_controller", lambda root, candidate: (state, {"commands": [argv]}))

    def context(pid, command, cwd, marker_name, marker, environment):
        actual = processes[pid]
        if (
            actual["command"] != command
            or marker != "command-marker"
            or actual["group"] != pid
            or actual["session"] != pid
        ):
            raise RuntimeError("CPU process executable/marker/session boundary differs")
        return actual

    monkeypatch.setattr(owner, "process_context", context)
    monkeypatch.setattr(
        owner, "process_details", lambda pid: {"environment": {contract.CHILD_MARKER: "validator-marker"}}
    )
    for name, value in {
        "WORLD_SIZE": "8",
        "LOCAL_WORLD_SIZE": "8",
        "RANK": "3",
        "LOCAL_RANK": "3",
        contract.COMMAND_MARKER: "command-marker",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "argv", ["train.py", *argv[7:]])
    if fault == "launcher_parent":
        launcher["parent"] = 840000
    elif fault == "rank_session":
        own["session"] = launcher["session"]
    elif fault == "rank_topology":
        monkeypatch.setenv("LOCAL_RANK", "4")
    elif fault == "command":
        state["active_command"] = [*argv[:-1], "5120"]
    elif fault == "marker":
        monkeypatch.delenv(contract.COMMAND_MARKER)
    args = SimpleNamespace(config=candidate / "config.json", launch_release=contract.locations(tmp_path)[0])
    if fault == "none":
        owner.require_rank(tmp_path, candidate, args)
        assert state["active_command"] == argv
    else:
        with pytest.raises(RuntimeError):
            owner.require_rank(tmp_path, candidate, args)


@pytest.mark.parametrize("outcome", ("zero", "spawn_failure"))
def test_one_shot_start_records_original_detached_acknowledgment(tmp_path, monkeypatch, outcome):
    binding, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    record(binding, {"fixture": "CPU frozen source"})
    intent_path = helpers.parent / "controller.json"
    program = helpers / "controller_fixture.py"
    program.write_text(
        "import json,os,select,sys\nfrom pathlib import Path\n"
        "intent=Path(sys.argv[1]);poll=select.poll()\n"
        "while json.loads(intent.read_text()).get('pid')!=os.getpid():poll.poll(1)\n"
    )
    argv, accepted = [sys.executable, str(program), str(intent_path)], keeper()
    monkeypatch.setattr(
        owner,
        "require_binding",
        lambda root: {
            "allocation_predecessor": {"restored_keeper": accepted},
            "predecessor_acceptance": {"seal_sha256": "CPU-fixture-seal"},
            "cpu_receipt_sha256": "CPU-fixture-receipt",
            "authenticated_files": {"CPU-fixture": {"sha256": "CPU-fixture"}},
        },
    )
    monkeypatch.setattr(owner, "controller_command", lambda root: argv)
    monkeypatch.setattr(owner, "require_predecessor_keeper", lambda *args: accepted)
    monkeypatch.setattr(controller, "local_node", lambda: owner.NODE)
    monkeypatch.setattr(controller, "idle_training", lambda: None)
    monkeypatch.setattr(controller, "keeper_guard", lambda: support.locked(tmp_path / "manager.lock"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    children, original_spawn = [], subprocess.Popen

    def spawn(*args, **kwargs):
        if outcome == "spawn_failure":
            return support.fail_spawn(*args, **kwargs)
        child = original_spawn(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(controller.subprocess, "Popen", spawn)
    try:
        if outcome == "zero":
            intent = controller.start(tmp_path)
            own = intent["generation"]
            assert intent["status"] == "acknowledged" and own["parent"] == os.getpid()
            assert own["session"] == own["group"] == own["pid"]
            assert children[0].wait(timeout=10) == 0 and handoff.observed(own["pid"]) is None
        else:
            with pytest.raises(OSError):
                controller.start(tmp_path)
            assert read(intent_path)["status"] == "needs_attention"
        assert not state_path.exists()
        with pytest.raises(RuntimeError):
            controller.start(tmp_path)
    finally:
        for child in children:
            controller.finish_marked_child(child, read(intent_path)["marker"], {}, contract.CONTROLLER_MARKER)


@pytest.mark.parametrize("fault", ("none", "decisions", "keeper", "live_validator"))
def test_terminal_retains_all_rejected_cases_and_requires_exact_restoration(tmp_path, monkeypatch, fault):
    state, intent, result, restored = support.completed_terminal(tmp_path, monkeypatch)
    if fault == "decisions":
        state.update(accepted_cases=result["rejected_cases"], rejected_cases=[], all_cases_accepted=True)
        record(contract.locations(tmp_path)[2], state)
    elif fault == "keeper":
        observed = json.loads(json.dumps(restored))
        observed["generation"]["100"]["start_ticks"] = "different-restoration"
        monkeypatch.setattr(controller.handoff, "keeper_observation", lambda: observed)
    elif fault == "live_validator":
        monkeypatch.setattr(controller.handoff, "observed", lambda pid: {"pid": pid})
    out, attempt, outcome, _ = controller.terminal_paths(tmp_path, intent)
    if fault == "none":
        receipt = controller.terminal(tmp_path, out)
        assert receipt["rejected_cases"] == result["rejected_cases"] and receipt["all_cases_accepted"] is False
        assert receipt["production_authorized"] is False and receipt["keeper"] == restored
        assert receipt["controller_pid_absent"] == intent["pid"]
        assert read(attempt)["output"] == str(out)
        assert read(outcome)["status"] == "complete" and read(outcome)["terminal_sha256"] == sha256(out)
        assert receipt["reconstruction_intent_sha256"] == sha256(attempt)
        original = out.read_bytes(), attempt.read_bytes(), outcome.read_bytes()
        monkeypatch.setattr(controller.validation, "reconstruct", support.fail_spawn)
        with pytest.raises(RuntimeError):
            controller.terminal(tmp_path, out)
        with pytest.raises(RuntimeError):
            controller.terminal(tmp_path, tmp_path / "run-preparation/alternate-terminal.json")
        assert (out.read_bytes(), attempt.read_bytes(), outcome.read_bytes()) == original
    else:
        with pytest.raises((RuntimeError, ValueError)):
            controller.terminal(tmp_path, out)
        assert not out.exists()
        assert read(attempt)["status"] == "reconstruction_claimed" and read(outcome)["status"] == "needs_attention"
        original = attempt.read_bytes(), outcome.read_bytes()
        monkeypatch.setattr(controller.validation, "reconstruct", support.fail_spawn)
        with pytest.raises(RuntimeError):
            controller.terminal(tmp_path, out)
        assert (attempt.read_bytes(), outcome.read_bytes()) == original


def test_terminal_rejects_arbitrary_output_before_claiming_reconstruction(tmp_path, monkeypatch):
    _, intent, _, _ = support.completed_terminal(tmp_path, monkeypatch)
    out, attempt, outcome, _ = controller.terminal_paths(tmp_path, intent)
    monkeypatch.setattr(controller.validation, "reconstruct", support.fail_spawn)
    with pytest.raises(RuntimeError):
        controller.terminal(tmp_path, tmp_path / "run-preparation/unused-terminal.json")
    assert not any(path.exists() for path in (out, attempt, outcome))


def test_terminal_serializes_contenders_and_never_overwrites_publication(tmp_path, monkeypatch):
    _, intent, _, _ = support.completed_terminal(tmp_path, monkeypatch)
    out, attempt, outcome, lock = controller.terminal_paths(tmp_path, intent)
    with support.locked(lock):
        with pytest.raises(BlockingIOError):
            controller.terminal(tmp_path, out)
        assert not any(path.exists() for path in (out, attempt, outcome))
    receipt = controller.terminal(tmp_path, out)
    original = out.read_bytes(), attempt.read_bytes(), outcome.read_bytes()
    with pytest.raises(FileExistsError):
        controller.publish_terminal_record(out, {"status": "competing_publication"})
    assert receipt["production_authorized"] is False
    assert (out.read_bytes(), attempt.read_bytes(), outcome.read_bytes()) == original


def test_terminal_failed_reconstruction_persists_one_shot_attention_evidence(tmp_path, monkeypatch):
    _, intent, _, _ = support.completed_terminal(tmp_path, monkeypatch)
    out, attempt, outcome, _ = controller.terminal_paths(tmp_path, intent)
    monkeypatch.setattr(controller.validation, "reconstruct", support.fail_spawn)
    with pytest.raises(OSError):
        controller.terminal(tmp_path, out)
    assert not out.exists()
    assert read(outcome)["status"] == "needs_attention"
    assert read(outcome)["reconstruction_intent_sha256"] == sha256(attempt)
    original = attempt.read_bytes(), outcome.read_bytes()
    with pytest.raises(RuntimeError):
        controller.terminal(tmp_path, out)
    assert (attempt.read_bytes(), outcome.read_bytes()) == original


def test_terminal_changed_generation_cannot_claim_another_reconstruction(tmp_path, monkeypatch):
    _, intent, _, _ = support.completed_terminal(tmp_path, monkeypatch)
    out, attempt, outcome, _ = controller.terminal_paths(tmp_path, intent)
    controller.terminal(tmp_path, out)
    original = out.read_bytes(), attempt.read_bytes(), outcome.read_bytes()
    intent["marker"] = "substituted-generation"
    record(contract.locations(tmp_path)[1].parent / "controller.json", intent)
    alternate = controller.terminal_paths(tmp_path, intent)[0]
    assert alternate != out
    monkeypatch.setattr(controller.validation, "reconstruct", support.fail_spawn)
    with pytest.raises(RuntimeError):
        controller.terminal(tmp_path, alternate)
    assert not alternate.exists()
    assert (out.read_bytes(), attempt.read_bytes(), outcome.read_bytes()) == original


@pytest.mark.parametrize("fault", ("none", "modified", "replaced", "alias"))
def test_operation_inventory_rechecks_authenticated_files_without_payload_cache(tmp_path, fault):
    path = tmp_path / "source.py"
    path.write_bytes(b"accepted source\n")
    inventory = owner.authenticated_inventory({str(path): sha256(path)})
    if fault == "modified":
        path.write_bytes(b"changed source\n")
    elif fault == "replaced":
        alternate = tmp_path / "replacement.py"
        alternate.write_bytes(path.read_bytes())
        alternate.replace(path)
    elif fault == "alias":
        alternate = tmp_path / "alternate.py"
        alternate.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(alternate)
    if fault == "none":
        owner.require_operation_inventory(inventory)
        assert inventory[str(path)]["sha256"] == sha256(path)
    else:
        with pytest.raises(ValueError):
            owner.require_operation_inventory(inventory)


def test_candidate_namespace_detects_added_removed_and_aliased_source(tmp_path):
    candidate = tmp_path / "candidate"
    record(candidate / "experiment/config.json", {"fixture": "independent configuration"})
    pinned = candidate / "artifacts/pinned-sources-materialized"
    pinned.mkdir(parents=True)
    (pinned / "engine.py").write_bytes(b"accepted numerical source\n")
    (candidate / "artifacts/external-sources").symlink_to("pinned-sources-materialized", target_is_directory=True)
    original = owner.candidate_file_names(candidate)
    extra = candidate / "experiment/extra.py"
    extra.write_bytes(b"unaccepted source\n")
    assert owner.candidate_file_names(candidate) == original | {"experiment/extra.py"}
    (candidate / "experiment/config.json").unlink()
    assert "experiment/config.json" not in owner.candidate_file_names(candidate)
    extra.unlink()
    extra.symlink_to(pinned / "engine.py")
    with pytest.raises(ValueError):
        owner.candidate_file_names(candidate)


@pytest.mark.parametrize("fault", ("none", "receipt", "inventory", "controller_environment", "caller_environment"))
def test_visible_parent_proof_rejects_forged_cpu_acceptance_without_hidden_predecessor_calls(
    tmp_path, monkeypatch, fault
):
    binding, helpers, state_path, _, _ = contract.locations(tmp_path)
    helpers.mkdir(parents=True)
    frozen = {
        "cpu_receipt_sha256": "independently-accepted-CPU-receipt",
        "authenticated_files": {"source.py": {"identity": {"inode": 42}, "sha256": "accepted-source"}},
    }
    record(binding, frozen)
    raw = binding.read_bytes()
    proof = owner.parent_proof(raw, frozen)
    own = support.generation(850001, 1, owner.controller_command(tmp_path))
    record(
        helpers.parent / "controller.json",
        {
            "status": "acknowledged",
            "binding_sha256": sha256(binding),
            "parent_proof": proof,
            "pid": own["pid"],
            "start_ticks": own["start_ticks"],
            "command": own["command"],
            "generation": own,
            "marker": "accepted-controller",
        },
    )
    record(
        state_path,
        {
            "controller_pid": own["pid"],
            "controller_start_ticks": own["start_ticks"],
            "binding_sha256": sha256(binding),
        },
    )
    environment = owner.CONTROLLER_ENVIRONMENT | {owner.PARENT_PROOF_MARKER: proof}

    def context(pid, command, cwd, marker_name, marker, required):
        if (
            pid != own["pid"]
            or command != own["command"]
            or cwd != helpers
            or marker != "accepted-controller"
            or any(environment.get(name) != value for name, value in required.items())
        ):
            raise RuntimeError("CPU process environment does not carry independently accepted proof")
        return own

    monkeypatch.setattr(owner, "process_context", context)
    monkeypatch.setenv(owner.PARENT_PROOF_MARKER, proof)
    if fault == "receipt":
        frozen["cpu_receipt_sha256"] = "substituted-CPU-receipt"
    elif fault == "inventory":
        frozen["authenticated_files"]["source.py"]["sha256"] = "substituted-source"
    elif fault == "controller_environment":
        environment[owner.PARENT_PROOF_MARKER] = "unaccepted-parent-proof"
    elif fault == "caller_environment":
        monkeypatch.setenv(owner.PARENT_PROOF_MARKER, "unaccepted-caller-proof")
    if fault == "none":
        owner.require_parent_proof(tmp_path, raw, frozen)
        assert read(helpers.parent / "controller.json")["binding_sha256"] == sha256(binding)
    else:
        with pytest.raises(RuntimeError):
            owner.require_parent_proof(tmp_path, raw, frozen)
