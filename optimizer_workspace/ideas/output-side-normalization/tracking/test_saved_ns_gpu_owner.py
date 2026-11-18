# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise terminal and kernel lock ownership before any replay allocation transfer."""

import fcntl
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import saved_ns_gpu_owner as owner


def terminal_pair():
    intent = {
        "status": "acknowledged",
        "binding_sha256": owner.PREVIOUS_BINDING_SHA256,
        "pid": 101,
        "start_ticks": "202",
    }
    state = {
        "status": "diagnostic_complete",
        "binding_sha256": owner.PREVIOUS_BINDING_SHA256,
        "validation_sha256": owner.RECEIPT_SHA256,
        "exit_code": 0,
        "cleanup_exit_code": 0,
        "marked_descendants_absent": True,
        "keeper_restore": 0,
        "keeper_release_complete": True,
        "controller_pid": 101,
        "controller_start_ticks": "202",
        "restored_keeper": {"pid": 303},
    }
    return state, intent


@pytest.mark.parametrize(
    "fault",
    (
        {"status": "validating"},
        {"exit_code": False},
        {"cleanup_exit_code": 1},
        {"keeper_restore": None},
        {"keeper_restore": False},
        {"marked_descendants_absent": False},
        {"validation_sha256": "0" * 64},
        {"controller_start_ticks": "different"},
        {"error": "reconstruction failed"},
    ),
)
def test_terminal_acceptance_refuses_incomplete_or_changed_owner_evidence(fault):
    state, intent = terminal_pair()
    owner.require_terminal(state, intent, owner.RECEIPT_SHA256)
    with pytest.raises(RuntimeError, match="incomplete"):
        owner.require_terminal(state | fault, intent, owner.RECEIPT_SHA256)


def test_live_predecessor_state_refuses_before_source_inventory_or_keeper_access(tmp_path, monkeypatch):
    paths = [tmp_path / name for name in ("binding.json", "helpers", "state.json", "work", "diagnostic.json")]
    paths[1].mkdir()
    paths[2].write_text(json.dumps({"status": "validating"}) + "\n")
    (tmp_path / "controller.json").write_text(json.dumps({"status": "acknowledged"}) + "\n")
    monkeypatch.setattr(owner, "previous_locations", lambda root: tuple(paths))
    with pytest.raises(RuntimeError, match="incomplete"):
        owner.predecessor(tmp_path)
    assert not paths[0].exists() and not paths[3].exists() and not paths[4].exists()


def test_exact_exclusive_parent_fd_is_recognized_without_unlocking_it(tmp_path):
    path = tmp_path / "allocation.lock"
    with path.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner.require_held_lock(os.getpid(), held.fileno(), path)
        with path.open("r+") as independent:
            with pytest.raises(BlockingIOError):
                fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with path.open("r+") as released:
        fcntl.flock(released, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_unlocked_parent_fd_cannot_borrow_an_unrelated_process_exclusive_lock(tmp_path):
    path = tmp_path / "allocation.lock"
    with path.open("a") as merely_open:
        command = [
            sys.executable,
            "-c",
            "import fcntl,sys; "
            "f=open(sys.argv[1],'r+'); fcntl.flock(f,fcntl.LOCK_EX); "
            "print('held',flush=True); sys.stdin.read()",
            str(path),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as child:
            try:
                assert child.stdout.readline().strip() == "held"
                with pytest.raises(RuntimeError, match="exclusive flock"):
                    owner.require_held_lock(os.getpid(), merely_open.fileno(), path)
            finally:
                child.stdin.close()
                child.wait(timeout=5)
        with path.open("r+") as independent:
            fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_shared_parent_lock_is_not_an_exclusive_transfer_owner(tmp_path):
    path = tmp_path / "allocation.lock"
    with path.open("a") as shared:
        fcntl.flock(shared, fcntl.LOCK_SH)
        with pytest.raises(RuntimeError, match="exclusive flock"):
            owner.require_held_lock(os.getpid(), shared.fileno(), path)
        with path.open("r+") as independent:
            fcntl.flock(independent, fcntl.LOCK_SH | fcntl.LOCK_NB)


def test_replaced_lock_inode_does_not_validate_the_old_held_descriptor(tmp_path):
    path = tmp_path / "allocation.lock"
    with path.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        path.rename(tmp_path / "preserved.lock")
        path.write_text("")
        with pytest.raises(RuntimeError, match="different inode"):
            owner.require_held_lock(os.getpid(), held.fileno(), path)


def test_lock_alias_is_refused_without_acquiring_or_creating_targets(tmp_path):
    path = tmp_path / "allocation.lock"
    alias = tmp_path / "alias.lock"
    with path.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        alias.symlink_to(path)
        with pytest.raises(RuntimeError, match="canonical"):
            owner.require_held_lock(os.getpid(), held.fileno(), alias)
    assert alias.is_symlink()


def test_worker_request_without_issued_binding_cannot_touch_cuda_or_create_state(tmp_path):
    state = owner.locations(tmp_path)[2]
    with pytest.raises(FileNotFoundError):
        owner.require_gpu_owner(tmp_path, "0" * 64, state, 0)
    assert not state.exists()


@pytest.mark.parametrize("fault", ("command", "omit_owner"))
def test_cpu_contract_rejects_substituted_command_or_omitted_owner_helper(tmp_path, monkeypatch, fault):
    path = tmp_path / "inherited_owner.py"
    path.write_text("verified = True\n")
    monkeypatch.setattr(owner, "helper_sources", lambda root: {path.name: path})
    helpers = tmp_path / "tracking"
    cpu = {
        "helpers": str(helpers),
        "command": owner.cpu_command(helpers),
        "helpers_sha256": {path.name: owner.sha256(path)},
    }
    frozen = {"cpu_command": cpu["command"], "helpers_sha256": dict(cpu["helpers_sha256"])}
    owner.require_cpu_contract(tmp_path, cpu, frozen)
    if fault == "command":
        cpu["command"] = frozen["cpu_command"] = [sys.executable, "-c", "print('GREEN')"]
    else:
        cpu["helpers_sha256"] = frozen["helpers_sha256"] = {}
    with pytest.raises(ValueError, match="prescribed"):
        owner.require_cpu_contract(tmp_path, cpu, frozen)


def test_preparation_refuses_loaded_inherited_helper_with_different_bytes(tmp_path, monkeypatch):
    expected, loaded = tmp_path / "verified.py", tmp_path / "mutable.py"
    expected.write_text("verified = True\n")
    loaded.write_text("verified = False\n")
    monkeypatch.setitem(sys.modules, "inherited_test_helper", SimpleNamespace(__file__=str(loaded)))
    with pytest.raises(RuntimeError, match="different helper bytes"):
        owner.require_loaded_sources({"inherited_test_helper.py": expected})
    loaded.write_bytes(expected.read_bytes())
    owner.require_loaded_sources({"inherited_test_helper.py": expected})
