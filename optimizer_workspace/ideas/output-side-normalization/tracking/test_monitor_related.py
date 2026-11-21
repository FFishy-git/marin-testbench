# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only progress observations distinguish unknown and reused process handles."""

import json
import subprocess

from monitor_related import observe


def registration(tmp_path, state):
    path = tmp_path / "owner.json"
    path.write_text(json.dumps(state))
    return [{"node": "28.216.17.9", "state": str(path)}]


def identity(pid):
    return {
        "pid": pid,
        "status": "S",
        "parent": 1,
        "group": pid,
        "session": pid,
        "start_ticks": "1234",
        "command": ["/cpu/owned.py"],
    }


def test_running_owner_reused_pid_requires_attention_without_changing_state(tmp_path):
    expected = identity(123)
    registrations = registration(tmp_path, {"status": "running", "pid": 123, "generation": expected})
    before = (tmp_path / "owner.json").read_bytes()
    replacement = expected | {"start_ticks": "5678"}
    result = observe(
        tmp_path,
        registrations,
        inspect_node=lambda *_: {"processes": {"123": replacement}, "gpus": "", "compute": ""},
    )
    assert result["needs_attention"]
    assert result["owners"][0]["owner_liveness"] == "exited_or_changed"
    assert not result["all_registered_owners_complete_and_restored"]
    assert (tmp_path / "owner.json").read_bytes() == before


def test_observation_timeout_does_not_claim_process_exit_or_restoration(tmp_path):
    registrations = registration(tmp_path, {"status": "running", "pid": 123, "generation": identity(123)})

    def unavailable(*_):
        raise subprocess.TimeoutExpired("read-only-probe", 45)

    result = observe(tmp_path, registrations, inspect_node=unavailable)
    assert result["nodes"]["28.216.17.9"]["observation_status"] == "unavailable"
    assert result["owners"][0]["owner_liveness"] == "not_observed"
    assert not result["needs_attention"]
    assert not result["all_registered_owners_complete_and_restored"]


def test_completed_owner_requires_live_restored_ranks_and_exact_gpu_occupancy(tmp_path):
    keeper = identity(200)
    state = {
        "status": "complete",
        "pid": 123,
        "generation": identity(123),
        "keeper_restore": 0,
        "restored_keeper": {"generation": {"200": keeper}, "occupied": [["200", "GPU-physical"]]},
    }
    registrations = registration(tmp_path, state)
    observation = {
        "processes": {"123": None, "200": keeper},
        "gpus": "0, GPU-physical, 100, 1000\n",
        "compute": "200, GPU-physical\n",
    }
    complete = observe(tmp_path, registrations, inspect_node=lambda *_: observation)
    assert complete["all_registered_owners_complete_and_restored"]
    assert complete["owners"][0]["restored_keeper_live"]
    observation["compute"] = "999, GPU-physical\n"
    changed = observe(tmp_path, registrations, inspect_node=lambda *_: observation)
    assert not changed["all_registered_owners_complete_and_restored"]
    assert not changed["owners"][0]["restored_keeper_live"]


def test_missing_receipt_preserves_fresh_live_process_observation(tmp_path):
    expected = identity(123)
    registrations = registration(tmp_path, {"status": "running", "pid": 123, "generation": expected})
    registrations[0]["receipt"] = str(tmp_path / "unavailable-receipt.json")
    result = observe(
        tmp_path,
        registrations,
        inspect_node=lambda *_: {"processes": {"123": expected}, "gpus": "", "compute": ""},
    )
    assert result["owners"][0]["receipt_observation"] == "unavailable"
    assert result["owners"][0]["owner_liveness"] == "live"
    assert not result["needs_attention"]


def test_unavailable_owner_state_withholds_liveness_and_completion(tmp_path):
    registrations = [{"node": "28.216.17.9", "state": str(tmp_path / "unavailable-owner.json")}]
    result = observe(
        tmp_path,
        registrations,
        inspect_node=lambda *_: {"processes": {}, "gpus": "", "compute": ""},
    )
    assert result["owners"][0]["status"] == "unavailable"
    assert "owner_liveness" not in result["owners"][0]
    assert not result["all_registered_owners_complete_and_restored"]
