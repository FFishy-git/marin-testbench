# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test unchanged gates and source-read concurrency through the actual ownership reader."""

import ast
import copy
import fcntl
import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, current_thread

import ns_training_path_query_serialized_gpu_owner as owner
import ns_training_path_query_serialized_predecessor as predecessor
import ns_training_path_query_serialized_source as source
import ns_training_path_serialized_source as parent_source
import prepare_ns_training_path_query_serialized as preparation
import pytest
from ns_training_path_query_serialized_gpu_test_support import locked, physical_owner, record

ROOT = Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization")


def test_query_source_preserves_numeric_commands_and_exact_owner_statement_order(tmp_path):
    parent = source.recorded_parent(ROOT)
    generated = source.derived_sources(parent)
    path = tmp_path / "contract.py"
    path.write_bytes(generated["ns_training_path_query_serialized_contract.py"])
    spec = importlib.util.spec_from_file_location("query_serialized_contract", path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    assert contract.specs() == predecessor.original.specs()
    assert contract.TOLERANCE == predecessor.original.TOLERANCE
    assert len(contract.commands(Path(parent["candidate"]), ROOT, contract.locations(ROOT)[3])) == 54
    assert set(contract.locations(ROOT)).isdisjoint(predecessor.original.locations(ROOT))
    hashes = preparation.digest_bytes(preparation.helper_bytes(parent))
    assert all(hashes[name] == digest for name, digest in predecessor.original.NUMERICAL_READERS.items())
    assert hashes["owned_output_handoff.py"] == source.HANDOFF_SHA256
    assert hashes["test_owned_output_handoff.py"] == source.HANDOFF_TEST_SHA256
    before = source.QuerySerializedNamespace().visit(
        ast.parse(parent_source.derived_sources(parent)["ns_training_path_serialized_gpu_owner.py"])
    )
    after = ast.parse(generated["ns_training_path_query_serialized_gpu_owner.py"])
    original = next(
        node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == "require_controller"
    )
    query = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == "require_controller")
    flattened = [item for node in query.body for item in (node.body if isinstance(node, ast.With) else [node])]
    assert ast.dump(ast.Module(body=flattened, type_ignores=[])) == ast.dump(
        ast.Module(body=original.body[0].body, type_ignores=[])
    )
    assert ast.unparse(query.body[0]) == "frozen = require_binding(root)"
    locked = next(node for node in query.body if isinstance(node, ast.With))
    assert "hostname" in ast.unparse(locked.body[0].test)


@pytest.mark.parametrize("fault", ("none", "coverage", "decision", "cleanup", "owner", "failed_command", "waiter"))
def test_query_predecessor_preserves_zero_progress_and_refuses_substitution(monkeypatch, fault):
    frozen, intent, state, result = copy.deepcopy(predecessor.recorded(ROOT))
    if fault == "coverage":
        result["gpu_commands"].append({})
    elif fault == "decision":
        result["results"].append({"accepted": True})
    elif fault == "cleanup":
        state["cleanup_exit_code"] = 0
    elif fault == "owner":
        state["child_generation"]["parent"] += 1
    elif fault == "failed_command":
        frozen["commands"][0] = ["unaccepted"]
    elif fault == "waiter":
        original = predecessor.read

        def changed(path):
            value = original(path)
            if path == ROOT / predecessor.WAITER / "state.json":
                value["status"] = "complete"
            return value

        monkeypatch.setattr(predecessor, "read", changed)
    if fault == "none":
        saved = predecessor.transition_structure(ROOT, frozen, intent, state, result)
        assert (saved["completed_commands"], saved["case_decisions"]) == (0, 0)
        assert saved["accepted_cases"] == saved["rejected_cases"] == []
        assert not saved["coverage_complete"] and not saved["numerically_accepted"]
        assert saved["queue"]["status"] == "cpu_archive_accepted_unissued"
    else:
        with pytest.raises(ValueError):
            predecessor.transition_structure(ROOT, frozen, intent, state, result)


@pytest.mark.parametrize("owner", ("controller", "validator", "waiter", "failed", "marked"))
def test_query_predecessor_refuses_consumed_owner_or_descendant(tmp_path, monkeypatch, owner):
    frozen, intent, state, result = predecessor.recorded(ROOT)
    monkeypatch.setattr(predecessor.previous.handoff, "PROC_ROOT", tmp_path)
    failed = predecessor.read(ROOT / predecessor.FAILED_LOG.replace(".log", ".execution.json"))
    handles = {
        "controller": intent["pid"],
        "validator": state["child_pid"],
        "waiter": predecessor.read(ROOT / predecessor.WAITER / "intent.json")["pid"],
        "failed": failed["generation"]["pid"],
    }
    if owner == "marked":
        monkeypatch.setattr(predecessor.previous.handoff, "marker_processes", lambda *args: {999999: {}})
    else:
        (tmp_path / str(handles[owner])).mkdir()
    with pytest.raises(RuntimeError):
        predecessor.require_absent(ROOT, frozen, intent, state, result)


def test_query_predecessor_refuses_launching_preserved_actual_lr_queue(tmp_path):
    (tmp_path / predecessor.QUEUE).mkdir(parents=True)
    with pytest.raises(ValueError):
        predecessor.queue_structure(tmp_path)
    assert (tmp_path / predecessor.QUEUE).is_dir()


def test_blocked_source_read_does_not_block_other_rank_but_queries_and_markers_are_locked(tmp_path, monkeypatch):
    root, candidate, state, frozen, _, _ = physical_owner(tmp_path, monkeypatch)
    path = owner.contract.locations(root)[1].parent / "rank-owner-inspection.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    blocked, release = Event(), Event()
    slow_thread = None
    query = owner.subprocess.check_output
    marker_processes = owner.handoff.marker_processes

    def source_read(base):
        if current_thread() == slow_thread:
            blocked.set()
            if not release.wait(timeout=20):
                raise RuntimeError("CPU source boundary was not released")
        return frozen

    def require_locked():
        with pytest.raises(BlockingIOError):
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def locked_query(argv, **kwargs):
        require_locked()
        return query(argv, **kwargs)

    def locked_markers(*args):
        require_locked()
        return marker_processes(*args)

    def slow_rank():
        nonlocal slow_thread
        slow_thread = current_thread()
        return owner.require_controller(root, candidate)

    monkeypatch.setattr(owner, "require_binding", source_read)
    monkeypatch.setattr(owner.subprocess, "check_output", locked_query)
    monkeypatch.setattr(owner.handoff, "marker_processes", locked_markers)
    node = root / "run-preparation" / f".wave7-node-{owner.NODE}.lock"
    manager = root / "manager.lock"
    monkeypatch.setattr(owner, "MANAGER_LOCK", manager)
    try:
        with locked(node) as node_fd, locked(manager) as manager_fd, ThreadPoolExecutor(max_workers=2) as pool:
            state["lock_descriptors"] = {"node": node_fd, "manager": manager_fd}
            record(owner.contract.locations(root)[2], state)
            slow = pool.submit(slow_rank)
            try:
                assert blocked.wait(timeout=5)
                other = pool.submit(owner.require_controller, root, candidate)
                assert other.result(timeout=5) == (state, frozen)
                assert not slow.done()
            finally:
                release.set()
            assert slow.result(timeout=5) == (state, frozen)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        release.set()
        os.close(descriptor)
