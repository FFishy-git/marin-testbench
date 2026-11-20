# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare complete scanner outcomes at the filesystem observation boundary."""

import ast
import copy
import hashlib
import importlib.util
import shutil
import types
from pathlib import Path

import owned_output_streaming_scan as streaming
import pytest
from owned_output_helper_churn import SOURCE, fake_process, source_bytes

PARENT_TEST_SHA256 = "78616824a05786c33bd904244400505ffbada4c52325e031ae8e715a26608aad"
QUERY = SOURCE.parents[1] / "ns-training-path-query-serialized-20261120/tracking"


def inherited_tests() -> bytes:
    path = QUERY / "test_owned_output_handoff.py"
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_TEST_SHA256:
        raise ValueError("Streaming scan original handoff regression bytes changed")
    return payload


exec(compile(inherited_tests(), __file__, "exec"), globals())


@pytest.fixture(autouse=True)
def replace_handoff(tmp_path, monkeypatch):
    sources = source_bytes(QUERY)
    directory = tmp_path / "streaming-handoff"
    directory.mkdir()
    for name, payload in sources.items():
        if name == "owned_output_handoff.py":
            payload = streaming.derived_source(QUERY / name)
        (directory / name).write_bytes(payload)
    path = directory / "owned_output_handoff.py"
    spec = importlib.util.spec_from_file_location("streaming_handoff_under_test", path)
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    monkeypatch.setitem(globals(), "handoff", scan)
    return scan


def test_streaming_scan_retains_complete_namespace_and_every_acceptance_statement():
    before = ast.parse((QUERY / "owned_output_handoff.py").read_bytes())
    after = ast.parse(streaming.derived_source(QUERY / "owned_output_handoff.py"))
    scanner = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == "marker_processes")
    normalized = copy.deepcopy(scanner)
    for node in ast.walk(normalized):
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "scan_processes()":
            node.iter = ast.parse("live_processes().items()", mode="eval").body
    expected = next(
        node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == "marker_processes"
    )
    assert ast.dump(normalized) == ast.dump(expected)
    retained = ast.Module(
        body=[node for node in after.body if not (isinstance(node, ast.FunctionDef) and node.name == "scan_processes")],
        type_ignores=[],
    )
    retained.body[retained.body.index(scanner)] = normalized
    assert ast.dump(retained) == ast.dump(before)


@pytest.mark.parametrize("reader", ("materialized", "streaming"))
def test_unmarked_process_exits_while_observing_later_process_do_not_stale_streaming_scan(tmp_path, monkeypatch, reader):
    scan = types.ModuleType("complete_scan_observation_order")
    scan.__file__ = str(QUERY / "owned_output_handoff.py")
    payload = (
        (QUERY / "owned_output_handoff.py").read_bytes()
        if reader == "materialized"
        else streaming.derived_source(QUERY / "owned_output_handoff.py")
    )
    exec(compile(payload, scan.__file__, "exec"), scan.__dict__)
    proc = tmp_path / "proc"
    proc.mkdir()
    scan.PROC_ROOT = proc
    fake_process(proc, 900)
    fake_process(proc, 901)
    for pid in (950, 1000):
        fake_process(proc, pid)
        (proc / str(pid) / "environ").write_bytes(b"UNRELATED_PROCESS=fixture\0")
    read_bytes = Path.read_bytes
    iterdir = Path.iterdir
    retired = []

    def ordered_process_directories(path):
        entries = list(iterdir(path))
        return iter(sorted(entries, key=lambda item: int(item.name))) if path == proc else iter(entries)

    def later_process_observation(path):
        payload = read_bytes(path)
        if path == proc / "1000/cmdline" and len(retired) < 3:
            previous = proc / str(950 + len(retired))
            replacement = proc / str(951 + len(retired))
            fake_process(proc, int(replacement.name))
            (replacement / "environ").write_bytes(b"UNRELATED_PROCESS=fixture\0")
            shutil.rmtree(previous)
            retired.append(int(previous.name))
        return payload

    monkeypatch.setattr(Path, "iterdir", ordered_process_directories)
    monkeypatch.setattr(Path, "read_bytes", later_process_observation)
    if reader == "materialized":
        with pytest.raises(scan.attribution_handler.OwnershipObservationError) as failure:
            scan.marker_processes("HELPER_CHURN_STUDY", "owned")
        assert failure.value.evidence["stage"] == "complete_scan_exhausted"
        assert failure.value.evidence["snapshots"]["previously_confirmed_marked"] == {"952": False}
    else:
        assert set(scan.marker_processes("HELPER_CHURN_STUDY", "owned")) == {900, 901}
    expected = [950, 951, 952] if reader == "materialized" else [950]
    assert retired == expected
    assert sorted(int(path.name) for path in proc.iterdir()) == [900, 901, 950 + len(expected), 1000]


def test_streaming_derivation_refuses_changed_parent_before_executing_source(tmp_path):
    path = tmp_path / "owned_output_handoff.py"
    path.write_bytes(
        (QUERY / "owned_output_handoff.py").read_bytes() + b"\nPath(__file__).with_name('unsafe').touch()\n"
    )
    with pytest.raises(ValueError):
        streaming.derived_source(path)
    assert not (tmp_path / "unsafe").exists()


@pytest.mark.parametrize("transition", ("absent", "terminal_absent", "surviving_thread", "partial_file", "reused_pid"))
def test_streaming_final_recheck_requires_identity_bound_whole_process_exit(
    tmp_path, monkeypatch, transition, replace_handoff
):
    proc = tmp_path / "proc"
    proc.mkdir()
    scan = replace_handoff
    scan.PROC_ROOT = proc
    fake_process(proc, 900)
    fake_process(proc, 901)
    read_bytes = Path.read_bytes
    observations = 0
    elapsed = 0
    drain = scan.exit_drain_handler.drain_exit

    def clock():
        return elapsed

    def pause(interval):
        nonlocal elapsed
        elapsed += interval
        if transition == "terminal_absent":
            shutil.rmtree(proc / "900")

    def recheck_read(path):
        nonlocal observations
        if path == proc / "900/cmdline":
            observations += 1
            if observations == 2:
                fake_process(proc, 902)
                if transition == "absent":
                    shutil.rmtree(proc / "900")
                elif transition == "partial_file":
                    (proc / "900/stat").unlink()
                else:
                    stat = proc / "900/stat"
                    prefix, fields = stat.read_text().rsplit(")", 1)
                    fields = fields.split()
                    fields[0], fields[6], fields[20] = "Z", "4", "0"
                    if transition == "reused_pid":
                        fields[19] = "999"
                    stat.write_text(prefix + ") " + " ".join(fields))
                    (proc / "900/cmdline").write_bytes(b"")
                    (proc / "900/environ").write_bytes(b"")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", recheck_read)
    monkeypatch.setattr(scan.exit_drain_handler, "whole_process_exited", lambda root, owner: False)
    monkeypatch.setattr(
        scan.exit_drain_handler,
        "drain_exit",
        lambda root, owner, **kwargs: drain(root, owner, clock=clock, pause=pause),
    )
    if transition in ("absent", "terminal_absent"):
        assert set(scan.marker_processes("HELPER_CHURN_STUDY", "owned")) == {901, 902}
        assert not (proc / "900").exists()
    else:
        with pytest.raises(RuntimeError):
            scan.marker_processes("HELPER_CHURN_STUDY", "owned")
        assert (proc / "900").is_dir()
    if transition == "surviving_thread":
        assert elapsed >= scan.exit_drain_handler.EXIT_DRAIN_TIMEOUT
    elif transition == "terminal_absent":
        assert elapsed > 0


def test_streaming_present_unmarked_process_with_missing_environment_refuses(tmp_path, replace_handoff):
    scan = replace_handoff
    scan.PROC_ROOT = tmp_path
    fake_process(tmp_path, 900)
    fake_process(tmp_path, 950)
    (tmp_path / "950/environ").unlink()
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(RuntimeError):
        scan.marker_processes("HELPER_CHURN_STUDY", "owned")

    assert (tmp_path / "950").is_dir()
    assert {path: path.read_bytes() for path in before} == before
    assert not (tmp_path / "950/environ").exists()
