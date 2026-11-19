# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise redacted attribution through the pinned scan's public rejection path."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import owned_output_attribution as attribution
import pytest


def process(proc, pid, *, marker="test-marker", ticks="900"):
    directory = proc / str(pid)
    directory.mkdir(exist_ok=True)
    fields = ["0"] * 20
    fields[:4] = ["S", "1", str(pid), str(pid)]
    fields[19] = ticks
    (directory / "stat").write_text(f"{pid} (fixture) {' '.join(fields)}")
    (directory / "cmdline").write_bytes(b"helper\0--secret=argv-credential\0")
    (directory / "environ").write_bytes(f"TOKEN={marker}\0PASSWORD=environment-credential\0".encode())


@pytest.fixture
def diagnostic(tmp_path):
    path = tmp_path / "attributed_handoff.py"
    path.write_bytes(attribution.derived_source(Path(__file__).with_name("owned_output_handoff.py")))
    handler_path = tmp_path / "owned_output_attribution.py"
    handler_path.write_bytes(Path(attribution.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location("attributed_handoff", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proc = tmp_path / "proc"
    proc.mkdir()
    module.PROC_ROOT = proc
    return module, proc


@pytest.mark.parametrize("change", ("command_empty", "command_replaced", "parent", "group", "session", "pid_reuse"))
def test_live_generation_changes_remain_rejected_with_exact_redacted_evidence(
    diagnostic, monkeypatch, change, record_property
):
    scan, proc = diagnostic
    process(proc, 900)
    read_bytes = Path.read_bytes
    changed = False

    def transitioned(path):
        nonlocal changed
        payload = read_bytes(path)
        if path == proc / "900/environ" and not changed:
            changed = True
            if change.startswith("command"):
                (proc / "900/cmdline").write_bytes(b"" if change == "command_empty" else b"replacement\0")
            else:
                stat = proc / "900/stat"
                prefix, fields = stat.read_text().rsplit(")", 1)
                fields = fields.split()
                fields[{"parent": 1, "group": 2, "session": 3, "pid_reuse": 19}[change]] = "999"
                stat.write_text(prefix + ") " + " ".join(fields))
        return payload

    monkeypatch.setattr(Path, "read_bytes", transitioned)
    with pytest.raises(RuntimeError) as failure:
        scan.marker_processes("TOKEN", "test-marker")
    evidence = failure.value.evidence
    record_property("ownership_failure_evidence", json.dumps(evidence, sort_keys=True))
    snapshots = evidence["snapshots"]
    assert evidence["stage"] == "recheck_identity_drift"
    assert evidence["pid"] == 900
    assert snapshots["scan"]["start_ticks"] == "900"
    if change.startswith("command"):
        assert snapshots["scan"]["command"]["argument_count"] == 2
        assert snapshots["recheck"]["command"]["argument_count"] == (0 if change == "command_empty" else 1)
        assert snapshots["scan"]["command"]["sha256"] != snapshots["recheck"]["command"]["sha256"]
    else:
        key = "start_ticks" if change == "pid_reuse" else change
        assert snapshots["recheck"][key] == ("999" if change == "pid_reuse" else 999)
    assert evidence["authority"] == "failure_attribution_only_no_ownership_or_launch"
    assert snapshots["previously_confirmed_marked"] is True
    rendered = str(failure.value)
    assert not any(secret in rendered for secret in ("argv-credential", "environment-credential", "test-marker"))


@pytest.mark.parametrize("replacement", (b"", b"PASSWORD=changed-secret\0"))
def test_live_marker_removal_is_rejected_with_presence_only_evidence(
    diagnostic, monkeypatch, replacement, record_property
):
    scan, proc = diagnostic
    for pid in (900, 901):
        process(proc, pid)
    read_bytes = Path.read_bytes
    changed = False

    def transitioned(path):
        nonlocal changed
        payload = read_bytes(path)
        if path == proc / "901/environ" and not changed:
            changed = True
            shutil.rmtree(proc / "900")
            (proc / "901/environ").write_bytes(replacement)
        return payload

    monkeypatch.setattr(Path, "read_bytes", transitioned)
    with pytest.raises(RuntimeError) as failure:
        scan.marker_processes("TOKEN", "test-marker")
    evidence = failure.value.evidence
    record_property("ownership_failure_evidence", json.dumps(evidence, sort_keys=True))
    assert evidence["stage"] == "scan_marker_removed"
    assert evidence["pid"] == 901
    assert evidence["snapshots"]["marker_present"] is False
    assert evidence["snapshots"]["scan"]["status"] == "S"
    assert evidence["snapshots"]["environment_byte_count"] == len(replacement)
    assert evidence["snapshots"]["previously_confirmed_marked"] is True
    assert "changed-secret" not in str(failure.value)


def test_unreadable_environment_never_returns_partial_absence(diagnostic):
    scan, proc = diagnostic
    process(proc, 900)
    (proc / "900/environ").unlink()
    with pytest.raises(RuntimeError) as failure:
        scan.marker_processes("TOKEN", "test-marker")
    evidence = failure.value.evidence
    assert evidence["stage"] == "complete_scan_exhausted"
    assert evidence["snapshots"]["unstable_pids"] == [900]
    assert evidence["snapshots"]["retained"]["900"]["start_ticks"] == "900"
    assert evidence["snapshots"]["previously_confirmed_marked"] == {"900": False}


def test_stat_drift_is_recorded_without_accepting_a_mixed_observation(diagnostic, monkeypatch, record_property):
    scan, proc = diagnostic
    process(proc, 900)
    read_bytes = Path.read_bytes

    def transitioned(path):
        payload = read_bytes(path)
        if path == proc / "900/cmdline":
            process(proc, 900, ticks="999")
        return payload

    monkeypatch.setattr(Path, "read_bytes", transitioned)
    with pytest.raises(RuntimeError) as failure:
        scan.observed(900)
    snapshots = failure.value.evidence["snapshots"]
    record_property("ownership_failure_evidence", json.dumps(failure.value.evidence, sort_keys=True))
    assert failure.value.evidence["stage"] == "observed_stat_drift"
    assert snapshots["stat_before"]["start_ticks"] == "900"
    assert snapshots["stat_after"]["start_ticks"] == "999"
    assert snapshots["command_between_stats"]["argument_count"] == 2


def test_delayed_sample_cannot_overwrite_the_exact_failure_snapshot(diagnostic, monkeypatch):
    scan, proc = diagnostic
    process(proc, 900)
    read_bytes = Path.read_bytes
    command_reads = 0

    def transitioned(path):
        nonlocal command_reads
        payload = read_bytes(path)
        if path == proc / "900/environ":
            (proc / "900/cmdline").write_bytes(b"")
        elif path == proc / "900/cmdline":
            command_reads += 1
            if command_reads == 3:
                process(proc, 900, ticks="999")
                return b"replacement-after-failure\0"
        return payload

    monkeypatch.setattr(Path, "read_bytes", transitioned)
    with pytest.raises(RuntimeError) as failure:
        scan.marker_processes("TOKEN", "test-marker")
    evidence = failure.value.evidence
    assert evidence["snapshots"]["recheck"]["command"]["argument_count"] == 0
    assert evidence["snapshots"]["recheck"]["start_ticks"] == "900"
    assert evidence["delayed_samples"][0]["cmdline"]["argument_count"] == 1


def test_real_linux_marked_child_retains_normal_result_without_raw_secrets(diagnostic):
    scan, _ = diagnostic
    scan.PROC_ROOT = Path("/proc")
    marker = uuid4().hex
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.buffer.read()"],
        env=os.environ | {"TOKEN": marker},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ) as child:
        try:
            assert child.stdout.readline().strip() == "ready"
            found = scan.marker_processes("TOKEN", marker)
            assert set(found) == {child.pid}
            assert found[child.pid]["command"][:2] == [sys.executable, "-c"]
        finally:
            child.stdin.close()
            child.wait(timeout=10)
    assert child.returncode == 0


def test_materialized_cpu_source_records_parent_and_derived_identity_without_launch(tmp_path):
    out = tmp_path / "derived"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(attribution.__file__)),
            "--parent",
            str(Path(__file__).with_name("owned_output_handoff.py")),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(result.stdout)
    assert receipt == json.loads((out / "source.json").read_text())
    assert receipt["parent_sha256"] == attribution.PARENT_SHA256
    assert (out / "owned_output_handoff.py").read_bytes() == attribution.derived_source(
        Path(__file__).with_name("owned_output_handoff.py")
    )
    assert receipt["authority"] == "no_scan_behavior_change_no_acceptance_or_launch"
    handler_path = out / "owned_output_attribution.py"
    assert handler_path.read_bytes() == Path(attribution.__file__).read_bytes()
    entry = subprocess.run(
        [sys.executable, "-c", "import owned_output_handoff; print('pinned_dependency_loaded')"],
        cwd=out,
        check=True,
        capture_output=True,
        text=True,
    )
    assert entry.stdout.strip() == "pinned_dependency_loaded"
    handler_path.write_bytes(
        handler_path.read_bytes() + b"\nPath(__file__).with_name('handler_executed').write_text('unsafe')\n"
    )
    refused = subprocess.run(
        [sys.executable, "-c", "import owned_output_handoff"],
        cwd=out,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert not (out / "handler_executed").exists()
    duplicate = subprocess.run(
        [
            sys.executable,
            str(Path(attribution.__file__)),
            "--parent",
            str(Path(__file__).with_name("owned_output_handoff.py")),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert duplicate.returncode != 0
    assert receipt == json.loads((out / "source.json").read_text())
