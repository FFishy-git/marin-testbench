# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify terminal-exit retries without accepting live ownership or partial absence."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import owned_output_attribution as attribution
import owned_output_exit_drain as drain
import pytest

PARENT = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/"
    "run-preparation/ns-training-path-marker-exit-recovery-20261119-cpu/20261119T1415/"
    "tracking/owned_output_handoff.py"
)


def process(proc, pid, *, status="S", flags=0, vsize=4096, ticks=None, parent=1, command=b"helper\0", marker=True):
    directory = proc / str(pid)
    directory.mkdir(exist_ok=True)
    fields = ["0"] * 21
    fields[:4] = [status, str(parent), str(pid), str(pid)]
    fields[6] = str(flags)
    fields[19] = str(pid if ticks is None else ticks)
    fields[20] = str(vsize)
    (directory / "stat").write_text(f"{pid} (fixture) {' '.join(fields)}")
    (directory / "cmdline").write_bytes(command)
    (directory / "environ").write_bytes(b"TOKEN=owned\0SECRET=hidden\0" if marker else b"")


def expected(pid=900):
    return {
        "pid": pid,
        "status": "S",
        "parent": 1,
        "group": pid,
        "session": pid,
        "start_ticks": str(pid),
        "command": ["helper"],
    }


@pytest.fixture
def diagnostic(tmp_path, monkeypatch):
    directory = tmp_path / "derived"
    directory.mkdir()
    sources = {
        "owned_output_handoff.py": drain.derived_source(PARENT),
        "owned_output_exit_drain.py": Path(drain.__file__).read_bytes(),
        "owned_output_attribution.py": Path(attribution.__file__).read_bytes(),
    }
    for name, content in sources.items():
        (directory / name).write_bytes(content)
    spec = importlib.util.spec_from_file_location("exit_drain_handoff", directory / "owned_output_handoff.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proc = tmp_path / "proc"
    proc.mkdir()
    module.PROC_ROOT = proc
    monkeypatch.setattr(
        module.exit_drain_handler,
        "whole_process_exited",
        lambda root, owner: (root / str(owner["pid"]) / "whole_group_exited").is_file(),
    )
    return module, proc


@pytest.mark.parametrize("terminal", ("absent", "Z"))
def test_empty_command_kernel_exit_retries_whole_scan_and_retains_new_child(diagnostic, monkeypatch, terminal):
    scan, proc = diagnostic
    process(proc, 900)
    process(proc, 901)
    read_bytes = Path.read_bytes
    changed = False
    stat_reads = 0

    def transition(path):
        nonlocal changed, stat_reads
        payload = read_bytes(path)
        if path == proc / "900/environ" and not changed:
            changed = True
            process(proc, 900, status="R", flags=4, vsize=0, command=b"", marker=False)
        return payload

    read_text = Path.read_text

    def terminal_transition(path, *args, **kwargs):
        nonlocal stat_reads
        value = read_text(path, *args, **kwargs)
        if changed and path == proc / "900/stat":
            stat_reads += 1
            if stat_reads == 4:
                process(proc, 902)
                if terminal == "absent":
                    shutil.rmtree(proc / "900")
                else:
                    process(proc, 900, status="Z", flags=4, vsize=0, command=b"", marker=False)
                    (proc / "900/whole_group_exited").touch()
        return value

    monkeypatch.setattr(Path, "read_bytes", transition)
    monkeypatch.setattr(Path, "read_text", terminal_transition)
    result = scan.marker_processes("TOKEN", "owned")
    assert set(result) == {901, 902}
    assert result[902]["command"] == ["helper"]


@pytest.mark.parametrize("fault", ("no_flag", "live_memory", "exec_command", "exec_environment", "parent", "pid_reuse"))
def test_empty_process_files_never_waive_missing_exit_or_changed_owner(diagnostic, fault):
    _, proc = diagnostic
    process(proc, 900, status="R", flags=4, vsize=0, command=b"", marker=False)
    if fault == "no_flag":
        process(proc, 900, status="S", vsize=0, command=b"", marker=False)
    elif fault == "live_memory":
        process(proc, 900, status="R", flags=4, command=b"", marker=False)
    elif fault == "exec_command":
        (proc / "900/cmdline").write_bytes(b"replacement\0")
    elif fault == "exec_environment":
        (proc / "900/environ").write_bytes(b"SECRET=hidden\0")
    else:
        process(
            proc,
            900,
            status="R",
            flags=4,
            vsize=0,
            command=b"",
            marker=False,
            parent=999 if fault == "parent" else 1,
            ticks=999 if fault == "pid_reuse" else 900,
        )
    with pytest.raises(RuntimeError):
        drain.drain_exit(proc, expected())


def test_persistent_teardown_exhausts_deadline_and_never_returns_absence(diagnostic):
    _, proc = diagnostic
    process(proc, 900, status="R", flags=4, vsize=0, command=b"", marker=False)
    elapsed = 0

    def clock():
        return elapsed

    def pause(interval):
        nonlocal elapsed
        elapsed += interval

    with pytest.raises(RuntimeError) as failure:
        drain.drain_exit(proc, expected(), clock=clock, pause=pause)
    assert "deadline exhausted" in str(failure.value)
    assert elapsed >= drain.EXIT_DRAIN_TIMEOUT
    assert (proc / "900").exists()


def test_missing_environment_file_is_not_process_absence(diagnostic):
    _, proc = diagnostic
    process(proc, 900, status="R", flags=4, vsize=0, command=b"", marker=False)
    (proc / "900/environ").unlink()
    with pytest.raises(RuntimeError):
        drain.drain_exit(proc, expected())
    assert (proc / "900/stat").exists()


def test_zombie_leader_with_surviving_threads_never_proves_terminal_exit(diagnostic, monkeypatch):
    _, proc = diagnostic
    process(proc, 900, status="Z", flags=4, vsize=0, command=b"", marker=False)
    elapsed = 0

    def clock():
        return elapsed

    def pause(interval):
        nonlocal elapsed
        elapsed += interval

    monkeypatch.setattr(drain, "whole_process_exited", lambda *args: False)
    with pytest.raises(RuntimeError):
        drain.drain_exit(proc, expected(), clock=clock, pause=pause)
    assert elapsed >= drain.EXIT_DRAIN_TIMEOUT


def test_drained_pid_reappearance_is_rejected_even_with_matching_start(diagnostic, monkeypatch):
    scan, proc = diagnostic
    process(proc, 900)
    process(proc, 901)
    read_bytes = Path.read_bytes
    changed = False
    reappeared = False

    def transition(path):
        nonlocal changed, reappeared
        payload = read_bytes(path)
        if path == proc / "900/environ" and not changed:
            changed = True
            process(proc, 900, status="R", flags=4, vsize=0, command=b"", marker=False)
        elif path == proc / "901/cmdline" and changed and not (proc / "900").exists() and not reappeared:
            reappeared = True
            process(proc, 900)
        return payload

    # Empty command is observed first, then directory disappearance is accepted only by the drain.
    read_text = Path.read_text
    reads = 0

    def disappearing(path, *args, **kwargs):
        nonlocal reads
        text = read_text(path, *args, **kwargs)
        if path == proc / "900/stat" and changed:
            reads += 1
            if reads == 4:
                shutil.rmtree(proc / "900")
        return text

    monkeypatch.setattr(Path, "read_text", disappearing)
    monkeypatch.setattr(Path, "read_bytes", transition)
    with pytest.raises(RuntimeError):
        scan.marker_processes("TOKEN", "owned")
    assert reappeared


def test_materialization_authenticates_exit_handler_before_side_effects(tmp_path):
    directory = tmp_path / "materialized"
    command = [
        sys.executable,
        str(Path(drain.__file__)),
        "--parent",
        str(PARENT),
        "--out",
        str(directory),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    receipt = json.loads(result.stdout)
    assert receipt == json.loads((directory / "source.json").read_text())
    loaded = subprocess.run(
        [sys.executable, "-c", "import owned_output_handoff"],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    assert loaded.returncode == 0
    path = directory / "owned_output_exit_drain.py"
    path.write_bytes(path.read_bytes() + b"\nPath(__file__).with_name('unsafe').write_text('executed')\n")
    refused = subprocess.run(
        [sys.executable, "-c", "import owned_output_handoff"],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert not (directory / "unsafe").exists()


def test_real_pidfd_confirms_whole_process_exit_before_reaping():
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; print('ready', flush=True); sys.stdin.buffer.read()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={},
        text=True,
    ) as child:
        try:
            assert child.stdout.readline().strip() == "ready"
            fields = (Path("/proc") / str(child.pid) / "stat").read_text().rsplit(")", 1)[1].split()
            owner = attribution.stat_identity(child.pid, fields)
            child.stdin.close()
            terminal = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
            assert terminal.si_status == 0
            assert drain.whole_process_exited(Path("/proc"), owner)
            assert drain.drain_exit(Path("/proc"), owner)[-1]["status"] == "terminal"
        finally:
            child.wait(timeout=10)
    assert not (Path("/proc") / str(child.pid)).exists()


def test_pidfd_in_different_procfs_pid_namespace_never_proves_owner_exit(tmp_path):
    directory = tmp_path / "self/fdinfo"
    directory.mkdir(parents=True)
    (directory / "7").write_text("pos:\t0\nflags:\t02\nPid:\t999\nNSpid:\t999 900\n")
    with pytest.raises(RuntimeError):
        drain.require_pidfd_identity(tmp_path, 7, 900)
