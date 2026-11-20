# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify missing-environment resolution through the complete source-derived marker scan."""

import ast
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import owned_output_attribution as attribution
import owned_output_exit_transition as transition
import owned_output_missing_environment as missing
import pytest

PARENT = Path(
    "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization/"
    "run-preparation/ns-training-path-marker-exit-recovery-20261119-cpu/20261119T1415/"
    "tracking/owned_output_handoff.py"
)
PARENT_TEST_SHA256 = "edfc521694492c706013d7288114b14040f499c876a2ad15283fef96a6fdc090"


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_owned_output_exit_transition.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_TEST_SHA256:
        raise ValueError("Missing-environment accepted lifecycle tests changed")
    tree = ast.parse(payload)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "inherited_tests")
    parent_pin = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PARENT_TEST_SHA256" for target in node.targets)
    )
    namespace = {"__file__": str(path), "ast": ast, "hashlib": hashlib, "Path": Path, "PARENT_TEST_SHA256": parent_pin}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    base = ast.parse(namespace["inherited_tests"]())
    base.body.extend(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_mixed_exit")
    )
    for node in ast.walk(base):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "owned_output_exit_transition":
                    alias.name = "owned_output_missing_environment"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("owned_output_exit_transition.py", "owned_output_missing_environment.py")
    return (ast.unparse(ast.fix_missing_locations(base)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())


@pytest.fixture
def diagnostic(tmp_path, monkeypatch):
    directory = tmp_path / "derived"
    directory.mkdir()
    namespace = {"__file__": str(Path(missing.__file__).resolve())}
    exec(compile(missing.runtime_source(), missing.__file__, "exec"), namespace)
    sources = {
        "owned_output_handoff.py": namespace["derived_source"](PARENT),
        "owned_output_missing_environment.py": Path(missing.__file__).read_bytes(),
        "owned_output_exit_transition.py": Path(transition.__file__).read_bytes(),
        "owned_output_exit_drain.py": Path(transition.__file__).with_name("owned_output_exit_drain.py").read_bytes(),
        "owned_output_attribution.py": Path(attribution.__file__).read_bytes(),
    }
    for name, content in sources.items():
        (directory / name).write_bytes(content)
    spec = importlib.util.spec_from_file_location("missing_environment_handoff", directory / "owned_output_handoff.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proc = tmp_path / "proc"
    proc.mkdir()
    module.PROC_ROOT = proc
    monkeypatch.setattr(
        module.exit_drain_handler.transition,
        "whole_process_exited",
        lambda root, owner: (root / str(owner["pid"]) / "whole_group_exited").is_file(),
    )
    monkeypatch.setattr(
        module.exit_drain_handler,
        "whole_process_exited",
        lambda root, owner: (root / str(owner["pid"]) / "whole_group_exited").is_file(),
    )
    return module, proc


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
    (directory / "environ").write_bytes(b"TOKEN=owned\0" if marker else b"")


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


@pytest.mark.parametrize("terminal", ("directory_absent", "bound_teardown"))
def test_missing_environment_complete_rescan_retains_new_marked_child(diagnostic, monkeypatch, terminal):
    scan, proc = diagnostic
    process(proc, 900, marker=False)
    process(proc, 901)
    read_bytes = Path.read_bytes
    changed = False

    def disappearing(path):
        nonlocal changed
        if path == proc / "900/environ" and not changed:
            changed = True
            process(proc, 902)
            if terminal == "directory_absent":
                shutil.rmtree(proc / "900")
            else:
                process(proc, 900, status="R", flags=4, vsize=0, command=b"", marker=False)
                (proc / "900/environ").unlink()
            raise FileNotFoundError(str(path))
        return read_bytes(path)

    elapsed = 0

    def clock():
        return elapsed

    def pause(interval):
        nonlocal elapsed
        elapsed += interval
        shutil.rmtree(proc / "900")

    original = scan.exit_drain_handler.missing_environment_exit
    monkeypatch.setattr(
        scan.exit_drain_handler,
        "missing_environment_exit",
        lambda root, owner: original(root, owner, clock=clock, pause=pause),
    )
    monkeypatch.setattr(Path, "read_bytes", disappearing)
    result = scan.marker_processes("TOKEN", "owned")
    assert set(result) == {901, 902}
    assert changed
    assert elapsed > 0 if terminal == "bound_teardown" else elapsed == 0


@pytest.mark.parametrize("fault", ("live", "owner", "memory", "flag", "stat", "environment", "alias"))
def test_missing_environment_never_accepts_partial_live_or_unbound_process(diagnostic, fault):
    _, proc = diagnostic
    process(proc, 900, status="R", flags=4, vsize=0, command=b"", marker=False)
    (proc / "900/environ").unlink()
    if fault == "live":
        (proc / "900/cmdline").write_bytes(b"live\0")
    elif fault == "owner":
        process(proc, 900, status="R", flags=4, vsize=0, ticks=999, command=b"", marker=False)
        (proc / "900/environ").unlink()
    elif fault == "memory":
        process(proc, 900, status="R", flags=4, command=b"", marker=False)
        (proc / "900/environ").unlink()
    elif fault == "flag":
        process(proc, 900, status="S", vsize=0, command=b"", marker=False)
        (proc / "900/environ").unlink()
    elif fault == "stat":
        (proc / "900/stat").unlink()
    elif fault == "environment":
        (proc / "900/environ").write_bytes(b"TOKEN=unowned\0")
    else:
        (proc / "900").rename(proc / "target")
        (proc / "900").symlink_to(proc / "target", target_is_directory=True)
    with pytest.raises((RuntimeError, OSError)):
        missing.missing_environment_exit(proc, expected())
    assert (proc / "900").exists()


def test_missing_environment_surviving_thread_exhausts_original_deadline(diagnostic):
    _, proc = diagnostic
    process(proc, 900, status="Z", flags=4, vsize=0, command=b"", marker=False)
    (proc / "900/environ").unlink()
    elapsed = 0

    def clock():
        return elapsed

    def pause(interval):
        nonlocal elapsed
        elapsed += interval

    with pytest.raises(RuntimeError):
        missing.missing_environment_exit(proc, expected(), clock=clock, pause=pause)
    assert elapsed >= transition.EXIT_DRAIN_TIMEOUT
    assert (proc / "900").exists()


def test_missing_environment_pinned_directory_replacement_refuses_same_pid_and_ticks(diagnostic, monkeypatch):
    _, proc = diagnostic
    process(proc, 900, status="R", flags=4, vsize=0, command=b"", marker=False)
    (proc / "900/environ").unlink()

    def pause(interval):
        (proc / "900").rename(proc / "old")
        process(proc, 900, status="R", flags=4, vsize=0, command=b"", marker=False)
        (proc / "900/environ").unlink()

    with pytest.raises(RuntimeError):
        missing.missing_environment_exit(proc, expected(), pause=pause)
    assert (proc / "old/stat").exists()
    assert (proc / "900/stat").exists()


def test_missing_environment_materialization_binds_all_handlers_before_execution(tmp_path):
    directory = tmp_path / "materialized"
    result = subprocess.run(
        [sys.executable, missing.__file__, "--parent", str(PARENT), "--out", str(directory)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    env = os.environ | {"PYTHONPATH": str(directory)}
    loaded = subprocess.run(
        [sys.executable, "-P", "-c", "import owned_output_handoff"],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert loaded.returncode == 0, loaded.stderr
    path = directory / "owned_output_exit_transition.py"
    path.write_bytes(path.read_bytes() + b"\nPath(__file__).with_name('unsafe').touch()\n")
    refused = subprocess.run(
        [sys.executable, "-P", "-c", "import owned_output_handoff"],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert refused.returncode != 0
    assert not (directory / "unsafe").exists()
