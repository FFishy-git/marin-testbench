# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise complete rechecks after proven exits and retain refusal boundaries."""

import ast
import hashlib
import importlib.util
import shutil
from pathlib import Path

import owned_output_missing_environment as original
import owned_output_recheck_exit as successor
import pytest

PARENT_TEST_SHA256 = "0ff9216a327964c805fcfd42656ddd54b9965fc0e9900ceb43275882c837e254"
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
    (directory / "environ").write_bytes(b"TOKEN=owned\0" if marker else b"")


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_owned_output_missing_environment.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_TEST_SHA256:
        raise ValueError("Recheck-exit accepted missing-environment tests changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    tree = ast.parse(payload)
    inherited = ast.parse(namespace["inherited_tests"]())
    tree.body = [
        node
        for node in tree.body
        if not (
            (isinstance(node, ast.FunctionDef) and node.name in ("inherited_tests", "process"))
            or (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "exec"
            )
        )
    ]
    tree.body = inherited.body + tree.body
    fixture = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "diagnostic")
    assignment = next(
        node for node in fixture.body if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "sources"
    )
    assignment.value = ast.parse(
        """{
    "owned_output_handoff.py": namespace["derived_source"](PARENT),
    **{name: Path(missing.__file__).with_name(name).read_bytes() for name in (
        "owned_output_recheck_exit.py", "owned_output_missing_environment.py",
        "owned_output_exit_transition.py", "owned_output_exit_drain.py",
        "owned_output_attribution.py"
    )}
}""",
        mode="eval",
    ).body
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "owned_output_missing_environment":
                    alias.name = "owned_output_recheck_exit"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("owned_output_missing_environment.py", "owned_output_recheck_exit.py")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())


def handoff(diagnostic, module):
    scan, proc = diagnostic
    namespace = {"__file__": str(Path(module.__file__).resolve())}
    exec(compile(module.runtime_source(), module.__file__, "exec"), namespace)
    path = Path(scan.__file__)
    path.write_bytes(namespace["derived_source"](PARENT))
    if module is original:
        path.with_name("owned_output_missing_environment.py").write_bytes(Path(original.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location(f"recheck_{module.__name__}", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    loaded.PROC_ROOT = proc
    return loaded


@pytest.mark.parametrize("observed_state", ("disappeared", "terminal"))
def test_recheck_retains_new_marked_child_after_bound_exit(diagnostic, monkeypatch, observed_state):
    scan, proc = diagnostic
    process(proc, 900)
    process(proc, 901)
    observe = scan.observed
    elapsed, changed, observations = 0, False, 0
    drain = scan.exit_drain_handler.drain_exit
    missing = scan.exit_drain_handler.missing_environment_exit

    def clock():
        return elapsed

    def pause(interval):
        nonlocal elapsed
        elapsed += interval
        shutil.rmtree(proc / "900")

    def recheck(pid):
        nonlocal changed, observations
        if pid == 900:
            observations += 1
        if pid == 900 and observations == 2 and not changed:
            changed = True
            process(proc, 902)
            process(proc, 900, status="Z", flags=4, vsize=0, command=b"", marker=False)
            return None if observed_state == "disappeared" else observe(pid)
        return observe(pid)

    monkeypatch.setattr(scan, "observed", recheck)
    monkeypatch.setattr(
        scan.exit_drain_handler, "drain_exit", lambda root, owner, **kwargs: drain(root, owner, clock=clock, pause=pause)
    )
    monkeypatch.setattr(
        scan.exit_drain_handler,
        "missing_environment_exit",
        lambda root, owner: missing(root, owner, clock=clock, pause=pause),
    )
    monkeypatch.setattr(scan.exit_drain_handler.transition, "drain_exit", scan.exit_drain_handler.drain_exit)
    result = scan.marker_processes("TOKEN", "owned")
    assert set(result) == {901, 902}
    assert elapsed > 0 and not (proc / "900").exists()


@pytest.mark.parametrize("observed_state", ("disappeared", "terminal"))
def test_recheck_surviving_thread_refuses_original_deadline(diagnostic, monkeypatch, observed_state):
    scan, proc = diagnostic
    process(proc, 900)
    observe, drain = scan.observed, scan.exit_drain_handler.drain_exit
    missing = scan.exit_drain_handler.missing_environment_exit
    elapsed, changed, observations = 0, False, 0

    def clock():
        return elapsed

    def pause(interval):
        nonlocal elapsed
        elapsed += interval

    def recheck(pid):
        nonlocal changed, observations
        if pid == 900:
            observations += 1
        if pid == 900 and observations == 2 and not changed:
            changed = True
            process(proc, 900, status="Z", flags=4, vsize=0, command=b"", marker=False)
            return None if observed_state == "disappeared" else observe(pid)
        return observe(pid)

    monkeypatch.setattr(scan, "observed", recheck)
    monkeypatch.setattr(
        scan.exit_drain_handler, "drain_exit", lambda root, owner, **kwargs: drain(root, owner, clock=clock, pause=pause)
    )
    monkeypatch.setattr(
        scan.exit_drain_handler,
        "missing_environment_exit",
        lambda root, owner: missing(root, owner, clock=clock, pause=pause),
    )
    monkeypatch.setattr(scan.exit_drain_handler.transition, "drain_exit", scan.exit_drain_handler.drain_exit)
    with pytest.raises(RuntimeError):
        scan.marker_processes("TOKEN", "owned")
    assert elapsed >= successor.EXIT_DRAIN_TIMEOUT
    assert (proc / "900").exists()


@pytest.mark.parametrize("fault", ("partial_file", "reused_pid", "terminal_owner_drift"))
def test_recheck_incomplete_or_reused_process_cannot_supply_absence(diagnostic, monkeypatch, fault):
    scan, proc = diagnostic
    process(proc, 900)
    observe, changed, observations = scan.observed, False, 0

    def recheck(pid):
        nonlocal changed, observations
        if pid == 900:
            observations += 1
        if pid == 900 and observations == 2 and not changed:
            changed = True
            if fault == "partial_file":
                (proc / "900/stat").unlink()
                return None
            process(proc, 900, status="Z", flags=4, vsize=0, ticks=999, command=b"", marker=False)
            return observe(pid) if fault == "terminal_owner_drift" else None
        return observe(pid)

    monkeypatch.setattr(scan, "observed", recheck)
    with pytest.raises(RuntimeError):
        scan.marker_processes("TOKEN", "owned")
    assert (proc / "900").exists()


def test_original_recheck_failure_is_reproduced_without_changing_accepted_source(diagnostic, monkeypatch):
    scan = handoff(diagnostic, original)
    _, proc = diagnostic
    process(proc, 900)
    process(proc, 901)
    observe = scan.observed
    observations = 0

    def recheck(pid):
        nonlocal observations
        if pid == 900:
            observations += 1
            if observations % 2 == 0:
                return None
        return observe(pid)

    monkeypatch.setattr(scan, "observed", recheck)
    with pytest.raises(RuntimeError, match="3 attempts"):
        scan.marker_processes("TOKEN", "owned")
    assert (proc / "900/stat").exists()
    assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() == successor.PARENT_SOURCE_SHA256
