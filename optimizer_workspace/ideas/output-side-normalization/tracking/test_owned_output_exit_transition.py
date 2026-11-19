# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the measured live-to-zombie bracket without accepting leader exit alone."""

import ast
import hashlib
from pathlib import Path

import pytest

PARENT_TEST_SHA256 = "8a311baac576e9e99882afecd95cb1036511c05c6d9fcbf05014a40d23b6c5ad"


def inherited_tests() -> bytes:
    path = Path(__file__).with_name("test_owned_output_exit_drain.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_TEST_SHA256:
        raise ValueError("Exit-transition inherited lifecycle tests changed")
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "owned_output_exit_drain":
                    alias.name = "owned_output_exit_transition"
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("owned_output_exit_drain.py", "owned_output_exit_transition.py")
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and any(
            isinstance(key, ast.Constant) and key.value == "owned_output_handoff.py" for key in node.keys
        ):
            node.keys.append(ast.Constant("owned_output_exit_drain.py"))
            node.values.append(
                ast.parse('Path(drain.__file__).with_name("owned_output_exit_drain.py").read_bytes()', mode="eval").body
            )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())
process = globals()["process"]
expected = globals()["expected"]
drain = globals()["drain"]


@pytest.mark.parametrize("outcome", ("whole_exit", "live_thread", "live_exec"))
def test_mixed_exit_bracket_waits_for_whole_process_and_rejects_surviving_execution(diagnostic, monkeypatch, outcome):
    _, proc = diagnostic
    process(proc, 900, status="S", flags=4, vsize=0, command=b"", marker=False)
    read_text = Path.read_text
    reads = 0

    def bracket(path, *args, **kwargs):
        nonlocal reads
        result = read_text(path, *args, **kwargs)
        if path == proc / "900/stat":
            reads += 1
            if reads == 1:
                process(proc, 900, status="Z", flags=4, vsize=0, command=b"", marker=False)
        return result

    monkeypatch.setattr(Path, "read_text", bracket)
    monkeypatch.setattr(drain, "whole_process_exited", lambda root, owner: (root / "900/whole_group_exited").is_file())
    elapsed = 0

    def clock():
        return elapsed

    def pause(interval):
        nonlocal elapsed
        elapsed += interval
        if outcome == "whole_exit":
            (proc / "900/whole_group_exited").touch()
        elif outcome == "live_exec":
            process(proc, 900, status="R", command=b"replacement\0", marker=False)

    if outcome == "whole_exit":
        evidence = drain.drain_exit(proc, expected(), clock=clock, pause=pause)
        assert evidence[0]["status"] == "teardown"
        assert [entry["status"] for entry in evidence[0]["stats"]] == ["S", "Z"]
        assert evidence[-1]["status"] == "terminal"
        assert elapsed > 0
    else:
        with pytest.raises(RuntimeError):
            drain.drain_exit(proc, expected(), clock=clock, pause=pause)
        assert (proc / "900").exists()
        assert elapsed >= drain.EXIT_DRAIN_TIMEOUT if outcome == "live_thread" else elapsed > 0


@pytest.mark.parametrize("fault", ("owner", "memory", "flag"))
def test_mixed_exit_bracket_preserves_owner_memory_and_kernel_flag_checks(diagnostic, monkeypatch, fault):
    _, proc = diagnostic
    process(proc, 900, status="S", flags=4, vsize=0, command=b"", marker=False)
    read_text = Path.read_text
    changed = False

    def bracket(path, *args, **kwargs):
        nonlocal changed
        result = read_text(path, *args, **kwargs)
        if path == proc / "900/stat" and not changed:
            changed = True
            process(
                proc,
                900,
                status="Z",
                flags=0 if fault == "flag" else 4,
                vsize=4096 if fault == "memory" else 0,
                ticks=999 if fault == "owner" else 900,
                command=b"",
                marker=False,
            )
        return result

    monkeypatch.setattr(Path, "read_text", bracket)
    with pytest.raises(RuntimeError):
        drain.drain_exit(proc, expected())
    assert (proc / "900").exists()
