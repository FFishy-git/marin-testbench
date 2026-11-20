# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve missing-environment exit races before repeating the complete ownership scan."""

import ast
import hashlib
import os
import stat
import time
import types
from pathlib import Path

PARENT_SHA256 = "e3ae48eea72cd3d9828f06c407551b8f316716498994f54b9d5bb77a55fa9a59"
SCOPE = (
    "Ownership-only missing-environment exit correction. Require directory absence or "
    "identity-bound inherited whole-process exit before a fresh complete scan. Missing "
    "individual files, live drift, PID reuse and surviving threads remain refusals. "
    "No numerical, production, efficiency or SwanLab acceptance."
)
transition_path = Path(__file__).with_name("owned_output_exit_transition.py")
transition_bytes = transition_path.read_bytes()
if transition_path.is_symlink() or hashlib.sha256(transition_bytes).hexdigest() != PARENT_SHA256:
    raise ValueError("Missing-environment accepted mixed-bracket parent changed")
transition = types.ModuleType("missing_environment_transition_pinned")
transition.__file__ = str(transition_path)
exec(compile(transition_bytes, str(transition_path), "exec"), transition.__dict__)


def absent_directory(directory: Path) -> bool:
    """Do not treat an aliased or inaccessible process directory as absence."""
    try:
        current = directory.lstat()
    except (FileNotFoundError, ProcessLookupError):
        return True
    if not stat.S_ISDIR(current.st_mode):
        raise RuntimeError("Missing-environment process directory is not a direct directory")
    return False


def pinned_bytes(descriptor: int, name: str) -> bytes:
    file = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=descriptor)
    with os.fdopen(file, "rb") as stream:
        return stream.read()


def require_directory_identity(directory: Path, descriptor: int) -> bool:
    try:
        current = directory.lstat()
    except (FileNotFoundError, ProcessLookupError):
        return False
    pinned = os.fstat(descriptor)
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
        raise RuntimeError("Missing-environment process directory identity changed")
    return True


def missing_environment_exit(root: Path, expected: dict, *, clock=time.monotonic, pause=time.sleep) -> list[dict]:
    """Drain only a bound kernel-exit observation; an unavailable environment is not empty."""
    directory = root / str(expected["pid"])
    deadline = clock() + transition.EXIT_DRAIN_TIMEOUT
    evidence = []
    try:
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except (FileNotFoundError, ProcessLookupError):
        if absent_directory(directory):
            return [{"status": "absent", "pid": expected["pid"]}]
        raise RuntimeError("Missing-environment directory open failed while process remains") from None
    try:
        while True:
            if not require_directory_identity(directory, descriptor):
                return [*evidence, {"status": "absent", "pid": expected["pid"]}]
            try:
                before = pinned_bytes(descriptor, "stat").decode().rsplit(")", 1)[1].split()
                command = pinned_bytes(descriptor, "cmdline")
                after = pinned_bytes(descriptor, "stat").decode().rsplit(")", 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                if not require_directory_identity(directory, descriptor):
                    return [*evidence, {"status": "absent", "pid": expected["pid"]}]
                raise RuntimeError("Missing-environment exit lacks complete process identity") from None
            if len(before) < 21 or len(after) < 21:
                raise RuntimeError("Missing-environment exit lacks complete kernel stat")
            identities = [transition.attribution.stat_identity(expected["pid"], fields) for fields in (before, after)]
            if any(any(row[key] != expected[key] for key in transition.OWNER_FIELDS) for row in identities):
                raise RuntimeError("Missing-environment exit process ownership changed")
            if (
                command
                or any(int(fields[20]) for fields in (before, after))
                or not all(row["pf_exiting"] or row["status"] in ("Z", "X", "x") for row in identities)
            ):
                raise RuntimeError("Missing-environment observation does not establish kernel teardown")
            try:
                environment = pinned_bytes(descriptor, "environ")
            except (FileNotFoundError, ProcessLookupError):
                environment = None
            if not require_directory_identity(directory, descriptor):
                return [*evidence, {"status": "absent", "pid": expected["pid"]}]
            if environment is not None:
                if environment:
                    raise RuntimeError("Missing-environment exit has a nonempty environment")
                return [*evidence, *transition.drain_exit(root, expected, clock=clock, pause=pause)]
            evidence.append({"status": "bound_teardown_environment_unavailable", "pid": expected["pid"]})
            if clock() >= deadline:
                raise RuntimeError("Missing-environment exit deadline exhausted without whole-process proof")
            pause(transition.EXIT_DRAIN_INTERVAL)
    finally:
        os.close(descriptor)


def runtime_source() -> bytes:
    """Retain the pinned mixed-bracket implementation, adding only missing-file resolution."""
    path = Path(__file__).with_name("owned_output_exit_transition.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Missing-environment accepted mixed-bracket parent changed")
    tree = ast.parse(transition.runtime_source())
    replacement = ast.parse(
        """identities.setdefault(pid, process)
exit_drain_handler.missing_environment_exit(PROC_ROOT, identities[pid])
drained.add(pid)
unstable.add(pid)
continue
"""
    ).body
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "MARKER_SCAN" for target in node.targets
        ):
            scan = ast.parse(ast.literal_eval(node.value))
            for handler in ast.walk(scan):
                if isinstance(handler, ast.ExceptHandler) and ast.dump(handler.type) == ast.dump(
                    ast.parse("(FileNotFoundError, ProcessLookupError)", mode="eval").body
                ):
                    handler.body = replacement
                    changed += 1
            node.value = ast.Constant(ast.unparse(ast.fix_missing_locations(scan)) + "\n")
    if changed != 1:
        raise ValueError("Missing-environment parent scan refusal boundary differs")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = node.value.replace("owned_output_exit_transition.py", "owned_output_missing_environment.py")
    boundary = next(index for index, node in enumerate(tree.body) if isinstance(node, ast.FunctionDef))
    tree.body[boundary:boundary] = ast.parse(
        """import stat
import types
"""
    ).body
    boundary += 2
    tree.body[boundary:boundary] = ast.parse(
        f"""transition_path = Path(__file__).with_name("owned_output_exit_transition.py")
transition_bytes = transition_path.read_bytes()
if transition_path.is_symlink() or hashlib.sha256(transition_bytes).hexdigest() != {PARENT_SHA256!r}:
    raise ValueError("Missing-environment accepted mixed-bracket parent changed")
transition = types.ModuleType("missing_environment_transition_pinned")
transition.__file__ = str(transition_path)
exec(compile(transition_bytes, str(transition_path), "exec"), transition.__dict__)
"""
    ).body
    boundary += 7
    own = ast.parse(Path(__file__).read_bytes())
    tree.body[boundary:boundary] = [
        node
        for node in own.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("absent_directory", "pinned_bytes", "require_directory_identity", "missing_environment_exit")
    ]
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name != "main":
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Tuple) and any(
                isinstance(element, ast.Call)
                and isinstance(element.func, ast.Name)
                and element.func.id == "Path"
                and any(isinstance(argument, ast.Name) and argument.id == "__file__" for argument in element.args)
                for element in node.elts
            ):
                node.elts.append(
                    ast.parse('Path(__file__).with_name("owned_output_exit_transition.py")', mode="eval").body
                )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
