# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drain kernel-confirmed exit teardown before repeating a complete ownership scan."""

import argparse
import ast
import ctypes
import hashlib
import json
import os
import platform
import select
import time
from pathlib import Path

import owned_output_attribution as attribution

EXIT_DRAIN_TIMEOUT = 1.0
EXIT_DRAIN_INTERVAL = 0.001
OWNER_FIELDS = ("pid", "start_ticks", "parent", "group", "session")
PIDFD_OPEN_X86_64 = 434
MARKER_SCAN = '''def marker_processes(name: str, value: str) -> dict[int, dict]:
    """Repeat complete scans after verified exits, never after unexplained live drift."""
    token = f"{name}={value}".encode()
    identities, marked, drained = {}, set(), set()
    for _ in range(MARKER_SCAN_ATTEMPTS):
        result, unstable = {}, set()
        for pid, process in live_processes().items():
            if pid in drained:
                raise observation_error("scan_identity_drift",
                    f"Drained process PID reappeared during handoff: pid={pid}",
                    locals() | {"PROC_ROOT": PROC_ROOT})
            if process["status"] in EXIT_STATES:
                continue
            if pid in identities and not same_generation(identities[pid], process):
                if exit_drain_handler.empty_command_transition(identities[pid], process):
                    exit_drain_handler.drain_exit(PROC_ROOT, identities[pid])
                    drained.add(pid)
                    unstable.add(pid)
                    continue
                raise observation_error("scan_identity_drift",
                    f"Marked process generation changed during handoff: pid={pid}",
                    locals() | {"PROC_ROOT": PROC_ROOT})
            try:
                tokens = (PROC_ROOT / str(pid) / "environ").read_bytes().split(b"\\0")
            except (FileNotFoundError, ProcessLookupError):
                identities.setdefault(pid, process)
                unstable.add(pid)
                continue
            if token not in tokens:
                if pid in marked:
                    if tokens == [b""]:
                        exit_drain_handler.drain_exit(PROC_ROOT, identities[pid])
                        drained.add(pid)
                        unstable.add(pid)
                        continue
                    raise observation_error("scan_marker_removed",
                        f"Marked process environment changed during handoff: pid={pid}",
                        locals() | {"PROC_ROOT": PROC_ROOT})
                continue
            identities.setdefault(pid, process)
            marked.add(pid)
            result[pid] = process
        for pid, process in result.items():
            current = observed(pid)
            if current is None:
                unstable.add(pid)
            elif current["status"] in EXIT_STATES:
                if any(current[key] != process[key] for key in ("pid", "start_ticks", "parent", "group", "session")):
                    raise observation_error("recheck_exit_owner_drift",
                        f"Marked process generation changed during handoff: pid={pid}",
                        locals() | {"PROC_ROOT": PROC_ROOT})
                unstable.add(pid)
            elif not same_generation(process, current):
                if exit_drain_handler.empty_command_transition(process, current):
                    exit_drain_handler.drain_exit(PROC_ROOT, process)
                    drained.add(pid)
                    unstable.add(pid)
                    continue
                raise observation_error("recheck_identity_drift",
                    f"Marked process generation changed during handoff: pid={pid}",
                    locals() | {"PROC_ROOT": PROC_ROOT})
        if not unstable:
            for pid in drained:
                exit_drain_handler.drain_exit(PROC_ROOT, identities[pid])
            return result
    raise observation_error("complete_scan_exhausted",
        f"Marked process scan remained unstable during handoff after {MARKER_SCAN_ATTEMPTS} attempts: "
        f"pids={sorted(unstable)}",
        locals() | {"PROC_ROOT": PROC_ROOT})
'''


def empty_command_transition(before: dict, after: dict) -> bool:
    """Recognize only command loss with all ownership fields unchanged."""
    return bool(before["command"]) and not after["command"] and all(before[key] == after[key] for key in OWNER_FIELDS)


def exit_sample(root: Path, expected: dict) -> dict:
    """Bracket empty process files with exact ownership and kernel teardown state."""
    directory = root / str(expected["pid"])
    try:
        before = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        command = (directory / "cmdline").read_bytes()
        environment = (directory / "environ").read_bytes()
        after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        # A missing individual file is not proof that its process directory disappeared.
        if directory.exists():
            raise RuntimeError(f"Exit drain process files are incomplete: pid={expected['pid']}") from None
        return {"status": "absent", "pid": expected["pid"]}
    if len(before) < 21 or len(after) < 21:
        raise RuntimeError(f"Exit drain lacks complete kernel stat evidence: pid={expected['pid']}")
    snapshots = [attribution.stat_identity(expected["pid"], fields) for fields in (before, after)]
    evidence = {
        "status": "observed",
        "pid": expected["pid"],
        "stats": snapshots,
        "command_bytes": len(command),
        "environment_bytes": len(environment),
        "vsize": [int(fields[20]) for fields in (before, after)],
    }
    if any(any(sample[key] != expected[key] for key in OWNER_FIELDS) for sample in snapshots):
        raise RuntimeError("Exit drain ownership changed: " + json.dumps(evidence, sort_keys=True))
    if command or environment:
        raise RuntimeError("Exit drain process files are nonempty: " + json.dumps(evidence, sort_keys=True))
    terminal = all(sample["status"] in ("Z", "X", "x") for sample in snapshots)
    if terminal:
        evidence["status"] = "terminal"
        return evidence
    if (
        any(sample["status"] in ("Z", "X", "x") for sample in snapshots)
        or not all(sample["pf_exiting"] for sample in snapshots)
        or any(evidence["vsize"])
    ):
        raise RuntimeError("Exit drain lacks bracketed kernel teardown: " + json.dumps(evidence, sort_keys=True))
    evidence["status"] = "teardown"
    return evidence


def whole_process_exited(root: Path, expected: dict) -> bool:
    """Use process-wide pidfd readiness; a zombie leader alone is insufficient."""
    if root != Path("/proc"):
        raise RuntimeError("Whole-process terminal proof requires the actual Linux procfs")
    if int((root / "self/stat").read_text().split(" ", 1)[0]) != os.getpid():
        raise RuntimeError("Exit drain procfs and pidfd PID namespaces differ")
    if platform.machine() != "x86_64":
        raise RuntimeError("Exit drain pidfd syscall requires the measured x86_64 host")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    descriptor = libc.syscall(ctypes.c_long(PIDFD_OPEN_X86_64), ctypes.c_int(expected["pid"]), ctypes.c_uint(0))
    if descriptor < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    try:
        # Bind the newly opened handle to the same still-present process generation.
        sample = exit_sample(root, expected)
        if sample["status"] == "absent":
            return True
        require_pidfd_identity(root, descriptor, expected["pid"])
        poll = select.poll()
        poll.register(descriptor, select.POLLIN)
        ready = any(flags & select.POLLIN for _, flags in poll.poll(0))
        final = exit_sample(root, expected)
        if final["status"] == "absent":
            return True
        require_pidfd_identity(root, descriptor, expected["pid"])
        return ready and final["status"] == "terminal"
    finally:
        os.close(descriptor)


def require_pidfd_identity(root: Path, descriptor: int, pid: int) -> None:
    """Compare pidfd's procfs-mount PID with the observed process directory."""
    rows = (root / "self/fdinfo" / str(descriptor)).read_text().splitlines()
    fields = {name: value.strip() for row in rows for name, separator, value in [row.partition(":")] if separator}
    if fields.get("Pid") != str(pid):
        raise RuntimeError(f"Exit drain pidfd and procfs identities differ: expected_pid={pid}")


def drain_exit(root: Path, expected: dict, *, clock=time.monotonic, pause=time.sleep) -> list[dict]:
    """Wait for directory absence or identity-bound process-wide terminal proof."""
    deadline = clock() + EXIT_DRAIN_TIMEOUT
    evidence = []
    while True:
        sample = exit_sample(root, expected)
        evidence.append(sample)
        if sample["status"] == "absent":
            return evidence
        if sample["status"] == "terminal":
            try:
                complete = whole_process_exited(root, expected)
            except ProcessLookupError:
                if (root / str(expected["pid"])).exists():
                    raise RuntimeError("Exit drain process handle lookup failed while directory remains") from None
                complete = True
            if complete:
                return evidence
        if clock() >= deadline:
            raise RuntimeError("Exit drain deadline exhausted: " + json.dumps(evidence[-1], sort_keys=True))
        pause(EXIT_DRAIN_INTERVAL)


def derived_source(parent: Path) -> bytes:
    """Derive the separately attributed scan with only a terminal-exit retry boundary."""
    tree = ast.parse(attribution.derived_source(parent))
    functions = [
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == "marker_processes"
    ]
    if len(functions) != 1:
        raise ValueError("Exit-drain parent scan boundary differs")
    tree.body[functions[0]] = ast.parse(MARKER_SCAN).body[0]
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    header = ast.parse(
        f"""exit_drain_path = AttributionPath(__file__).with_name("owned_output_exit_drain.py")
if exit_drain_path.is_symlink():
    raise RuntimeError("Exit drain handler is aliased")
exit_drain_bytes = exit_drain_path.read_bytes()
if attribution_hashlib.sha256(exit_drain_bytes).hexdigest() != {digest!r}:
    raise RuntimeError("Exit drain handler source identity differs")
exit_drain_handler = attribution_types.ModuleType("owned_output_exit_drain_pinned")
exit_drain_handler.__file__ = str(exit_drain_path)
exit_drain_handler.attribution = attribution_handler
import ast as exit_drain_ast
exit_drain_tree = exit_drain_ast.parse(exit_drain_bytes)
exit_drain_tree.body = [
    node for node in exit_drain_tree.body
    if not (isinstance(node, exit_drain_ast.Import)
        and len(node.names) == 1 and node.names[0].name == "owned_output_attribution")
]
exec(compile(exit_drain_tree, str(exit_drain_path), "exec"), exit_drain_handler.__dict__)
"""
    ).body
    boundary = next(
        index + 1
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "observation_error" for target in node.targets)
    )
    tree.body[boundary:boundary] = header
    return (
        "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
        + ast.unparse(ast.fix_missing_locations(tree))
        + "\n"
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = derived_source(args.parent)
    args.out.mkdir(parents=True, exist_ok=False)
    files = {"owned_output_handoff.py": payload}
    files |= {path.name: path.read_bytes() for path in (Path(__file__), Path(attribution.__file__))}
    for name, content in files.items():
        with (args.out / name).open("xb") as stream:
            stream.write(content)
    receipt = {
        "status": "exit_drain_materialized_cpu_diagnostic_gpu_unissued",
        "parent_sha256": attribution.PARENT_SHA256,
        "files_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
        "timeout": EXIT_DRAIN_TIMEOUT,
        "scope": "Kernel-confirmed teardown must reach terminal/absent before full rescan; no GPU acceptance.",
    }
    with (args.out / "source.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
