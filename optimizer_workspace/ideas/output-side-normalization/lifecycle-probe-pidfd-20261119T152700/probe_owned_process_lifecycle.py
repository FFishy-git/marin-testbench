# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain bounded, synthetic Linux lifecycle evidence without granting ownership.

Run with --out pointing to an unused directory. The exec control includes both
leader exec and a nonleader exec following leader-thread exit. Group rescans are
complete within the private probe scope; they are not a production marker scan.
"""

import argparse
import ctypes
import hashlib
import json
import math
import mmap
import os
import select
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from subprocess import DEVNULL, PIPE, Popen

MARKER_NAME = "LIFECYCLE_MARKER"
MARKER_VALUE = "synthetic"
MARKER_TOKEN = f"{MARKER_NAME}={MARKER_VALUE}".encode()
PF_EXITING = 0x00000004
PIDFD_OPEN_X86_64 = 434
EXIT_STATES = ("Z", "X", "x")
EXIT_MAPS = 8192
EXIT_STATUS = 37
DEFAULT_TIMEOUT = 10.0
CLEANUP_TIMEOUT = 10.0
MAX_EXIT_ATTEMPTS = 4
MAX_PROTOCOL_BYTES = 4096
IDENTITY_FIELDS = ("pid", "start_ticks", "parent", "group", "session")
LIMITATIONS = [
    "Synthetic CPU controls establish possible mechanisms, not the recovery failure's cause.",
    "8192 anonymous mappings widen exit teardown; no training frequency is measured.",
    "A missed pre-zombie window is inconclusive and is retained as not observed.",
    "PF_EXITING describes a task; nonleader exec can preserve the process PID and start time.",
    "Pidfd readiness is measured only for owned synthetic processes through x86_64 syscall 434.",
    "Separate /proc reads and complete group rescans are not atomic kernel snapshots.",
    "Rescans cover only private probe process groups, not all production marked owners.",
    "A changed cmdline or missing marker cannot alone distinguish exec, mutation, or teardown.",
    "No GPU validation, successor acceptance, production result, or kernel-cause attribution is issued.",
]
KERNEL_REFERENCES = {
    "repository": "gregkh/linux",
    "tag": "v5.4.241",
    "fetched_by_this_probe": False,
    "anchors": {
        "kernel/exit.c": [530, 707, 814, 845, 879],
        "fs/proc/base.c": [346, 348, 939, 948],
        "fs/exec.c": [1061, 1168, 1169, 1185, 1189],
    },
    "limit": "Previously inspected upstream source; not the installed vendor kernel source.",
}


def timestamp() -> dict:
    wall = time.time_ns()
    instant = datetime.fromtimestamp(wall / 1_000_000_000, UTC)
    return {
        "unix_ns": wall,
        "utc": instant.isoformat(timespec="microseconds"),
        "local": instant.astimezone().isoformat(timespec="microseconds"),
        "monotonic_ns": time.monotonic_ns(),
    }


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def remaining(deadline: float) -> float:
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("Synthetic lifecycle control deadline expired")
    return value


def process_stat(path: Path) -> dict:
    """Parse identity and teardown fields without retaining comm or raw stat."""
    payload = path.read_text()
    fields = payload.rsplit(")", 1)[1].split()
    return {
        "pid": int(payload.split(" ", 1)[0]),
        "status": fields[0],
        "parent": int(fields[1]),
        "group": int(fields[2]),
        "session": int(fields[3]),
        "flags": int(fields[6]),
        "pf_exiting": bool(int(fields[6]) & PF_EXITING),
        "start_ticks": fields[19],
        "vsize": int(fields[20]),
        "exit_code": int(fields[49]),
    }


def read_summary(path: Path, *, environment: bool = False) -> dict:
    try:
        payload = path.read_bytes()
    except OSError as error:
        return {"errno": error.errno, "read_error": type(error).__name__}
    tokens = payload.split(b"\0")
    result = {"byte_count": len(payload), "entry_count": sum(bool(token) for token in tokens)}
    if environment:
        return result | {"marker_present": MARKER_TOKEN in tokens}
    return result | {"sha256": sha256(payload)}


def snapshot(directory: Path) -> dict:
    """Bracket redacted memory reads with independently timestamped stat reads."""
    sample = {"started": timestamp()}
    for label, leaf in (("stat_before", "stat"), ("command", "cmdline"), ("environment", "environ")):
        try:
            sample[label] = (
                process_stat(directory / leaf)
                if leaf == "stat"
                else read_summary(directory / leaf, environment=leaf == "environ")
            )
        except OSError as error:
            sample[label] = {"errno": error.errno, "read_error": type(error).__name__}
        if leaf == "stat":
            sample[label]["timestamp"] = timestamp()
    try:
        executable = os.readlink(directory / "exe")
        sample["executable"] = {"path_sha256": sha256(os.fsencode(executable))}
    except OSError as error:
        sample["executable"] = {"errno": error.errno, "read_error": type(error).__name__}
    try:
        sample["stat_after"] = process_stat(directory / "stat")
    except OSError as error:
        sample["stat_after"] = {"errno": error.errno, "read_error": type(error).__name__}
    sample["stat_after"]["timestamp"] = timestamp()
    sample["finished"] = timestamp()
    return sample


def same_identity(before: dict, after: dict) -> bool:
    return all(field in before and field in after and before[field] == after[field] for field in IDENTITY_FIELDS)


def group_members(group: int) -> list[int]:
    """Enumerate the entire process set using stat only, selecting our session."""
    members = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            identity = process_stat(directory / "stat")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if identity["group"] == group and identity["session"] == group:
            members.append(identity["pid"])
    return sorted(members)


def group_rescan(group: int) -> dict:
    """Retain complete private-group membership and each member's thread set."""
    started = timestamp()
    before = group_members(group)
    processes = []
    for pid in before:
        directory = Path("/proc") / str(pid)
        process = {"pid": pid, "sample": snapshot(directory), "threads": []}
        try:
            threads = sorted((directory / "task").iterdir(), key=lambda path: int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            threads = []
        for thread in threads:
            process["threads"].append({"tid": int(thread.name), "sample": snapshot(thread)})
        processes.append(process)
    return {
        "started": started,
        "scope": "private_probe_group_and_session",
        "group": group,
        "members_before": before,
        "processes": processes,
        "members_after": group_members(group),
        "finished": timestamp(),
    }


def kernel_identity() -> dict:
    identity = os.uname()
    return {
        "sysname": identity.sysname,
        "nodename": identity.nodename,
        "release": identity.release,
        "version": identity.version,
        "machine": identity.machine,
        "proc_version": Path("/proc/version").read_text().strip(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
    }


def prctl(option: int, value: int) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    library.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    library.prctl.restype = ctypes.c_int
    if library.prctl(option, value, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Lifecycle prctl failed")


def process_pidfd(pid: int) -> int:
    """Open an owned process handle using the explicit Linux x86_64 ABI."""
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        raise RuntimeError("pidfd_open syscall 434 requires the measured Linux x86_64 ABI")
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    descriptor = library.syscall(ctypes.c_long(PIDFD_OPEN_X86_64), ctypes.c_int(pid), ctypes.c_uint(0))
    if descriptor == -1:
        raise OSError(ctypes.get_errno(), "Synthetic pidfd_open failed")
    return int(descriptor)


def pidfd_readiness(descriptor: int) -> dict:
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    events = poller.poll(0)
    mask = events[0][1] if events else 0
    if mask & (select.POLLERR | select.POLLNVAL):
        raise RuntimeError("Synthetic pidfd returned an invalid polling state")
    return {
        "timestamp": timestamp(),
        "descriptor": descriptor,
        "poll_event_mask": mask,
        "readable": bool(mask & select.POLLIN),
        "hangup": bool(mask & select.POLLHUP),
    }


def child_event(event: str, **values: int) -> None:
    print(json.dumps({"event": event, "pid": os.getpid()} | values), flush=True)


def child_command() -> None:
    if os.read(0, 1) != b"x":
        os._exit(2)


def child_exec(source: Path, timeout: float) -> None:
    os.execve(
        sys.executable,
        [sys.executable, "-I", "-S", "-B", str(source), "--child", "exec_after", "--timeout", str(timeout)],
        {},
    )


def child_main(role: str, timeout: float) -> None:
    """Run only synthetic, self-terminating controls in an isolated process."""
    signal.setitimer(signal.ITIMER_REAL, timeout)
    source = Path(__file__).resolve(strict=True)
    if role == "exit":
        regions = [mmap.mmap(-1, mmap.PAGESIZE) for _ in range(EXIT_MAPS)]
        child_event("ready", mapping_count=len(regions))
        child_command()
        os._exit(EXIT_STATUS)
    if role == "exec_after":
        child_event("exec_complete")
        child_command()
        os._exit(0)
    if role == "exec":
        child_event("ready")
        child_command()
        child_exec(source, timeout)
    if role == "exec_nonleader":
        ready = threading.Event()
        start_worker = threading.Event()

        def worker() -> None:
            ready.set()
            if not start_worker.wait(timeout):
                os._exit(3)
            child_command()
            child_exec(source, timeout)

        thread = threading.Thread(target=worker)
        thread.start()
        if not ready.wait(timeout):
            os._exit(3)
        child_event("ready")
        child_command()
        start_worker.set()
        # x86_64 SYS_exit terminates only the leader task; the worker keeps the TGID live.
        library = ctypes.CDLL(None)
        library.syscall(ctypes.c_long(60), ctypes.c_int(0))
        os._exit(4)
    if role == "reparent":
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            signal.setitimer(signal.ITIMER_REAL, timeout)
            os.close(write_fd)
            os.read(read_fd, 1)
            os.close(read_fd)
            child_event("adopted")
            child_command()
            os._exit(0)
        os.close(read_fd)
        child_event("ready", child_pid=pid)
        child_command()
        os.close(write_fd)
        os._exit(0)
    raise ValueError("Unknown synthetic child role")


@dataclass
class Child:
    process: Popen
    deadline: float
    pending: bytes = b""
    pidfd: int | None = None


class Children:
    """Own isolated process groups and reap both direct and adopted children."""

    def __init__(self, source: Path, timeout: float, log: Callable):
        self.source = source
        self.timeout = timeout
        self.log = log
        self.processes: dict[int, Child] = {}
        self.groups: set[int] = set()
        self.reaped: list[dict] = []

    def start(self, role: str) -> Child:
        deadline = time.monotonic() + self.timeout
        process = Popen(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(self.source),
                "--child",
                role,
                "--timeout",
                str(self.timeout),
            ],
            stdin=PIPE,
            stdout=PIPE,
            stderr=DEVNULL,
            env={MARKER_NAME: MARKER_VALUE},
            close_fds=True,
            start_new_session=True,
        )
        child = Child(process, deadline)
        self.processes[process.pid] = child
        self.groups.add(process.pid)
        self.log("child_started", role=role, pid=process.pid, deadline_monotonic=deadline)
        return child

    def event(self, child: Child, expected: str) -> dict:
        stream = child.process.stdout
        assert stream is not None
        while b"\n" not in child.pending:
            if not select.select([stream], [], [], remaining(child.deadline))[0]:
                raise TimeoutError("Synthetic child readiness timed out")
            chunk = os.read(stream.fileno(), MAX_PROTOCOL_BYTES)
            if not chunk:
                raise RuntimeError("Synthetic child protocol closed")
            child.pending += chunk
            if len(child.pending) > MAX_PROTOCOL_BYTES:
                raise RuntimeError("Synthetic child protocol exceeded its bound")
        line, child.pending = child.pending.split(b"\n", 1)
        event = json.loads(line)
        if event["event"] != expected:
            raise RuntimeError("Unexpected synthetic child event")
        self.log("child_event", protocol=event)
        return event

    def release(self, child: Child) -> None:
        remaining(child.deadline)
        stream = child.process.stdin
        assert stream is not None
        os.write(stream.fileno(), b"x")

    def terminal(self, pid: int, deadline: float, expected: int, *, pidfd: int | None = None) -> dict:
        readiness = None
        if pidfd is not None:
            poller = select.poll()
            poller.register(pidfd, select.POLLIN)
            while not poller.poll(max(1, math.ceil(remaining(deadline) * 1000))):
                remaining(deadline)
            readiness = pidfd_readiness(pidfd)
            if not readiness["readable"]:
                raise RuntimeError("Synthetic terminal pidfd was not readable")
        while True:
            remaining(deadline)
            info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
            if info is not None:
                break
            select.select([], [], [], min(0.001, remaining(deadline)))
        sample = snapshot(Path("/proc") / str(pid))
        _, status = os.waitpid(pid, 0)
        result = {
            "timestamp": timestamp(),
            "pid": pid,
            "waitid": {"si_pid": info.si_pid, "si_code": info.si_code, "si_status": info.si_status},
            "waitpid_status": status,
            "returncode": os.waitstatus_to_exitcode(status),
            "terminal_sample": sample,
            "proc_absent_after_reap": not (Path("/proc") / str(pid)).exists(),
            "pidfd_before_reap": readiness,
        }
        self.reaped.append(result)
        if pid in self.processes:
            self.processes[pid].process.returncode = result["returncode"]
        self.log("child_reaped", pid=pid, returncode=result["returncode"])
        if info.si_code != os.CLD_EXITED or info.si_status != expected:
            raise RuntimeError("Synthetic control did not exit with its expected result")
        return result

    def cleanup(self) -> dict:
        deadline = time.monotonic() + CLEANUP_TIMEOUT
        during_cleanup = []
        try:
            while True:
                children = Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()
                if not children:
                    return {"remaining_children": [], "reaped_during_cleanup": during_cleanup}
                groups = set()
                for pid in children:
                    try:
                        group = process_stat(Path("/proc") / pid / "stat")["group"]
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    if group not in self.groups:
                        raise RuntimeError("Cleanup found a child outside the private probe groups")
                    groups.add(group)
                for group in groups:
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                while True:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        break
                    result = {"pid": pid, "returncode": os.waitstatus_to_exitcode(status)}
                    during_cleanup.append(result)
                    if pid in self.processes:
                        self.processes[pid].process.returncode = result["returncode"]
                select.select([], [], [], min(0.001, remaining(deadline)))
        except ChildProcessError:
            return {"remaining_children": [], "reaped_during_cleanup": during_cleanup}
        finally:
            for child in self.processes.values():
                if child.pidfd is not None:
                    os.close(child.pidfd)
                    child.pidfd = None
                for stream in (child.process.stdin, child.process.stdout):
                    if stream is not None:
                        stream.close()


def exit_control(children: Children, attempts: int) -> dict:
    result = {"started": timestamp(), "mapping_count": EXIT_MAPS, "attempts": []}
    for _ in range(attempts):
        child = children.start("exit")
        children.event(child, "ready")
        directory = Path("/proc") / str(child.process.pid)
        live = snapshot(directory)
        if not live["environment"].get("marker_present"):
            raise RuntimeError("Synthetic exit child has no initial marker")
        observed = None
        samples = 0
        with (directory / "environ").open("rb", buffering=0) as old_environment:
            initial_fd_count = len(old_environment.read())
            children.release(child)
            while True:
                remaining(child.deadline)
                sample = snapshot(directory)
                samples += 1
                before, after = sample["stat_before"], sample["stat_after"]
                if (
                    same_identity(before, after)
                    and before["status"] not in EXIT_STATES
                    and after["status"] not in EXIT_STATES
                    and before["pf_exiting"]
                    and after["pf_exiting"]
                    and before["vsize"] == after["vsize"] == 0
                    and sample["command"].get("byte_count") == 0
                    and sample["environment"].get("byte_count") == 0
                ):
                    observed = sample
                    break
                if after.get("status") in EXIT_STATES:
                    break
            terminal = children.terminal(child.process.pid, child.deadline, EXIT_STATUS)
            old_environment.seek(0)
            final_fd_count = len(old_environment.read())
        result["attempts"].append(
            {
                "live": live,
                "last_sample": sample,
                "sample_count": samples,
                "nonterminal_teardown": observed,
                "terminal": terminal,
                "old_environment_fd_initial_bytes": initial_fd_count,
                "old_environment_fd_after_exit_bytes": final_fd_count,
            }
        )
        if observed is not None:
            break
    result["pre_zombie_teardown_observed"] = observed is not None
    result["finished"] = timestamp()
    return result


def exec_variant(children: Children, role: str) -> dict:
    started = timestamp()
    child = children.start(role)
    children.event(child, "ready")
    directory = Path("/proc") / str(child.process.pid)
    child.pidfd = process_pidfd(child.process.pid)
    opened = {
        "timestamp": timestamp(),
        "pid": child.process.pid,
        "syscall_number": PIDFD_OPEN_X86_64,
        "flags": 0,
        "descriptor": child.pidfd,
        "stat_at_open": snapshot(directory),
    }
    initial_readiness = pidfd_readiness(child.pidfd)
    if initial_readiness["readable"]:
        raise RuntimeError("Synthetic pidfd was readable while the initial leader was live")
    leader_exit = None
    leader_exit_readiness = None
    if role == "exec_nonleader":
        children.release(child)
        while True:
            remaining(child.deadline)
            leader_exit = group_rescan(child.process.pid)
            process = leader_exit["processes"][0]
            if process["sample"]["stat_after"].get("status") == "Z":
                live_threads = [
                    thread
                    for thread in process["threads"]
                    if thread["sample"]["stat_after"].get("status") not in EXIT_STATES
                    and thread["sample"]["environment"].get("marker_present")
                ]
                if live_threads:
                    break
        before = process["sample"]
        leader_exit_readiness = pidfd_readiness(child.pidfd)
        if leader_exit_readiness["readable"]:
            raise RuntimeError("Synthetic pidfd became readable while the worker was live")
    else:
        before = snapshot(directory)
    with (directory / "environ").open("rb", buffering=0) as old_environment:
        initial_fd_count = len(old_environment.read())
        children.release(child)
        children.event(child, "exec_complete")
        rescans = [group_rescan(child.process.pid), group_rescan(child.process.pid)]
        after = snapshot(directory)
        after_readiness = pidfd_readiness(child.pidfd)
        if after_readiness["readable"]:
            raise RuntimeError("Synthetic pidfd became readable before replacement termination")
        old_environment.seek(0)
        final_fd_count = len(old_environment.read())
    if not same_identity(before["stat_after"], after["stat_after"]):
        raise RuntimeError("Synthetic exec did not preserve the expected process identity")
    if after["environment"].get("marker_present") or after["stat_after"]["pf_exiting"]:
        raise RuntimeError("Synthetic replacement did not remain live without its marker")
    if after["stat_after"]["status"] in EXIT_STATES or after["command"].get("byte_count", 0) == 0:
        raise RuntimeError("Synthetic replacement is not a live user process")
    if role == "exec" and (not before["environment"].get("marker_present") or before["command"] == after["command"]):
        raise RuntimeError("Leader exec did not reproduce command and marker drift")
    if any(
        scan["members_before"] != [child.process.pid] or scan["members_after"] != [child.process.pid] for scan in rescans
    ):
        raise RuntimeError("Complete private-group rescan did not recover the replacement")
    children.release(child)
    terminal = children.terminal(child.process.pid, child.deadline, 0, pidfd=child.pidfd)
    return {
        "started": started,
        "variant": role,
        "before": before,
        "leader_exited_with_live_thread": leader_exit,
        "pidfd": {
            "opened": opened,
            "initial_leader_live": initial_readiness,
            "zombie_leader_live_worker": leader_exit_readiness,
            "replacement_live": after_readiness,
        },
        "complete_rescans_after_exec": rescans,
        "after": after,
        "same_pid_start_parent_group_session": True,
        "old_environment_fd_initial_bytes": initial_fd_count,
        "old_environment_fd_after_exec_bytes": final_fd_count,
        "terminal": terminal,
        "finished": timestamp(),
    }


def reparent_control(children: Children) -> dict:
    child = children.start("reparent")
    event = children.event(child, "ready")
    pid = event["child_pid"]
    directory = Path("/proc") / str(pid)
    before = snapshot(directory)
    if before["stat_after"]["parent"] != child.process.pid:
        raise RuntimeError("Synthetic child started with an unexpected parent")
    children.release(child)
    children.event(child, "adopted")
    parent_terminal = children.terminal(child.process.pid, child.deadline, 0)
    rescan = group_rescan(child.process.pid)
    after = snapshot(directory)
    if after["stat_after"]["parent"] != os.getpid():
        raise RuntimeError("Synthetic child was not adopted by this probe")
    unchanged = ("pid", "start_ticks", "group", "session")
    if any(before["stat_after"][field] != after["stat_after"][field] for field in unchanged):
        raise RuntimeError("Synthetic reparent changed fields beyond ancestry")
    if before["command"] != after["command"] or before["environment"] != after["environment"]:
        raise RuntimeError("Synthetic reparent changed command or environment")
    if not after["environment"].get("marker_present") or after["stat_after"]["pf_exiting"]:
        raise RuntimeError("Synthetic adopted child did not remain live and marked")
    children.release(child)
    terminal = children.terminal(pid, child.deadline, 0)
    return {
        "before": before,
        "after": after,
        "complete_rescan_after_reparent": rescan,
        "parent_terminal": parent_terminal,
        "child_terminal": terminal,
    }


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="ascii") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=True)
        stream.write("\n")


def interrupted(signum: int, frame: object) -> None:
    raise InterruptedError(f"Lifecycle probe interrupted by signal {signum}")


def run_probe(out: Path, timeout: float, attempts: int) -> int:
    """Retain source, evidence, log and a hashed receipt in an unused directory."""
    source_path = Path(__file__).resolve(strict=True)
    source_bytes = source_path.read_bytes()
    out.mkdir(parents=True, exist_ok=False)
    source_copy = out / source_path.name
    with source_copy.open("xb") as stream:
        stream.write(source_bytes)
    evidence = {
        "schema": "owned_process_lifecycle_v1",
        "authority": "cpu_synthetic_evidence_only_no_acceptance_or_launch",
        "started": timestamp(),
        "kernel": kernel_identity(),
        "configuration": {
            "timeout": timeout,
            "cleanup_timeout": CLEANUP_TIMEOUT,
            "max_exit_attempts": attempts,
            "exit_maps": EXIT_MAPS,
            "mapping_page_bytes": mmap.PAGESIZE,
            "mapping_virtual_bytes": EXIT_MAPS * mmap.PAGESIZE,
            "pidfd_open": {"architecture": "x86_64", "syscall_number": PIDFD_OPEN_X86_64, "flags": 0},
        },
        "redaction": "No raw child commands, comm, environment keys or environment values are retained.",
        "kernel_references": KERNEL_REFERENCES,
        "limitations": LIMITATIONS,
        "controls": {},
        "status": "running",
    }
    sources = {
        "source": {"path": str(source_path), "sha256": sha256(source_bytes)},
        "retained_source": {"path": str(source_copy.resolve()), "sha256": sha256(source_bytes)},
        "interpreter": {
            "path": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "sha256": sha256(Path(sys.executable).read_bytes()),
        },
        "invocation": sys.orig_argv,
        "replay_command": [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(source_path),
            "--out",
            str(out.resolve()),
            "--timeout",
            str(timeout),
            "--exit-attempts",
            str(attempts),
        ],
    }
    write_json(out / "source.json", sources)
    with (out / "probe.log").open("x", encoding="ascii") as log_stream:

        def log(event: str, **values: object) -> None:
            payload = {"timestamp": timestamp(), "event": event} | values
            log_stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
            log_stream.flush()

        children = Children(source_copy, timeout, log)
        try:
            if sys.platform != "linux" or os.uname().machine != "x86_64":
                raise RuntimeError("This probe requires Linux x86_64 for leader-only SYS_exit")
            prctl(36, 1)  # PR_SET_CHILD_SUBREAPER, restricted to this dedicated CLI process.
            evidence["controls"]["exit_teardown"] = exit_control(children, attempts)
            log("control_complete", control="exit_teardown")
            started = timestamp()
            variants = [exec_variant(children, role) for role in ("exec", "exec_nonleader")]
            evidence["controls"]["exec_marker_removal"] = {
                "started": started,
                "variants": variants,
                "finished": timestamp(),
            }
            log("control_complete", control="exec_marker_removal")
            started = timestamp()
            reparent = reparent_control(children)
            evidence["controls"]["reparenting"] = {"started": started, **reparent, "finished": timestamp()}
            log("control_complete", control="reparenting")
            evidence["status"] = "completed"
        except (Exception, KeyboardInterrupt) as error:
            evidence["status"] = "failed"
            evidence["error"] = {"type": type(error).__name__, "timestamp": timestamp()}
            log("probe_failed", error_type=type(error).__name__)
        finally:
            # Repeated termination signals must not interrupt the bounded reaping path.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                evidence["cleanup"] = children.cleanup()
            except Exception as error:
                evidence["status"] = "cleanup_failed"
                evidence["cleanup"] = {"error_type": type(error).__name__, "all_children_reaped": False}
                log("cleanup_failed", error_type=type(error).__name__)
            evidence["reaped_children"] = children.reaped
            evidence["finished"] = timestamp()
            log("probe_finished", status=evidence["status"], cleanup=evidence["cleanup"])
    write_json(out / "evidence.json", evidence)
    artifacts = {}
    for name in ("source.json", source_copy.name, "probe.log", "evidence.json"):
        path = out / name
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path.read_bytes()),
            "bytes": path.stat().st_size,
        }
    receipt = {
        "schema": "owned_process_lifecycle_receipt_v1",
        "status": evidence["status"],
        "authority": evidence["authority"],
        "started": evidence["started"],
        "finished": evidence["finished"],
        "kernel": evidence["kernel"],
        "artifacts": artifacts,
        "cleanup": evidence["cleanup"],
        "limitations": LIMITATIONS,
    }
    receipt_path = out / "receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path.resolve()), "sha256": sha256(receipt_path.read_bytes())}))
    return 0 if evidence["status"] == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--exit-attempts", type=int, default=MAX_EXIT_ATTEMPTS)
    parser.add_argument(
        "--child", choices=("exit", "exec", "exec_nonleader", "exec_after", "reparent"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 30 or not 1 <= args.exit_attempts <= MAX_EXIT_ATTEMPTS:
        parser.error("timeout must be in (0, 30] and exit-attempts in [1, 4]")
    if args.child is not None:
        child_main(args.child, args.timeout)
        return 0
    if args.out is None:
        parser.error("--out is required")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    return run_probe(args.out, args.timeout, args.exit_attempts)


if __name__ == "__main__":
    sys.exit(main())
