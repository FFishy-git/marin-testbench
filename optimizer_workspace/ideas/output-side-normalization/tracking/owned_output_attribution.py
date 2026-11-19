# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Add redacted failure evidence to an unchanged, source-pinned ownership scan."""

import argparse
import ast
import copy
import hashlib
import json
import time
from pathlib import Path

PARENT_SHA256 = "293eab75802b1eff8726cd2a886d713742064885c456f74572b8ffd62227fefe"
IDENTITY_FIELDS = ("pid", "status", "parent", "group", "session", "start_ticks")
PF_EXITING = 0x00000004
ERROR_SITES = (
    ("observed", "Handoff process generation changed during observation", "observed_stat_drift"),
    ("marker_processes", "Marked process generation changed during handoff: pid=", "scan_identity_drift"),
    ("marker_processes", "Marked process environment changed during handoff: pid=", "scan_marker_removed"),
    ("marker_processes", "Marked process generation changed during handoff: pid=", "recheck_exit_owner_drift"),
    ("marker_processes", "Marked process generation changed during handoff: pid=", "recheck_identity_drift"),
    (
        "marker_processes",
        "Marked process scan remained unstable during handoff after ",
        "complete_scan_exhausted",
    ),
)


def command_identity(command: list[str]) -> dict:
    """Retain exact command equality evidence without emitting argument values."""
    payload = json.dumps(command, ensure_ascii=True, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "argument_count": len(command),
        "argument_bytes": sum(len(argument.encode()) for argument in command),
    }


def process_identity(process: dict | None) -> dict | None:
    if process is None:
        return None
    return {key: process[key] for key in IDENTITY_FIELDS} | {"command": command_identity(process["command"])}


def stat_identity(pid: int, fields: list[str]) -> dict:
    result = {
        "pid": pid,
        "status": fields[0],
        "parent": int(fields[1]),
        "group": int(fields[2]),
        "session": int(fields[3]),
        "start_ticks": fields[19],
    }
    if len(fields) > 6:
        result |= {"flags": int(fields[6]), "pf_exiting": bool(int(fields[6]) & PF_EXITING)}
    return result


def delayed_process_sample(root: Path, pid: int) -> dict:
    """Record a later sample without interpreting it as the failed scan's identity."""
    directory = root / str(pid)
    result = {"pid": pid, "sample_monotonic_ns": time.monotonic_ns()}
    for field in ("stat", "cmdline"):
        try:
            data = (directory / field).read_bytes()
        except OSError as error:
            result[field] = {"read_error": type(error).__name__, "errno": error.errno}
            continue
        if field == "stat":
            try:
                result[field] = stat_identity(pid, data.decode().rsplit(")", 1)[1].split())
            except (ValueError, IndexError, UnicodeError) as error:
                result[field] = {"parse_error": type(error).__name__}
        else:
            result[field] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "argument_count": sum(bool(argument) for argument in data.split(b"\0")),
                "byte_count": len(data),
            }
    return result


class OwnershipObservationError(RuntimeError):
    """Preserve scan rejection while exposing JSON-safe ownership evidence."""

    def __init__(self, message: str, evidence: dict):
        self.evidence = evidence
        super().__init__(message + " ownership_evidence=" + json.dumps(evidence, sort_keys=True))


def observation_error(stage: str, message: str, context: dict) -> OwnershipObservationError:
    """Capture only allowlisted locals at the exact rejecting observation boundary."""
    evidence = {
        "schema": "owned_output_failure_attribution_v1",
        "stage": stage,
        "failure_monotonic_ns": time.monotonic_ns(),
        "authority": "failure_attribution_only_no_ownership_or_launch",
        "snapshots": {},
    }
    pid = context["pid"] if stage != "complete_scan_exhausted" else None
    snapshots = evidence["snapshots"]
    if stage == "observed_stat_drift":
        snapshots["stat_before"] = stat_identity(pid, context["before"])
        snapshots["stat_after"] = stat_identity(pid, context["after"])
        snapshots["command_between_stats"] = command_identity(context["command"])
    elif stage in ("scan_identity_drift", "scan_marker_removed"):
        snapshots["retained"] = process_identity(context["identities"][pid])
        snapshots["scan"] = process_identity(context["process"])
        snapshots["previously_confirmed_marked"] = pid in context["marked"]
    elif stage in ("recheck_exit_owner_drift", "recheck_identity_drift"):
        snapshots["scan"] = process_identity(context["process"])
        snapshots["recheck"] = process_identity(context["current"])
        snapshots["previously_confirmed_marked"] = pid in context["marked"]
    if stage == "scan_marker_removed":
        snapshots["marker_present"] = context["token"] in context["tokens"]
        snapshots["environment_byte_count"] = sum(len(token) for token in context["tokens"]) + len(context["tokens"]) - 1
    if stage == "complete_scan_exhausted":
        pids = sorted(context["unstable"])
        snapshots["unstable_pids"] = pids
        snapshots["retained"] = {str(worker): process_identity(context["identities"].get(worker)) for worker in pids}
        snapshots["previously_confirmed_marked"] = {str(worker): worker in context["marked"] for worker in pids}
    else:
        pids = [pid]
        evidence["pid"] = pid
    root = context["directory"].parent if stage == "observed_stat_drift" else context["PROC_ROOT"]
    evidence["delayed_samples"] = [delayed_process_sample(root, worker) for worker in pids]
    evidence["delayed_sample_limit"] = "Separate reads after failure; cannot establish cause or coherent identity."
    return OwnershipObservationError(message, evidence)


class AttributionSites(ast.NodeTransformer):
    """Replace exactly the pinned rejection constructors without changing scan branches."""

    def __init__(self):
        self.function = None
        self.sites = []

    def visit_FunctionDef(self, node):
        previous = self.function
        self.function = node.name
        node = self.generic_visit(node)
        self.function = previous
        return node

    def visit_Raise(self, node):
        call = node.exc
        if self.function not in ("observed", "marker_processes"):
            return node
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "RuntimeError":
            raise ValueError("Pinned attribution rejection constructor differs")
        argument = call.args[0]
        prefix = (
            argument.value
            if isinstance(argument, ast.Constant)
            else "".join(value.value for value in argument.values if isinstance(value, ast.Constant))
        )
        function, expected, stage = ERROR_SITES[len(self.sites)]
        if self.function != function or not prefix.startswith(expected):
            raise ValueError("Pinned attribution rejection site differs")
        node.exc = ast.Call(
            func=ast.Name(id="observation_error", ctx=ast.Load()),
            args=[
                ast.Constant(value=stage),
                argument,
                ast.BinOp(
                    left=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]),
                    op=ast.BitOr(),
                    right=ast.Dict(
                        keys=[ast.Constant(value="PROC_ROOT")], values=[ast.Name(id="PROC_ROOT", ctx=ast.Load())]
                    ),
                ),
            ],
            keywords=[],
        )
        self.sites.append(stage)
        return node


def derived_source(parent: Path) -> bytes:
    """Derive a CPU diagnostic helper from exactly the consumed recovery scan."""
    source = parent.read_bytes()
    if hashlib.sha256(source).hexdigest() != PARENT_SHA256:
        raise ValueError("Attribution parent is not the exact consumed recovery helper")
    tree = copy.deepcopy(ast.parse(source))
    transformer = AttributionSites()
    transformer.visit(tree)
    if len(transformer.sites) != len(ERROR_SITES):
        raise ValueError("Attribution must cover every pinned ownership rejection")
    handler_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tree.body[1:1] = ast.parse(
        f"""import hashlib as attribution_hashlib
import types as attribution_types
from pathlib import Path as AttributionPath
attribution_path = AttributionPath(__file__).with_name("owned_output_attribution.py")
if attribution_path.is_symlink():
    raise RuntimeError("CPU attribution handler source is aliased")
attribution_bytes = attribution_path.read_bytes()
if attribution_hashlib.sha256(attribution_bytes).hexdigest() != {handler_sha256!r}:
    raise RuntimeError("CPU attribution handler origin or source identity differs")
attribution_handler = attribution_types.ModuleType("owned_output_attribution_pinned")
attribution_handler.__file__ = str(attribution_path)
exec(compile(attribution_bytes, str(attribution_path), "exec"), attribution_handler.__dict__)
observation_error = attribution_handler.observation_error
"""
    ).body
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
    with (args.out / "owned_output_handoff.py").open("xb") as stream:
        stream.write(payload)
    handler = Path(__file__).read_bytes()
    with (args.out / "owned_output_attribution.py").open("xb") as stream:
        stream.write(handler)
    receipt = {
        "status": "failure_attribution_materialized_cpu_diagnostic_gpu_unissued",
        "parent": str(args.parent.resolve(strict=True)),
        "parent_sha256": PARENT_SHA256,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "handler_sha256": hashlib.sha256(handler).hexdigest(),
        "derived_sha256": hashlib.sha256(payload).hexdigest(),
        "error_stages": [stage for _, _, stage in ERROR_SITES],
        "authority": "no_scan_behavior_change_no_acceptance_or_launch",
    }
    with (args.out / "source.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
