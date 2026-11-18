# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and archive a distinct fused continuation diagnostic without GPU access."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from fused_attribution_validation_contract import CHILD_MARKER as PARENT_CHILD_MARKER
from fused_attribution_validation_contract import CONTROLLER_MARKER as PARENT_CONTROLLER_MARKER
from fused_attribution_validation_contract import locations as parent_locations
from fused_continuation_contract import CONFIG, CPU_STATUS, PREPARATION, PREPARED_STATUS, plan
from prepare_fused_attribution_production import original, replace_once
from prepare_selected_lr_stage_profile_validation import helper_names
from prepare_selected_lr_validation import require_archive
from selected_lr_release_owner import candidate_files, read, sha256
from upstream_production import cpu_command
from watch_wave5 import write

COPIES = {
    "fused_continuation_contract.py": "experiment/fused_continuation_contract.py",
    "fused_continuation_trace.py": "experiment/fused_continuation_trace.py",
    "test_fused_continuation_trace.py": "experiment/tests/test_fused_continuation_trace.py",
}
OWNER_TEST = "experiment/tests/test_fused_attribution_validation_owner.py"
CHANGES = {"experiment/train.py", "experiment/fused_attribution_owner.py", OWNER_TEST, *COPIES.values()}
TESTS = ("test_fused_continuation_trace.py", "test_prepare_fused_continuation.py")


def failed_attribution(root: Path, frozen: dict) -> dict:
    """Authenticate the original failure and exact saved restoration without a live probe."""
    binding_path, helpers, state_path, _, receipt_path = parent_locations(root)
    intent_path = helpers.parent / "controller.json"
    state, receipt, intent = (read(path) for path in (state_path, receipt_path, intent_path))
    digest = sha256(binding_path)
    if (
        frozen["helpers"] != str(helpers)
        or state["status"] != "needs_attention"
        or receipt["status"] != "needs_attention"
        or state["binding_sha256"] != digest
        or receipt["execution_binding_sha256"] != digest
        or intent["binding_sha256"] != digest
        or intent["status"] != "acknowledged"
        or intent["pid"] != state["controller_pid"]
        or intent["start_ticks"] != state["controller_start_ticks"]
        or receipt["owner"]
        != {"controller_pid": state["controller_pid"], "controller_start_ticks": state["controller_start_ticks"]}
        or state["keeper_restore"] != 0
        or type(state["keeper_restore"]) is not int
        or type(state["exit_code"]) is not int
        or state["exit_code"] != 1
        or state.get("marked_descendants_absent") is not True
        or state.get("keeper_release_complete") is not True
        or "cleanup_error" in state
        or len(receipt["gpu_commands"]) != 6
        or any(type(command["exit_code"]) is not int or command["exit_code"] != 0 for command in receipt["gpu_commands"])
        or receipt["error"] != "AssertionError: rank_0.model.body.emb.weight"
        or not intent["marker"]
        or not state["child_marker"]
    ):
        raise ValueError("Continuation diagnostic requires the original failed/restored fused attribution attempt")
    paths = [binding_path, state_path, receipt_path, intent_path, Path(state["log"])]
    if sha256(paths[-1]) != state["log_sha256"]:
        raise ValueError("Original failed validator log changed")
    for command in receipt["gpu_commands"]:
        path = Path(command["log"])
        if sha256(path) != command["log_sha256"]:
            raise ValueError("Original attribution command log changed")
        paths.append(path)
    return {
        "status": "failed_attribution_exact_keeper_pending_live_handoff",
        "node": state["node"],
        "evidence_sha256": {str(path.relative_to(root)): sha256(path) for path in paths},
        "controller_pid": state["controller_pid"],
        "controller_start_ticks": state["controller_start_ticks"],
        "child_pid": state["child_pid"],
        "child_start_ticks": state["child_start_ticks"],
        "controller_marker_name": PARENT_CONTROLLER_MARKER,
        "controller_marker": intent["marker"],
        "child_marker_name": PARENT_CHILD_MARKER,
        "child_marker": state["child_marker"],
        "restored_keeper": state["restored_keeper"],
    }


def transformed_train(previous: Path) -> bytes:
    text = (previous / "experiment/train.py").read_text()
    replacements = (
        (
            "from .block_toner import BLOCK_ARMS, BlockToner, DiagonalMoments\n",
            "from . import block_toner as block_module\n"
            "from toner.fast import roots as root_module\n"
            "from .fused_continuation_trace import ContinuationTrace\n"
            "from .block_toner import BLOCK_ARMS, BlockToner, DiagonalMoments\n",
        ),
        (
            "    args = parser.parse_args()\n",
            '    parser.add_argument("--trace-continuation", action="store_true")\n' "    args = parser.parse_args()\n",
        ),
        (
            '    compile_mode = runtime["compile"]\n',
            "    continuation_trace = None\n"
            "    if args.trace_continuation:\n"
            "        continuation_trace = ContinuationTrace(\n"
            '            args.out / "continuation-trace" / f"rank_{rank}",\n'
            "            rank=rank,\n"
            '            provenance={"identity": identity, "start_step": start_step},\n'
            "        )\n"
            "        continuation_trace.install(fast, block_module, root_module)\n"
            '    compile_mode = runtime["compile"]\n',
        ),
        (
            "            loss_sum = torch.zeros((), device=device)\n",
            "            if continuation_trace is not None:\n"
            "                continuation_trace.begin_update(step + 1)\n"
            "            loss_sum = torch.zeros((), device=device)\n",
        ),
        (
            "            if isinstance(fast, BlockToner):\n                fast.finish_backward()\n",
            "            if continuation_trace is not None:\n"
            '                continuation_trace.record("model/gradients", {\n'
            "                    name: parameter.grad for name, parameter in model.named_parameters()\n"
            "                })\n"
            "            if isinstance(fast, BlockToner):\n                fast.finish_backward()\n",
        ),
        (
            "                or completed == args.execution_boundary\n",
            "                or completed == args.execution_boundary\n" "                or completed == 23\n",
        ),
        (
            '            metrics["end_to_end_seconds"] = time.perf_counter() - wall_start\n',
            "            if continuation_trace is not None:\n"
            '                continuation_trace.record("model/after_update", model.state_dict())\n'
            "                continuation_trace.end_update(completed)\n"
            '            metrics["end_to_end_seconds"] = time.perf_counter() - wall_start\n',
        ),
    )
    for before, after in replacements:
        text = replace_once(text, before, after)
    return text.encode()


def contents(previous: Path) -> dict[str, bytes]:
    directory = Path(__file__).parent
    owner = replace_once(
        (previous / "experiment/fused_attribution_owner.py").read_text(),
        "from .fused_attribution_validation_contract import (",
        "from .fused_continuation_contract import (",
    )
    owner_test = replace_once(
        (previous / OWNER_TEST).read_text(),
        '        args.config = Path(item.frozen["commands"][0][item.command.index("--config") + 1])',
        '        args.config = item.candidate / contract.config_name("muont", "lower")',
    )
    return {
        "experiment/train.py": transformed_train(previous),
        "experiment/fused_attribution_owner.py": owner.encode(),
        OWNER_TEST: owner_test.encode(),
        **{destination: (directory / source).read_bytes() for source, destination in COPIES.items()},
    }


def preparation_record(root: Path, candidate: Path, frozen: dict) -> dict:
    previous = Path(frozen["candidate"])
    return {
        "status": PREPARED_STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_binding_sha256": sha256(parent_locations(root)[0]),
        "parent_metadata_sha256": {path.name: sha256(path) for path in previous.glob("*.json")},
        "parent_files_sha256": frozen["files_sha256"],
        "files_sha256": candidate_files(candidate),
        "data_receipt_sha256": frozen["data_receipt_sha256"],
        "config_sha256": sha256(candidate / CONFIG),
        "changed_files": sorted(CHANGES),
        "helpers_sha256": {
            name: sha256(Path(__file__).parent / name)
            for name in helper_names(Path(__file__).parent, (Path(__file__).name, *COPIES, *TESTS))
        },
        "allocation_predecessor": failed_attribution(root, frozen),
        "plan": plan(candidate, root),
        "scope": (
            "Distinct diagnostic-only owner/source; original model, kernels, optimizer and nine configs unchanged. "
            "Additional update23 checkpoint and optional rank-local intermediate copies. CPU/archive evidence "
            "does not authorize GPUs. All original gates remain required before full-budget production."
        ),
    }


def prepare(root: Path, candidate: Path) -> dict:
    root, candidate = root.resolve(), candidate.resolve()
    frozen = original(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.is_relative_to(previous) or previous.is_relative_to(candidate):
        raise FileExistsError("Continuation diagnostic needs a new separate source identity")
    payload = contents(previous)
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, raw in payload.items():
        path = candidate / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    prepared = preparation_record(root, candidate, frozen)
    changed = {
        name
        for name in prepared["files_sha256"].keys() | frozen["files_sha256"].keys()
        if prepared["files_sha256"].get(name) != frozen["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError("Continuation diagnostic exceeded owner/trainer/trace/test scope")
    write(candidate / PREPARATION, prepared)
    return prepared


def require_preparation(root: Path, candidate: Path) -> dict:
    root, candidate = root.resolve(), candidate.resolve()
    prepared = read(candidate / PREPARATION)
    frozen = original(root)
    expected = preparation_record(root, candidate, frozen)
    changed = {
        name
        for name in expected["files_sha256"].keys() | frozen["files_sha256"].keys()
        if expected["files_sha256"].get(name) != frozen["files_sha256"].get(name)
    }
    if (
        prepared.keys() != expected.keys() | {"time"}
        or any(prepared[key] != value for key, value in expected.items())
        or changed != CHANGES
        or any((candidate / name).read_bytes() != raw for name, raw in contents(Path(frozen["candidate"])).items())
        or any(sha256(candidate / name) != digest for name, digest in prepared["parent_metadata_sha256"].items())
    ):
        raise ValueError("Continuation diagnostic source, config, failure or transformation changed")
    return prepared


def archive_inventory(prepared: dict, digest: str) -> dict:
    return (
        prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: digest}
        | {f"tracking/{name}": value for name, value in prepared["helpers_sha256"].items()}
    )


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    """Run the entire candidate CPU suite and preserve a separate source archive."""
    root, candidate, out = root.resolve(), candidate.resolve(), out.resolve()
    prepared = require_preparation(root, candidate)
    if out.exists() or out.is_relative_to(candidate) or candidate.is_relative_to(out):
        raise FileExistsError("Continuation CPU evidence needs an unused external path")
    out.mkdir(parents=True)
    helpers = out / "tracking"
    helpers.mkdir()
    for name, digest in prepared["helpers_sha256"].items():
        shutil.copy2(Path(__file__).parent / name, helpers / name)
        if sha256(helpers / name) != digest:
            raise ValueError("Continuation helper changed before CPU verification")
    command, log, archive = cpu_command(), out / "candidate.log", out / "source.tar"
    error = None
    with log.open("x") as stream:
        try:
            code = subprocess.run(
                command,
                cwd=candidate,
                env=os.environ
                | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(candidate)},
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=900,
            ).returncode
        except (OSError, subprocess.TimeoutExpired) as failure:
            code, error = 124, f"{type(failure).__name__}: {failure}"
    digest = sha256(candidate / PREPARATION)
    expected = archive_inventory(prepared, digest)
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in expected:
                source = helpers / Path(name).name if name.startswith("tracking/") else candidate / name
                bundle.add(source, arcname=name, recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
        require_archive(archive, expected, alias=True)
        if require_preparation(root, candidate) != prepared:
            raise ValueError("Continuation source changed during candidate CPU suite")
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError, tarfile.TarError) as failure:
        error = f"{type(failure).__name__}: {failure}"
    result = {
        "status": CPU_STATUS if code == 0 and error is None else "needs_attention",
        "root": str(root),
        "candidate": str(candidate),
        "preparation_sha256": digest,
        "files_sha256": prepared["files_sha256"],
        "helpers_sha256": prepared["helpers_sha256"],
        "helpers": str(helpers),
        "command": command,
        "exit_code": code,
        "cuda_visible_devices": "",
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive) if archive.exists() else None,
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CPU_STATUS:
        raise RuntimeError(f"Continuation CPU/archive failure preserved at {out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "check", "cpu"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU evidence requires --out")
    if args.phase == "prepare":
        result = prepare(args.root, args.candidate)
    elif args.phase == "check":
        result = require_preparation(args.root, args.candidate)
    else:
        result = cpu(args.root, args.candidate, args.out)
    print(json.dumps({"status": result["status"], "candidate": result["candidate"]}))


if __name__ == "__main__":
    main()
