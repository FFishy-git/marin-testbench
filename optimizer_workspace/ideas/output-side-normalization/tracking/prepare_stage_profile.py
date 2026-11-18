# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a GPU-denied actual-training profiler from the original frozen Wave-9 source."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from boundary_handoff import release
from launch_wave5 import PYTHON, sha256
from prepare_upstream_outer import replace_once
from watch_wave5 import write
from wave7_release import candidate_inventory

FIRST_UPDATE = 201
LAST_UPDATE = 220
CHANGES = {
    "experiment/train.py",
    "experiment/block_toner.py",
    "experiment/stage_profiler.py",
    "experiment/tests/test_stage_profiler.py",
}
HELPERS = ("stage_profiler.py", "test_stage_profiler.py", "prepare_stage_profile.py")


def prepare(root: Path, candidate: Path) -> dict:
    """Copy exact predecessor source and add only opt-in, bounded stage instrumentation."""
    frozen = release(root, wave="wave9")
    checked = subprocess.run(
        [
            PYTHON,
            "-c",
            "import json,sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
            "from upstream_run_history import binding; print(json.dumps(binding(Path(sys.argv[2]))))",
            frozen["helpers"],
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if json.loads(checked.stdout) != frozen:
        raise ValueError("Stage profiling predecessor does not match its frozen production release")
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.resolve().is_relative_to(previous.resolve()):
        raise FileExistsError("Stage profiling requires a distinct unused candidate identity")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    helpers = Path(__file__).parent
    shutil.copy2(helpers / "stage_profiler.py", candidate / "experiment/stage_profiler.py")
    shutil.copy2(helpers / "test_stage_profiler.py", candidate / "experiment/tests/test_stage_profiler.py")
    test_path = candidate / "experiment/tests/test_stage_profiler.py"
    test_path.write_text(
        replace_once(test_path.read_text(), "from stage_profiler import", "from experiment.stage_profiler import")
    )
    path = candidate / "experiment/train.py"
    text = replace_once(
        path.read_text(),
        "from pathlib import Path",
        "from contextlib import nullcontext\nfrom pathlib import Path",
    )
    text = replace_once(
        text,
        "from .block_toner import BLOCK_ARMS, BlockToner, DiagonalMoments",
        "from . import block_toner\nfrom .block_toner import BLOCK_ARMS, BlockToner, DiagonalMoments",
    )
    text = replace_once(
        text,
        "from .factors import OutputSettings",
        "from .factors import OutputSettings\nfrom .stage_profiler import StageProfiler",
    )
    text = replace_once(
        text,
        "    args = parser.parse_args()",
        '    parser.add_argument("--profile-updates", type=int, nargs=2, metavar=("FIRST", "LAST"))\n'
        "    args = parser.parse_args()",
    )
    text = replace_once(
        text,
        "    require_training_owner(args, config)",
        "    if args.profile_updates is not None:\n"
        '        raise RuntimeError("Stage-profile candidate has no distinct GPU execution binding")\n'
        "    require_training_owner(args, config)",
    )
    text = replace_once(
        text,
        "    start_step = 0",
        "    stage_profile = None\n"
        "    if args.profile_updates is not None:\n"
        "        first_update, last_update = args.profile_updates\n"
        "        stage_profile = StageProfiler(\n"
        '            args.out / "profiles" / f"rank{rank}",\n'
        "            first=first_update,\n            last=last_update,\n            rank=rank,\n"
        '            provenance={"arm": arm, "config": config, "checkpoint_identity": identity},\n'
        "        )\n"
        "        stage_profile.install(fast, block_toner)\n"
        "    profile_scope = (lambda name: nullcontext()) if stage_profile is None else stage_profile.scope\n"
        "    start_step = 0",
    )
    text = replace_once(
        text,
        '                with torch.autocast("cuda", dtype=torch.bfloat16):\n'
        "                    loss = run(*to_device(mixture.batch(start, micro)))",
        '                with profile_scope("forward_and_data"), torch.autocast("cuda", dtype=torch.bfloat16):\n'
        "                    loss = run(*to_device(mixture.batch(start, micro)))",
    )
    text = replace_once(
        text,
        "                (loss / accumulation).backward()",
        '                with profile_scope("backward_and_output_collection"):\n'
        "                    (loss / accumulation).backward()",
    )
    text = replace_once(
        text,
        '            clip_gradients(clipping, config["optimizer"]["clip_norm"])\n'
        "            for optimizer in optimizers:\n"
        "                optimizer.step()",
        '            with profile_scope("clip_and_optimizer"):\n'
        '                clip_gradients(clipping, config["optimizer"]["clip_norm"])\n'
        "                for optimizer in optimizers:\n"
        "                    optimizer.step()",
    )
    text = replace_once(
        text,
        "            if world > 1:\n"
        "                dist.all_reduce(loss_sum)\n"
        "                loss_sum /= world\n"
        "            if not torch.isfinite(loss_sum).item():\n"
        '                raise FloatingPointError("Nonfinite training loss")\n'
        "            torch.cuda.synchronize()",
        '            with profile_scope("loss_collective_and_final_sync"):\n'
        "                if world > 1:\n"
        "                    dist.all_reduce(loss_sum)\n"
        "                    loss_sum /= world\n"
        "                if not torch.isfinite(loss_sum).item():\n"
        '                    raise FloatingPointError("Nonfinite training loss")\n'
        "                torch.cuda.synchronize()",
    )
    text = replace_once(
        text,
        "            step_start = time.perf_counter()",
        "            if stage_profile is not None:\n"
        "                stage_profile.begin_update(step + 1)\n"
        "            step_start = time.perf_counter()",
    )
    text = replace_once(
        text,
        "            metrics = {\n",
        "            measured_step_seconds = time.perf_counter() - step_start\n"
        "            if stage_profile is not None:\n"
        "                stage_profile.end_update(completed)\n"
        "            metrics = {\n",
    )
    text = replace_once(
        text,
        '                "step_seconds": time.perf_counter() - step_start,',
        '                "step_seconds": measured_step_seconds,',
    )
    text = replace_once(
        text,
        "    finally:\n"
        "        for engine in toner_engines(fast):\n"
        "            engine.side.synchronize()\n"
        "            engine.pool.shutdown(wait=True)\n"
        "        if world > 1:\n"
        "            dist.destroy_process_group()",
        "    finally:\n"
        "        try:\n"
        "            if stage_profile is not None:\n"
        "                stage_profile.finish()\n"
        "        finally:\n"
        "            for engine in toner_engines(fast):\n"
        "                engine.side.synchronize()\n"
        "                engine.pool.shutdown(wait=True)\n"
        "            if world > 1:\n"
        "                dist.destroy_process_group()",
    )
    path.write_text(text)
    path = candidate / "experiment/block_toner.py"
    text = replace_once(
        path.read_text(),
        "from dataclasses import dataclass",
        "from contextlib import nullcontext\nfrom dataclasses import dataclass",
    )
    text = replace_once(
        text,
        "BLOCK_KINDS =",
        "def profile_scope(name):\n" "    return nullcontext()\n\n\n" "BLOCK_KINDS =",
    )
    text = replace_once(
        text,
        "                weights = torch.stack([parameter for _, _, parameter in mine])\n"
        "                gradients = torch.stack([parameter.grad.float() for _, _, parameter in mine])",
        '                with profile_scope("parameter_gradient_pack"):\n'
        "                    weights = torch.stack([parameter for _, _, parameter in mine])\n"
        "                    gradients = torch.stack([parameter.grad.float() for _, _, parameter in mine])",
    )
    text = replace_once(
        text,
        "        checks = torch.stack(finite).tolist()",
        '        with profile_scope("finite_check_wait"):\n' "            checks = torch.stack(finite).tolist()",
    )
    text = replace_once(
        text,
        "            if work is not None:\n                work.wait()",
        "            if work is not None:\n"
        '                with profile_scope("update_gather_wait"):\n'
        "                    work.wait()",
    )
    text = replace_once(
        text,
        "                if entries:\n"
        "                    torch._foreach_add_(\n"
        "                        [parameter for _, _, parameter in entries],\n"
        "                        list(updates.unbind(0)),\n"
        "                    )",
        "                if entries:\n"
        '                    with profile_scope("parameter_apply"):\n'
        "                        torch._foreach_add_(\n"
        "                            [parameter for _, _, parameter in entries],\n"
        "                            list(updates.unbind(0)),\n"
        "                        )",
    )
    path.write_text(text)
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | frozen["files_sha256"] if files.get(name) != frozen["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError(f"Profiling preparation exceeds its declared source changes: {sorted(changed ^ CHANGES)}")
    result = {
        "status": "stage_profile_prepared_gpu_execution_denied",
        "candidate": str(candidate.resolve()),
        "predecessor_candidate": str(previous),
        "predecessor_release": str(root / "run-preparation/wave9-launch-binding.json"),
        "predecessor_release_sha256": sha256(root / "run-preparation/wave9-launch-binding.json"),
        "files_sha256": files,
        "changed_files": sorted(changed),
        "helpers_sha256": {name: sha256(helpers / name) for name in HELPERS},
        "profile_updates": [FIRST_UPDATE, LAST_UPDATE],
        "arms": frozen["arms"],
        "world_size": 8,
        "policy": (
            "CPU-only profiling preparation. Optimizer/model/config/pinned sources unchanged. "
            "No CUDA binding, controller, queue or production authorization. Existing Wave-9 and ceiling-LR "
            "allocations remain occupied; later execution requires its own owner and cleanup gates."
        ),
    }
    write(candidate / "stage-profile-preparation.json", result)
    return result


def cpu_evidence(root: Path, candidate: Path, output: Path) -> dict:
    """Preserve source, command logs and CPU pass/fail without authorizing GPU execution."""
    preparation = candidate / "stage-profile-preparation.json"
    record = json.loads(preparation.read_text())
    frozen = release(root, wave="wave9")
    files = candidate_inventory(candidate)
    helpers = Path(__file__).parent
    helper_hashes = {name: sha256(helpers / name) for name in HELPERS}
    changed = {
        name for name in files.keys() | frozen["files_sha256"] if files.get(name) != frozen["files_sha256"].get(name)
    }
    if (
        record["candidate"] != str(candidate.resolve())
        or record["predecessor_release_sha256"] != sha256(root / "run-preparation/wave9-launch-binding.json")
        or record["files_sha256"] != files
        or record["helpers_sha256"] != helper_hashes
        or changed != CHANGES
        or record["profile_updates"] != [FIRST_UPDATE, LAST_UPDATE]
    ):
        raise ValueError("Stage profiling source or preparation changed before CPU evidence")
    if output.exists() or output.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Stage profiling CPU evidence requires a new external directory")
    output.mkdir(parents=True, exist_ok=False)
    command = [
        PYTHON,
        "-m",
        "pytest",
        "-c",
        "/dev/null",
        "-p",
        "no:cacheprovider",
        "-m",
        "not slow and not integration and not data_integration and not cluster "
        "and not requires_cluster and not docker and not manual",
        "experiment/tests",
        "-q",
    ]
    log = output / "candidate.log"
    with log.open("x") as stream:
        completed = subprocess.run(
            command,
            cwd=candidate,
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(helpers)},
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    if files != candidate_inventory(candidate) or helper_hashes != {name: sha256(helpers / name) for name in HELPERS}:
        raise ValueError("Stage profiling sources changed during CPU evidence")
    archive = output / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in files:
            bundle.add(candidate / name, arcname=f"candidate/{name}", recursive=False)
        bundle.add(
            candidate / "artifacts/external-sources", arcname="candidate/artifacts/external-sources", recursive=False
        )
        bundle.add(preparation, arcname="candidate/stage-profile-preparation.json", recursive=False)
        for name in HELPERS:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    result = {
        **record,
        "status": "stage_profile_cpu_verified_gpu_denied" if completed.returncode == 0 else "stage_profile_cpu_failed",
        "preparation": str(preparation),
        "preparation_sha256": sha256(preparation),
        "commands": [
            {
                "command": command,
                "cwd": str(candidate),
                "exit_code": completed.returncode,
                "log": str(log),
                "log_sha256": sha256(log),
            }
        ],
        "cuda_visible_devices": "",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
    write(output / "receipt.json", result)
    if completed.returncode:
        raise RuntimeError(f"Stage profiling CPU evidence failed; preserve {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "cpu"), default="prepare")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU evidence requires --out")
    result = (
        prepare(args.root, args.candidate)
        if args.phase == "prepare"
        else cpu_evidence(args.root, args.candidate, args.out)
    )
    print(json.dumps({"status": result["status"], "candidate": result["candidate"]}))


if __name__ == "__main__":
    main()
