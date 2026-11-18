# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a GPU-denied stage profiler from the exact frozen Wave12 production source."""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from launch_wave5 import NODES, PYTHON, read, sha256
from prepare_selected_lr_validation import require_archive
from prepare_upstream_outer import replace_once
from selected_lr_validation import ARMS, ENVIRONMENT
from upstream_production import cpu_command
from watch_wave5 import write
from wave7_release import candidate_inventory

PREPARATION = "selected-lr-stage-profile-preparation.json"
PARENT_PREPARATION = "selected-lr-release-preparation.json"
PARENT_STATUS = "frozen_selected_lr_production_after_original_gpu_validation"
STATUS = "selected_lr_stage_profile_prepared_gpu_denied"
CPU_STATUS = "selected_lr_stage_profile_cpu_verified_gpu_denied"
FIRST_UPDATE, LAST_UPDATE = 201, 220
COPIES = {
    "selected_lr_stage_profiler.py": "experiment/selected_lr_stage_profiler.py",
    "selected_lr_stage_profile_owner.py": "experiment/selected_lr_stage_profile_owner.py",
    "test_selected_lr_stage_profiler.py": "experiment/tests/test_selected_lr_stage_profiler.py",
}
CHANGES = {"experiment/train.py", "experiment/block_toner.py", *COPIES.values()}
HELPERS = (
    "prepare_selected_lr_stage_profile.py",
    *COPIES,
    "test_prepare_selected_lr_stage_profile.py",
)
EXTERNAL_MODULES = sys.stdlib_module_names | {"pytest", "torch"}
POLICY = (
    "CPU-only preparation, not GPU execution or launch authorization. Preserve Wave12 numerical/configuration "
    "bytes and historical owners. A separate profiling release must establish same-trajectory on/off parity, "
    "cached-root continuation, all-rank trace reconstruction and exact cleanup/restoration under serialized "
    "keeper-manager ownership. Do not resume historical Wave12 checkpoints or mutate its runs. Trace interval "
    "201-220 retains the full 5120-update schedule; no throughput or quality claim follows."
)


def paths(root: Path) -> tuple[Path, Path]:
    return root / "run-preparation/wave12-launch-binding.json", root / "controllers/wave12-launch-20261117/tracking"


def helper_binding(frozen: dict) -> dict:
    """Bind every local AST import, with inherited bytes fixed by the Wave12 manifest."""
    directory, parent = Path(__file__).parent, Path(frozen["helpers"])
    inherited = frozen["helpers_sha256"]
    if any(Path(name).name != name for name in inherited):
        raise ValueError("Frozen Wave12 helper manifest contains a nonlocal path")
    allowed = set(HELPERS) | set(inherited)
    hashes, imports = {}, {}
    pending = set(HELPERS)
    while pending:
        name = pending.pop()
        source = directory / name
        if name not in allowed or source.is_symlink() or not source.is_file():
            raise ValueError(f"Stage-profile local helper dependency missing or unexpected: {name}")
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if name not in HELPERS:
            original_source = parent / name
            if (
                original_source.is_symlink()
                or not original_source.is_file()
                or digest != inherited[name]
                or sha256(original_source) != inherited[name]
            ):
                raise ValueError(f"Stage-profile inherited helper differs from frozen Wave12: {name}")
        dependencies = set()
        try:
            tree = ast.parse(content, filename=str(source))
        except SyntaxError as failure:
            raise ValueError(f"Stage-profile helper cannot be parsed: {name}") from failure
        for item in ast.walk(tree):
            if isinstance(item, ast.Import):
                modules = [alias.name for alias in item.names]
            elif isinstance(item, ast.ImportFrom):
                if item.level:
                    raise ValueError(f"Stage-profile flat helper has an unsupported relative import: {name}")
                modules = [item.module] if item.module else []
            else:
                continue
            for module in modules:
                head = module.split(".")[0]
                dependency = f"{head}.py"
                if dependency in allowed or (directory / dependency).exists() or (directory / head).exists():
                    if dependency not in allowed or module != head:
                        raise ValueError(f"Stage-profile helper has an unexpected local dependency: {name}: {module}")
                    dependencies.add(dependency)
                elif head not in EXTERNAL_MODULES:
                    raise ValueError(f"Stage-profile helper has an unresolved dependency: {name}: {module}")
        hashes[name], imports[name] = digest, sorted(dependencies)
        pending.update(dependencies - hashes.keys())
    for name, digest in hashes.items():
        source = directory / name
        if source.is_symlink() or not source.is_file() or sha256(source) != digest:
            raise ValueError(f"Stage-profile helper changed during closure reconstruction: {name}")
        if name not in HELPERS:
            source = parent / name
            if source.is_symlink() or not source.is_file() or sha256(source) != inherited[name]:
                raise ValueError(f"Frozen Wave12 helper changed during closure reconstruction: {name}")
    return {
        "helpers_sha256": dict(sorted(hashes.items())),
        "helper_imports": dict(sorted(imports.items())),
        "new_helpers_sha256": {name: hashes[name] for name in sorted(HELPERS)},
        "inherited_helpers_sha256": {name: hashes[name] for name in sorted(hashes.keys() - set(HELPERS))},
    }


def original(root: Path) -> dict:
    """Validate the parent with its frozen read-only check and independently verify source archives."""
    root = root.resolve()
    binding_path, helpers = paths(root)
    binding_digest, frozen = sha256(binding_path), read(binding_path)
    local_binding = helper_binding(frozen)
    previous = Path(frozen["candidate"])
    if (
        frozen["status"] != PARENT_STATUS
        or frozen["root"] != str(root)
        or frozen["helpers"] != str(helpers)
        or frozen["candidate"] != str(previous.resolve())
        or frozen["environment"] != ENVIRONMENT
        or frozen["arms"] != list(ARMS)
        or frozen["nodes"] != list(NODES)
        or candidate_inventory(previous) != frozen["files_sha256"]
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
    ):
        raise ValueError("Stage profiling requires the exact frozen Wave12 production source")
    checked = subprocess.run(
        [PYTHON, "-P", str(helpers / "selected_lr_production.py"), "--root", str(root), "--phase", "check"],
        cwd=helpers,
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(helpers)},
        capture_output=True,
        text=True,
        check=True,
        timeout=900,
    )
    if json.loads(checked.stdout) != frozen:
        raise ValueError("Frozen Wave12 check returned a different production binding")
    preparation_path, cpu_path = previous / PARENT_PREPARATION, Path(frozen["cpu_receipt"])
    prepared, cpu = read(preparation_path), read(cpu_path)
    if (
        frozen["preparation"] != str(preparation_path)
        or frozen["preparation_sha256"] != sha256(preparation_path)
        or prepared["status"] != "prepared_release_owner_gpu_pending"
        or prepared["candidate"] != str(previous)
        or prepared["files_sha256"] != frozen["files_sha256"]
        or frozen["cpu_receipt_sha256"] != sha256(cpu_path)
        or cpu["status"] != "selected_lr_release_cpu_verified_gpu_pending"
        or cpu["root"] != str(root)
        or cpu["candidate"] != str(previous)
        or cpu["files_sha256"] != frozen["files_sha256"]
        or cpu["preparation_sha256"] != frozen["preparation_sha256"]
        or cpu["command"] != cpu_command()
        or cpu["cwd"] != str(previous)
        or cpu["cuda_visible_devices"] != ""
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or any(key in cpu for key in ("error", "errors"))
        or cpu["log_sha256"] != sha256(Path(cpu["log"]))
        or cpu["archive_sha256"] != sha256(Path(cpu["archive"]))
        or frozen["archive_sha256"] != sha256(Path(frozen["archive"]))
        or frozen["data_receipt_sha256"] != sha256(root / "data/experiment_receipt.json")
    ):
        raise ValueError("Frozen Wave12 preparation or CPU source evidence differs")
    require_archive(
        Path(cpu["archive"]),
        frozen["files_sha256"]
        | {PARENT_PREPARATION: frozen["preparation_sha256"]}
        | {f"tracking/{name}": digest for name, digest in cpu["helpers_sha256"].items()},
        alias=True,
    )
    require_archive(
        Path(frozen["archive"]),
        frozen["files_sha256"]
        | {"preparation.json": frozen["preparation_sha256"]}
        | {f"tracking/{name}": digest for name, digest in frozen["helpers_sha256"].items()},
        alias=True,
    )
    if (
        sha256(binding_path) != binding_digest
        or candidate_inventory(previous) != frozen["files_sha256"]
        or sha256(cpu_path) != frozen["cpu_receipt_sha256"]
        or any(sha256(helpers / name) != digest for name, digest in frozen["helpers_sha256"].items())
        or helper_binding(frozen) != local_binding
    ):
        raise ValueError("Frozen Wave12 identities changed during read-only verification")
    return frozen


def transformed_train(previous: Path) -> bytes:
    text = replace_once(
        (previous / "experiment/train.py").read_text(),
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
        "from .selected_lr_release_owner import require_selected_lr_release_owner",
        "from .selected_lr_stage_profile_owner import require_selected_lr_stage_profile_owner\n"
        "from .selected_lr_stage_profiler import StageProfiler",
    )
    text = replace_once(
        text,
        "    args = parser.parse_args()",
        '    parser.add_argument("--profile-updates", type=int, nargs=2, metavar=("FIRST", "LAST"))\n'
        "    args = parser.parse_args()",
    )
    text = replace_once(
        text, "    require_selected_lr_release_owner(args)", "    require_selected_lr_stage_profile_owner(args)"
    )
    text = replace_once(
        text,
        "    start_step = 0",
        "    stage_profile = None\n"
        "    if args.profile_updates is not None:\n"
        "        first_update, last_update = args.profile_updates\n"
        "        stage_profile = StageProfiler(\n"
        '            args.out / "profiles" / f"rank{rank}",\n'
        "            first=first_update,\n"
        "            last=last_update,\n"
        "            rank=rank,\n"
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
    return text.encode()


def transformed_block(previous: Path) -> bytes:
    text = replace_once(
        (previous / "experiment/block_toner.py").read_text(),
        "from dataclasses import dataclass",
        "from contextlib import nullcontext\nfrom dataclasses import dataclass",
    )
    text = replace_once(text, "BLOCK_KINDS =", "\ndef profile_scope(name):\n    return nullcontext()\n\n\nBLOCK_KINDS =")
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
        '        with profile_scope("finite_check_wait"):\n            checks = torch.stack(finite).tolist()',
    )
    text = replace_once(
        text,
        "            if work is not None:\n                work.wait()",
        '            if work is not None:\n                with profile_scope("update_gather_wait"):\n'
        "                    work.wait()",
    )
    text = replace_once(
        text,
        "                if entries:\n"
        "                    torch._foreach_add_([parameter for _, _, parameter in entries], list(updates.unbind(0)))",
        "                if entries:\n"
        '                    with profile_scope("parameter_apply"):\n'
        "                        torch._foreach_add_("
        "[parameter for _, _, parameter in entries], list(updates.unbind(0)))",
    )
    return text.encode()


def transformed_files(previous: Path) -> dict[str, bytes]:
    contents = {destination: (Path(__file__).parent / source).read_bytes() for source, destination in COPIES.items()}
    test_path = COPIES["test_selected_lr_stage_profiler.py"]
    text = replace_once(
        contents[test_path].decode(),
        "from selected_lr_stage_profiler import",
        "from experiment.selected_lr_stage_profiler import",
    )
    contents[test_path] = text.encode()
    return {
        **contents,
        "experiment/train.py": transformed_train(previous),
        "experiment/block_toner.py": transformed_block(previous),
    }


def preparation_record(root: Path, candidate: Path, frozen: dict, files: dict, helpers: dict) -> dict:
    previous, cpu = Path(frozen["candidate"]), read(Path(frozen["cpu_receipt"]))
    return {
        "status": STATUS,
        "root": str(root),
        "candidate": str(candidate),
        "parent_candidate": str(previous),
        "parent_release": str(paths(root)[0]),
        "parent_release_sha256": sha256(paths(root)[0]),
        "parent_preparation_sha256": frozen["preparation_sha256"],
        "parent_cpu_receipt": frozen["cpu_receipt"],
        "parent_cpu_receipt_sha256": frozen["cpu_receipt_sha256"],
        "parent_cpu_archive": cpu["archive"],
        "parent_cpu_archive_sha256": cpu["archive_sha256"],
        "parent_controller_cpu_receipt": frozen["controller_cpu_receipt"],
        "parent_controller_cpu_receipt_sha256": frozen["controller_cpu_receipt_sha256"],
        "parent_archive": frozen["archive"],
        "parent_archive_sha256": frozen["archive_sha256"],
        "parent_metadata_sha256": {path.name: sha256(path) for path in sorted(previous.glob("*.json"))},
        "data_receipt_sha256": frozen["data_receipt_sha256"],
        "files_sha256": files,
        "changed_files": sorted(CHANGES),
        **helpers,
        "arms": list(ARMS),
        "world_size": 8,
        "profile_updates": [FIRST_UPDATE, LAST_UPDATE],
        "policy": POLICY,
    }


def prepare(root: Path, candidate: Path) -> dict:
    root, candidate = root.resolve(), candidate.resolve()
    frozen = original(root)
    previous = Path(frozen["candidate"])
    if candidate.exists() or candidate.is_relative_to(previous) or candidate.is_relative_to(paths(root)[1]):
        raise FileExistsError("Stage profiling requires a distinct unused candidate")
    if (previous / PREPARATION).exists() or any(name in frozen["files_sha256"] for name in COPIES.values()):
        raise ValueError("Frozen Wave12 source already contains a stage-profile successor")
    helpers = helper_binding(frozen)
    contents = transformed_files(previous)
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name, content in contents.items():
        (candidate / name).write_bytes(content)
    files = candidate_inventory(candidate)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    if changed != CHANGES or helper_binding(frozen) != helpers:
        raise ValueError("Stage-profile preparation exceeds its exact five-file scope or helper identity")
    result = preparation_record(root, candidate, frozen, files, helpers)
    write(candidate / PREPARATION, result)
    require_preparation(root, candidate)
    return result


def require_preparation(root: Path, candidate: Path) -> dict:
    root, candidate = root.resolve(), candidate.resolve()
    frozen = original(root)
    helpers = helper_binding(frozen)
    prepared, files = read(candidate / PREPARATION), candidate_inventory(candidate)
    expected = preparation_record(root, candidate, frozen, files, helpers)
    changed = {
        name
        for name in files.keys() | frozen["files_sha256"].keys()
        if files.get(name) != frozen["files_sha256"].get(name)
    }
    if (
        prepared.keys() != expected.keys() | {"time"}
        or any(prepared[key] != value for key, value in expected.items())
        or changed != CHANGES
        or any(
            (candidate / name).read_bytes() != content
            for name, content in transformed_files(Path(frozen["candidate"])).items()
        )
        or any(sha256(candidate / name) != digest for name, digest in expected["parent_metadata_sha256"].items())
        or helper_binding(frozen) != helpers
    ):
        raise ValueError("Stage-profile preparation differs from its exact source and instrumentation scope")
    return prepared


def archive_inventory(prepared: dict, digest: str) -> dict:
    return (
        prepared["files_sha256"]
        | prepared["parent_metadata_sha256"]
        | {PREPARATION: digest}
        | {f"tracking/{name}": digest for name, digest in prepared["helpers_sha256"].items()}
    )


def cpu(root: Path, candidate: Path, out: Path) -> dict:
    root, candidate, out = root.resolve(), candidate.resolve(), out.resolve()
    prepared = require_preparation(root, candidate)
    if out.exists() or any(
        out.is_relative_to(path) for path in (candidate, Path(prepared["parent_candidate"]), paths(root)[1])
    ):
        raise FileExistsError("Stage-profile CPU evidence requires an unused external directory")
    digest = sha256(candidate / PREPARATION)
    out.mkdir(parents=True, exist_ok=False)
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
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            code, error = 124, "CPU timeout at 600 seconds"
        except OSError as failure:
            code, error = 127, f"{type(failure).__name__}: {failure}"
    try:
        with tarfile.open(archive, "x") as bundle:
            for name in prepared["files_sha256"] | prepared["parent_metadata_sha256"] | {PREPARATION: digest}:
                bundle.add(candidate / name, arcname=name, recursive=False)
            bundle.add(candidate / "artifacts/external-sources", arcname="artifacts/external-sources", recursive=False)
            for name in prepared["helpers_sha256"]:
                bundle.add(Path(__file__).parent / name, arcname=f"tracking/{name}", recursive=False)
        require_archive(archive, archive_inventory(prepared, digest), alias=True)
        if require_preparation(root, candidate) != prepared or sha256(candidate / PREPARATION) != digest:
            raise ValueError("Stage-profile source changed during CPU verification")
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, tarfile.TarError) as failure:
        error = str(failure) if error is None else f"{error}; {failure}"
    result = {
        "status": (
            CPU_STATUS if type(code) is int and code == 0 and error is None else "selected_lr_stage_profile_cpu_failed"
        ),
        "root": str(root),
        "candidate": str(candidate),
        "preparation_sha256": digest,
        "files_sha256": prepared["files_sha256"],
        "helpers_sha256": prepared["helpers_sha256"],
        "parent_release_sha256": prepared["parent_release_sha256"],
        "command": command,
        "cwd": str(candidate),
        "cuda_visible_devices": "",
        "exit_code": code,
        "log": str(log),
        "log_sha256": sha256(log),
        "archive": str(archive),
        "archive_sha256": sha256(archive) if archive.is_file() else None,
    }
    if error is not None:
        result["error"] = error
    write(out / "receipt.json", result)
    if result["status"] != CPU_STATUS:
        raise RuntimeError(f"Stage-profile CPU failure preserved at {out}")
    return result


def require_cpu(root: Path, receipt_path: Path) -> dict:
    receipt = read(receipt_path)
    candidate = Path(receipt["candidate"])
    prepared = require_preparation(root, candidate)
    if (
        receipt["status"] != CPU_STATUS
        or receipt["root"] != str(root.resolve())
        or receipt["files_sha256"] != prepared["files_sha256"]
        or receipt["helpers_sha256"] != prepared["helpers_sha256"]
        or receipt["parent_release_sha256"] != prepared["parent_release_sha256"]
        or receipt["preparation_sha256"] != sha256(candidate / PREPARATION)
        or receipt["command"] != cpu_command()
        or receipt["cwd"] != str(candidate.resolve())
        or receipt["cuda_visible_devices"] != ""
        or type(receipt["exit_code"]) is not int
        or receipt["exit_code"] != 0
        or any(key in receipt for key in ("error", "errors"))
        or receipt["log_sha256"] != sha256(Path(receipt["log"]))
        or receipt["archive_sha256"] != sha256(Path(receipt["archive"]))
    ):
        raise ValueError("Stage-profile CPU command, source, archive or outcome differs")
    require_archive(Path(receipt["archive"]), archive_inventory(prepared, receipt["preparation_sha256"]), alias=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "cpu", "check"), default="prepare")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "cpu" and args.out is None:
        parser.error("CPU evidence requires --out")
    if args.phase == "cpu":
        result = cpu(args.root, args.candidate, args.out)
    elif args.phase == "check":
        result = require_preparation(args.root, args.candidate)
    else:
        result = prepare(args.root, args.candidate)
    print(json.dumps({"status": result["status"], "candidate": result["candidate"]}))


if __name__ == "__main__":
    main()
