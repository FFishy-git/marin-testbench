# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy an isolated upstream-product/null candidate without granting GPU execution."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from boundary_handoff import release
from launch_wave5 import PYTHON, read, sha256
from watch_wave5 import write
from wave7_release import candidate_inventory

ARMS = ("outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused")
BATCHES = ("0p5m",)


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError("Upstream-product source insertion differs from its expected frozen reference")
    return text.replace(before, after, 1)


def prepare(root: Path, candidate: Path) -> dict:
    frozen = release(root)
    previous = Path(frozen["candidate"])
    original = read(root / "run-preparation/wave8-candidate.json")
    if candidate_inventory(previous) != original["files_sha256"]:
        raise ValueError("Original Wave-8 source changed before isolated candidate preparation")
    if candidate.exists():
        raise FileExistsError("Candidate already has evidence; preserve it without preparing again")
    configurations = {
        batch: read(previous / f"experiment/configs/outer_leftdiag_fused_{batch}_seed0.json") for batch in BATCHES
    }
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    prototype = Path(__file__).with_name("upstream_outer_products.py")
    module = candidate / "experiment/upstream_outer_products.py"
    shutil.copy2(prototype, module)
    block_path = candidate / "experiment/block_toner.py"
    block = block_path.read_text()
    block = replace_once(
        block,
        "from .toner_state import engine_state, restore_engine",
        "from .toner_state import engine_state, restore_engine\n"
        "from .upstream_outer_products import upstream_outer_update_stack",
    )
    block = replace_once(
        block,
        "    combine_outer_products: bool = False",
        "    combine_outer_products: bool = False\n    upstream_product_kernels: bool = False",
    )
    block = replace_once(
        block,
        '    "outer_leftdiag_combined_fused": BlockArm(',
        '    "outer_leftdiag_upstream_fused": BlockArm(\n'
        "        Side.FULL, Side.DIAGONAL, Side.DIAGONAL,\n"
        "        apply_left=True, apply_left_inner=False, upstream_product_kernels=True,\n"
        "    ),\n"
        '    "null_leftdiag_upstream_fused": BlockArm(\n'
        "        Side.FULL, Side.DIAGONAL, Side.DIAGONAL,\n"
        "        apply_left=False, apply_left_inner=False, upstream_product_kernels=True,\n"
        "    ),\n"
        '    "outer_leftdiag_combined_fused": BlockArm(',
    )
    block = replace_once(
        block,
        "        self.steps = 0\n        # Left factors",
        "        self.steps = 0\n"
        "        self.update_scalars = torch.empty(2, device=device) if self.arm.upstream_product_kernels else None\n"
        "        # Left factors",
    )
    block = replace_once(
        block,
        "        group = self.param_groups[0]\n        right = self._factors",
        "        group = self.param_groups[0]\n"
        '        update_lr, update_decay = group["lr"], group["weight_decay"]\n'
        "        if self.arm.upstream_product_kernels:\n"
        "            scalars = torch.tensor([update_lr, update_decay])\n"
        '            if self.update_scalars.device.type == "cuda":\n'
        "                scalars = scalars.pin_memory()\n"
        "            self.update_scalars.copy_(scalars, non_blocking=True)\n"
        "            update_lr, update_decay = self.update_scalars.unbind(0)\n"
        "        right = self._factors",
    )
    block = replace_once(
        block,
        "                update = combined_outer_update_stack "
        "if self.arm.combine_outer_products else block_update_stack",
        "                update = (\n"
        "                    upstream_outer_update_stack if self.arm.upstream_product_kernels\n"
        "                    else combined_outer_update_stack "
        "if self.arm.combine_outer_products else block_update_stack\n"
        "                )",
    )
    block = replace_once(
        block,
        '                    lr=group["lr"],\n'
        '                    weight_decay=group["weight_decay"],\n'
        "                    trace=trace,",
        "                    lr=update_lr,\n"
        "                    weight_decay=update_decay,\n"
        "                    trace=trace,",
    )
    block_path.write_text(block)
    configure_path = candidate / "experiment/configure.py"
    configure = replace_once(
        configure_path.read_text(),
        '    "outer_leftdiag_combined_fused",',
        '    "outer_leftdiag_combined_fused",\n'
        '    "outer_leftdiag_upstream_fused",\n'
        '    "null_leftdiag_upstream_fused",',
    )
    configure_path.write_text(configure)
    for arm in ARMS:
        for batch in BATCHES:
            config = json.loads(json.dumps(configurations[batch]))
            config["experiment"]["arm"] = arm
            path = candidate / f"experiment/configs/{arm}_{batch}_seed0.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
    owner = candidate / "experiment/execution_owner.py"
    owner.write_text(
        "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
        '"""Unvalidated upstream-product candidate: CUDA execution is not released."""\n\n'
        "def require_gpu_owner():\n"
        '    raise RuntimeError("Upstream-product candidate has no GPU validation release")\n\n'
        "def require_training_owner(args, config):\n"
        '    raise RuntimeError("Upstream-product candidate has no bounded validation or training release")\n'
    )
    cluster_path = candidate / "experiment/cluster.json"
    cluster = read(cluster_path)
    cluster["wave_name"] = "upstream-products-unvalidated"
    cluster["wave"] = [{"arm": arm, "seed": 0, "batch": "0p5m"} for arm in ("muont", *ARMS)]
    write(cluster_path, cluster)
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | original["files_sha256"] if files.get(name) != original["files_sha256"].get(name)
    }
    expected = {
        "experiment/block_toner.py",
        "experiment/configure.py",
        "experiment/cluster.json",
        "experiment/execution_owner.py",
        "experiment/upstream_outer_products.py",
        *(f"experiment/configs/{arm}_{batch}_seed0.json" for arm in ARMS for batch in BATCHES),
    }
    if changed != expected:
        raise ValueError(f"Isolated candidate source scope differs: {sorted(changed ^ expected)}")
    result = {
        "status": "implemented_cpu_checks_pending_gpu_execution_denied",
        "candidate": str(candidate.resolve()),
        "predecessor_release_sha256": sha256(root / "run-preparation/wave8-launch-binding.json"),
        "predecessor_candidate_sha256": sha256(root / "run-preparation/wave8-candidate.json"),
        "preparation_helper_sha256": sha256(Path(__file__)),
        "prototype_sha256": sha256(prototype),
        "files_sha256": files,
        "changed_files": sorted(changed),
        "arms": ARMS,
        "policy": (
            "No GPU release, queue or launch. Independently validate pinned split kernels, null exactness, "
            "production arithmetic, all-rank continuation and measured cost after active jobs restore allocations."
        ),
    }
    write(candidate / "preparation.json", result)
    return result


def cpu_evidence(root: Path, candidate: Path, output: Path) -> dict:
    """Save fresh CPU-only test logs and a separate inventory without replacing preparation evidence."""
    initial_path = candidate / "preparation.json"
    initial = read(initial_path)
    frozen = release(root)
    previous = Path(frozen["candidate"])
    original = read(root / "run-preparation/wave8-candidate.json")
    if (
        initial["candidate"] != str(candidate.resolve())
        or initial["predecessor_release_sha256"] != sha256(root / "run-preparation/wave8-launch-binding.json")
        or initial["predecessor_candidate_sha256"] != sha256(root / "run-preparation/wave8-candidate.json")
        or candidate_inventory(previous) != original["files_sha256"]
    ):
        raise ValueError("CPU candidate preparation no longer matches the preserved Wave-8 predecessor")
    files = candidate_inventory(candidate)
    changed = sorted(
        name for name in files.keys() | original["files_sha256"] if files.get(name) != original["files_sha256"].get(name)
    )
    expected = set(initial["changed_files"]) | {
        "experiment/tests/test_block_toner.py",
        "experiment/tests/test_upstream_release.py",
        "experiment/tests/test_validation.py",
    }
    if set(changed) != expected:
        raise ValueError("CPU candidate changed outside its implementation and candidate-specific tests")
    if output.exists() or output.resolve().is_relative_to(candidate.resolve()):
        raise ValueError("CPU evidence needs a new directory outside the candidate")
    output.mkdir(parents=True, exist_ok=False)
    helpers = Path(__file__).resolve().parent
    helper_files = {
        name: sha256(helpers / name)
        for name in ("prepare_upstream_outer.py", "upstream_outer_products.py", "test_upstream_outer_products.py")
    }
    prefix = [
        "uv",
        "run",
        "--no-project",
        "--python",
        PYTHON,
        "--with",
        "pytest",
        "--with",
        "pytest-timeout",
        "--with",
        "pytest-xdist",
        "python",
        "-m",
        "pytest",
        "-n",
        "0",
        "-q",
    ]
    environment = os.environ | {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": f"{helpers}:/opt/conda/envs/torch-base/lib/python3.13/site-packages",
    }
    results = []
    for label, directory, tests in (
        ("candidate", candidate, ["experiment/tests"]),
        ("equations", helpers.parent, ["tracking/test_upstream_outer_products.py"]),
    ):
        log = output / f"{label}.log"
        command = [*prefix, *tests]
        with log.open("w") as stream:
            completed = subprocess.run(
                command, cwd=directory, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=600
            )
        results.append(
            {
                "command": command,
                "cwd": str(directory),
                "exit_code": completed.returncode,
                "log": str(log),
                "log_sha256": sha256(log),
            }
        )
        if completed.returncode:
            raise RuntimeError(f"CPU verification failed; preserve its evidence at {log}")
    if files != candidate_inventory(candidate) or helper_files != {
        name: sha256(helpers / name) for name in helper_files
    }:
        raise ValueError("Candidate or helper source changed while collecting CPU evidence")
    archive = output / "source.tar"
    with tarfile.open(archive, "w") as bundle:
        for name in files:
            bundle.add(candidate / name, arcname=f"candidate/{name}", recursive=False)
        bundle.add(
            candidate / "artifacts/external-sources", arcname="candidate/artifacts/external-sources", recursive=False
        )
        bundle.add(initial_path, arcname="candidate/preparation.json", recursive=False)
        for name in helper_files:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    result = {
        "status": "cpu_verified_gpu_execution_denied",
        "candidate": str(candidate.resolve()),
        "initial_preparation": str(initial_path),
        "initial_preparation_sha256": sha256(initial_path),
        "predecessor_release_sha256": initial["predecessor_release_sha256"],
        "predecessor_candidate_sha256": initial["predecessor_candidate_sha256"],
        "files_sha256": files,
        "changed_files": changed,
        "helpers": str(helpers),
        "helpers_sha256": helper_files,
        "commands": results,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "arms": initial["arms"],
        "cuda_visible_devices": "",
        "scope": (
            "CPU algebra, candidate behavior and fail-closed ownership only. Direct products do not emulate "
            "CUDA split kernels. No actual GPU parity, continuation, timing, validation release, queue or launch."
        ),
    }
    write(output / "receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "cpu"), default="prepare")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.phase == "cpu":
        if args.out is None:
            parser.error("--out is required for separate CPU evidence")
        result = cpu_evidence(args.root, args.candidate, args.out)
    else:
        result = prepare(args.root, args.candidate)
    print(json.dumps({"status": result["status"], "candidate": result["candidate"], "arms": result["arms"]}))


if __name__ == "__main__":
    main()
