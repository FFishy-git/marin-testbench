# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare rank-owned outer factors without changing the issued Wave-9 source or GPU release."""

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

ARMS = ("outer_leftdiag_upstream_fused", "null_leftdiag_upstream_fused")
CHANGES = {"experiment/block_toner.py", "experiment/tests/test_owned_output_factors.py"}
HELPERS = ("prepare_owned_output.py", "owned_output_factor_checks.py")
OWNED_CLASS = '''class OwnedOutputMoments(DiagonalMoments):
    """Keep all pooled moments, constructing only each rank's used outer factors."""

    def __init__(self, bufs, *, rank, layers, **settings):
        super().__init__(bufs, **settings)
        counts = layer_counts(layers, self.world)
        first, last = sum(counts[:rank]), sum(counts[: rank + 1])
        self.owned = {}
        for width, keys in self.keys.items():
            selected = [(index, key) for index, key in enumerate(keys) if first <= key[0] < last]
            if selected:
                begin = selected[0][0]
                owned_keys = tuple(key for _, key in selected)
                if keys[begin : begin + len(selected)] != list(owned_keys):
                    raise ValueError("Owned output rows must form a contiguous layer partition")
                self.owned[width] = (slice(begin, begin + len(selected)), owned_keys)

    @torch.no_grad()
    def factors(self):
        factors = {}
        for width, (selection, keys) in self.owned.items():
            moment = self.moment[width][selection]
            trace = moment.sum(-1, keepdim=True)
            positive = trace > 0
            normalized = moment * (width / trace.clamp_min(torch.finfo(moment.dtype).tiny))
            normalized = torch.where(positive, normalized.clamp_min(0), torch.ones_like(moment)) + self.damping
            outer = torch.where(positive, normalized.pow(-self.outer_power), torch.ones_like(moment))
            for index, key in enumerate(keys):
                factors[key] = (None, outer[index])
        return factors


'''


def prepare(root: Path, candidate: Path) -> dict:
    """Copy verified production bytes and retain their unchanged owner, configs and statistics."""
    frozen = release(root, wave="wave9")
    previous = Path(frozen["candidate"])
    if candidate_inventory(previous) != frozen["files_sha256"]:
        raise ValueError("Issued Wave-9 numerical source changed before preparation")
    if candidate.exists() or candidate.resolve().is_relative_to(previous.resolve()):
        raise FileExistsError("Owned output preparation requires a distinct unused candidate")
    shutil.copytree(previous, candidate, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    block_path = candidate / "experiment/block_toner.py"
    text = replace_once(block_path.read_text(), "def diagonal_roots(", OWNED_CLASS + "def diagonal_roots(")
    text = replace_once(
        text,
        "        self.output_diagonal = (\n"
        "            DiagonalMoments(\n"
        "                {key: rows for key, rows in self.output_rows.items() if key[1] in diagonal_kinds},\n"
        "                **diagonal,\n"
        "            )",
        "        output_moments = OwnedOutputMoments if self.arm.upstream_product_kernels else DiagonalMoments\n"
        '        output_ownership = {"rank": rank, "layers": layers} if self.arm.upstream_product_kernels else {}\n'
        "        self.output_diagonal = (\n"
        "            output_moments(\n"
        "                {key: rows for key, rows in self.output_rows.items() if key[1] in diagonal_kinds},\n"
        "                **diagonal,\n"
        "                **output_ownership,\n"
        "            )",
    )
    block_path.write_text(text)
    helpers = Path(__file__).parent
    shutil.copy2(helpers / "owned_output_factor_checks.py", candidate / "experiment/tests/test_owned_output_factors.py")
    files = candidate_inventory(candidate)
    changed = {
        name for name in files.keys() | frozen["files_sha256"] if files.get(name) != frozen["files_sha256"].get(name)
    }
    if changed != CHANGES:
        raise ValueError(f"Owned output preparation exceeds scope: {sorted(changed ^ CHANGES)}")
    record = {
        "status": "owned_output_prepared_gpu_denied",
        "candidate": str(candidate.resolve()),
        "predecessor_candidate": str(previous),
        "predecessor_release_sha256": sha256(root / "run-preparation/wave9-launch-binding.json"),
        "files_sha256": files,
        "changed_files": sorted(changed),
        "helpers_sha256": {name: sha256(helpers / name) for name in HELPERS},
        "arms": ARMS,
        "policy": (
            "Unchanged full pooled EMA/reduction/checkpoints, cotangent collection, input kernels and configurations. "
            "Only rank-owned outer output factors constructed; null retains the same factor work. "
            "Existing ownership rejects this distinct source. No GPU release, queue or training launch."
        ),
    }
    write(candidate / "owned-output-preparation.json", record)
    return record


def cpu(root: Path, candidate: Path, output: Path) -> dict:
    """Record real candidate tests and immutable source evidence with CUDA hidden."""
    path = candidate / "owned-output-preparation.json"
    prepared = json.loads(path.read_text())
    frozen = release(root, wave="wave9")
    files = candidate_inventory(candidate)
    helpers = Path(__file__).parent
    hashes = {name: sha256(helpers / name) for name in HELPERS}
    if (
        prepared["candidate"] != str(candidate.resolve())
        or prepared["files_sha256"] != files
        or prepared["helpers_sha256"] != hashes
        or prepared["predecessor_release_sha256"] != sha256(root / "run-preparation/wave9-launch-binding.json")
        or candidate_inventory(Path(frozen["candidate"])) != frozen["files_sha256"]
    ):
        raise ValueError("Owned output source, helper or predecessor identity changed")
    if output.exists() or output.resolve().is_relative_to(candidate.resolve()):
        raise FileExistsError("Owned output CPU evidence needs a new external directory")
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
            env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(candidate)},
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    if files != candidate_inventory(candidate) or hashes != {name: sha256(helpers / name) for name in HELPERS}:
        raise ValueError("Owned output source changed during CPU tests")
    archive = output / "source.tar"
    with tarfile.open(archive, "x") as bundle:
        for name in files:
            bundle.add(candidate / name, arcname=f"candidate/{name}", recursive=False)
        bundle.add(
            candidate / "artifacts/external-sources", arcname="candidate/artifacts/external-sources", recursive=False
        )
        bundle.add(path, arcname="candidate/owned-output-preparation.json", recursive=False)
        for name in HELPERS:
            bundle.add(helpers / name, arcname=f"tracking/{name}", recursive=False)
    result = {
        **prepared,
        "status": "owned_output_cpu_verified_gpu_denied" if completed.returncode == 0 else "owned_output_cpu_failed",
        "preparation": str(path),
        "preparation_sha256": sha256(path),
        "command": command,
        "cwd": str(candidate),
        "exit_code": completed.returncode,
        "log": str(log),
        "log_sha256": sha256(log),
        "cuda_visible_devices": "",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
    }
    write(output / "receipt.json", result)
    if completed.returncode:
        raise RuntimeError(f"Owned output CPU tests failed; preserved at {output}")
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
    result = prepare(args.root, args.candidate) if args.phase == "prepare" else cpu(args.root, args.candidate, args.out)
    print(json.dumps({"status": result["status"], "candidate": result["candidate"]}))


if __name__ == "__main__":
    main()
