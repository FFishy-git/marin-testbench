# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atomic rank-local checkpoints and auditable execution manifests."""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.distributed as dist

from .optimizer import fast_runtime_state, restore_fast
from .reference import REVISION, TASK, UPSTREAM

CHECKPOINT_SCHEMA = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(UTC).isoformat()


def checkpoint_identity(config, data_receipt):
    files = sorted((TASK / "experiment").rglob("*.py"))
    files += sorted((UPSTREAM / "reproduce" / "marin").glob("*.py"))
    files += sorted((UPSTREAM / "src" / "toner").rglob("*.py"))
    return {
        "schema": CHECKPOINT_SCHEMA,
        "source_sha256": {str(path.relative_to(TASK)): sha256(path) for path in files},
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "data_receipt_sha256": sha256(data_receipt),
    }


def atomic_json(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(temporary, path)


def manifest(config, data, data_receipt):
    files = sorted((TASK / "experiment").rglob("*.py"))
    files += sorted((UPSTREAM / "reproduce" / "marin").glob("*.py"))
    files += sorted((UPSTREAM / "src" / "toner").rglob("*.py"))
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).splitlines()[0]
    repo = TASK.parents[2]
    revision = (
        subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if (repo / ".git").exists()
        else None
    )
    return {
        "start_time": utc_now(),
        "status": "running",
        "command": sys.argv,
        "pid": os.getpid(),
        "hostname": platform.node(),
        "source_revision": REVISION,
        "experiment_git_revision": revision,
        "source_sha256": {str(p.relative_to(TASK)): sha256(p) for p in files},
        "config": config,
        "data": str(data.resolve()),
        "data_receipt_sha256": sha256(data_receipt),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": driver,
        "gpu": torch.cuda.get_device_name(),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
    }


def save_checkpoint(out, model, optimizers, fast, output, *, step, config, rank, world, identity):
    directory = out / "checkpoints" / f"step_{step:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "next_global_sequence": step * config["training"]["global_batch_sequences"],
        "config": config,
        "rank": rank,
        "world": world,
        "identity": identity,
        "optimizers": [None if optimizer is fast else optimizer.state_dict() for optimizer in optimizers],
        "fast": None if fast is None else fast_runtime_state(fast),
        "output": None if output is None else output.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None,
        "python_rng": random.getstate(),
    }
    if rank == 0:
        state["model"] = model.state_dict()
    temporary = directory / f"rank_{rank}.pt.tmp"
    path = directory / f"rank_{rank}.pt"
    if path.exists():
        raise FileExistsError(path)
    torch.save(state, temporary)
    os.replace(temporary, path)
    if world > 1:
        dist.barrier()
    if rank == 0:
        atomic_json(directory / "complete.json", {"step": step, "world": world, "time": utc_now()})
        atomic_json(out / "latest_checkpoint.json", {"path": str(directory.resolve()), "step": step})


def compare_continuation(uninterrupted, resumed, world):
    """Compare actual checkpoint tensors and counters for all ranks."""
    comparisons = []
    for rank in range(world):
        actual = torch.load(resumed / f"rank_{rank}.pt", map_location="cpu", weights_only=False)
        expected = torch.load(uninterrupted / f"rank_{rank}.pt", map_location="cpu", weights_only=False)
        if actual["step"] != expected["step"] or actual["next_global_sequence"] != expected["next_global_sequence"]:
            raise ValueError("Checkpoint step or data position differs after continuation")
        if actual["identity"] != expected["identity"]:
            raise ValueError("Checkpoint source or data identity differs after continuation")
        for name in ("model", "optimizers", "fast", "output", "torch_rng", "cuda_rng", "python_rng"):
            if name not in expected:
                continue
            compare_tree(actual[name], expected[name], f"rank_{rank}.{name}", comparisons)
    return {"tensor_comparisons": len(comparisons), "max_absolute_error": max(comparisons, default=0.0)}


def compare_tree(actual, expected, name, comparisons):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6, msg=name)
        comparisons.append((actual.double() - expected.double()).abs().max().item() if expected.numel() else 0.0)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"Checkpoint keys differ: {name}")
        for key in expected:
            compare_tree(actual[key], expected[key], f"{name}.{key}", comparisons)
    elif isinstance(expected, (tuple, list)):
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            compare_tree(a, e, f"{name}.{index}", comparisons)
    elif actual != expected:
        raise ValueError(f"Checkpoint scalar differs: {name}")


def restore_checkpoint(directory, model, optimizers, fast, output, *, config, rank, world, identity):
    receipt = json.loads((directory / "complete.json").read_text())
    if receipt["world"] != world:
        raise ValueError("Checkpoint ownership requires the original world size")
    master = torch.load(directory / "rank_0.pt", map_location="cpu", weights_only=False)
    if master["config"] != config:
        raise ValueError("Resume configuration differs from saved configuration")
    if master["identity"] != identity:
        raise ValueError("Resume source, configuration, or prepared-data identity differs from checkpoint")
    state = master if rank == 0 else torch.load(directory / f"rank_{rank}.pt", map_location="cpu", weights_only=False)
    if state["identity"] != identity or state["rank"] != rank or state["world"] != world:
        raise ValueError("Rank-local checkpoint identity or ownership differs")
    if master["rank"] != 0 or master["world"] != world:
        raise ValueError("Master checkpoint ownership differs")
    if master["step"] != receipt["step"] or state["step"] != receipt["step"]:
        raise ValueError("Rank-local checkpoint step differs from completion receipt")
    model.load_state_dict(master["model"])
    for optimizer, saved in zip(optimizers, state["optimizers"], strict=True):
        if optimizer is not fast:
            optimizer.load_state_dict(saved)
    if fast is not None:
        restore_fast(fast, state["fast"])
    if output is not None:
        output.load_state_dict(state["output"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return state["step"]
