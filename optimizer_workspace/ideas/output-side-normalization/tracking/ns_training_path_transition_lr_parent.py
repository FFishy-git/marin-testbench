# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep numerical and terminal archives isolated across the transition-LR boundary."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

from ns_training_path_predecessor import accepted_archive
from root_stage_failed_predecessor import file_identity
from selected_lr_release_owner import sha256

CPU = "run-preparation/ns-training-path-transition-actual-lr-predecessor-cpu/staged-20261119T1748/receipt.json"
CPU_SHA256 = "c119d1b53146cbfec8065b496535e1b9c8ebd73b78687f9d589fbf6ba0777762"
STATUS = "transition_actual_lr_predecessor_cpu_archive_verified_execution_archives_separate"
VERIFIER = "ns_training_path_transition_lr_predecessor.py"
VERIFIER_SHA256 = "f38cf4e4c991ed960b20f9d6849e248100bc96cfb5672362c080dc39078bc5f9"
CHECKER = "prepare_ns_training_path_transition_lr_predecessor.py"
CHECKER_SHA256 = "49472305ac5b0f9a30f0e29ba2c40aa7f2f2419363f955463aadeb317f5cb726"
SEAL = "run-preparation/ns-training-path-transition-actual-lr-predecessor-20261119-seal.json"
NUMERICAL = (
    "run-preparation/ns-training-path-actual-lr-component-cpu/20261119T0926/receipt.json",
    "aca115258fca1cfb4adf4bbfe58b6606f1b63b022bf0d10525efcb3d9d7a4251",
    "ns_training_path_actual_lr_component_cpu_archive_verified_owner_unissued",
    29,
)
ADAPTER = (
    "run-preparation/ns-training-path-actual-lr-adapter-cpu/20261119T1020/receipt.json",
    "82c1ad5b154041485d80333954260e7b8b2b67aca7feedaf3868ca3e87dee5f1",
    "ns_training_path_actual_lr_adapter_cpu_archive_verified_execution_owner_unissued",
    14,
)
PYTHON = "/opt/conda/envs/torch-base/bin/python"
ENVIRONMENT = {"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONFAULTHANDLER": "1"}
PROGRAM = """import json,sys
from pathlib import Path
import prepare_ns_training_path_transition_lr_predecessor as preparation
import ns_training_path_transition_lr_predecessor as predecessor
root,path,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
cpu=preparation.accepted(root,path)
value=cpu if mode=='cpu' else predecessor.require_seal(root,sys.argv[4])
print(json.dumps(value,allow_nan=False))
"""


def require_root(root: Path) -> None:
    if root != root.resolve(strict=True) or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Transition-LR parent inspection requires canonical root and hidden CUDA")


def cpu_sources(root: Path) -> dict:
    """Authenticate the separate terminal-reader closure before starting its read-only child."""
    require_root(root)
    path = root / CPU
    identity = file_identity(path)
    raw = path.read_bytes()
    cpu = json.loads(raw)
    helpers = path.parent / "tracking"
    if (
        hashlib.sha256(raw).hexdigest() != CPU_SHA256
        or cpu["root"] != str(root)
        or cpu["status"] != STATUS
        or cpu["helpers"] != str(helpers)
        or cpu["environment"] != ENVIRONMENT
        or type(cpu["exit_code"]) is not int
        or cpu["exit_code"] != 0
        or "error" in cpu
        or cpu["helpers_sha256"][VERIFIER] != VERIFIER_SHA256
        or cpu["helpers_sha256"][CHECKER] != CHECKER_SHA256
        or sha256(helpers / VERIFIER) != VERIFIER_SHA256
        or sha256(helpers / CHECKER) != CHECKER_SHA256
        or file_identity(path) != identity
    ):
        raise ValueError("Transition-LR separate terminal CPU provenance differs")
    return cpu


def inspect(root: Path, mode: str, digest: str = "") -> dict:
    cpu = cpu_sources(root)
    helpers = Path(cpu["helpers"])
    result = subprocess.run(
        [PYTHON, "-B", "-P", "-c", PROGRAM, str(root), str(root / CPU), mode, digest],
        cwd=helpers,
        env=os.environ | ENVIRONMENT | {"PYTHONPATH": str(helpers), "UV_OFFLINE": "1"},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Transition-LR isolated terminal reader refused evidence (exit {result.returncode})")
    if cpu_sources(root) != cpu:
        raise ValueError("Transition-LR terminal parent changed during isolated inspection")
    return json.loads(result.stdout)


def require_cpu(root: Path) -> dict:
    cpu = cpu_sources(root)
    if inspect(root, "cpu") != cpu:
        raise ValueError("Transition-LR complete separate CPU authentication differs")
    return cpu


def require_seal(root: Path, digest: str) -> dict:
    require_root(root)
    return inspect(root, "seal", digest)


def adapter_cpu(root: Path) -> dict:
    """Authenticate existing numerical preparation without executing its old ownership gate."""
    require_root(root)
    numerical, adapter = (accepted_archive(root, spec) for spec in (NUMERICAL, ADAPTER))
    changed = {
        name
        for name in numerical["files_sha256"].keys() | adapter["files_sha256"].keys()
        if numerical["files_sha256"].get(name) != adapter["files_sha256"].get(name)
    }
    if (
        changed != {"experiment/fused_attribution_owner.py"}
        or any(adapter["helpers_sha256"].get(name) != digest for name, digest in numerical["helpers_sha256"].items())
        or adapter["parent_cpu_receipt_sha256"] != "2ad5cee992e97594b0a3b9d79680e6fa273435dbb6885bafbb2fa82d19b36230"
    ):
        raise ValueError("Transition-LR numerical adapter changed its accepted component")
    return adapter
