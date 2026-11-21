# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-bound two-rank A100 acceptance gates for related optimizer controls."""

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist

from .go_muon_gpu_checks import run_checks as check_go_muon
from .launch import RELATED_CHECKS
from .mousse_gpu_checks import run_checks as check_mousse
from .records import atomic_json, sha256, utc_now
from .reference import TASK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("Related-method acceptance gates require A100 SM80")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("nccl", device_id=device)
    try:
        results = {**check_go_muon(device), **check_mousse(device)}
        checks = {name: bool(results[name]) for name in RELATED_CHECKS}
        if not all(checks.values()):
            raise RuntimeError("Related-method numerical acceptance failed")
        if dist.get_rank() == 0:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(
                args.receipt,
                {
                    "time": utc_now(),
                    "checks": checks,
                    "evidence": results,
                    "world_size": dist.get_world_size(),
                    "torch": torch.__version__,
                    "source_sha256": {
                        str(path.relative_to(TASK)): sha256(path) for path in sorted((TASK / "experiment").rglob("*.py"))
                    },
                    "limitation": "Small numerical acceptance probes do not establish quality or full-model memory fit.",
                },
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
