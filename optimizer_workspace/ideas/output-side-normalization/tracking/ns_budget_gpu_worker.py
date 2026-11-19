# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Execute the unchanged archived NS component inside its separately acknowledged CUDA owner."""

import argparse
from pathlib import Path

import ns_budget_execution as execution
import ns_budget_gpu_owner as owner


def run(root: Path, binding_hash: str, state: Path, index: int) -> dict:
    acknowledged = owner.require_gpu_owner(root, binding_hash, state, index)
    _, helpers, _, work = owner.locations(root)
    if Path(__file__).resolve() != helpers / "ns_budget_gpu_worker.py":
        raise RuntimeError("NS budget worker is outside its frozen source directory")
    return execution.run(
        helpers,
        acknowledged["binding"]["helpers_sha256"],
        index,
        work / f"process-{index}.pt",
        work / f"process-{index}.json",
        lambda: owner.require_gpu_owner(root, binding_hash, state, index),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--worker-index", required=True, type=int, choices=(0, 1))
    args = parser.parse_args()
    result = run(args.root.resolve(strict=True), args.binding_sha256, args.state.resolve(), args.worker_index)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
