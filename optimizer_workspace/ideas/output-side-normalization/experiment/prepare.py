# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned corpus with a local credential and unchanged Toner code."""

import argparse
import fcntl
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from reproduce.marin import prepare_data

from .bootstrap import verify_snapshot
from .data_access import verify_data
from .records import atomic_json, utc_now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    verify_snapshot()
    token = dotenv_values(args.env_file).get("HF_TOKEN")
    if not token:
        raise ValueError("The local credential file must contain HF_TOKEN")
    os.environ["HF_TOKEN"] = token
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ[variable] = args.proxy
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / ".preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {
            "started_at": utc_now(),
            "pid": os.getpid(),
            "out": str(args.out.resolve()),
            "workers": args.workers,
            "status": "preparing",
        }
        atomic_json(args.state, state)
        try:
            sys.argv = ["prepare_data", "--out", str(args.out), "--workers", str(args.workers)]
            prepare_data.main()
            state.update(status="verifying", preparation_finished_at=utc_now())
            atomic_json(args.state, state)
            sys.argv.append("--check-only")
            prepare_data.main()
            receipt = verify_data(args.out, args.receipt)
            state.update(
                status="complete",
                finished_at=utc_now(),
                verified_training_and_c4_shards=len(receipt["files"]),
                receipt=str(args.receipt.resolve()),
            )
            atomic_json(args.state, state)
            print(f"Full pinned corpus prepared; training receipt: {args.receipt}", flush=True)
        except BaseException as error:
            state.update(status="failed", finished_at=utc_now(), error_type=type(error).__name__)
            atomic_json(args.state, state)
            raise


if __name__ == "__main__":
    main()
