# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the full pinned plan and mint content receipts for training."""

import argparse
import json
from pathlib import Path

import numpy as np
from reproduce.marin.data import validate_data_plan

from .records import atomic_json, sha256, utc_now
from .reference import UPSTREAM

PLAN = UPSTREAM / "reproduce" / "marin" / "data_plan.json"
REQUIRED = {"dclm", "starcoder", "proofpile", "paloma/c4_en"}


def verify_data(root, receipt_path):
    validate_data_plan(root, PLAN)
    plan = json.loads(PLAN.read_text())
    records = []
    for record in plan["files"]:
        if record["component"] not in REQUIRED:
            continue
        relative = Path(record["component"]) / record["file"]
        path = root / relative
        docs = 0
        if path.stat().st_size:
            tokens = np.memmap(path, mode="r", dtype="<u4")
            for start in range(0, len(tokens), 1_000_000):
                chunk = tokens[start : start + 1_000_000]
                if chunk.max() >= 128256:
                    raise ValueError(f"Invalid Llama token ID in {path}")
                docs += int(np.count_nonzero(chunk == 128001))
        if docs != record["docs"]:
            raise ValueError(f"Document count differs for {path}: {docs} != {record['docs']}")
        records.append(
            {
                "file": str(relative),
                "tokens": record["tokens"],
                "docs": docs,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "mtime_ns": path.stat().st_mtime_ns,
            }
        )
    receipt = {
        "verified_at": utc_now(),
        "root": str(root.resolve()),
        "data_plan_sha256": sha256(PLAN),
        "tokenizer": plan["tokenizer"],
        "tokenizer_revision": plan["tokenizer_revision"],
        "sources": plan["sources"],
        "files": records,
        "limitation": "Generated hashes prove file integrity, not unavailable original cache identity.",
    }
    atomic_json(receipt_path, receipt)
    return receipt


def require_receipt(root, receipt_path):
    """Check a previously fully verified dataset without rehashing 102GB per rank."""
    receipt = json.loads(receipt_path.read_text())
    if receipt["root"] != str(root.resolve()) or receipt["data_plan_sha256"] != sha256(PLAN):
        raise ValueError("Data receipt belongs to a different root or plan")
    plan = json.loads(PLAN.read_text())
    expected = {
        str(Path(r["component"]) / r["file"]): (r["tokens"], r["docs"])
        for r in plan["files"]
        if r["component"] in REQUIRED
    }
    if {r["file"]: (r["tokens"], r["docs"]) for r in receipt["files"]} != expected:
        raise ValueError("Incomplete data verification receipt")
    validate_data_plan(root, PLAN)
    for record in receipt["files"]:
        stat = (root / record["file"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
            raise ValueError(f"Data changed since full verification: {record['file']}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify_data(args.data, args.receipt)
    print(f"Verified {len(receipt['files'])} pinned shards with document counts and SHA256.")


if __name__ == "__main__":
    main()
