# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare wave-7 histories only after verifying their exact same-source binding."""

import sys
from pathlib import Path

import analyze_wave
from launch_wave5 import read, sha256
from snapshot_wave7 import verify_inventory
from wave7 import ARMS, binding


def executed_sources(candidate: Path, frozen: dict) -> dict:
    """Reconstruct the complete manifest inventory without importing experiment code."""
    upstream = read(candidate / "experiment/upstream_manifest.json")
    prefix = candidate / "artifacts/external-sources" / f"Toner-{upstream['revision']}"
    paths = sorted((candidate / "experiment").rglob("*.py"))
    pinned = [
        *sorted((prefix / "reproduce/marin").glob("*.py")),
        *sorted((prefix / "src/toner").rglob("*.py")),
    ]
    if not pinned:
        raise ValueError("Wave-7 executed upstream inventory is missing")
    for path in pinned:
        physical = str(path.resolve().relative_to(candidate.resolve()))
        digest = sha256(path)
        if (
            upstream["files"].get(str(path.relative_to(prefix))) != digest
            or frozen["files_sha256"].get(physical) != digest
        ):
            raise ValueError("Wave-7 executed upstream source differs from the frozen inventory")
    return {str(path.relative_to(candidate)): sha256(path) for path in [*paths, *pinned]}


def verify_histories(root: Path) -> None:
    candidate, frozen = binding(root)
    verify_inventory(root, require_full=True)
    expected_sources = executed_sources(candidate, frozen)
    for arm in ARMS:
        manifest = read(root / "runs/wave7" / arm / "run.json")
        expected = read(candidate / "experiment/configs" / f"{arm}_0p5m_seed0.json")
        if manifest["config"] != expected or manifest["source_sha256"] != expected_sources:
            raise ValueError("Wave-7 executed configuration or source differs from its binding")
        if manifest["data_receipt_sha256"] != frozen["data_receipt_sha256"]:
            raise ValueError("Wave-7 executed data differs from its binding")


def main() -> None:
    arguments = sys.argv[1:]
    if len(arguments) != 2 or arguments[0] != "--root":
        raise SystemExit("usage: analyze_wave7.py --root OUTPUT_ROOT")
    verify_histories(Path(arguments[1]))
    analyze_wave.COMPARISONS["wave7"] = (
        ("wave7/muont", tuple(f"wave7/{arm}" for arm in ARMS[1:])),
        ("wave7/inner_outer_leftdiag", ("wave7/outer_leftdiag",)),
        ("wave7/outer_leftdiag", ("wave7/outer_leftdiag_fused",)),
    )
    sys.argv = [sys.argv[0], *arguments, "--wave", "wave7"]
    analyze_wave.main()


if __name__ == "__main__":
    main()
