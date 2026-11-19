# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install a source-bound start variant before full-model warm-up."""

import hashlib
import json
from pathlib import Path

import torch

if __package__:
    from .root_start_rational import rational_start
else:
    from root_start_rational import rational_start

ROOTS_SHA256 = "6a61bdefa6fd3fca6365e571baac4208c53d73e9187978be5d75a6e1951f331c"
RATIONAL_SHA256 = "e509251567fd885257eb5ab60e334be600ce1bcc52ffb0e9fc8671ff490a327e"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install_start(roots, variant: str) -> dict:
    """Replace only the compiled start callable before any engine or root execution."""
    source = Path(roots.__file__).resolve(strict=True)
    rational_source = Path(__file__).with_name("root_start_rational.py")
    if (
        variant not in ("pinned", "rational")
        or digest(source) != ROOTS_SHA256
        or digest(rational_source) != RATIONAL_SHA256
        or roots.start._torchdynamo_orig_callable is not roots._start
    ):
        raise ValueError("Root start installation lacks original compiled/source identity")
    if variant == "rational":
        roots.start = torch.compile(rational_start, dynamic=False)
        if roots.start._torchdynamo_orig_callable is not rational_start:
            raise RuntimeError("Rational start lost its original compiled callable")
    return {
        "status": "source_bound_start_installed_before_model_and_root_warmup",
        "variant": variant,
        "pinned_roots_sha256": ROOTS_SHA256,
        "rational_start_sha256": RATIONAL_SHA256,
        "installer_sha256": digest(Path(__file__)),
        "compilation_preserved": True,
        "compile_options": {"dynamic": False},
        "root_graph_lifecycle_unchanged": True,
    }


def require_resume_start(out: Path, rank: int, variant: str, identity: dict) -> None:
    """Bind a resume to the executed start receipt from the same source and rank."""
    path = out / f"root_start_rank_{rank}_from_000000.json"
    receipt = json.loads(path.read_bytes())
    if (
        path.is_symlink()
        or path.stat().st_nlink != 1
        or receipt
        != {
            "status": "source_bound_start_installed_before_model_and_root_warmup",
            "variant": variant,
            "pinned_roots_sha256": ROOTS_SHA256,
            "rational_start_sha256": RATIONAL_SHA256,
            "installer_sha256": digest(Path(__file__)),
            "compilation_preserved": True,
            "compile_options": {"dynamic": False},
            "root_graph_lifecycle_unchanged": True,
            "rank": rank,
            "start_step": 0,
            "identity": identity,
        }
    ):
        raise ValueError("Checkpoint start variant, rank or source identity differs")
