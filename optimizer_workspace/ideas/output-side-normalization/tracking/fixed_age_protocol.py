# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Route explicit fixed-age LR protocols without accepting campaign-defined search grids."""

import os
from pathlib import Path

from analyze_fixed_age_tuning import analyze as analyze_initial
from analyze_lr_boundary import analyze as analyze_boundary
from analyze_lr_ceiling import analyze as analyze_ceiling
from prepare_fixed_age_tuning import STUDY as INITIAL_STUDY
from prepare_lr_boundary import MULTIPLIERS as BOUNDARY_MULTIPLIERS
from prepare_lr_boundary import STUDY as BOUNDARY_STUDY
from prepare_lr_ceiling import MULTIPLIERS as CEILING_MULTIPLIERS
from prepare_lr_ceiling import STUDY as CEILING_STUDY
from prepare_lr_ceiling import require_recovery
from prepare_tuning import MULTIPLIERS as INITIAL_MULTIPLIERS
from validate_wave5 import ENVIRONMENT
from wave7_release import read


def environment(campaign_path: Path) -> dict[str, str]:
    campaign = read(campaign_path)
    if "recovery" not in campaign:
        return ENVIRONMENT
    require_recovery(campaign)
    return ENVIRONMENT | {"CUDA_VISIBLE_DEVICES": campaign["recovery"]["cuda_visible_devices"]}


def require_gpu_environment(campaign_path: Path) -> None:
    expected = environment(campaign_path).get("CUDA_VISIBLE_DEVICES")
    if expected is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
        raise RuntimeError("GPU execution visibility differs from the bound ceiling recovery environment")


def multipliers(campaign_path: Path) -> tuple[tuple[str, float], ...]:
    study = read(campaign_path).get("study")
    if study == INITIAL_STUDY:
        return INITIAL_MULTIPLIERS
    if study == BOUNDARY_STUDY:
        return BOUNDARY_MULTIPLIERS
    if study == CEILING_STUDY:
        return CEILING_MULTIPLIERS
    raise ValueError("Unrecognized fixed-age LR protocol; campaign-defined grids are not executable")


def is_boundary(campaign_path: Path) -> bool:
    multipliers(campaign_path)
    return read(campaign_path)["study"] in (BOUNDARY_STUDY, CEILING_STUDY)


def predecessor_wave(campaign_path: Path) -> str:
    multipliers(campaign_path)
    study = read(campaign_path)["study"]
    if study == CEILING_STUDY:
        return "wave9"
    return "wave8" if study == BOUNDARY_STUDY else "wave7"


def analyze(campaign_path: Path) -> dict:
    multipliers(campaign_path)
    study = read(campaign_path)["study"]
    if study == CEILING_STUDY:
        return analyze_ceiling(campaign_path)
    return analyze_boundary(campaign_path) if study == BOUNDARY_STUDY else analyze_initial(campaign_path)


def analyzer_name(campaign_path: Path) -> str:
    if predecessor_wave(campaign_path) == "wave9":
        return "analyze_lr_ceiling.py"
    return "analyze_lr_boundary.py" if is_boundary(campaign_path) else "analyze_fixed_age_tuning.py"
