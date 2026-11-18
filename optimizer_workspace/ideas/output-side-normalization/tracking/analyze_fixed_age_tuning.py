# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rank the complete fixed-age four-family LR study by reserved mixture loss."""

import argparse
import json
from pathlib import Path

from analyze_tuning import analyze as analyze_trials
from prepare_fixed_age_tuning import BATCH, STUDY, require_source
from wave7_release import ARMS, read


def analyze(campaign_path: Path) -> dict:
    """Reject changed scheduling evidence and retain the equal-budget selection protocol."""
    campaign = read(campaign_path)
    frozen = require_source(Path(campaign["root"]), Path(campaign["source_receipt"]))
    if (
        campaign["study"] != STUDY
        or campaign["batch"] != BATCH
        or type(campaign["refresh_install_age"]) is not int
        or campaign["refresh_install_age"] != 1
        or campaign["original_gpu_validation_sha256"] != frozen["original_gpu_validation_sha256"]
        or campaign["original_validation_state_sha256"] != frozen["original_validation_state_sha256"]
    ):
        raise ValueError("Fixed-age tuning campaign differs from its scheduling/validation protocol")
    return analyze_trials(campaign_path, families=ARMS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.campaign)
    temporary = args.out.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
