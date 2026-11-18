# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import only complete, analyzed, restored attribution waves into private SwanLab histories."""

import sys
from pathlib import Path

import swanlab_refresh_upload
from fused_attribution_production_contract import FAMILIES, WAVES
from fused_attribution_run_history import require_analysis, require_completed_snapshot, verify_histories


def main() -> None:
    arguments = list(sys.argv[1:])
    root = Path(arguments[arguments.index("--root") + 1])
    index = arguments.index("--wave")
    wave = arguments[index + 1]
    if wave not in WAVES:
        raise ValueError("Unknown attribution production wave")
    release = verify_histories(root, wave)
    require_completed_snapshot(root, wave, release)
    require_analysis(root, wave, release)
    swanlab_refresh_upload.WAVES[wave] = FAMILIES
    swanlab_refresh_upload.main()


if __name__ == "__main__":
    main()
