# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use the established historical uploader for the separately bound wave-6 inventory."""

import sys
from pathlib import Path

import swanlab_upload
from analyze_wave6 import verify_histories
from wave6 import ARMS


def main() -> None:
    arguments = sys.argv[1:]
    index = arguments.index("--root")
    verify_histories(Path(arguments[index + 1]))
    swanlab_upload.WAVES["wave6"] = ARMS
    sys.argv = [sys.argv[0], *arguments, "--wave", "wave6"]
    swanlab_upload.main()


if __name__ == "__main__":
    main()
