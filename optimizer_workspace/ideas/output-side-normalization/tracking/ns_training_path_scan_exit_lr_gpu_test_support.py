# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain execution lifecycle fixtures under the separate scan-exit actual-LR namespace."""

import ast
from pathlib import Path

import ns_training_path_scan_exit_lr_gpu_source as derivation

SOURCE_SHA256 = "5bebd17d06d0e790cd51f883742556b6238b11a6226d84a70876eca557333b60"


def runtime_source() -> bytes:
    tree = derivation.transform(
        Path(__file__).with_name("ns_training_path_transition_lr_gpu_test_support.py"), SOURCE_SHA256
    )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
