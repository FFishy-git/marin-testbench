# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real lifecycle, CUDA identity and one-shot terminal boundaries for scan-exit LR."""

import ast
from pathlib import Path

import ns_training_path_scan_exit_lr_gpu_source as derivation

SOURCE_SHA256 = "d4a035515c2188972f1c799866b2b2ac4529539eaecb3530e5a8408043ff5828"


def inherited_tests() -> bytes:
    tree = derivation.transform(
        Path(__file__).with_name("test_ns_training_path_transition_lr_gpu.py"), SOURCE_SHA256, "inherited_tests"
    )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())
