# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise complete scan-exit LR execution archives and the distinct terminal handoff."""

import ast
from pathlib import Path

import ns_training_path_scan_exit_lr_gpu_source as derivation

SOURCE_SHA256 = "9b4cd44df192aa23743221df2a60335a2821eb037af15efd3938cb085661ed8a"


def inherited_tests() -> bytes:
    tree = derivation.transform(
        Path(__file__).with_name("test_ns_training_path_transition_lr_gpu_source.py"), SOURCE_SHA256, "inherited_tests"
    )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())
