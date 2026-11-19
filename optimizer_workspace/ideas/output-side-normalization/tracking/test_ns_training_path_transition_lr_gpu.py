# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run inherited lifecycle, CUDA-identity and one-shot terminal regressions for transition-LR."""

import ast
from pathlib import Path

import ns_training_path_transition_lr_gpu_source as derivation

SOURCE_SHA256 = "cbefe5b297bb8eaa87055a828d9503d00b6e4aea9cc7caf3d7356549fe252815"


def inherited_tests() -> bytes:
    tree = derivation.transform(Path(__file__).with_name("test_ns_training_path_lr_gpu.py"), SOURCE_SHA256)
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())
