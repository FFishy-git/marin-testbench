# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain real lifecycle fixtures under the distinct transition-LR execution namespace."""

import ast
from pathlib import Path

import ns_training_path_transition_lr_gpu_source as derivation

SOURCE_SHA256 = "30c86d472e229829f61300d4c74ac9b70cb7d86ea8a95eda849d2fe772c72668"


def runtime_source() -> bytes:
    tree = derivation.transform(Path(__file__).with_name("ns_training_path_lr_gpu_test_support.py"), SOURCE_SHA256)
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
