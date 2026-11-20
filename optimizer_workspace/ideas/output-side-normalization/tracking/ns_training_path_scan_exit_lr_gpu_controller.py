# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue scan-exit actual-LR commands only after separate terminal sealing."""

import ast
from pathlib import Path

import ns_training_path_scan_exit_lr_gpu_source as derivation

SOURCE_SHA256 = "d5e980426783bd97c648a11784b45eb72bde29892e093766d93635d5eeeaed12"


def runtime_source() -> bytes:
    tree = derivation.transform(
        Path(__file__).with_name("ns_training_path_transition_lr_gpu_controller.py"), SOURCE_SHA256
    )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
