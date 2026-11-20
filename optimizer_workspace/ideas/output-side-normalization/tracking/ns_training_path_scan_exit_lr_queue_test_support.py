# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain the queue's executable lifecycle fixtures under the scan-exit actual-LR namespace."""

import ast
from pathlib import Path

import ns_training_path_scan_exit_lr_queue as derivation


def runtime_source() -> bytes:
    tree = derivation.transform(Path(__file__).with_name("ns_training_path_transition_lr_queue_test_support.py"))
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
