# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain execution-source regressions while binding the separate transition terminal reader."""

import ast
from pathlib import Path

import ns_training_path_transition_lr_gpu_source as derivation

SOURCE_SHA256 = "27e0f6d7d858f302e015a723c8547450ec701db346cf82913ea97dde14331f46"


def inherited_tests() -> bytes:
    tree = derivation.transform(Path(__file__).with_name("test_ns_training_path_lr_gpu_source.py"), SOURCE_SHA256)
    position = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.FunctionDef) and node.name == "complete_cpu"
    )
    tree.body.insert(position, ast.parse("real_run = subprocess.run").body[0])
    completed = tree.body[position + 1]
    completed.body[:0] = ast.parse(
        """if kwargs.get("capture_output"):
    return real_run(command,cwd=cwd,**kwargs)
"""
    ).body
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        if function.name == "test_freeze_and_visible_identity_check_bind_terminal_rejections_without_hidden_calls":
            for node in ast.walk(function):
                if isinstance(node, ast.Dict):
                    if any(isinstance(key, ast.Constant) and key.value == "checkpoint_payloads" for key in node.keys):
                        node.keys.append(ast.Constant(value="failed_checkpoint_payloads"))
                        node.values.append(ast.Dict(keys=[], values=[]))
            function.body.insert(
                0,
                ast.parse(
                    'monkeypatch.setattr(controller.bridge,"require_cpu",'
                    'lambda root: source.accepted_archive(root,()) | {"helpers_sha256":{}})'
                ).body[0],
            )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(inherited_tests(), __file__, "exec"), globals())
