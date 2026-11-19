# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue transition-LR work only after the external canonical terminal is sealed."""

import ast
from pathlib import Path

import ns_training_path_transition_lr_gpu_source as derivation

SOURCE_SHA256 = "5d980fe070624452dcc7b3d030ddbad7de49e2e2ed90164d53a5e4f8fa977e47"


def runtime_source() -> bytes:
    tree = derivation.transform(Path(__file__).with_name("ns_training_path_lr_gpu_controller.py"), SOURCE_SHA256)
    tree.body.insert(1, ast.parse("import ns_training_path_transition_lr_parent as bridge").body[0])
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name != "freeze":
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id == "accepted_cpu":
                    node.value = ast.parse("bridge.require_cpu(root)", mode="eval").body
            if isinstance(node, ast.Tuple) and ast.unparse(node) == "('evidence', 'checkpoint_payloads')":
                node.elts.append(ast.Constant(value="failed_checkpoint_payloads"))
            if isinstance(node, ast.Subscript) and ast.unparse(node) == "accepted_cpu['inventory_sha256']":
                node.value = ast.parse(
                    'dict(inventory_sha256={f"tracking/{n}":h for n,h in accepted_cpu["helpers_sha256"].items()})',
                    mode="eval",
                ).body
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
