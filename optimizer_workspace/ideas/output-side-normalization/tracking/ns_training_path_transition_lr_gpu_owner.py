# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate transition-LR ownership through a separate pinned terminal archive."""

import ast
from pathlib import Path

import ns_training_path_transition_lr_gpu_source as derivation
import ns_training_path_transition_lr_parent as bridge

SOURCE_SHA256 = "8424a3c1ecd55d081bda1f7c1f6147893accb0bde0c6f756b1a83e4f81f7bb32"


def runtime_source() -> bytes:
    tree = derivation.transform(Path(__file__).with_name("ns_training_path_lr_gpu_owner.py"), SOURCE_SHA256)
    replacements = {
        "PREDECESSOR_RECEIPT": repr(bridge.CPU),
        "PREDECESSOR_RECEIPT_SHA256": repr(bridge.CPU_SHA256),
        "PREDECESSOR_VERIFIER_SHA256": repr(bridge.VERIFIER_SHA256),
        "PREDECESSOR_SEALER_SHA256": repr(bridge.CHECKER_SHA256),
        "ARCHIVE": repr(f"run-preparation/source-{derivation.adapter_source.NAMESPACE}.tar"),
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(replacements[target.id], mode="eval").body
    derivation.replace_function(
        tree,
        "predecessor_binding",
        """def predecessor_binding(root,seal_sha256):
    path = root/PREDECESSOR_RECEIPT
    before = file_identity(path)
    raw = path.read_bytes()
    cpu = json.loads(raw)
    helpers = path.parent/"tracking"
    if (
        hashlib.sha256(raw).hexdigest() != PREDECESSOR_RECEIPT_SHA256
        or cpu["root"] != str(root)
        or cpu["status"] != bridge.STATUS
        or cpu["environment"] != bridge.ENVIRONMENT
        or type(cpu["exit_code"]) is not int or cpu["exit_code"] != 0 or "error" in cpu
        or cpu["helpers"] != str(helpers)
        or cpu["helpers_sha256"][bridge.VERIFIER] != PREDECESSOR_VERIFIER_SHA256
        or cpu["helpers_sha256"][bridge.CHECKER] != PREDECESSOR_SEALER_SHA256
        or sha256(helpers/bridge.VERIFIER) != PREDECESSOR_VERIFIER_SHA256
        or sha256(helpers/bridge.CHECKER) != PREDECESSOR_SEALER_SHA256
        or file_identity(path) != before
        or len(seal_sha256) != 64
        or any(character not in "0123456789abcdef" for character in seal_sha256)
    ):
        raise ValueError("Transition-LR complete separate predecessor provenance differs")
    return dict(cpu_receipt=str(path),cpu_receipt_sha256=PREDECESSOR_RECEIPT_SHA256,
        verifier_sha256=PREDECESSOR_VERIFIER_SHA256,sealer_sha256=PREDECESSOR_SEALER_SHA256,
        seal=str(root/bridge.SEAL),seal_sha256=seal_sha256)
""",
    )
    derivation.replace_function(
        tree,
        "require_sealed_predecessor",
        """def require_sealed_predecessor(root,seal_sha256):
    require_hidden_cuda()
    predecessor_binding(root,seal_sha256)
    return bridge.require_seal(root,seal_sha256)
""",
    )
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or function.name != "require_binding":
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Tuple) and ast.unparse(node) == "('evidence', 'checkpoint_payloads')":
                node.elts.append(ast.Constant(value="failed_checkpoint_payloads"))
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
