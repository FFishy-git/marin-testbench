# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accept distinct transition-LR ownership without changing its prepared numerical candidate."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_transition_lr_source as adapter_source

SOURCE_SHA256 = "9e51b4366091a650b3665a292679039bde03a06354280e49fb36e07f159019bf"
ADAPTER_CPU = "run-preparation/ns-training-path-transition-actual-lr-adapter-cpu/corrected-20261119T1807/receipt.json"
ADAPTER_CPU_SHA256: str | None = "8c9ea7cf47c3fcb437bc591cc90a82341651767ea19305082963363ebb05b9f7"
PARENT_FILES = {
    "ns_training_path_lr_gpu_source.py": "9e51b4366091a650b3665a292679039bde03a06354280e49fb36e07f159019bf",
    "ns_training_path_lr_gpu_owner.py": "8424a3c1ecd55d081bda1f7c1f6147893accb0bde0c6f756b1a83e4f81f7bb32",
    "ns_training_path_lr_gpu_controller.py": "5d980fe070624452dcc7b3d030ddbad7de49e2e2ed90164d53a5e4f8fa977e47",
    "ns_training_path_lr_gpu_test_support.py": "30c86d472e229829f61300d4c74ac9b70cb7d86ea8a95eda849d2fe772c72668",
    "test_ns_training_path_lr_gpu.py": "cbefe5b297bb8eaa87055a828d9503d00b6e4aea9cc7caf3d7356549fe252815",
    "test_ns_training_path_lr_gpu_source.py": "27e0f6d7d858f302e015a723c8547450ec701db346cf82913ea97dde14331f46",
}
NEW_FILES = (
    "ns_training_path_transition_lr_gpu_source.py",
    "ns_training_path_transition_lr_gpu_owner.py",
    "ns_training_path_transition_lr_gpu_controller.py",
    "ns_training_path_transition_lr_gpu_test_support.py",
    "test_ns_training_path_transition_lr_gpu.py",
    "test_ns_training_path_transition_lr_gpu_source.py",
    *PARENT_FILES,
)
MODULES = adapter_source.MODULES | {
    "ns_training_path_lr_gpu_source": "ns_training_path_transition_lr_gpu_source",
    "ns_training_path_lr_gpu_test_support": "ns_training_path_transition_lr_gpu_test_support",
    "test_ns_training_path_lr_gpu": "test_ns_training_path_transition_lr_gpu",
    "test_ns_training_path_lr_gpu_source": "test_ns_training_path_transition_lr_gpu_source",
}
SCOPE_VALUE = (
    "Distinct transition-LR execution owner; all54configs162commands retain the prepared "
    "candidate, numerical helpers and explicit measured handoff transform unchanged. "
    "Freeze requires the separately accepted complete transition canonical terminal and "
    "caller-bound seal, retained rejections/failed-checkpoint identities, actual owner/"
    "waiter/descendant absence and exact restored keeper. Separate terminal archive remains "
    "external. Retain physical UUIDs, node/manager locks, exact launcher/rank ancestry, "
    "visible parent proof and one-shot cleanup/restoration/terminal. No production, quality, "
    "whole-training efficiency, numerical waiver or SwanLab authority."
)


class Namespace(adapter_source.Namespace):
    def visit_Import(self, node):
        for alias in node.names:
            alias.name = MODULES.get(alias.name, alias.name)
        return node

    def visit_ImportFrom(self, node):
        node.module = MODULES.get(node.module, node.module)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = literal(node.value)
        return node


def literal(value: str) -> str:
    value = adapter_source.literal(value)
    for before, after in MODULES.items():
        value = value.replace(before, after)
    return value.replace("ns-training-path-actual-lr-terminal-", f"{adapter_source.NAMESPACE}-terminal-")


def transform(path: Path, digest: str) -> ast.Module:
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Transition-LR pinned execution parent changed")
    return Namespace().visit(ast.parse(raw))


def replace_function(tree: ast.Module, name: str, content: str) -> None:
    matches = [index for index, node in enumerate(tree.body) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"Transition-LR execution parent function differs: {name}")
    tree.body[matches[0]] = ast.parse(content).body[0]


def runtime_source() -> bytes:
    tree = transform(Path(__file__).with_name("ns_training_path_lr_gpu_source.py"), SOURCE_SHA256)
    replacements = {
        "PARENT": repr(ADAPTER_CPU),
        "PARENT_SHA256": repr(ADAPTER_CPU_SHA256),
        "NEW_HELPERS": repr(NEW_FILES),
        "TEST_COUNTS": repr(
            {"test_ns_training_path_transition_lr_gpu": 45, "test_ns_training_path_transition_lr_gpu_source": 8}
        ),
        "SCOPE": repr(SCOPE_VALUE),
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(replacements[target.id], mode="eval").body
    replace_function(
        tree,
        "inherited",
        """def inherited(root):
    if PARENT_SHA256 is None:
        raise RuntimeError("Transition-LR execution adapter CPU acceptance is unbound")
    previous = accepted_archive(root,(PARENT,PARENT_SHA256,adapter.CPU_STATUS,adapter.TEST_COUNT))
    if adapter.require_cpu(root,root/PARENT) != previous:
        raise ValueError("Transition-LR complete accepted adapter differs")
    return previous
""",
    )
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execution_sources")
    function.body.insert(
        len(function.body) - 1,
        ast.parse(
            """for name,digest in PARENT_FILES.items():
    if sha256(sources[name]) != digest:
        raise ValueError("Transition-LR retained execution parent source changed")
"""
        ).body[0],
    )
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
