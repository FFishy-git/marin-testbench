# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Accept scan-exit LR execution ownership without changing the prepared numerical candidate."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_scan_exit_lr_source as adapter_source

SOURCE_SHA256 = "0ef66343a365678213f69ab608f3d300edca70ba742b4e69b74f68af20d90651"
ADAPTER_CPU = "run-preparation/ns-training-path-scan-exit-actual-lr-adapter-cpu/20261119T2146/receipt.json"
ADAPTER_CPU_SHA256: str | None = "b0fc9541f22c8bf41da079a3369a1c722890882812aa676399eef5092f5b80a6"
PARENT_FILES = {
    "ns_training_path_lr_gpu_source.py": "9e51b4366091a650b3665a292679039bde03a06354280e49fb36e07f159019bf",
    "ns_training_path_lr_gpu_owner.py": "8424a3c1ecd55d081bda1f7c1f6147893accb0bde0c6f756b1a83e4f81f7bb32",
    "ns_training_path_lr_gpu_controller.py": "5d980fe070624452dcc7b3d030ddbad7de49e2e2ed90164d53a5e4f8fa977e47",
    "ns_training_path_lr_gpu_test_support.py": "30c86d472e229829f61300d4c74ac9b70cb7d86ea8a95eda849d2fe772c72668",
    "test_ns_training_path_lr_gpu.py": "cbefe5b297bb8eaa87055a828d9503d00b6e4aea9cc7caf3d7356549fe252815",
    "test_ns_training_path_lr_gpu_source.py": "27e0f6d7d858f302e015a723c8547450ec701db346cf82913ea97dde14331f46",
    "ns_training_path_transition_lr_gpu_source.py": SOURCE_SHA256,
    "ns_training_path_transition_lr_gpu_owner.py": "4399c9ff5aeff03c15033d34668bba46566d6b628d36d0f5130d5be786682679",
    "ns_training_path_transition_lr_gpu_controller.py": (
        "d5e980426783bd97c648a11784b45eb72bde29892e093766d93635d5eeeaed12"
    ),
    "ns_training_path_transition_lr_gpu_test_support.py": (
        "5bebd17d06d0e790cd51f883742556b6238b11a6226d84a70876eca557333b60"
    ),
    "test_ns_training_path_transition_lr_gpu.py": "d4a035515c2188972f1c799866b2b2ac4529539eaecb3530e5a8408043ff5828",
    "test_ns_training_path_transition_lr_gpu_source.py": (
        "9b4cd44df192aa23743221df2a60335a2821eb037af15efd3938cb085661ed8a"
    ),
}
NEW_FILES = (
    "ns_training_path_scan_exit_lr_gpu_source.py",
    "ns_training_path_scan_exit_lr_gpu_owner.py",
    "ns_training_path_scan_exit_lr_gpu_controller.py",
    "ns_training_path_scan_exit_lr_gpu_test_support.py",
    "test_ns_training_path_scan_exit_lr_gpu.py",
    "test_ns_training_path_scan_exit_lr_gpu_source.py",
    *PARENT_FILES,
)
MODULES = adapter_source.MODULES | {
    "ns_training_path_transition_lr_gpu_source": "ns_training_path_scan_exit_lr_gpu_source",
    "ns_training_path_transition_lr_gpu_test_support": "ns_training_path_scan_exit_lr_gpu_test_support",
    "test_ns_training_path_transition_lr_gpu": "test_ns_training_path_scan_exit_lr_gpu",
    "test_ns_training_path_transition_lr_gpu_source": "test_ns_training_path_scan_exit_lr_gpu_source",
}
SCOPE_VALUE = (
    "Distinct scan-exit actual-LR execution owner. Preserve the accepted numerical candidate, "
    "all54configs162commands and explicit accepted scan-exit handoff transform unchanged. "
    "Retain transitive source parents and keep the independent terminal archive external. "
    "Freeze requires the complete54-command18-decision original canonical terminal and "
    "caller-bound seal, retained rejections/failed payload identities, actual owner/waiter/"
    "descendant absence and exact restored keeper. Retain physical UUIDs, node/manager locks, "
    "launcher/rank ancestry, visible parent proof, compilation, cached-root CUDA graphs and "
    "one-shot cleanup/restoration/terminal. No numerical waiver, production, quality, "
    "whole-training efficiency, private SwanLab or optimizer-superiority acceptance."
)


def literal(value: str) -> str:
    value = adapter_source.literal(value)
    for before, after in sorted(MODULES.items(), key=lambda item: len(item[0]), reverse=True):
        value = value.replace(before, after)
    return value


class Namespace(ast.NodeTransformer):
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


def transform(path: Path, digest: str, entry: str = "runtime_source") -> ast.Module:
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Scan-exit LR pinned execution derivation changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    return Namespace().visit(ast.parse(namespace[entry]()))


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_transition_lr_gpu_source.py")
    tree = transform(path, SOURCE_SHA256)
    replacements = {
        "PARENT": ADAPTER_CPU,
        "PARENT_SHA256": ADAPTER_CPU_SHA256,
        "NEW_HELPERS": NEW_FILES,
        "TEST_COUNTS": {
            "test_ns_training_path_scan_exit_lr_gpu": 45,
            "test_ns_training_path_scan_exit_lr_gpu_source": 8,
        },
        "SCOPE": SCOPE_VALUE,
    }
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
                    changed.add(target.id)
    if changed != set(replacements):
        raise ValueError("Scan-exit LR execution closure receipt replacement differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
