# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wait for the independent scan-exit terminal before issuing one accepted actual-LR diagnostic."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_scan_exit_lr_gpu_source as derivation

PARENT_SHA256 = "7a0cc4b23eb0a52f959552033779b99714def017a0538fc156052186116a7f37"
PARENT_FILES = {
    "ns_training_path_lr_queue.py": "84e9f3750cd0dea7fdf0fdf48bff0b1acb259b06f420d6970dbad0bdaaf49a18",
    "ns_training_path_lr_queue_test_support.py": "5196e706d284dcdae73679a40c113fb0c8d313153723ae36061830ac690ae235",
    "test_ns_training_path_lr_queue.py": "e41e4d662b0b128e47bee010353fe4bb0af797f75c57854b8acc7e3d38455cd6",
    "ns_training_path_transition_lr_queue.py": PARENT_SHA256,
    "ns_training_path_transition_lr_queue_parent.py": "2c14caa054f7c3b7478af3e037522c72a3f7d494c0998d93bafbde6e9274df1d",
    "ns_training_path_transition_lr_queue_test_support.py": (
        "4f85af3f3fff0ccefa1d66f84ce9dbd7ac1c44d63557e158d1a69535cdcc1737"
    ),
    "test_ns_training_path_transition_lr_queue.py": "1817a037e675409f5fd66a79e824123567876ba50ee0fb2a873f7219a74d1448",
}
MODULES = derivation.MODULES | {
    "ns_training_path_transition_lr_queue": "ns_training_path_scan_exit_lr_queue",
    "ns_training_path_transition_lr_queue_parent": "ns_training_path_scan_exit_lr_queue_parent",
    "ns_training_path_transition_lr_queue_test_support": "ns_training_path_scan_exit_lr_queue_test_support",
    "test_ns_training_path_transition_lr_queue": "test_ns_training_path_scan_exit_lr_queue",
}
EXECUTION_CPU = "run-preparation/ns-training-path-scan-exit-actual-lr-execution-cpu/20261119T2154/receipt.json"
EXECUTION_CPU_SHA256: str | None = "34f40a430563fecda50c8490b16f32602a53c43b61a4ad0ce7afe72e08a25bdc"
NEW_FILES = (
    "ns_training_path_scan_exit_lr_queue.py",
    "ns_training_path_scan_exit_lr_queue_parent.py",
    "ns_training_path_scan_exit_lr_queue_test_support.py",
    "test_ns_training_path_scan_exit_lr_queue.py",
    *PARENT_FILES,
)
SCOPE_VALUE = (
    "One distinct hidden-CUDA scan-exit actual-LR queue. Preserve the separately accepted "
    "execution archive, numerical candidate and54configs162commands unchanged. Authenticate "
    "the separate canonical-terminal reader with acknowledged isolated subprocesses. Never "
    "merge or invoke the original scan-exit terminal. Wait every900seconds for complete "
    "original54-command18-decision terminal, exact original owner/waiter/descendant absence "
    "and restored keeper. Issue only one seal/check-seal/freeze/check/start/terminal sequence "
    "with caller-bound digest and exact issued owner cleanup/restoration. Retain rejections "
    "and failed one-shots, unchanged tolerances, compilation and cached-root CUDA graphs. "
    "No retry, direct keeper operation, tensor reader, production, quality/efficiency or "
    "private SwanLab authority."
)


def literal(value: str) -> str:
    value = derivation.literal(value)
    for before, after in sorted(MODULES.items(), key=lambda item: len(item[0]), reverse=True):
        value = value.replace(before, after)
    return (
        value.replace(
            "OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION_LR_QUEUE", "OUTPUT_MARIN_NS_TRAINING_PATH_SCAN_EXIT_LR_QUEUE"
        )
        .replace(
            "prepare_ns_training_path_transition_lr_predecessor.py",
            "prepare_ns_training_path_scan_exit_lr_predecessor.py",
        )
        .replace(
            "run-preparation/ns-training-path-transition-actual-lr-queue-20261119",
            "run-preparation/ns-training-path-scan-exit-actual-lr-queue-20261119",
        )
    )


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


def transform(path: Path, entry: str = "runtime_source") -> ast.Module:
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_FILES[path.name]:
        raise ValueError("Scan-exit LR queue pinned derivation source changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
    return Namespace().visit(ast.parse(namespace[entry]()))


def runtime_source() -> bytes:
    tree = transform(Path(__file__).with_name("ns_training_path_transition_lr_queue.py"))
    replacements = {
        "PARENT": EXECUTION_CPU,
        "PARENT_SHA256": EXECUTION_CPU_SHA256,
        "WORK": "run-preparation/ns-training-path-scan-exit-actual-lr-queue-20261119",
        "MARKER": "OUTPUT_MARIN_NS_TRAINING_PATH_SCAN_EXIT_LR_QUEUE",
        "CPU_STATUS": "ns_training_path_scan_exit_actual_lr_queue_cpu_archive_verified_gpu_unissued",
        "NEW_HELPERS": NEW_FILES,
        "TEST_COUNT": 47,
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
        raise ValueError("Scan-exit LR queue receipt and namespace replacement differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
