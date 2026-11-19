# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a distinct actual-LR queue bound to the accepted transition execution archive."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_transition_lr_gpu_source as derivation

PARENT_FILES = {
    "ns_training_path_lr_queue.py": "84e9f3750cd0dea7fdf0fdf48bff0b1acb259b06f420d6970dbad0bdaaf49a18",
    "ns_training_path_lr_queue_test_support.py": "5196e706d284dcdae73679a40c113fb0c8d313153723ae36061830ac690ae235",
    "test_ns_training_path_lr_queue.py": "e41e4d662b0b128e47bee010353fe4bb0af797f75c57854b8acc7e3d38455cd6",
}
MODULES = derivation.MODULES | {
    "ns_training_path_lr_queue": "ns_training_path_transition_lr_queue",
    "ns_training_path_lr_queue_test_support": "ns_training_path_transition_lr_queue_test_support",
    "test_ns_training_path_lr_queue": "test_ns_training_path_transition_lr_queue",
    "ns_training_path_lr_predecessor": "ns_training_path_transition_lr_queue_parent",
    "prepare_ns_training_path_lr_predecessor": "prepare_ns_training_path_transition_lr_predecessor",
}
EXECUTION_CPU = "run-preparation/ns-training-path-transition-actual-lr-execution-cpu/20261119T1822/receipt.json"
EXECUTION_CPU_SHA256 = "5ec5d6155d7ab97642917f7b65432572b9d7c4de7d838cd089a0ce2072cdda27"
NEW_FILES = (
    "ns_training_path_transition_lr_queue.py",
    "ns_training_path_transition_lr_queue_parent.py",
    "ns_training_path_transition_lr_queue_test_support.py",
    "test_ns_training_path_transition_lr_queue.py",
    *PARENT_FILES,
)
SCOPE_VALUE = (
    "One distinct hidden-CUDA transition actual-LR queue. Preserve all191 accepted "
    "execution helpers, candidate and54configs162commands unchanged. Authenticate the "
    "separate external terminal archive in isolated subprocesses; never merge or invoke "
    "the original transition terminal. Wait every900seconds for complete original "
    "54-command18-decision canonical terminal, exact owner/waiter/descendant absence "
    "and restored keeper. Issue one seal/check-seal/freeze/check/start/terminal sequence "
    "with caller-bound seal digest and exact issued owner/descendant cleanup/restoration. "
    "Retain all rejections and consumed failures. No retry, direct keeper operation, "
    "new tensor reader, numerical waiver, production, quality/efficiency or SwanLab authority."
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


def literal(value: str) -> str:
    value = derivation.adapter_source.transition_contract.literal(value)
    for before, after in sorted(MODULES.items(), key=lambda item: len(item[0]), reverse=True):
        value = value.replace(before, after)
    return (
        value.replace(
            "prepare_ns_training_path_lr_predecessor.py", "prepare_ns_training_path_transition_lr_predecessor.py"
        )
        .replace("ns-training-path-actual-lr-terminal-", "ns-training-path-transition-actual-lr-20261119-terminal-")
        .replace("ns_training_path_lr_gpu_source", "ns_training_path_transition_lr_gpu_source")
    )


def transform(path: Path) -> ast.Module:
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != PARENT_FILES[path.name]:
        raise ValueError("Transition-LR queue retained parent source changed")
    return Namespace().visit(ast.parse(raw))


def runtime_source() -> bytes:
    tree = transform(Path(__file__).with_name("ns_training_path_lr_queue.py"))
    tree.body.insert(1, ast.parse("import ns_training_path_transition_lr_parent as bridge").body[0])
    replacements = {
        "PARENT": repr(EXECUTION_CPU),
        "PARENT_SHA256": repr(EXECUTION_CPU_SHA256),
        "WORK": "'run-preparation/ns-training-path-transition-actual-lr-queue-20261119'",
        "MARKER": "'OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION_LR_QUEUE'",
        "CPU_STATUS": "'ns_training_path_transition_actual_lr_queue_cpu_archive_verified_gpu_unissued'",
        "NEW_HELPERS": repr(NEW_FILES),
        "TEST_COUNT": "47",
        "SCOPE": repr(SCOPE_VALUE),
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(replacements[target.id], mode="eval").body
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "cpu_command":
            for item in ast.walk(node):
                if isinstance(item, ast.Subscript) and ast.unparse(item) == "NEW_HELPERS[-1]":
                    item.slice = ast.Constant(value=3)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "require_cpu")
    for item in ast.walk(function):
        if isinstance(item, ast.Subscript) and ast.unparse(item) == "NEW_HELPERS[-1]":
            item.slice = ast.Constant(value=3)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "helper_sources")
    function.body.insert(
        len(function.body) - 1,
        ast.parse(
            """for name,digest in PARENT_FILES.items():
    if sha256(sources[name]) != digest:
        raise ValueError("Transition-LR queue retained parent bytes changed")
"""
        ).body[0],
    )
    derivation.replace_function(
        tree,
        "predecessor_command",
        """def predecessor_command(root,phase,seal_sha256=None):
    if phase not in ("seal","check-seal"):
        raise ValueError("Transition-LR queue predecessor phase differs")
    receipt=bridge.require_cpu(root)
    helpers=Path(receipt["helpers"])
    argv=[contract.PYTHON,"-P",str(helpers/bridge.CHECKER),"--root",str(root),
        "--phase",phase,"--cpu-receipt",str(root/bridge.CPU)]
    if seal_sha256 is not None:
        argv.extend(("--seal-sha256",seal_sha256))
    return argv,helpers
""",
    )
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "issued_readiness")
    index = next(index for index, node in enumerate(function.body) if isinstance(node, ast.If))
    index = next(
        index
        for index, node in enumerate(function.body[index:], start=index)
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith("any(")
    )
    function.body[index:index] = ast.parse(
        """keeper = {
    int(pid): process
    for pid, process in saved.get("restored_keeper", {}).get("generation", {}).items()
}
for pid, process in handoff.marker_processes(contract.CONTROLLER_MARKER, intent["marker"]).items():
    if pid not in keeper or not handoff.same_generation(keeper[pid], process):
        marked[pid] = process
"""
    ).body
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
