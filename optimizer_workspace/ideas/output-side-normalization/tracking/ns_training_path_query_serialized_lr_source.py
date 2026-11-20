# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a separate actual-LR adapter with the accepted query execution-helper bytes."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_query_serialized_lr_contract as query_contract
import ns_training_path_query_serialized_lr_parent as bridge

DERIVATION_SHA256 = "da8dc3a706704f2b910e7731ad67a767e57508ab187cc745bd9aa7c5e5509ded"
NAMESPACE = query_contract.NAMESPACE
HANDOFF = "owned_output_handoff.py"
HANDOFF_SHA256 = "7c653b0a08e3159599ef00ba4767ef3291b7d581744581d000ca14c3a6ad4e24"
NUMERICAL_HANDOFF_SHA256 = "b359c02fd184f1afa7c32f3a4fbab8693ea41f7c4a7225f38cd17d7ab52b43f6"
HANDOFF_FIXTURE = "fixtures/query-serialized-lr-original-owned_output_handoff.py"
EXECUTION_PINS = {
    HANDOFF: HANDOFF_SHA256,
    "owned_output_attribution.py": "c999637ad12729d767f2d9539e7adeb359fd38260f6e09b089b35d00be235653",
    "owned_output_exit_drain.py": "981deaf0ec73f43c69e64c0409212100afafb0929e6c0cb11ccf8892edd17db6",
    "owned_output_exit_transition.py": "e3ae48eea72cd3d9828f06c407551b8f316716498994f54b9d5bb77a55fa9a59",
    "owned_output_missing_environment.py": "06c18d6387607339ceaa320b8ae4a156c172f0054b93fb32c5e2b67b1feb7dde",
    "owned_output_recheck_exit.py": "59bef187d102f41367062437c8b15f563a7268d6ad5b233c3af49d43773e60f3",
    "ns_training_owner_serialization.py": "0ac178a8e3b302c5767d162a52b2b9dbd703fa2e9c17a66a603237d7316ed9e3",
}
SOURCE_PARENTS = {
    "ns_training_path_transition_lr_parent.py": "58ccfde4c5d4683162204bc438186f538824b3df9f01a74c2c76e34990904274",
    "ns_training_path_transition_lr_contract.py": "1d51d28530607690c3716885f18df79d9bdb0bd0608e405d8f6df510ccf420e2",
    "ns_training_path_transition_lr_validation.py": "56817963148654c7c964ddfb3b729346627270e7400e5c32398a120b6820ae58",
    "ns_training_path_transition_lr_source.py": DERIVATION_SHA256,
    "test_ns_training_path_transition_lr_source.py": "71364d8e8932c34ee3642c0484a1cfb2f6860b3a450f41bfff248f3fdfd7e25b",
}
NEW_FILES = (
    "ns_training_path_query_serialized_lr_parent.py",
    "ns_training_path_query_serialized_lr_contract.py",
    "ns_training_path_query_serialized_lr_validation.py",
    "ns_training_path_query_serialized_lr_source.py",
    "test_ns_training_path_query_serialized_lr_source.py",
    *SOURCE_PARENTS,
)
MODULES = {
    "ns_training_path_transition_lr_contract": "ns_training_path_query_serialized_lr_contract",
    "ns_training_path_transition_lr_validation": "ns_training_path_query_serialized_lr_validation",
    "ns_training_path_transition_lr_source": "ns_training_path_query_serialized_lr_source",
    "ns_training_path_transition_lr_parent": "ns_training_path_query_serialized_lr_parent",
    "ns_training_path_transition_lr_gpu_owner": "ns_training_path_query_serialized_lr_gpu_owner",
    "ns_training_path_transition_lr_gpu_controller": "ns_training_path_query_serialized_lr_gpu_controller",
    "test_ns_training_path_transition_lr_source": "test_ns_training_path_query_serialized_lr_source",
}
SCOPE_VALUE = (
    "Distinct query-serialized actual-LR adapter preparation, not execution ownership or launch. "
    "Retain all accepted numerical trainer/configuration/reader bytes and54configs162commands "
    "with the original5120schedule. Change only the candidate owner adapter and namespace; "
    "explicitly transform the conflicting handoff to accepted query helper bytes, preserving "
    "the original numerical helper as a fixture. Pin all transitive exit/recheck handlers and "
    "the unchanged30-second inspection-lock helper. Keep the independent query terminal reader "
    "archive external and separately authenticated. Preserve all prior rejections/failed evidence, "
    "compilation, cached-root CUDA graphs and numerical tolerances. No seal, GPU transfer, "
    "production, optimizer quality, whole-training efficiency or SwanLab authority."
)


def literal(value: str) -> str:
    value = value.replace("ns-training-path-transition-actual-lr-20261119", NAMESPACE).replace(
        "validation/ns_training_path_transition_lr_20261119",
        "validation/ns_training_path_query_serialized_lr_20261120",
    )
    value = value.replace(
        "OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION_LR_", "OUTPUT_MARIN_NS_TRAINING_PATH_QUERY_SERIALIZED_LR_"
    ).replace("ns_training_path_transition_actual_lr_", "ns_training_path_query_serialized_actual_lr_")
    for before, after in sorted(MODULES.items(), key=lambda item: len(item[0]), reverse=True):
        value = value.replace(before, after)
    return value.replace("fixtures/transition-lr-parent-owner.py", "fixtures/query-serialized-lr-parent-owner.py")


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


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_transition_lr_source.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != DERIVATION_SHA256:
        raise ValueError("Query actual-LR pinned adapter implementation changed")
    parent = ast.parse(payload)
    parent.body = [
        node
        for node in parent.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "exec"
        )
    ]
    namespace = {"__file__": str(path)}
    exec(compile(parent, str(path), "exec"), namespace)
    tree = Namespace().visit(ast.parse(namespace["runtime_source"]()))
    replacements = {
        "PARENT": bridge.ADAPTER[0],
        "PARENT_SHA256": bridge.ADAPTER[1],
        "VERIFIER_SHA256": bridge.VERIFIER_SHA256,
        "CHECKER_SHA256": bridge.CHECKER_SHA256,
        "NEW_HELPERS": NEW_FILES,
        "TEST_COUNT": 16,
        "SCOPE": SCOPE_VALUE,
    }
    changed = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(repr(replacements[target.id]), mode="eval").body
                    changed.add(target.id)
        if isinstance(node, ast.FunctionDef) and node.name == "execution_sources":
            node.body[-1:-1] = ast.parse(
                """for name, digest in SOURCE_PARENTS.items():
    if sha256(sources[name]) != digest:
        raise ValueError("Query actual-LR retained derivation parent changed")
"""
            ).body
    if changed != set(replacements):
        raise ValueError("Query actual-LR adapter source/receipt replacement differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
