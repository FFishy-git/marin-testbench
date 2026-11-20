# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the separate actual-LR candidate with the accepted scan-exit execution handler."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_scan_exit_lr_contract as scan_contract
import ns_training_path_scan_exit_lr_parent as bridge

PARENT_SHA256 = "da8dc3a706704f2b910e7731ad67a767e57508ab187cc745bd9aa7c5e5509ded"
NAMESPACE = scan_contract.NAMESPACE
HANDOFF = "owned_output_handoff.py"
HANDOFF_SHA256 = "09b0a33e370fd683a0b16f18d036226930c56dab46b662f64196dee2d6eb1276"
NUMERICAL_HANDOFF_SHA256 = "b359c02fd184f1afa7c32f3a4fbab8693ea41f7c4a7225f38cd17d7ab52b43f6"
HANDOFF_FIXTURE = "fixtures/scan-exit-lr-original-owned_output_handoff.py"
EXECUTION_PINS = {
    HANDOFF: HANDOFF_SHA256,
    "owned_output_attribution.py": "c999637ad12729d767f2d9539e7adeb359fd38260f6e09b089b35d00be235653",
    "owned_output_exit_drain.py": "981deaf0ec73f43c69e64c0409212100afafb0929e6c0cb11ccf8892edd17db6",
    "owned_output_exit_transition.py": "e3ae48eea72cd3d9828f06c407551b8f316716498994f54b9d5bb77a55fa9a59",
    "owned_output_missing_environment.py": "06c18d6387607339ceaa320b8ae4a156c172f0054b93fb32c5e2b67b1feb7dde",
}
SOURCE_PARENTS = {
    "ns_training_path_transition_lr_parent.py": "58ccfde4c5d4683162204bc438186f538824b3df9f01a74c2c76e34990904274",
    "ns_training_path_transition_lr_contract.py": "1d51d28530607690c3716885f18df79d9bdb0bd0608e405d8f6df510ccf420e2",
    "ns_training_path_transition_lr_validation.py": "56817963148654c7c964ddfb3b729346627270e7400e5c32398a120b6820ae58",
    "ns_training_path_transition_lr_source.py": PARENT_SHA256,
    "test_ns_training_path_transition_lr_source.py": "71364d8e8932c34ee3642c0484a1cfb2f6860b3a450f41bfff248f3fdfd7e25b",
}
NEW_FILES = (
    "ns_training_path_scan_exit_lr_parent.py",
    "ns_training_path_scan_exit_lr_contract.py",
    "ns_training_path_scan_exit_lr_validation.py",
    "ns_training_path_scan_exit_lr_source.py",
    "test_ns_training_path_scan_exit_lr_source.py",
    *SOURCE_PARENTS,
)
MODULES = {
    "ns_training_path_transition_lr_contract": "ns_training_path_scan_exit_lr_contract",
    "ns_training_path_transition_lr_validation": "ns_training_path_scan_exit_lr_validation",
    "ns_training_path_transition_lr_source": "ns_training_path_scan_exit_lr_source",
    "ns_training_path_transition_lr_parent": "ns_training_path_scan_exit_lr_parent",
    "ns_training_path_transition_lr_gpu_owner": "ns_training_path_scan_exit_lr_gpu_owner",
    "ns_training_path_transition_lr_gpu_controller": "ns_training_path_scan_exit_lr_gpu_controller",
}
SCOPE_VALUE = (
    "Distinct scan-exit actual-LR adapter preparation, not ownership or launch. Preserve "
    "all accepted numerical training/configuration/reader bytes. Change only the candidate "
    "owner adapter and namespace; explicitly transform the conflicting execution handoff "
    "to the accepted scan-exit bytes, retaining its original numerical source in the inventory. "
    "Include pinned missing-environment/exit handlers and transitive derivation parents. "
    "The independent terminal archive remains external and separately authenticated. "
    "Retain54configs162commands, original5120schedule, all rejections, compilation, cached-root "
    "CUDA graphs and unchanged tolerances. No seal, GPU transfer, production, quality, "
    "whole-training efficiency, SwanLab or optimizer-superiority authority."
)


def literal(value: str) -> str:
    value = scan_contract.literal(value)
    for before, after in sorted(MODULES.items(), key=lambda item: len(item[0]), reverse=True):
        value = value.replace(before, after)
    return value.replace("fixtures/transition-lr-parent-owner.py", "fixtures/scan-exit-lr-parent-owner.py").replace(
        "test_ns_training_path_transition_lr_source", "test_ns_training_path_scan_exit_lr_source"
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


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_transition_lr_source.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_SHA256:
        raise ValueError("Scan-exit LR pinned adapter source changed")
    namespace = {"__file__": str(path)}
    exec(compile(payload, str(path), "exec"), namespace)
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
            node.body.insert(
                len(node.body) - 1,
                ast.parse(
                    """for name,digest in SOURCE_PARENTS.items():
    if sha256(sources[name]) != digest:
        raise ValueError("Scan-exit LR retained derivation parent changed")
"""
                ).body[0],
            )
    if changed != set(replacements):
        raise ValueError("Scan-exit LR adapter source or receipt replacement differs")
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
