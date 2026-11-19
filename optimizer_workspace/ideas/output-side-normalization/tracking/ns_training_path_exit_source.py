# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a fresh exit-drain execution identity without editing accepted recovery code."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_exit_predecessor as predecessor
import ns_training_path_recovery_source as parent_source
import owned_output_exit_drain as drain
from selected_lr_release_owner import read, sha256

NAMESPACE = "ns-training-path-exit-drain-20261119"
PREPARATION = "ns-training-path-exit-drain-preparation.json"
OWNER = parent_source.OWNER
CPU_STATUS = "ns_training_path_exit_drain_cpu_archive_verified_gpu_unissued"
PREPARED_STATUS = "ns_training_path_exit_drain_candidate_prepared_gpu_unissued"
ENVIRONMENT = parent_source.ENVIRONMENT
SCOPE = (
    "Distinct exit-drain original-training-path diagnostic. Preserve original and recovery "
    "incomplete failures, all rejections and consumed identities. Fresh controller/validator/"
    "commands/outputs/terminal;18cases54commands, unchanged lower LR,5120schedule, numerical "
    "readers, training/configuration bytes, tolerances, compilation and cached-root CUDA graphs. "
    "Only ownership observation gains redacted attribution and bounded kernel-confirmed exit "
    "drain with whole-process terminal proof and full rescan. Failed handoff must be separately "
    "sealed and exact keeper retained. No production or SwanLab authority."
)
MODULES = {
    "ns_training_path_recovery_predecessor": "ns_training_path_exit_predecessor",
    "ns_training_path_recovery_source": "ns_training_path_exit_source",
    "prepare_ns_training_path_recovery": "prepare_ns_training_path_exit",
    "test_ns_training_path_recovery": "test_ns_training_path_exit",
    **{
        f"ns_training_path_recovery_{suffix}": f"ns_training_path_exit_{suffix}"
        for suffix in ("contract", "validation", "gpu_owner", "gpu_controller", "gpu_test_support")
    },
    "test_ns_training_path_recovery_gpu": "test_ns_training_path_exit_gpu",
}


def literal(value: str) -> str:
    """Map only execution namespace and module identities, leaving numerical strings intact."""
    return (
        value.replace(parent_source.NAMESPACE, NAMESPACE)
        .replace("ns_training_path_marker_exit_recovery", "ns_training_path_exit_drain")
        .replace("ns_training_path_recovery", "ns_training_path_exit")
        .replace("ns-training-path-recovery-preparation.json", PREPARATION)
        .replace("OUTPUT_MARIN_NS_TRAINING_PATH_RECOVERY", "OUTPUT_MARIN_NS_TRAINING_PATH_EXIT")
    )


LITERALS = {key: literal(value) for key, value in parent_source.LITERALS.items()}
HANDOFF_TEST_SHA256 = parent_source.HANDOFF_TEST_SHA256


class ExitNamespace(ast.NodeTransformer):
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


ADAPTER = ast.unparse(ExitNamespace().visit(ast.parse(parent_source.ADAPTER))) + "\n"


def recorded_parent(root: Path) -> dict:
    path = root / predecessor.PARENT_SPEC[0]
    if sha256(path) != predecessor.PARENT_SPEC[1]:
        raise ValueError("Exit-drain original numerical parent receipt changed")
    return read(path)


def derived_sources(parent: dict) -> dict[str, bytes]:
    result = {}
    for name, content in parent_source.derived_sources(parent).items():
        target = MODULES.get(name.removesuffix(".py"), name.removesuffix(".py")) + ".py"
        tree = ExitNamespace().visit(ast.parse(content))
        result[target] = (
            "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
            + ast.unparse(ast.fix_missing_locations(tree))
            + "\n"
        ).encode()
    return result


def handoff_bytes(parent: dict) -> bytes:
    path = Path(parent["root"]) / predecessor.RECOVERY_SPEC[0]
    if sha256(path) != predecessor.RECOVERY_SPEC[1]:
        raise ValueError("Exit-drain measured recovery parent differs")
    helper = Path(read(path)["helpers"]) / "owned_output_handoff.py"
    return drain.derived_source(helper)


HANDOFF_SHA256 = hashlib.sha256(
    handoff_bytes(
        recorded_parent(Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"))
    )
).hexdigest()
