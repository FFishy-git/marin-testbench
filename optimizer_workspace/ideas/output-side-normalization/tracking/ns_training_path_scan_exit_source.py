# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a fresh original-training-path diagnostic with identity-bound missing-file exit resolution."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_scan_exit_predecessor as predecessor
import ns_training_path_transition_source as parent_source
import owned_output_missing_environment as missing
from selected_lr_release_owner import read, sha256

NAMESPACE = "ns-training-path-scan-exit-20261119"
PREPARATION = "ns-training-path-scan-exit-preparation.json"
OWNER = parent_source.OWNER
CPU_STATUS = "ns_training_path_scan_exit_cpu_archive_verified_gpu_unissued"
PREPARED_STATUS = "ns_training_path_scan_exit_candidate_prepared_gpu_unissued"
ENVIRONMENT = parent_source.ENVIRONMENT
HANDOFF_TEST_SHA256 = parent_source.HANDOFF_TEST_SHA256
SCOPE = (
    "Distinct missing-environment scan-exit original-training-path diagnostic. Preserve "
    "the failed transition23/54commands7/18decisions, all earlier failures/rejections and "
    "stopped waiters/queues. Fresh18cases54commands with unchanged lower LR,5120schedule, "
    "training/configuration bytes, numerical readers/tolerances, compilation and CUDA graphs. "
    "Only ownership observation resolves missing environment using a pinned process-directory "
    "identity and verified disappearance or inherited whole-process exit before full rescan. "
    "Separate source acceptance, failed-evidence seal and exact keeper handoff required. "
    "No production, optimizer quality/efficiency or SwanLab authority."
)
MODULES = {
    "ns_training_path_transition_predecessor": "ns_training_path_scan_exit_predecessor",
    "ns_training_path_transition_source": "ns_training_path_scan_exit_source",
    "prepare_ns_training_path_transition": "prepare_ns_training_path_scan_exit",
    "test_ns_training_path_transition": "test_ns_training_path_scan_exit",
    **{
        f"ns_training_path_transition_{suffix}": f"ns_training_path_scan_exit_{suffix}"
        for suffix in ("contract", "validation", "gpu_owner", "gpu_controller", "gpu_test_support")
    },
    "test_ns_training_path_transition_gpu": "test_ns_training_path_scan_exit_gpu",
}


def literal(value: str) -> str:
    return (
        value.replace(parent_source.NAMESPACE, NAMESPACE)
        .replace("ns_training_path_transition", "ns_training_path_scan_exit")
        .replace("ns-training-path-exit-transition-preparation.json", PREPARATION)
        .replace("OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION", "OUTPUT_MARIN_NS_TRAINING_PATH_SCAN_EXIT")
    )


class ScanExitNamespace(ast.NodeTransformer):
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


ADAPTER = ast.unparse(ScanExitNamespace().visit(ast.parse(parent_source.ADAPTER))) + "\n"
LITERALS = {key: literal(value) for key, value in parent_source.LITERALS.items()}
recorded_parent = parent_source.recorded_parent


def derived_sources(parent: dict) -> dict[str, bytes]:
    sources = {}
    for name, payload in parent_source.derived_sources(parent).items():
        target = MODULES.get(name.removesuffix(".py"), name.removesuffix(".py")) + ".py"
        tree = ScanExitNamespace().visit(ast.parse(payload))
        sources[target] = (
            "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
            + ast.unparse(ast.fix_missing_locations(tree))
            + "\n"
        ).encode()
    return sources


def handoff_bytes(parent: dict) -> bytes:
    path = Path(parent["root"]) / predecessor.RECOVERY_SPEC[0]
    if sha256(path) != predecessor.RECOVERY_SPEC[1]:
        raise ValueError("Scan-exit measured ownership parent receipt changed")
    return missing.derived_source(Path(read(path)["helpers"]) / "owned_output_handoff.py")


HANDOFF_SHA256 = hashlib.sha256(
    handoff_bytes(
        recorded_parent(Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"))
    )
).hexdigest()
