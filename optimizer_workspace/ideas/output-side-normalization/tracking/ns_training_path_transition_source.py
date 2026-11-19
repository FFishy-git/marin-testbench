# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue a distinct mixed-exit-bracket diagnostic while retaining numerical sources."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_exit_source as parent_source
import ns_training_path_transition_predecessor as predecessor
import owned_output_exit_transition as transition
from selected_lr_release_owner import read, sha256

NAMESPACE = "ns-training-path-exit-transition-20261119"
PREPARATION = "ns-training-path-exit-transition-preparation.json"
OWNER = parent_source.OWNER
CPU_STATUS = "ns_training_path_exit_transition_cpu_archive_verified_gpu_unissued"
PREPARED_STATUS = "ns_training_path_exit_transition_candidate_prepared_gpu_unissued"
ENVIRONMENT = parent_source.ENVIRONMENT
HANDOFF_TEST_SHA256 = parent_source.HANDOFF_TEST_SHA256
SCOPE = (
    "Distinct mixed-exit-bracket original-training-path diagnostic. Preserve every failed "
    "original/recovery/exit-drain attempt and waiter. Fresh18cases54commands and terminal; "
    "unchanged lower LR,5120schedule, all numerical readers, training/configuration bytes, "
    "tolerances, compilation and cached-root CUDA graphs. Only measured ownership scan gains "
    "bounded mixed live/terminal observation; require exact ownership, PF_EXITING, empty "
    "files/zero memory and whole-process pidfd exit or directory absence before full rescan. "
    "Separate CPU archive and failed handoff seal required; no production or SwanLab authority."
)
MODULES = {
    "ns_training_path_exit_predecessor": "ns_training_path_transition_predecessor",
    "ns_training_path_exit_source": "ns_training_path_transition_source",
    "prepare_ns_training_path_exit": "prepare_ns_training_path_transition",
    "test_ns_training_path_exit": "test_ns_training_path_transition",
    **{
        f"ns_training_path_exit_{suffix}": f"ns_training_path_transition_{suffix}"
        for suffix in ("contract", "validation", "gpu_owner", "gpu_controller", "gpu_test_support")
    },
    "test_ns_training_path_exit_gpu": "test_ns_training_path_transition_gpu",
}


def literal(value: str) -> str:
    return (
        value.replace(parent_source.NAMESPACE, NAMESPACE)
        .replace("ns_training_path_exit_drain", "ns_training_path_transition")
        .replace("ns_training_path_exit_", "ns_training_path_transition_")
        .replace("ns_training_path_exit", "ns_training_path_transition")
        .replace("ns-training-path-exit-drain-preparation.json", PREPARATION)
        .replace("OUTPUT_MARIN_NS_TRAINING_PATH_EXIT", "OUTPUT_MARIN_NS_TRAINING_PATH_TRANSITION")
    )


LITERALS = {key: literal(value) for key, value in parent_source.LITERALS.items()}


class TransitionNamespace(ast.NodeTransformer):
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


ADAPTER = ast.unparse(TransitionNamespace().visit(ast.parse(parent_source.ADAPTER))) + "\n"


def recorded_parent(root: Path) -> dict:
    return parent_source.recorded_parent(root)


def derived_sources(parent: dict) -> dict[str, bytes]:
    result = {}
    for name, content in parent_source.derived_sources(parent).items():
        target = MODULES.get(name.removesuffix(".py"), name.removesuffix(".py")) + ".py"
        tree = TransitionNamespace().visit(ast.parse(content))
        result[target] = (
            "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
            + ast.unparse(ast.fix_missing_locations(tree))
            + "\n"
        ).encode()
    return result


def handoff_bytes(parent: dict) -> bytes:
    path = Path(parent["root"]) / predecessor.RECOVERY_SPEC[0]
    if sha256(path) != predecessor.RECOVERY_SPEC[1]:
        raise ValueError("Exit-transition inherited measured recovery receipt differs")
    return transition.derived_source(Path(read(path)["helpers"]) / "owned_output_handoff.py")


HANDOFF_SHA256 = hashlib.sha256(
    handoff_bytes(
        recorded_parent(Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"))
    )
).hexdigest()
