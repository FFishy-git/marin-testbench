# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive unchanged numerical commands with serialized full ownership inspection."""

import ast

import ns_training_path_recheck_exit_source as parent_source

NAMESPACE = "ns-training-path-serialized-20261119"
PREPARATION = "ns-training-path-serialized-preparation.json"
OWNER = parent_source.OWNER
CPU_STATUS = "ns_training_path_serialized_cpu_archive_verified_gpu_unissued"
PREPARED_STATUS = "ns_training_path_serialized_candidate_prepared_gpu_unissued"
ENVIRONMENT = parent_source.ENVIRONMENT
SCOPE = (
    "Distinct ownership-serialization diagnostic. Preserve failed recheck1/54commands "
    "0/18decisions, its waiter and every earlier failure. Retain18cases54commands, lower "
    "LR/full5120schedule, trainer/config bytes, numerical readers/tolerances, compilation "
    "and cached-root graphs. Serialize unchanged complete require_controller inspections "
    "under one canonical30-second lock so concurrent rank NVIDIA queries cannot overlap. "
    "Three complete scans and one-second exit proof remain unchanged. Separate CPU/source "
    "archive and failed-predecessor seal required; no production or SwanLab authority."
)
MODULES = {
    "ns_training_path_recheck_exit_predecessor": "ns_training_path_serialized_predecessor",
    "ns_training_path_recheck_exit_source": "ns_training_path_serialized_source",
    "prepare_ns_training_path_recheck_exit": "prepare_ns_training_path_serialized",
    "test_ns_training_path_recheck_exit": "test_ns_training_path_serialized",
    **{
        f"ns_training_path_recheck_exit_{suffix}": f"ns_training_path_serialized_{suffix}"
        for suffix in ("contract", "validation", "gpu_owner", "gpu_controller", "gpu_test_support")
    },
    "test_ns_training_path_recheck_exit_gpu": "test_ns_training_path_serialized_gpu",
}


def literal(value: str) -> str:
    return (
        value.replace(parent_source.NAMESPACE, NAMESPACE)
        .replace("ns_training_path_recheck_exit", "ns_training_path_serialized")
        .replace("ns-training-path-recheck-exit-preparation.json", PREPARATION)
        .replace("OUTPUT_MARIN_NS_TRAINING_PATH_RECHECK_EXIT", "OUTPUT_MARIN_NS_TRAINING_PATH_SERIALIZED")
    )


class SerializedNamespace(ast.NodeTransformer):
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


ADAPTER = ast.unparse(SerializedNamespace().visit(ast.parse(parent_source.ADAPTER))) + "\n"
LITERALS = {key: literal(value) for key, value in parent_source.LITERALS.items()}
recorded_parent = parent_source.recorded_parent
HANDOFF_SHA256 = parent_source.HANDOFF_SHA256
HANDOFF_TEST_SHA256 = parent_source.HANDOFF_TEST_SHA256
handoff_bytes = parent_source.handoff_bytes
handoff_test_bytes = parent_source.handoff_test_bytes


def derived_sources(parent: dict) -> dict[str, bytes]:
    result = {}
    for name, payload in parent_source.derived_sources(parent).items():
        target = MODULES.get(name.removesuffix(".py"), name.removesuffix(".py")) + ".py"
        tree = SerializedNamespace().visit(ast.parse(payload))
        if target == "ns_training_path_serialized_gpu_owner.py":
            boundary = next(index for index, node in enumerate(tree.body) if isinstance(node, ast.FunctionDef))
            tree.body[boundary:boundary] = ast.parse(
                "from ns_training_owner_serialization import inspection_lock\n"
            ).body
            function = next(
                node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "require_controller"
            )
            lock = ast.parse(
                """with inspection_lock(contract.locations(root)[1].parent / "rank-owner-inspection.lock"):
    pass
"""
            ).body[0]
            lock.body = function.body
            function.body = [lock]
        result[target] = (
            "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
            + ast.unparse(ast.fix_missing_locations(tree))
            + "\n"
        ).encode()
    return result
