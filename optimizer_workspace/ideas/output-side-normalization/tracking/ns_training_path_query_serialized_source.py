# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retain every ownership statement while excluding source reads from the scan lock."""

import ast

import ns_training_path_serialized_source as parent_source

NAMESPACE = "ns-training-path-query-serialized-20261120"
PREPARATION = "ns-training-path-query-serialized-preparation.json"
OWNER = parent_source.OWNER
CPU_STATUS = "ns_training_path_query_serialized_cpu_archive_verified_gpu_unissued"
PREPARED_STATUS = "ns_training_path_query_serialized_candidate_prepared_gpu_unissued"
ENVIRONMENT = parent_source.ENVIRONMENT
SCOPE = (
    "Distinct query-serialization diagnostic preserving failed whole-inspection0/54commands "
    "0/18decisions, its original waiter, all earlier failures and accepted unissued queues. "
    "Retain18cases54commands, lower LR/full5120schedule, trainer/config bytes, numerical "
    "readers/tolerances, compilation and cached-root graphs. Keep authenticated binding "
    "reads and acknowledgment wait outside the unchanged30-second lock; serialize every "
    "statement from the hostname query through final ownership/generation recheck. Preserve "
    "statement order, all ancestry/source/GPU/lock/environment gates, three complete scans "
    "and one-second exit proof. Separate CPU/source archive and failed-predecessor seal "
    "required. No production, optimizer quality/efficiency or SwanLab authority."
)
MODULES = {
    "ns_training_path_serialized_predecessor": "ns_training_path_query_serialized_predecessor",
    "ns_training_path_serialized_source": "ns_training_path_query_serialized_source",
    "prepare_ns_training_path_serialized": "prepare_ns_training_path_query_serialized",
    "test_ns_training_path_serialized": "test_ns_training_path_query_serialized",
    **{
        f"ns_training_path_serialized_{suffix}": f"ns_training_path_query_serialized_{suffix}"
        for suffix in ("contract", "validation", "gpu_owner", "gpu_controller", "gpu_test_support")
    },
    "test_ns_training_path_serialized_gpu": "test_ns_training_path_query_serialized_gpu",
}


def literal(value: str) -> str:
    return (
        value.replace(parent_source.NAMESPACE, NAMESPACE)
        .replace("ns_training_path_serialized", "ns_training_path_query_serialized")
        .replace("ns-training-path-serialized-preparation.json", PREPARATION)
        .replace("OUTPUT_MARIN_NS_TRAINING_PATH_SERIALIZED", "OUTPUT_MARIN_NS_TRAINING_PATH_QUERY_SERIALIZED")
    )


class QuerySerializedNamespace(ast.NodeTransformer):
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


ADAPTER = ast.unparse(QuerySerializedNamespace().visit(ast.parse(parent_source.ADAPTER))) + "\n"
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
        tree = QuerySerializedNamespace().visit(ast.parse(payload))
        if target == "ns_training_path_query_serialized_gpu_owner.py":
            function = next(
                node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "require_controller"
            )
            if len(function.body) != 1 or not isinstance(function.body[0], ast.With):
                raise ValueError("Query serialization parent no longer encloses the complete inspection")
            lock = function.body[0]
            statements = lock.body
            boundaries = [
                index
                for index, node in enumerate(statements)
                if isinstance(node, ast.If)
                and any(
                    isinstance(item, ast.Call)
                    and isinstance(item.func, ast.Attribute)
                    and isinstance(item.func.value, ast.Name)
                    and item.func.value.id == "subprocess"
                    and item.func.attr == "check_output"
                    for item in ast.walk(node.test)
                )
            ]
            if len(boundaries) != 1 or not isinstance(statements[-1], ast.Return):
                raise ValueError("Query serialization hostname/return boundary changed")
            boundary = boundaries[0]
            if ast.unparse(statements[0]) != "frozen = require_binding(root)":
                raise ValueError("Query serialization source authentication boundary changed")
            lock.body = statements[boundary:-1]
            function.body = [*statements[:boundary], lock, statements[-1]]
        result[target] = (
            "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
            + ast.unparse(ast.fix_missing_locations(tree))
            + "\n"
        ).encode()
    return result
