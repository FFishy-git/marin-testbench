# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a separately owned streaming observation diagnostic with unchanged numerics."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_query_serialized_source as parent_source
import ns_training_path_streaming_predecessor as predecessor
import owned_output_streaming_scan as streaming
from selected_lr_release_owner import read, sha256

NAMESPACE = "ns-training-path-streaming-20261120"
PREPARATION = "ns-training-path-streaming-preparation.json"
OWNER = parent_source.OWNER
CPU_STATUS = "ns_training_path_streaming_cpu_archive_verified_gpu_unissued"
PREPARED_STATUS = "ns_training_path_streaming_candidate_prepared_gpu_unissued"
ENVIRONMENT = parent_source.ENVIRONMENT
SCOPE = (
    "Distinct streaming/query ownership diagnostic. Preserve failed query6/54commands "
    "2/18accepted controls, pre-command ten-step refusal, stopped waiter and all earlier "
    "failures/rejections/seals and accepted unissued actual-LR preparations/queues. Retain "
    "18cases54commands, lower LR and original5120schedule, trainer/configuration bytes, "
    "numerical readers/tolerances, compilation and cached-root CUDA graphs. Change only "
    "marker-scan identity/environment observation interleaving; every complete namespace "
    "filter, scan statement, marked-process recheck, three attempts, one-second exit proof "
    "and query-lock boundary remains unchanged. Separate source/CPU acceptance and "
    "failed-precommand predecessor seal required. No production, optimizer quality/"
    "efficiency or SwanLab authority."
)
MODULES = {
    "ns_training_path_query_serialized_predecessor": "ns_training_path_streaming_predecessor",
    "ns_training_path_query_serialized_source": "ns_training_path_streaming_source",
    "prepare_ns_training_path_query_serialized": "prepare_ns_training_path_streaming",
    "test_ns_training_path_query_serialized": "test_ns_training_path_streaming",
    **{
        f"ns_training_path_query_serialized_{suffix}": f"ns_training_path_streaming_{suffix}"
        for suffix in ("contract", "validation", "gpu_owner", "gpu_controller", "gpu_test_support")
    },
    "test_ns_training_path_query_serialized_gpu": "test_ns_training_path_streaming_gpu",
}


def literal(value: str) -> str:
    return (
        value.replace(parent_source.NAMESPACE, NAMESPACE)
        .replace("ns_training_path_query_serialized", "ns_training_path_streaming")
        .replace(parent_source.PREPARATION, PREPARATION)
        .replace("OUTPUT_MARIN_NS_TRAINING_PATH_QUERY_SERIALIZED", "OUTPUT_MARIN_NS_TRAINING_PATH_STREAMING")
    )


class StreamingNamespace(ast.NodeTransformer):
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


ADAPTER = ast.unparse(StreamingNamespace().visit(ast.parse(parent_source.ADAPTER))) + "\n"
LITERALS = {key: literal(value) for key, value in parent_source.LITERALS.items()}
recorded_parent = parent_source.recorded_parent


def query_helpers(parent: dict) -> Path:
    path = Path(parent["root"]) / predecessor.CPU_SPEC[0]
    if sha256(path) != predecessor.CPU_SPEC[1]:
        raise ValueError("Streaming source accepted query CPU receipt changed")
    return Path(read(path)["helpers"])


def handoff_bytes(parent: dict) -> bytes:
    return streaming.derived_source(query_helpers(parent) / "owned_output_handoff.py")


def handoff_test_bytes(parent: dict) -> bytes:
    path = query_helpers(parent) / "test_owned_output_handoff.py"
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != parent_source.HANDOFF_TEST_SHA256:
        raise ValueError("Streaming source accepted query handoff cases changed")
    return payload


def derived_sources(parent: dict) -> dict[str, bytes]:
    sources = {}
    for name, payload in parent_source.derived_sources(parent).items():
        target = MODULES.get(name.removesuffix(".py"), name.removesuffix(".py")) + ".py"
        tree = StreamingNamespace().visit(ast.parse(payload))
        sources[target] = (
            "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
            + ast.unparse(ast.fix_missing_locations(tree))
            + "\n"
        ).encode()
    sources["test_owned_output_handoff.py"] = handoff_test_bytes(parent)
    return sources


def study_sources(parent: dict) -> dict[str, bytes]:
    """Bind fake-tree studies to flat archived copies of the original query scanner."""
    query = query_helpers(parent)
    sources = {
        "query_original_owned_output_handoff.py": (query / "owned_output_handoff.py").read_bytes(),
        "query_original_test_owned_output_handoff.py": handoff_test_bytes(parent),
    }
    if hashlib.sha256(sources["query_original_owned_output_handoff.py"]).hexdigest() != streaming.PARENT_SHA256:
        raise ValueError("Streaming study original scanner fixture changed")
    local = Path(__file__).resolve().parent
    for name in ("owned_output_helper_churn.py", "test_owned_output_streaming_scan.py"):
        tree = ast.parse((local / name).read_bytes())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in ("SOURCE", "QUERY") for target in node.targets
            ):
                node.value = ast.parse("Path(__file__).resolve().parent", mode="eval").body
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                base = ast.unparse(node.left)
                if base == "directory" and ast.unparse(node.right) == "name":
                    node.right = ast.parse(
                        '"query_original_owned_output_handoff.py" if name == "owned_output_handoff.py" else name',
                        mode="eval",
                    ).body
                elif base == "QUERY":
                    if isinstance(node.right, ast.Constant) and node.right.value == "owned_output_handoff.py":
                        node.right.value = "query_original_owned_output_handoff.py"
                    elif isinstance(node.right, ast.Name) and node.right.id == "name":
                        node.right = ast.parse(
                            '"query_original_owned_output_handoff.py" if name == "owned_output_handoff.py" else name',
                            mode="eval",
                        ).body
                    elif isinstance(node.right, ast.Constant) and node.right.value == "test_owned_output_handoff.py":
                        node.right.value = "query_original_test_owned_output_handoff.py"
        sources[name] = (
            "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
            + ast.unparse(ast.fix_missing_locations(tree))
            + "\n"
        ).encode()
    return sources


HANDOFF_SHA256 = hashlib.sha256(
    handoff_bytes(
        recorded_parent(Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"))
    )
).hexdigest()
HANDOFF_TEST_SHA256 = parent_source.HANDOFF_TEST_SHA256
