# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Archive the precision CUDA component, preserving the reviewed CPU-study parent unchanged."""

import ast
from pathlib import Path

import replay_saved_ns as replay

PARENT = (
    "run-preparation/ns-precision-placement-cpu/reviewed-20261119T1935/receipt.json",
    "f43d4f5bd6806c7abcaf066e705c77a67185c415ccdfa0b7f45b776bda6b8c55",
    "cpu_precision_placement_sources_tests_verified_no_gpu_authority",
    64,
)
TEMPLATE_SHA256 = "fc91c17412936ff4394fa34140e1e3227ced84f4f823b30cb24aa209b5750d4e"
NEW = (
    "ns_precision_placement_maps.py",
    "test_ns_precision_placement_maps.py",
    "ns_precision_placement_execution.py",
    "test_ns_precision_placement_execution.py",
    "prepare_ns_precision_placement_execution.py",
)
TEST_COUNTS = {
    "test_ns_precision_placement_maps": 22,
    "test_ns_precision_placement_execution": 27,
}
STATUS = "ns_precision_placement_execution_cpu_archive_verified_gpu_unissued"


def retained_preparer() -> dict:
    """Reuse the authenticated archive lifecycle, changing only successor inventory and test scope."""
    path = (replay.ROOT / PARENT[0]).parent / "tracking/prepare_ns_precision_placement.py"
    with replay.analyzer._bound_file(path, TEMPLATE_SHA256) as stream:
        module = ast.parse(stream.read(), filename=str(path))
    substitutions = {
        "PARENT": PARENT,
        "CHANGED": (),
        "NEW": NEW,
        "TEST_FILES": tuple(f"{name}.py" for name in TEST_COUNTS),
        "TEST_COUNTS": TEST_COUNTS,
        "STATUS": STATUS,
    }
    changed = set()
    for node in module.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "ns_precision_placement" and alias.asname == "placement":
                    alias.name = "ns_precision_placement_execution"
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in substitutions:
                node.value = ast.parse(repr(substitutions[name]), mode="eval").body
                changed.add(name)
    if changed != set(substitutions):
        raise ValueError("Precision execution archive template constants differ")
    namespace = {"__name__": "precision_placement_execution_archive_template", "__file__": str(Path(__file__).resolve())}
    exec(compile(ast.fix_missing_locations(module), str(Path(__file__).resolve()), "exec"), namespace)
    return namespace


_retained = retained_preparer()
cpu = _retained["cpu"]
require_cpu = _retained["require_cpu"]
source_inventory = _retained["source_inventory"]
command = _retained["command"]


if __name__ == "__main__":
    _retained["main"]()
