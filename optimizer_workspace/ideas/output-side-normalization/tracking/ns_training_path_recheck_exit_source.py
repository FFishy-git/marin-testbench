# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a distinct recheck-exit diagnostic with unchanged numerical trajectories."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_recheck_exit_predecessor as predecessor
import ns_training_path_scan_exit_source as parent_source
import owned_output_recheck_exit as recheck
from selected_lr_release_owner import read, sha256

NAMESPACE = "ns-training-path-recheck-exit-20261119"
PREPARATION = "ns-training-path-recheck-exit-preparation.json"
OWNER = parent_source.OWNER
CPU_STATUS = "ns_training_path_recheck_exit_cpu_archive_verified_gpu_unissued"
PREPARED_STATUS = "ns_training_path_recheck_exit_candidate_prepared_gpu_unissued"
ENVIRONMENT = parent_source.ENVIRONMENT
PARENT_HANDOFF_TEST_SHA256 = parent_source.HANDOFF_TEST_SHA256
SCOPE = (
    "Distinct recheck-exit original-training-path diagnostic. Retain scan-exit8/54commands "
    "2/18decisions and failed resumed command/waiter, accepted unissued actual-LR queue "
    "and all previous failures. Fresh18cases54commands retain lower LR, original5120schedule, "
    "training/configuration bytes, numerical readers/tolerances, compilation and cached-root "
    "graphs. Only marked recheck disappearance/terminal state gains existing identity-bound "
    "whole-process exit proof before fresh complete scan. Retain three scan attempts and "
    "one-second exit deadline. Separate CPU/source acceptance and preservation seal required. "
    "No production, optimizer quality/efficiency or private SwanLab authority."
)
MODULES = {
    "ns_training_path_scan_exit_predecessor": "ns_training_path_recheck_exit_predecessor",
    "ns_training_path_scan_exit_source": "ns_training_path_recheck_exit_source",
    "prepare_ns_training_path_scan_exit": "prepare_ns_training_path_recheck_exit",
    "test_ns_training_path_scan_exit": "test_ns_training_path_recheck_exit",
    **{
        f"ns_training_path_scan_exit_{suffix}": f"ns_training_path_recheck_exit_{suffix}"
        for suffix in ("contract", "validation", "gpu_owner", "gpu_controller", "gpu_test_support")
    },
    "test_ns_training_path_scan_exit_gpu": "test_ns_training_path_recheck_exit_gpu",
}


def literal(value: str) -> str:
    return (
        value.replace(parent_source.NAMESPACE, NAMESPACE)
        .replace("ns_training_path_scan_exit", "ns_training_path_recheck_exit")
        .replace("ns-training-path-scan-exit-preparation.json", PREPARATION)
        .replace("OUTPUT_MARIN_NS_TRAINING_PATH_SCAN_EXIT", "OUTPUT_MARIN_NS_TRAINING_PATH_RECHECK_EXIT")
    )


class RecheckExitNamespace(ast.NodeTransformer):
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


ADAPTER = ast.unparse(RecheckExitNamespace().visit(ast.parse(parent_source.ADAPTER))) + "\n"
LITERALS = {key: literal(value) for key, value in parent_source.LITERALS.items()}
recorded_parent = parent_source.recorded_parent


def derived_sources(parent: dict) -> dict[str, bytes]:
    sources = {}
    for name, payload in parent_source.derived_sources(parent).items():
        target = MODULES.get(name.removesuffix(".py"), name.removesuffix(".py")) + ".py"
        tree = RecheckExitNamespace().visit(ast.parse(payload))
        sources[target] = (
            "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
            + ast.unparse(ast.fix_missing_locations(tree))
            + "\n"
        ).encode()
    sources["test_owned_output_handoff.py"] = handoff_test_bytes(parent)
    return sources


def handoff_bytes(parent: dict) -> bytes:
    path = Path(parent["root"]) / predecessor.RECOVERY_SPEC[0]
    if sha256(path) != predecessor.RECOVERY_SPEC[1]:
        raise ValueError("Recheck-exit measured ownership parent receipt changed")
    return recheck.derived_source(Path(read(path)["helpers"]) / "owned_output_handoff.py")


def handoff_test_bytes(parent: dict) -> bytes:
    """Retain handoff cases with complete stat evidence and observed exit disappearance."""
    measured = read(Path(parent["root"]) / predecessor.RECOVERY_SPEC[0])
    path = Path(measured["helpers"]) / "test_owned_output_handoff.py"
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_HANDOFF_TEST_SHA256:
        raise ValueError("Recheck-exit inherited handoff tests changed")
    tree = ast.parse(payload)
    test = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "test_marker_scan_rescans_exiting_helpers_and_retains_new_live_children"
    )
    before = next(
        index for index, node in enumerate(test.body) if isinstance(node, ast.FunctionDef) and node.name == "exiting"
    )
    test.body[before:before] = ast.parse(
        """elapsed = 0
drain = handoff.exit_drain_handler.drain_exit

def clock():
    return elapsed

def pause(interval):
    nonlocal elapsed
    elapsed += interval
    shutil.rmtree(proc / "900")

monkeypatch.setattr(handoff.exit_drain_handler, "whole_process_exited", lambda root, owner: False)
monkeypatch.setattr(
    handoff.exit_drain_handler, "drain_exit",
    lambda root, owner, **kwargs: drain(root, owner, clock=clock, pause=pause),
)
"""
    ).body
    exiting = next(node for node in test.body if isinstance(node, ast.FunctionDef) and node.name == "exiting")
    branch = next(
        node
        for node in ast.walk(exiting)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "exit_state == 'zombie'"
    )
    status = next(
        index
        for index, node in enumerate(branch.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "fields[0]"
    )
    branch.body[status + 1 : status + 1] = ast.parse('fields[6] = "4"\nfields.append("0")').body
    branch.body.extend(ast.parse('(proc / "900/environ").write_bytes(b"")').body)
    test.body.extend(
        ast.parse(
            """if exit_state == "zombie":
    assert elapsed > 0 and not (proc / "900").exists()
"""
        ).body
    )
    return (
        "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
        + ast.unparse(ast.fix_missing_locations(tree))
        + "\n"
    ).encode()


HANDOFF_SHA256 = hashlib.sha256(
    handoff_bytes(
        recorded_parent(Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"))
    )
).hexdigest()
HANDOFF_TEST_SHA256 = hashlib.sha256(
    handoff_test_bytes(
        recorded_parent(Path("/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"))
    )
).hexdigest()
