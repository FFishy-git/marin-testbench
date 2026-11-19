# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Derive a fresh diagnostic namespace from accepted ownership orchestration."""

import ast
import copy
from pathlib import Path

import ns_training_path_recovery_predecessor as predecessor
from root_stage_failed_predecessor import file_identity
from selected_lr_release_owner import read, sha256

NAMESPACE = "ns-training-path-marker-exit-recovery-20261119"
PREPARATION = "ns-training-path-recovery-preparation.json"
OWNER = "experiment/fused_attribution_owner.py"
CPU_STATUS = "ns_training_path_marker_exit_recovery_cpu_archive_verified_gpu_unissued"
PREPARED_STATUS = "ns_training_path_marker_exit_recovery_candidate_prepared_gpu_unissued"
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_INSPECTOR_ENABLE": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONFAULTHANDLER": "1",
}
HANDOFF_SHA256 = "293eab75802b1eff8726cd2a886d713742064885c456f74572b8ffd62227fefe"
HANDOFF_TEST_SHA256 = "92cc01fab464f52488213d1458085608b55c163986e591bf2fd1dbc86d406441"
MODULES = {
    "ns_training_path_contract": "ns_training_path_recovery_contract",
    "ns_training_path_validation": "ns_training_path_recovery_validation",
    "ns_training_path_gpu_owner": "ns_training_path_recovery_gpu_owner",
    "ns_training_path_gpu_controller": "ns_training_path_recovery_gpu_controller",
    "ns_training_path_gpu_test_support": "ns_training_path_recovery_gpu_test_support",
}
LITERALS = {
    "run-preparation/ns-training-path-binding-20261119.json": f"run-preparation/{NAMESPACE}-binding.json",
    "controllers/ns-training-path-20261119/tracking": f"controllers/{NAMESPACE}/tracking",
    "run-preparation/ns-training-path-state-20261119.json": f"run-preparation/{NAMESPACE}-state.json",
    "validation/ns_training_path_20261119": "validation/ns_training_path_marker_exit_recovery_20261119",
    "run-preparation/validation-ns-training-path-gpu-20261119.json": f"run-preparation/validation-{NAMESPACE}-gpu.json",
    "OUTPUT_MARIN_NS_TRAINING_PATH_CONTROLLER": "OUTPUT_MARIN_NS_TRAINING_PATH_RECOVERY_CONTROLLER",
    "OUTPUT_MARIN_NS_TRAINING_PATH_VALIDATOR": "OUTPUT_MARIN_NS_TRAINING_PATH_RECOVERY_VALIDATOR",
    "OUTPUT_MARIN_NS_TRAINING_PATH_COMMAND": "OUTPUT_MARIN_NS_TRAINING_PATH_RECOVERY_COMMAND",
    "OUTPUT_MARIN_NS_TRAINING_PATH_PARENT_PROOF": "OUTPUT_MARIN_NS_TRAINING_PATH_RECOVERY_PARENT_PROOF",
    "ns-training-path-terminal-": f"{NAMESPACE}-terminal-",
    ".ns-training-path-terminal-": f".{NAMESPACE}-terminal-",
    "frozen_ns_training_path_diagnostic_not_production": (
        "frozen_ns_training_path_marker_exit_recovery_diagnostic_not_production"
    ),
    "ns_training_path_actual_config_gates_reconstructed_not_production": (
        "ns_training_path_marker_exit_recovery_gates_reconstructed_not_production"
    ),
    "ns_training_path_terminal_independently_reconstructed_exact_restoration_not_production": (
        "ns_training_path_marker_exit_recovery_terminal_independently_reconstructed_not_production"
    ),
}
SCOPE = (
    "Distinct marker-exit recovery of the original training-path diagnostic. Preserve every "
    "original failure and rejected decision. Fresh candidate, controller, commands, outputs and "
    "terminal identities;18cases/54commands, unchanged numerical readers/training sources/"
    "configurations/tolerances, compilation and cached-root CUDA graphs. Accepted source-derived "
    "ownership orchestration and CPU-tested bounded marker rescan; authenticate failed original "
    "handoff and exact restored keeper. Complete diagnostic and independent terminal are still "
    "required before successor gates. No production, quality/efficiency or SwanLab authority."
)
ADAPTER = '''# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticate the separately frozen marker-exit recovery owner before training."""

import importlib
import sys
from pathlib import Path

from .selected_lr_release_owner import read, sha256

TASK = Path(__file__).resolve().parents[1]


def execution_owner():
    preparation = TASK / "ns-training-path-recovery-preparation.json"
    if not preparation.is_file() or preparation.is_symlink():
        raise RuntimeError("Training-path recovery is not separately prepared")
    prepared = read(preparation)
    root = Path(prepared["root"])
    binding = root / "run-preparation/ns-training-path-marker-exit-recovery-20261119-binding.json"
    helpers = root / "controllers/ns-training-path-marker-exit-recovery-20261119/tracking"
    frozen = read(binding)
    helper = helpers / "ns_training_path_recovery_gpu_owner.py"
    if (
        root != root.resolve(strict=True)
        or prepared["candidate"] != str(TASK)
        or frozen["candidate"] != str(TASK)
        or frozen["helpers"] != str(helpers)
        or helpers.is_symlink()
        or helpers != helpers.resolve(strict=True)
        or frozen["preparation_sha256"] != sha256(preparation)
        or helper.is_symlink()
        or sha256(helper) != frozen["helpers_sha256"][helper.name]
        or sha256(Path(__file__)) != frozen["files_sha256"]["experiment/fused_attribution_owner.py"]
    ):
        raise RuntimeError("Training-path recovery candidate or issued owner changed")
    sys.path.insert(0, str(helpers))
    owner = importlib.import_module("ns_training_path_recovery_gpu_owner")
    if Path(owner.__file__).resolve(strict=True) != helper:
        raise RuntimeError("Training-path recovery imported an unissued owner")
    return root, owner


def require_ns_training_path_gpu_owner(root):
    expected, owner = execution_owner()
    if root != expected:
        raise RuntimeError("Training-path recovery validator root differs")
    return owner.require_gpu_owner(root, TASK)


def require_fused_attribution_owner(args):
    root, owner = execution_owner()
    owner.require_rank(root, TASK, args)
'''


class RecoveryNamespace(ast.NodeTransformer):
    """Rewrite only exact accepted module names and protocol namespace literals."""

    def visit_Import(self, node):
        for alias in node.names:
            alias.name = MODULES.get(alias.name, alias.name)
        return node

    def visit_ImportFrom(self, node):
        node.module = MODULES.get(node.module, node.module)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            if node.value in LITERALS:
                node.value = LITERALS[node.value]
            elif node.value.endswith(".py") and node.value[:-3] in MODULES:
                node.value = MODULES[node.value[:-3]] + ".py"
        return node


def replace_function(module: ast.Module, name: str, source: str) -> None:
    matches = [
        index for index, node in enumerate(module.body) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Accepted recovery orchestration function differs: {name}")
    replacement = ast.parse(source).body
    if len(replacement) != 1 or not isinstance(replacement[0], ast.FunctionDef) or replacement[0].name != name:
        raise ValueError("Recovery replacement must name its exact accepted boundary")
    module.body[matches[0]] = replacement[0]


def replace_assignment(module: ast.Module, name: str, value: str) -> None:
    matches = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Accepted recovery orchestration assignment differs: {name}")
    matches[0].value = ast.parse(value, mode="eval").body


def derived_module(parent: dict, name: str) -> bytes:
    """Generate namespaced code with an explicit owner acceptance boundary."""
    path = Path(parent["helpers"]) / f"{name}.py"
    file_identity(path)
    if sha256(path) != parent["helpers_sha256"][path.name]:
        raise ValueError("Recovery derivation parent helper changed")
    module = RecoveryNamespace().visit(copy.deepcopy(ast.parse(path.read_bytes(), filename=str(path))))
    if name == "ns_training_path_gpu_owner":
        for node in module.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ns_training_path_execution_source":
                        alias.name = "prepare_ns_training_path_recovery"
                    elif alias.name == "ns_training_path_predecessor":
                        alias.name = "ns_training_path_recovery_predecessor"
        replace_assignment(module, "ARCHIVE", repr(f"run-preparation/source-{NAMESPACE}.tar"))
        for node in list(module.body):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id.startswith("PREDECESSOR_") for target in node.targets
            ):
                module.body.remove(node)
        replace_function(
            module,
            "predecessor_binding",
            """def predecessor_binding(root):
    return source.predecessor_binding(root)
""",
        )
        replace_function(
            module,
            "require_sealed_predecessor",
            """def require_sealed_predecessor(root):
    require_hidden_cuda()
    return source.require_predecessor(root)
""",
        )
        replace_function(
            module,
            "require_binding",
            """def require_binding(root):
    return source.require_binding(root, globals())
""",
        )
    elif name == "ns_training_path_gpu_controller":
        for node in module.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ns_training_path_execution_source":
                        alias.name = "prepare_ns_training_path_recovery"
        replace_function(
            module,
            "freeze",
            """def freeze(root, cpu_receipt):
    return source.freeze(root, cpu_receipt, contract, owner, validation)
""",
        )
        replace_assignment(
            module,
            "TERMINAL_STATUS",
            repr(LITERALS["ns_training_path_terminal_independently_reconstructed_exact_restoration_not_production"]),
        )
    transformed = ast.fix_missing_locations(module)
    header = "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
    return (header + ast.unparse(transformed) + "\n").encode()


def derived_sources(parent: dict) -> dict[str, bytes]:
    result = {f"{target}.py": derived_module(parent, name) for name, target in MODULES.items()}
    original_test = Path(parent["helpers"]) / "test_ns_training_path_gpu.py"
    if sha256(original_test) != parent["helpers_sha256"][original_test.name]:
        raise ValueError("Recovery lifecycle tests differ from their accepted parent")
    parsed = RecoveryNamespace().visit(ast.parse(original_test.read_bytes()))
    result["test_ns_training_path_recovery_gpu.py"] = (
        "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n" + ast.unparse(parsed) + "\n"
    ).encode()
    return result


def recorded_parent(root: Path) -> dict:
    path = root / predecessor.PARENT_SPEC[0]
    if sha256(path) != predecessor.PARENT_SPEC[1]:
        raise ValueError("Recovery recorded parent receipt changed")
    return read(path)
