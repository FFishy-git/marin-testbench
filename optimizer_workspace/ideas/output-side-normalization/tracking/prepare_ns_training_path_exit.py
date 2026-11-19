# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use separately derived accepted orchestration for the fresh exit-drain diagnostic."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_exit_source as exit_source

PARENT_PREPARER_SHA256 = "7c8f9e7460555bfab7b88afc4b18b6b31febc26badec14f6b48e722c856f5252"
CPU_PATH_VALUE = f"run-preparation/{exit_source.NAMESPACE}-cpu/pidfd-bound-20261119T1542/receipt.json"


def runtime_source() -> bytes:
    """Reuse accepted preparation functions with explicit successor closure changes."""
    path = Path(__file__).with_name("prepare_ns_training_path_recovery.py")
    payload = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(payload).hexdigest() != PARENT_PREPARER_SHA256:
        raise ValueError("Exit-drain inherited preparer source changed")
    tree = exit_source.ExitNamespace().visit(ast.parse(payload))
    replacements = {
        "CPU_PATH": repr(CPU_PATH_VALUE),
        "LOCAL_HELPERS": repr(
            (
                "ns_training_path_exit_predecessor.py",
                "ns_training_path_exit_source.py",
                "prepare_ns_training_path_exit.py",
                "test_ns_training_path_exit.py",
                "owned_output_attribution.py",
                "owned_output_exit_drain.py",
                "test_owned_output_exit_drain.py",
            )
        ),
        "TEST_COUNTS": repr(
            {
                "test_ns_training_path_exit_gpu": 45,
                "test_owned_output_handoff": 36,
                "test_owned_output_exit_drain": 15,
            }
        ),
        "RECOVERY_TEST_COUNT": "29",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(replacements[target.id], mode="eval").body
    replacement = ast.parse(
        """def helper_bytes(parent):
    directory = Path(parent["helpers"])
    sources = {}
    for name, digest in parent["helpers_sha256"].items():
        path = directory / name
        if sha256(path) != digest:
            raise ValueError("Exit-drain inherited helper changed")
        sources[name] = path.read_bytes()
    local = Path(__file__).resolve().parent
    sources["owned_output_handoff.py"] = source.handoff_bytes(parent)
    measured = read(Path(parent["root"]) / predecessor.RECOVERY_SPEC[0])
    test_path = Path(measured["helpers"]) / "test_owned_output_handoff.py"
    if sha256(test_path) != source.HANDOFF_TEST_SHA256:
        raise ValueError("Exit-drain inherited ownership tests changed")
    sources[test_path.name] = test_path.read_bytes()
    sources["test_ns_training_path_recovery.py"] = (local / "test_ns_training_path_recovery.py").read_bytes()
    sources["prepare_ns_training_path_recovery.py"] = (local / "prepare_ns_training_path_recovery.py").read_bytes()
    for name in LOCAL_HELPERS:
        if name in sources:
            raise ValueError("Exit-drain new helper collides with original source closure")
        sources[name] = (local / name).read_bytes()
    sources |= source.derived_sources(parent)
    external = sys.stdlib_module_names | {"torch", "triton", "numpy", "pytest", "pytest_timeout"}
    pending, inspected = [*LOCAL_HELPERS, *source.derived_sources(parent), "owned_output_handoff.py"], set()
    while pending:
        name = pending.pop()
        if name in inspected or not name.endswith(".py"):
            continue
        inspected.add(name)
        for node in ast.walk(ast.parse(sources[name])):
            modules = ([item.name for item in node.names] if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else [])
            for module in modules:
                dependency = module.split(".")[0]
                if dependency in external:
                    continue
                filename = dependency + ".py"
                if filename not in sources:
                    if not (local / filename).is_file():
                        raise ValueError("Exit-drain helper closure is incomplete: " + filename)
                    sources[filename] = (local / filename).read_bytes()
                    pending.append(filename)
    return sources
"""
    ).body[0]
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name == "helper_bytes":
            tree.body[index] = replacement
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


# Execute only the source-pinned parent's explicitly namespaced preparation implementation.
exec(compile(runtime_source(), __file__, "exec"), globals())
