# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a distinct LR adapter with one explicit measured execution-helper transformation."""

import ast
import hashlib
from pathlib import Path

import ns_training_path_transition_lr_contract as transition_contract
import ns_training_path_transition_lr_parent as bridge

ADAPTER_SOURCE_SHA256 = "87b4aceca18a95788474956fc03ff3616b3ed377cef39131ac41ea8805876521"
NAMESPACE = transition_contract.NAMESPACE
HANDOFF = "owned_output_handoff.py"
HANDOFF_SHA256 = "ec894019aa1af6ceadd81da2a4045a868a0fa34dff7c5c6da08a6230868defcb"
NUMERICAL_HANDOFF_SHA256 = "b359c02fd184f1afa7c32f3a4fbab8693ea41f7c4a7225f38cd17d7ab52b43f6"
HANDOFF_FIXTURE = "fixtures/transition-lr-original-owned_output_handoff.py"
EXECUTION_PINS = {
    HANDOFF: HANDOFF_SHA256,
    "owned_output_attribution.py": "c999637ad12729d767f2d9539e7adeb359fd38260f6e09b089b35d00be235653",
    "owned_output_exit_drain.py": "981deaf0ec73f43c69e64c0409212100afafb0929e6c0cb11ccf8892edd17db6",
    "owned_output_exit_transition.py": "e3ae48eea72cd3d9828f06c407551b8f316716498994f54b9d5bb77a55fa9a59",
}
NEW_FILES = (
    "ns_training_path_transition_lr_parent.py",
    "ns_training_path_transition_lr_contract.py",
    "ns_training_path_transition_lr_validation.py",
    "ns_training_path_transition_lr_source.py",
    "test_ns_training_path_transition_lr_source.py",
)
MODULES = {
    "ns_training_path_lr_contract": "ns_training_path_transition_lr_contract",
    "ns_training_path_lr_validation": "ns_training_path_transition_lr_validation",
    "ns_training_path_lr_execution_source": "ns_training_path_transition_lr_source",
    "ns_training_path_lr_predecessor": "ns_training_path_transition_lr_parent",
    "prepare_ns_training_path_lr_predecessor": "ns_training_path_transition_lr_parent",
    "ns_training_path_lr_gpu_owner": "ns_training_path_transition_lr_gpu_owner",
    "ns_training_path_lr_gpu_controller": "ns_training_path_transition_lr_gpu_controller",
}
SCOPE_VALUE = (
    "Distinct transition actual-LR adapter preparation, not ownership or launch. Preserve all "
    "accepted numerical candidate, configuration and reader bytes; change only the candidate "
    "owner adapter and execution namespace. Explicitly transform the one conflicting execution "
    "handoff to the measured transition bytes, retaining its original source in the inventory "
    "and all other numerical helpers unchanged. Keep the independent terminal archive external "
    "and authenticated through isolated subprocesses. Retain54configs162commands, original5120 "
    "schedule, all rejections, compilation, cached-root CUDA graphs and unchanged tolerances. "
    "No seal, GPU transfer, production, full-budget quality, efficiency or SwanLab authority."
)


class Namespace(ast.NodeTransformer):
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


def literal(value: str) -> str:
    value = transition_contract.literal(value)
    for before, after in MODULES.items():
        value = value.replace(before, after)
    return (
        value.replace("ns-training-path-actual-lr-owner-preparation.json", f"{NAMESPACE}-preparation.json")
        .replace("fixtures/ns-training-path-actual-lr-parent-owner.py", "fixtures/transition-lr-parent-owner.py")
        .replace("test_ns_training_path_lr_execution_source", "test_ns_training_path_transition_lr_source")
    )


def runtime_source() -> bytes:
    path = Path(__file__).with_name("ns_training_path_lr_execution_source.py")
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != ADAPTER_SOURCE_SHA256:
        raise ValueError("Transition-LR pinned adapter implementation changed")
    tree = Namespace().visit(ast.parse(raw))
    replacements = {
        "PARENT": repr(bridge.ADAPTER[0]),
        "PARENT_SHA256": repr(bridge.ADAPTER[1]),
        "VERIFIER_SHA256": repr(bridge.VERIFIER_SHA256),
        "CHECKER_SHA256": repr(bridge.CHECKER_SHA256),
        "NEW_HELPERS": repr(NEW_FILES),
        "TEST_COUNT": "15",
        "SCOPE": repr(SCOPE_VALUE),
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in replacements:
                    node.value = ast.parse(replacements[target.id], mode="eval").body
                elif isinstance(target, ast.Name) and target.id == "ADAPTER":
                    node.value = ast.Constant(value=literal(node.value.value))
    functions = {
        "inherited": (
            """def inherited(root):
    return bridge.adapter_cpu(root)
"""
        ),
        "execution_sources": (
            """def execution_sources(previous):
    sources = {name: Path(previous["helpers"]) / name for name in previous["helpers_sha256"]}
    if previous["helpers_sha256"][HANDOFF] != NUMERICAL_HANDOFF_SHA256:
        raise ValueError("Transition-LR original numerical handoff differs")
    terminal_cpu = bridge.require_cpu(Path(previous["root"]))
    external = Path(terminal_cpu["helpers"])
    for name,digest in EXECUTION_PINS.items():
        if terminal_cpu["helpers_sha256"].get(name) != digest or sha256(external/name) != digest:
            raise ValueError("Transition-LR measured handoff provenance differs")
        if name in sources and name != HANDOFF and sha256(sources[name]) != digest:
            raise ValueError("Transition-LR would change another inherited helper")
        sources[name] = external/name
    if set(sources).intersection(NEW_HELPERS):
        raise ValueError("Transition-LR would overwrite an inherited helper")
    local = Path(__file__).resolve().parent
    sources.update({name:local/name for name in NEW_HELPERS})
    allowed = sys.stdlib_module_names | {"torch","triton","numpy","pytest","pytest_timeout"}
    for name in NEW_HELPERS:
        path = sources[name]
        if path.is_symlink() or path != path.resolve(strict=True):
            raise ValueError("Transition-LR source is aliased")
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node,ast.ImportFrom) and node.level:
                raise ValueError("Transition-LR requires flat imports")
            modules = [a.name for a in node.names] if isinstance(node,ast.Import) else (
                [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            if any(m.split(".")[0]+".py" not in sources and m.split(".")[0] not in allowed for m in modules):
                raise ValueError("Transition-LR import escapes its accepted closure")
    if any(sha256(sources[n]) != h for n,h in previous["helpers_sha256"].items() if n != HANDOFF):
        raise ValueError("Transition-LR changed an unreviewed inherited helper")
    return sources
"""
        ),
    }
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            tree.body[index] = ast.parse(functions[node.name]).body[0]
        if isinstance(node, ast.FunctionDef) and node.name == "inventory":
            node.body = ast.parse(
                """return (
    previous["inventory_sha256"] | prepared["files_sha256"] | prepared["parent_metadata_sha256"]
    | {PREPARATION:sha256(candidate/PREPARATION), PARENT_OWNER:previous["inventory_sha256"][OWNER],
       HANDOFF_FIXTURE:NUMERICAL_HANDOFF_SHA256}
    | {f"tracking/{name}":digest for name,digest in prepared["helpers_sha256"].items()}
)
"""
            ).body
        if isinstance(node, ast.FunctionDef) and node.name == "preparation_record":
            for record in ast.walk(node):
                if not isinstance(record, ast.Dict):
                    continue
                for position, key in enumerate(record.keys):
                    if isinstance(key, ast.Constant) and key.value == "predecessor_cpu_receipt_sha256":
                        record.values[position] = ast.parse("bridge.CPU_SHA256", mode="eval").body
                if any(
                    isinstance(key, ast.Constant) and key.value == "required_predecessor_seal" for key in record.keys
                ):
                    record.keys.extend(
                        ast.parse(repr(value), mode="eval").body
                        for value in ("helper_transform", "numerical_cpu_receipt_sha256", "terminal_archive_merged")
                    )
                    record.values.extend(
                        ast.parse(value, mode="eval").body
                        for value in (
                            "{HANDOFF:dict(original_sha256=NUMERICAL_HANDOFF_SHA256,executed_sha256=HANDOFF_SHA256)}",
                            "bridge.NUMERICAL[1]",
                            "False",
                        )
                    )
        if isinstance(node, ast.FunctionDef) and node.name == "cpu":
            for branch in ast.walk(node):
                if isinstance(branch, ast.If) and ast.unparse(branch.test) == "name.startswith('tracking/')":
                    branch.orelse.insert(
                        0,
                        ast.parse(
                            """if name == HANDOFF_FIXTURE:
    original = Path(previous["helpers"])/HANDOFF
"""
                        ).body[0],
                    )
                    branch.orelse[0].orelse = branch.orelse[1:]
                    branch.orelse = branch.orelse[:1]
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


exec(compile(runtime_source(), __file__, "exec"), globals())
