# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Owner-gated full-batch saved NS CUDA replay; no orchestration or production acceptance."""

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
from uuid import UUID

import replay_saved_ns as replay
import saved_ns_gpu_owner as ownership
import torch
from benchmark_diagonal import source_definitions

STATUS = "saved_ns_cuda_replay_not_production_gate"


def tensor_layout(value: torch.Tensor) -> dict:
    """Describe the concrete tensor layout and process-local storage identity."""
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "storage_offset": value.storage_offset(),
        "storage_nbytes": value.untyped_storage().nbytes(),
        "data_ptr": value.data_ptr(),
        "storage_data_ptr": value.untyped_storage().data_ptr(),
        "contiguous": value.is_contiguous(),
    }


def clone_source(source: torch.Tensor) -> torch.Tensor:
    """Clone a finite CPU source without silently changing its batch layout."""
    if source.device.type != "cpu" or source.dtype != torch.float32 or source.ndim != 3:
        raise ValueError("Saved source staging requires an FP32 CPU batch")
    if source.numel() == 0 or not torch.isfinite(source).all():
        raise ValueError("Saved source staging requires finite nonempty tensors")
    result = source.detach().clone(memory_format=torch.preserve_format)
    if result.stride() != source.stride() or result.storage_offset() != source.storage_offset():
        raise ValueError("Cloning would change the saved source layout")
    return result


def source_mode(groups: dict, source: torch.Tensor) -> str:
    """Select resident C/R inputs by CPU object identity, never by their possibly equal values."""
    for mode in replay.analyzer.MODES:
        if source is groups[mode]["source"]:
            return mode
    raise ValueError("Replay callback received an unbound source identity")


def require_unchanged_source(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Reject altered resident input values after copying them back to CPU."""
    if (
        actual.device.type != "cpu"
        or expected.device.type != "cpu"
        or actual.dtype != expected.dtype
        or actual.shape != expected.shape
        or not torch.isfinite(actual).all()
        or not torch.equal(actual, expected)
    ):
        raise ValueError("Resident CUDA source changed during replay")


def write_outputs(path: Path, outputs: tuple[torch.Tensor, ...]) -> str:
    """Exclusively save four CPU clones and return their evidence digest."""
    if len(outputs) != len(replay.ORDER):
        raise ValueError("Replay output evidence requires four ordered tensors")
    for value in outputs:
        if value.device.type != "cpu" or value.dtype != torch.float32 or tuple(value.shape) != replay.SAVED_SHAPE:
            raise ValueError("Replay output evidence requires four full FP32 CPU batches")
    with path.open("xb") as stream:
        torch.save(outputs, stream)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bound_sources(stack: ExitStack, helpers: Path, frozen: dict, owner_source: Path) -> dict:
    paths = {
        Path(__file__).resolve(),
        Path(replay.__file__).resolve(),
        Path(replay.analyzer.__file__).resolve(),
        Path(replay.analyzer.compare_saved.__code__.co_filename).resolve(),
        Path(replay.differences.__code__.co_filename).resolve(),
        Path(source_definitions.__code__.co_filename).resolve(),
        owner_source,
    }
    sources = ownership.worker_sources(frozen)
    if paths != {Path(name) for name in sources}:
        raise ValueError("Loaded worker helpers differ from the complete bound execution source set")
    for path in sorted(paths):
        if path.parent != helpers:
            raise ValueError("Loaded worker helper is outside the frozen execution directory")
        expected = sources[str(path)]
        stack.enter_context(replay.analyzer._bound_file(path, expected))
    return sources


def run(root: Path, binding_sha256: str, state_path: Path, worker_index: int) -> dict:
    """Require the adapter's exact owner before any CUDA access, then publish one bounded replay."""
    owner = ownership.require_gpu_owner(root, binding_sha256, state_path, worker_index)
    binding_path, helpers, expected_state, work = ownership.locations(root)
    frozen = owner["binding"]
    if (
        state_path != expected_state
        or Path(frozen["helpers"]) != helpers
        or Path(frozen["work"]) != work
        or owner["device"] != "cuda:0"
        or worker_index not in (0, 1)
    ):
        raise ValueError("Worker owner device, locations or index differs")
    environment = dict(frozen["worker_environment"])
    if (
        environment.get("CUDA_VISIBLE_DEVICES") != owner["physical_uuid"]
        or environment.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or any(os.environ.get(key) != value for key, value in environment.items())
        or os.environ.get("NVIDIA_TF32_OVERRIDE") == "1"
    ):
        raise ValueError("Worker requires its UUID-bound deterministic CUDA environment")
    if work != work.resolve(strict=True) or not work.is_relative_to(root):
        raise ValueError("Worker output directory is not the canonical bound diagnostic location")
    report_path, outputs_path = work / f"process-{worker_index}.json", work / f"process-{worker_index}.pt"
    if report_path.exists() or outputs_path.exists():
        raise FileExistsError("Worker output evidence already exists; never overwrite or retry")
    with ExitStack() as stack:
        authenticated = replay.analyzer._json(
            stack.enter_context(replay.analyzer._bound_file(binding_path, binding_sha256))
        )
        if authenticated != frozen:
            raise ValueError("Worker binding differs from its acknowledged owner")
        sources = _bound_sources(
            stack, helpers, frozen, Path(ownership.require_gpu_owner.__code__.co_filename).resolve()
        )
        groups, provenance = replay.load_saved_group(stack)
        receipt = replay.analyzer._json(
            stack.enter_context(
                replay.analyzer._bound_file(root / replay.analyzer.RECEIPT_RELATIVE, replay.RECEIPT_SHA256)
            )
        )
        candidate = Path(receipt["candidate"])
        if work.is_relative_to(candidate):
            raise ValueError("Worker outputs must be outside the immutable candidate")
        config = replay.analyzer._json(
            stack.enter_context(replay.analyzer._bound_file(Path(receipt["config"]), provenance["config_sha256"]))
        )
        runtime = config["runtime"]
        if type(runtime["deterministic"]) is not bool:
            raise ValueError("Bound deterministic runtime setting is not boolean")
        groups = {
            mode: {"source": clone_source(group["source"]), "output": group["output"]} for mode, group in groups.items()
        }
        if any(tuple(group["source"].shape) != replay.SAVED_SHAPE for group in groups.values()):
            raise ValueError("CUDA replay requires the original full [16,512,512] source batches")
        muon_stream = stack.enter_context(
            replay.analyzer._bound_file(Path(provenance["pinned_muon"]), provenance["pinned_muon_sha256"])
        )
        definitions = source_definitions(
            Path(f"/proc/self/fd/{muon_stream.fileno()}"),
            ("ABC", "_bmm", "newton_schulz_batched"),
            {"torch": torch},
        )
        if ownership.require_gpu_owner(root, binding_sha256, state_path, worker_index) != owner:
            raise ValueError("Worker ownership changed before CUDA initialization")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(runtime["deterministic"])
        if torch.cuda.device_count() != 1:
            raise ValueError("Worker requires exactly one explicitly visible physical GPU")
        properties = torch.cuda.get_device_properties(0)
        actual_uuid = str(properties.uuid)
        if UUID(actual_uuid.removeprefix("GPU-")) != UUID(owner["physical_uuid"].removeprefix("GPU-")):
            raise ValueError("Actual CUDA device differs from the accepted physical UUID")
        torch.cuda.set_device(owner["device"])
        device = torch.device(owner["device"])
        resident = {
            mode: group["source"].to(device=device, copy=True, memory_format=torch.preserve_format)
            for mode, group in groups.items()
        }
        for mode, value in resident.items():
            source = groups[mode]["source"]
            if value.stride() != source.stride() or value.storage_offset() != source.storage_offset():
                raise ValueError("CUDA upload changed the saved source layout")
        layouts = {mode: tensor_layout(value) for mode, value in resident.items()}
        calls = []

        def numerical(source: torch.Tensor) -> torch.Tensor:
            mode = source_mode(groups, source)
            with torch.autocast(device_type="cuda", enabled=False):
                output = definitions["newton_schulz_batched"](resident[mode], replay.NS_STEPS, replay.NS_EPS, mode="tpu")
            torch.cuda.synchronize(device)
            if output.device != device or output.dtype != torch.float32 or tuple(output.shape) != replay.SAVED_SHAPE:
                raise ValueError("Pinned CUDA NS returned a different device, dtype or batch shape")
            calls.append({"source_mode": mode, "output_layout": tensor_layout(output)})
            result = output.detach().to(device="cpu", copy=True)
            for name, value in resident.items():
                if tensor_layout(value) != layouts[name]:
                    raise ValueError("Resident CUDA source storage or layout changed during replay")
                require_unchanged_source(value.detach().to(device="cpu", copy=True), groups[name]["source"])
            return result

        result = replay.replay_pair(groups, numerical)
        torch.cuda.synchronize(device)
        result.report.update(
            status=STATUS,
            gpu_status="owned_cuda_replay_not_acceptance",
            owner_context=owner,
            owner=owner,
            worker_index=worker_index,
            binding_sha256=binding_sha256,
            provenance=provenance,
            worker_sources_sha256=sources,
            replay_mode="tpu",
            ns_steps=replay.NS_STEPS,
            ns_eps=replay.NS_EPS,
            runtime=runtime,
            device={
                "logical": str(device),
                "physical_uuid": owner["physical_uuid"],
                "actual_uuid": actual_uuid,
                "name": properties.name,
            },
            source_layouts={
                mode: {"cpu": tensor_layout(groups[mode]["source"]), "cuda": layouts[mode]}
                for mode in replay.analyzer.MODES
            },
            calls=calls,
            environment=environment,
            numerical_settings={
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "autocast": False,
                "cuda_source_uploads": 2,
                "source_values_checked_after_each_call": True,
            },
            outputs=str(outputs_path),
            report_path=str(report_path),
        )
        if ownership.require_gpu_owner(root, binding_sha256, state_path, worker_index) != owner:
            raise ValueError("Worker ownership changed after CUDA replay")
        result.report["outputs_sha256"] = write_outputs(outputs_path, result.outputs)
    if ownership.require_gpu_owner(root, binding_sha256, state_path, worker_index) != owner:
        raise ValueError("Worker ownership changed before report publication")
    with report_path.open("x") as stream:
        stream.write(json.dumps(result.report, allow_nan=False) + "\n")
    return result.report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.binding_sha256, args.state, args.worker_index), allow_nan=False))


if __name__ == "__main__":
    main()
