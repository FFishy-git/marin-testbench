# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect source-bound retained root stages after independently reconstructed CUDA transparency."""

import argparse
import os
from contextlib import ExitStack
from pathlib import Path

import replay_saved_roots as replay
import root_stage_gpu_worker as worker
import root_stage_trace as tracing
import torch
from selected_lr_release_owner import sha256
from watch_wave5 import write

CAST_PAIRS = {
    "start": (("E", "Eb"), ("O", "Ob")),
    "iter_mid": (("P", "Pb"),),
    "iter_post": (("O", "Ob"),),
    "sqrt_start": (("Q", "Qb"),),
    "sqrt_post": (("Q", "Qb"),),
}
CPU = "run-preparation/root-stage-replay-cpu/sealed-failed-handoff-20261118T1638/receipt.json"
CPU_SHA256 = "2cebe11da69d328c4d1ddb2fa282566e0cf1938d842da008cdbff4008c0180d0"
TERMINAL_STATUS = "root_stage_replay_terminal_independently_reconstructed_exact_restoration_verified"
STATUS = "saved_root_stage_precision_localized_not_causal_or_continuation_acceptance"
LIMITATION = (
    "Consumes the supplied terminal receipt's independently reconstructed transparency and repeat predicates. "
    "Retained replay outputs and source identities are authenticated, not independently rerun here. "
    "A recorded FP16 value differing from independent rounding of its retained FP32 producer can reflect fused "
    "precision semantics; it does not prove the cause of training continuation failure. "
    "Rounding-threshold crossings can occur for FP32 sources that pass the original tolerance. "
    "Historical attribution additionally requires corresponding saved-output reproduction. "
    "No CUDA execution, numerical correction, production acceptance, optimizer quality or efficiency is measured."
)


def cast_records(snapshot: dict) -> list[dict]:
    """Compare every explicit retained FP32/FP16 pair with independent CPU FP16 rounding."""
    checked = tracing.compare_root_stages(snapshot, snapshot)
    if not checked["complete_stage_coverage"]:
        raise ValueError("Root precision inspection needs complete stage records")
    result = []
    for record in snapshot["records"]:
        for source_name, cast_name in CAST_PAIRS.get(record["stage"], ()):
            source, observed = (record["values"][name] for name in (source_name, cast_name))
            if (
                not isinstance(source, torch.Tensor)
                or not isinstance(observed, torch.Tensor)
                or source.device.type != "cpu"
                or observed.device.type != "cpu"
                or source.dtype != torch.float32
                or observed.dtype != torch.float16
                or source.shape != observed.shape
                or source.ndim != 3
                or not source.numel()
            ):
                raise ValueError("Root precision pair lacks full CPU FP32/FP16 batch coverage")
            rounded = source.to(torch.float16)
            difference = observed != rounded
            comparison = tracing.comparison(observed, rounded, f"{record['stage']}.{source_name}.{cast_name}")
            result.append(
                {
                    "stage": record["stage"],
                    "occurrence": record["occurrence"],
                    "source": source_name,
                    "cast": cast_name,
                    "shape": list(source.shape),
                    "comparison": comparison,
                    "mismatched_elements": int(torch.count_nonzero(difference)),
                    "mismatches_per_matrix": difference.reshape(source.shape[0], -1).sum(-1).tolist(),
                    "matches_independent_rounding": comparison["finite"] and comparison["exact"],
                }
            )
    if not result:
        raise ValueError("Root precision inspection found no explicit FP32/FP16 pairs")
    return result


def first_difference(comparison: dict, key: str) -> dict | None:
    for stage in comparison["stages"]:
        if not stage[key]:
            return {
                "stage": stage["stage"],
                "occurrence": stage["occurrence"],
                "comparison": stage,
            }
    return None


def paired_stages(actual: dict, expected: dict) -> dict:
    """Localize crossed root stages and distinguish source sensitivity from cast inconsistency."""
    compared = tracing.compare_root_stages(actual, expected)
    if not compared["complete_stage_coverage"]:
        raise ValueError("Root precision inspection does not prefix-pair unequal stage sequences")
    actual_casts, expected_casts = cast_records(actual), cast_records(expected)
    crossings = []
    for observed, reference in zip(actual["records"], expected["records"], strict=True):
        for source_name, cast_name in CAST_PAIRS.get(reference["stage"], ()):
            source_actual, source_expected = (record["values"][source_name] for record in (observed, reference))
            rounded_actual, rounded_expected = (source.to(torch.float16) for source in (source_actual, source_expected))
            source_comparison = tracing.comparison(source_actual, source_expected, source_name)
            observed_comparison = tracing.comparison(
                observed["values"][cast_name], reference["values"][cast_name], cast_name
            )
            independent_comparison = tracing.comparison(rounded_actual, rounded_expected, f"rounded.{source_name}")
            crossing = rounded_actual != rounded_expected
            crossings.append(
                {
                    "stage": reference["stage"],
                    "occurrence": reference["occurrence"],
                    "source": source_name,
                    "cast": cast_name,
                    "source_comparison": source_comparison,
                    "observed_cast_comparison": observed_comparison,
                    "independent_cast_comparison": independent_comparison,
                    "independent_rounding_crossings": int(torch.count_nonzero(crossing)),
                    "crossings_per_matrix": crossing.reshape(source_actual.shape[0], -1).sum(-1).tolist(),
                    "source_within_tolerance_but_rounded_values_differ": (
                        source_comparison["within_tolerance"]
                        and independent_comparison["finite"]
                        and not independent_comparison["exact"]
                    ),
                }
            )
    return {
        "stage_comparison": compared,
        "first_exact_difference": first_difference(compared, "exact"),
        "first_tolerance_failure": first_difference(compared, "within_tolerance"),
        "actual_casts": actual_casts,
        "expected_casts": expected_casts,
        "all_casts_match_independent_rounding": all(
            item["matches_independent_rounding"] for item in (*actual_casts, *expected_casts)
        ),
        "crossed_casts": crossings,
    }


def retained_precision(retained: list, index: int) -> dict:
    """Inspect a complete accepted process's crossed continuous/resumed stages."""
    if [{key: item[key] for key in ("protocol", "group", "mode", "phase")} for item in retained] != worker.coordinates(
        index
    ):
        raise ValueError("Root precision inspection lacks complete process coordinates")
    indexed = {}
    for item in retained:
        replay._require_cpu_tree(item["output"])
        if item["protocol"] == "trace":
            replay._require_cpu_tree(item["stages"])
            width = item["output"][1].shape[-1]
            if tuple(record["stage"] for record in item["stages"]["records"]) != worker.stage_sequence(width):
                raise ValueError("Root precision inspection lacks complete pinned split-root stages")
            output = tracing.comparison(item["stages"]["output"], item["output"], "retained.output")
            if not output["finite"] or not output["exact"]:
                raise ValueError("Root precision stage output differs from retained candidate")
            indexed[(item["group"], item["mode"], item["phase"])] = item["stages"]
    pairs = [
        {
            "group": group,
            "phase": phase,
            **paired_stages(indexed[(group, "resumed", phase)], indexed[(group, "continuous", phase)]),
        }
        for group in worker.GROUPS
        for phase in ("eager", "capture", "replay")
    ]
    return {
        "worker_index": index,
        "pairs": pairs,
        "all_recorded_casts_match_independent_rounding": all(
            pair["all_casts_match_independent_rounding"] for pair in pairs
        ),
    }


def require_terminal_predicates(terminal: dict) -> None:
    comparison = terminal["comparison"]
    reports = comparison["reports"]
    if (
        terminal["status"] != TERMINAL_STATUS
        or terminal["intermediates_interpretable"] is not True
        or comparison["intermediates_interpretable"] is not True
        or comparison["all_cross_process_outputs_exact"] is not True
        or comparison["all_cross_process_stages_exact"] is not True
        or len(reports) != 2
        or any(
            report["analysis"][key] is not True
            for report in reports
            for key in (
                "trace_transparency_exact",
                "capture_repeat_safety_exact",
                "capture_repeat_stages_exact",
                "trace_stages_finite",
                "intermediates_interpretable",
            )
        )
    ):
        raise RuntimeError("Root precision interpretation requires completed independent CUDA transparency/repeats")


def analyze(root: Path, terminal_path: Path, terminal_hash: str, out: Path) -> dict:
    """Authenticate retained data after separate terminal reconstruction; execute no numerical CUDA work."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Saved root precision inspection requires hidden CUDA")
    if (
        root != replay.ROOT
        or root != root.resolve(strict=True)
        or terminal_path.parent != root / "run-preparation"
        or out.exists()
        or out.parent != root / "run-preparation"
    ):
        raise ValueError("Root precision inspection requires canonical terminal and unused external output paths")
    with ExitStack() as stack:
        terminal = replay.analyzer._json(stack.enter_context(replay.analyzer._bound_file(terminal_path, terminal_hash)))
        require_terminal_predicates(terminal)
        paths = {root / name: value for name, value in terminal["evidence_sha256"].items()}
        if any(path.parent == root or not path.is_relative_to(root) for path in paths):
            raise ValueError("Root precision terminal evidence escapes the accepted output root")
        binding_path = root / "run-preparation/root-stage-replay-binding-20261118.json"
        if paths.get(binding_path) != terminal["binding_sha256"]:
            raise ValueError("Root precision terminal does not authenticate its diagnostic binding")
        binding = replay.analyzer._json(
            stack.enter_context(replay.analyzer._bound_file(binding_path, terminal["binding_sha256"]))
        )
        if (
            binding["status"] != "frozen_root_stage_replay_diagnostic_only"
            or binding["cpu_receipt"] != str(root / CPU)
            or binding["cpu_receipt_sha256"] != CPU_SHA256
            or terminal["verifier_sha256"] != binding["helpers_sha256"]["root_stage_gpu_controller.py"]
        ):
            raise ValueError("Root precision terminal controller/source acceptance differs")
        stack.enter_context(replay.analyzer._bound_file(root / CPU, CPU_SHA256))
        helpers = Path(binding["helpers"])
        for name, expected in binding["helpers_sha256"].items():
            stack.enter_context(replay.analyzer._bound_file(helpers / name, expected))
        streams = {
            path: stack.enter_context(replay.analyzer._bound_file(path, expected))
            for path, expected in paths.items()
            if path != binding_path
        }
        work = root / "diagnostics/root-stage-replay-20261118"
        processes = []
        for index, report in enumerate(terminal["comparison"]["reports"]):
            outputs, report_path = work / f"process-{index}.pt", work / f"process-{index}.json"
            if (
                report["worker_index"] != index
                or report["outputs"] != str(outputs)
                or report["report_path"] != str(report_path)
                or paths.get(outputs) != report["outputs_sha256"]
                or outputs not in streams
                or report_path not in streams
                or replay.analyzer._json(streams[report_path]) != report
            ):
                raise ValueError("Root precision retained process report/output binding differs")
            retained = torch.load(
                f"/proc/self/fd/{streams[outputs].fileno()}", map_location="cpu", weights_only=True, mmap=True
            )
            processes.append(retained_precision(retained, index))
        historical = all(
            report["analysis"]["historical_outputs_within_tolerance"] is True
            for report in terminal["comparison"]["reports"]
        )
        result = {
            "status": STATUS,
            "terminal": str(terminal_path),
            "terminal_sha256": terminal_hash,
            "binding_sha256": terminal["binding_sha256"],
            "analyzer_sha256": sha256(Path(__file__)),
            "processes": processes,
            "historical_outputs_reproduced_within_original_tolerance": historical,
            "historical_interpretation_allowed": historical,
            "tolerance": {"rtol": 1e-5, "atol": 1e-6},
            "limitation": LIMITATION,
        }
    write(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument("--terminal-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = analyze(args.root.resolve(strict=True), args.terminal.resolve(strict=True), args.terminal_sha256, args.out)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
