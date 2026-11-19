# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report fixed saved CPU start evidence without a continuation or production claim."""

from datetime import datetime
from pathlib import Path

from root_stage_precision_reporting import cpu_evidence
from selected_lr_release_owner import read, sha256

CPU = "run-preparation/root-start-rational-cpu/20261118T1750/receipt.json"
CPU_SHA256 = "fd873a15d96e9ffd2065e18d66912d96329acf37149b5c6518b6f324a0331cef"
RESULT = "run-preparation/saved-root-start-rational-20261118T1752.json"
RESULT_SHA256 = "1e994857030e22c3bd419065ff06c58e544bd434992e10e0b5dd313b0a0f53ab"
REFERENCE_CPU = "run-preparation/root-start-accuracy-cpu/20261118T1818/receipt.json"
REFERENCE_CPU_SHA256 = "034249d1eb6097c56323db3a740e2ca3ae3fcd66148c750084ae6f8c0e076444"
REFERENCE = "run-preparation/saved-root-start-log-reference-20261118T1822.json"
REFERENCE_SHA256 = "8860caf92240590ae24e615073d5bd8ddf7c8df567b4a371505f5eac7289f664"


def evidence(root: Path) -> tuple[dict, dict]:
    cpu = cpu_evidence(
        root, root / CPU, CPU_SHA256, "root_start_rational_cpu_archive_verified_saved_input_probe_pending", 84
    )
    result = read(root / RESULT)
    expected = [
        (group, mode, index)
        for group, count in (("d", 12), ("f", 4))
        for mode in ("continuous", "resumed")
        for index in range(count)
    ]
    if (
        sha256(root / RESULT) != RESULT_SHA256
        or result["status"] != "saved_root_start_rational_cpu_error_comparison_not_cuda_or_continuation"
        or result["cpu_receipt"] != str(root / CPU)
        or result["cpu_receipt_sha256"] != CPU_SHA256
        or result["probe_sha256"] != cpu["probe_sha256"]
        or cpu["probe_sha256"] != cpu["helpers_sha256"][Path(cpu["probe"]).name]
        or Path(cpu["probe"]).parent != Path(cpu["helpers"])
        or [(row["group"], row["mode"], row["matrix"]) for row in result["matrices"]] != expected
        or result["tolerance"] != {"rtol": 1e-5, "atol": 1e-6}
        or (result["rank"], result["update"]) != (0, 23)
        or any(
            row["positive_unclamped_diagonal"] is not True
            or any(value is not True for value in row["unaffected_outputs_exact"].values())
            for row in result["matrices"]
        )
        or datetime.fromisoformat(result["time"]) < datetime.fromisoformat(cpu["time"])
    ):
        raise ValueError("Saved rational-start CPU result/source or coverage differs")
    for group, count in (("d", 12), ("f", 4)):
        crossed = result["provenance"][group]["crossed_cpu_starts"]
        if [row["matrix"] for row in crossed] != list(range(count)):
            raise ValueError("Saved rational-start crossed CPU comparison coverage differs")
    return cpu, result


def independent_reference(root: Path) -> tuple[dict, dict]:
    """Authenticate the separately archived logarithmic reference and complete saved-matrix result."""
    cpu = cpu_evidence(
        root, root / REFERENCE_CPU, REFERENCE_CPU_SHA256, "root_start_accuracy_cpu_archive_verified_cuda_unissued", 97
    )
    result = read(root / REFERENCE)
    expected = [
        (group, mode, index)
        for group, count in (("d", 12), ("f", 4))
        for mode in ("continuous", "resumed")
        for index in range(count)
    ]
    if (
        sha256(root / REFERENCE) != REFERENCE_SHA256
        or result["status"] != "saved_start_independent_log_fp64_accuracy_verified_not_cuda_or_continuation"
        or result["cpu_receipt"] != str(root / REFERENCE_CPU)
        or result["cpu_receipt_sha256"] != REFERENCE_CPU_SHA256
        or result["probe_sha256"] != cpu["probe_sha256"]
        or cpu["probe_sha256"] != cpu["helpers_sha256"][Path(cpu["probe"]).name]
        or Path(cpu["probe"]).parent != Path(cpu["helpers"])
        or result["prior"] != str(root / RESULT)
        or result["prior_sha256"] != RESULT_SHA256
        or sha256(root / RESULT) != RESULT_SHA256
        or [(row["group"], row["mode"], row["matrix"]) for row in result["matrices"]] != expected
        or result["tolerance"] != {"rtol": 1e-5, "atol": 1e-6}
        or result["reference_equivalence_tolerance"] != {"rtol": 1e-12, "atol": 0}
        or (result["rank"], result["update"]) != (0, 23)
        or any(
            row["independent_log_vs_rational_fp64"]["within_tolerance"] is not True
            or any(value is not True for value in row["unaffected_outputs_exact"].values())
            for row in result["matrices"]
        )
        or datetime.fromisoformat(result["time"]) < datetime.fromisoformat(cpu["time"])
    ):
        raise ValueError("Independent saved start-reference identity or full matrix coverage differs")
    for group, count in (("d", 12), ("f", 4)):
        if [row["matrix"] for row in result["provenance"][group]["crossed_CPU_O"]] != list(range(count)):
            raise ValueError("Independent saved start-reference crossed coverage differs")
    return cpu, result


def lines(root: Path, local_time) -> list[str]:
    if not (root / CPU).is_file():
        return []
    cpu, result = evidence(root)
    lines = [
        "### Cancellation-Free Root-Start CPU Hypothesis",
        "",
        f"Saved acceptance: `PVC/{CPU}` at {local_time(cpu['time'])}; 84 passing archived CPU tests.",
        f"Saved start comparison: `PVC/{RESULT}` at {local_time(result['time'])}; rank 0/update 23, 32 matrices.",
    ]
    for group in ("d", "f"):
        rows = [row for row in result["matrices"] if row["group"] == group]
        failures = {
            variant: sum(not row[f"{variant}_O_vs_fp64"]["within_tolerance"] for row in rows)
            for variant in ("original", "rational")
        }
        maximum = {
            variant: max(row[f"{variant}_O_vs_fp64"]["max_absolute_error"] for row in rows)
            for variant in ("original", "rational")
        }
        exact = all(
            row[key]["exact"]
            for row in result["provenance"][group]["crossed_cpu_starts"]
            for key in ("d", "E", "original_O", "rational_O")
        )
        lines.append(
            f"Saved `{group}` CPU O-versus-FP64 failures: original {failures['original']}/{len(rows)}, "
            f"rational {failures['rational']}/{len(rows)}; max absolute errors "
            f"{maximum['original']:.9g}/{maximum['rational']:.9g}. "
            f"Crossed continuous/resumed CPU starts exact: `{exact}`."
        )
    lines.append(
        "Saved CPU accuracy does not explain the continuation divergence; original and rational crossed starts "
        "are exact here. No CUDA start reproduction, root iterations, NS, production, "
        "optimizer-quality or efficiency result is inferred. "
        "Numerical metrics are not independently rederived by this renderer."
    )
    if (root / REFERENCE).is_file():
        reference_cpu, reference = independent_reference(root)
        lines.extend(
            [
                "",
                f"Separate log1p/expm1 FP64 reference: `PVC/{REFERENCE}` at {local_time(reference['time'])}; "
                f"97 archived CPU tests at {local_time(reference_cpu['time'])}.",
            ]
        )
        for group in ("d", "f"):
            rows = [row for row in reference["matrices"] if row["group"] == group]
            failures = {
                variant: sum(not row[f"{variant}_O_vs_log_fp64"]["within_tolerance"] for row in rows)
                for variant in ("original", "rational")
            }
            relative = max(row["independent_log_vs_rational_fp64"]["relative_frobenius_error"] for row in rows)
            exact = all(row["exact"] for row in reference["provenance"][group]["crossed_CPU_O"])
            lines.append(
                f"Independent `{group}` CPU accuracy failures: original {failures['original']}/{len(rows)}, "
                f"rational {failures['rational']}/{len(rows)}; FP64-reference maximum relative disagreement "
                f"{relative:.9g}; crossed CPU O exact: `{exact}`."
            )
        lines.append(
            "Different FP64 coefficient algorithms confirm the CPU accuracy classification. The archived CUDA "
            "graph component was unexecuted and later corrected to replay before inspecting capture outputs; "
            "neither its historical CPU suite nor this saved probe validates CUDA graphs, continuation or runtime."
        )
    return [*lines, ""]
