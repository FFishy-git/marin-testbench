# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only public comparison and CLI regression tests with real JSONL files."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analyze_wave import analyze
from compare_runs import compare, history

HALF_MILLION = 524288
BUDGET = 2684354560
SCRIPT = Path(__file__).with_name("compare_runs.py")


def write_run(
    root: Path,
    name: str,
    batch_tokens: int = HALF_MILLION,
    *,
    seed: int = 0,
    observed_steps: int | None = None,
    status: str = "complete",
    eval_deltas: dict[int, float] | None = None,
) -> Path:
    """Write full-budget fixtures with independent, constant-window loss oracles."""
    directory = root / "runs" / name
    directory.mkdir(parents=True)
    expected_steps = BUDGET // batch_tokens
    count = expected_steps if observed_steps is None else observed_steps
    manifest = {
        "status": status,
        "completed_steps": count,
        "execution_check": False,
        "elapsed_seconds": 10 + count * 2 + count // 250,
        "data_receipt_sha256": "same-data-receipt",
        "config": {
            "training": {
                "steps": expected_steps,
                "tokens": BUDGET,
                "batch_tokens": batch_tokens,
                "seed": seed,
                "heldout_windows": 64,
            },
            "model": {"sequence_length": 4096, "embedding_init_scale": "sqrt_hidden"},
            "data": {"mixture_weights": {"dclm": 3.8, "starcoder": 0.25}, "eos_token_id": 128001},
            "runtime": {"eval_max_windows": 0},
        },
    }
    rows = []
    for step in range(1, count + 1):
        tokens = step * batch_tokens
        row = {
            "step": step,
            "tokens": tokens,
            "train_loss": 5.0 - (tokens - 1) // 536870912,
            "step_seconds": 2.0,
            "elapsed_seconds": 10.0 + step * 2,
            "end_to_end_seconds": 10.0 + step * 2 + step // 250,
            "peak_allocated_bytes": 2**30,
        }
        if step % 250 == 1 or step == count:
            row.update(c4_loss=3.0, selection_loss=2.0)
        if eval_deltas is not None:
            row.pop("c4_loss", None)
            row.pop("selection_loss", None)
            if tokens in eval_deltas:
                row.update(c4_loss=3.0 + eval_deltas[tokens], selection_loss=2.0 + eval_deltas[tokens])
        rows.append(row)
    (directory / "run.json").write_text(json.dumps(manifest))
    write_rows(directory, rows)
    return directory


def write_rows(directory: Path, rows: list[dict], filename: str = "metrics_from_000000.jsonl") -> None:
    (directory / filename).write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def read_rows(directory: Path) -> list[dict]:
    return [json.loads(line) for line in (directory / "metrics_from_000000.jsonl").read_text().splitlines()]


def cli(root: Path, *options: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "wave/arm", "--root", str(root), "--baseline", "wave/base", *options],
        text=True,
        capture_output=True,
        check=False,
    )


class CompareRunsTest(unittest.TestCase):
    def test_wave5_analysis_preserves_fused_reference_and_all_broad_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = (
                "wave5/muont",
                "wave5/inner_outer_mlpdiag",
                "wave5/inner_outer_mlpdiag_fused",
                "wave5/inner_outer_leftdiag_fused",
                "wave2/inner_outer_leftdiag",
            )
            for index, name in enumerate(names):
                path = write_run(root, name)
                rows = read_rows(path)
                for row in rows:
                    row["train_loss"] += index * 0.125
                write_rows(path, rows)
            result = analyze(root, "wave5")
            same_source = result["analyses"][1]
            comparison = same_source["comparisons"]["wave5/inner_outer_mlpdiag_fused"]
            self.assertEqual(same_source["baseline"], "wave5/inner_outer_mlpdiag")
            self.assertEqual([window["delta"] for window in comparison["training_windows"]], [0.125] * 5)
            self.assertTrue(comparison["final_comparison_eligible"])
            self.assertEqual(result["analyses"][0]["baseline"], "wave5/muont")
            self.assertEqual(result["unpaired_runs"], {})

    def test_complete_half_million_and_million_runs_match_token_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = write_run(root, "wave/base")
            arm_path = write_run(root, "wave/arm", 2 * HALF_MILLION)
            rows = read_rows(arm_path)
            for row in rows:
                row["train_loss"] += 0.25
                row["step_seconds"] = 3.0
                row["elapsed_seconds"] = 15.0 + row["step"] * 3
                row["end_to_end_seconds"] = 15.0 + row["step"] * 3 + row["step"] // 250
            write_rows(arm_path, rows)
            manifest = json.loads((arm_path / "run.json").read_text())
            manifest["elapsed_seconds"] = rows[-1]["end_to_end_seconds"] + 1
            (arm_path / "run.json").write_text(json.dumps(manifest))
            result = compare({"wave/base": history(base_path), "wave/arm": history(arm_path)}, "wave/base")
            item = result["comparisons"]["wave/arm"]
            self.assertEqual(result["window_tokens"], 536870912)
            self.assertEqual(
                [row["end_tokens"] for row in item["training_windows"]],
                [
                    536870912,
                    1073741824,
                    1610612736,
                    2147483648,
                    2684354560,
                ],
            )
            self.assertEqual([row["baseline_mean"] for row in item["training_windows"]], [5, 4, 3, 2, 1])
            self.assertEqual([row["delta"] for row in item["training_windows"]], [0.25] * 5)
            self.assertTrue(item["completion"]["complete"])
            self.assertTrue(item["final_comparison_eligible"])
            self.assertEqual(item["pairing"]["kind"], "matched_tokens")
            self.assertTrue(item["pairing"]["data_identity_match"])
            self.assertFalse(item["pairing"]["per_update_paired"])
            evaluation = item["evaluations"]["c4_loss"]
            # Update 2501 in each batch size is NOT the same token evaluation.
            self.assertEqual(evaluation["shared_tokens"], [BUDGET])
            self.assertEqual(evaluation["late_tokens"], [BUDGET])
            self.assertEqual(evaluation["final_delta"], 0.0)
            timing = item["runtime"]
            self.assertEqual(timing["delta"]["startup_seconds"], 5.0)
            self.assertEqual(timing["delta"]["shared_step_seconds"], -2560.0)
            self.assertEqual(timing["arm"]["median_step_seconds"], 3.0)
            self.assertAlmostEqual(timing["arm"]["steady_step_seconds_per_million_tokens"], 3e6 / (2 * HALF_MILLION))
            self.assertEqual(timing["delta"]["final_total_seconds"], -2564.0)

    def test_identical_seed_and_batch_are_paired_and_late_evaluations_use_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            midpoint = BUDGET // 2
            base_path = write_run(
                root, "wave/base", eval_deltas={HALF_MILLION: 0, midpoint: 0, midpoint + HALF_MILLION: 0, BUDGET: 0}
            )
            arm_path = write_run(
                root, "wave/arm", eval_deltas={HALF_MILLION: 8, midpoint: 4, midpoint + HALF_MILLION: 1, BUDGET: 3}
            )
            item = compare({"base": history(base_path), "arm": history(arm_path)}, "base")["comparisons"]["arm"]
            self.assertTrue(item["pairing"]["per_update_paired"])
            self.assertEqual(item["pairing"]["kind"], "paired_updates")
            for key in ("c4_loss", "selection_loss"):
                evaluation = item["evaluations"][key]
                self.assertEqual(evaluation["late_tokens"], [midpoint + HALF_MILLION, BUDGET])
                self.assertEqual(evaluation["mean_late_delta"], 2.0)
                self.assertEqual(evaluation["final_delta"], 3.0)

    def test_crossing_window_boundaries_are_token_weighted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = write_run(root, "wave/base")
            arm_path = write_run(root, "wave/arm", 2 * HALF_MILLION)
            for directory, losses in ((base_path, [0, 0, 6]), (arm_path, [0, 6])):
                rows = read_rows(directory)
                for row, loss in zip(rows[: len(losses)], losses, strict=True):
                    row["train_loss"] = loss
                write_rows(directory, rows)
            item = compare({"base": history(base_path), "arm": history(arm_path)}, "base", window=3 * HALF_MILLION)[
                "comparisons"
            ]["arm"]
            first = item["training_windows"][0]
            # Two zero-loss half-batches, then one six-loss half-batch: (0 + 0 + 6) / 3.
            self.assertEqual(first["baseline_mean"], 2.0)
            self.assertEqual(first["mean"], 2.0)
            self.assertEqual(first["delta"], 0.0)

    def test_seed_and_data_mismatches_require_explicit_descriptive_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run(root, "wave/base")
            arm_path = write_run(root, "wave/arm", seed=1)
            rejected = cli(root, "--json")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(rejected.stdout, "")
            allowed = cli(root, "--descriptive", "--json")
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            item = json.loads(allowed.stdout)["comparisons"]["wave/arm"]
            self.assertEqual(item["pairing"]["kind"], "descriptive_unpaired")
            self.assertFalse(item["pairing"]["per_update_paired"])
            self.assertFalse(item["pairing"]["seed_match"])
            self.assertIn("config.training.seed", item["pairing"]["mismatches"])
            self.assertEqual(item["evaluations"]["selection_loss"]["final_delta"], 0.0)
            manifest = json.loads((arm_path / "run.json").read_text())
            manifest["config"]["training"]["seed"] = 0
            for damage in ("receipt", "data_config", "missing_receipt"):
                with self.subTest(damage=damage):
                    altered = json.loads(json.dumps(manifest))
                    if damage == "receipt":
                        altered["data_receipt_sha256"] = "other-data"
                    elif damage == "data_config":
                        altered["config"]["data"]["mixture_weights"]["dclm"] = 1.0
                    else:
                        altered.pop("data_receipt_sha256")
                    (arm_path / "run.json").write_text(json.dumps(altered))
                    rejected = cli(root, "--json")
                    self.assertNotEqual(rejected.returncode, 0)
                    allowed = cli(root, "--descriptive", "--json")
                    self.assertEqual(allowed.returncode, 0, allowed.stderr)
                    identity = json.loads(allowed.stdout)["comparisons"]["wave/arm"]["pairing"]
                    self.assertFalse(identity["data_identity_match"])
                    self.assertFalse(identity["per_update_paired"])

    def test_truncated_or_unfinished_runs_have_progress_but_no_final_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = write_run(root, "wave/base")
            arm_path = write_run(root, "wave/arm", observed_steps=3072, status="running")
            for truncated in ("running", "stale_complete", "full_history_running", "execution_check"):
                with self.subTest(truncated=truncated):
                    manifest = json.loads((arm_path / "run.json").read_text())
                    if truncated == "stale_complete":
                        manifest.update(status="complete", completed_steps=5120)
                    elif truncated in ("full_history_running", "execution_check"):
                        write_rows(arm_path, read_rows(base_path))
                        manifest.update(status="running", completed_steps=5120)
                        if truncated == "execution_check":
                            manifest.update(status="complete", execution_check=True)
                    (arm_path / "run.json").write_text(json.dumps(manifest))
                    result = cli(root, "--json")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    item = json.loads(result.stdout)["comparisons"]["wave/arm"]
                    self.assertFalse(item["completion"]["complete"])
                    self.assertFalse(item["final_comparison_eligible"])
                    self.assertIsNone(item["evaluations"]["c4_loss"]["final_value"])
                    self.assertIsNone(item["evaluations"]["c4_loss"]["final_delta"])
                    self.assertIsNone(item["runtime"]["delta"]["final_total_seconds"])
                    self.assertFalse(item["runtime"]["arm"]["total_is_final"])
                    self.assertEqual(item["runtime"]["total_scope"], "observed_progress")
                    if truncated in ("running", "stale_complete"):
                        self.assertEqual(len(item["training_windows"]), 3)
                        self.assertEqual(item["common_tokens"], 1610612736)
                        self.assertEqual(item["evaluations"]["c4_loss"]["late_start_tokens_exclusive"], 805306368)

    def test_duplicate_steps_tokens_and_internal_gaps_fail_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run(root, "wave/base")
            arm_path = write_run(root, "wave/arm")
            original = read_rows(arm_path)
            for damage in (
                "duplicate_step",
                "duplicate_tokens",
                "gap",
                "missing_tokens",
                "missing_step",
                "segment_overlap",
                "out_of_order",
                "token_gap",
            ):
                with self.subTest(damage=damage):
                    rows = [dict(row) for row in original]
                    if damage == "duplicate_step":
                        rows[1]["step"] = 1
                    elif damage == "duplicate_tokens":
                        rows[1]["tokens"] = rows[0]["tokens"]
                    elif damage == "gap":
                        rows.pop(250)
                    elif damage == "missing_tokens":
                        rows[250].pop("tokens")
                    elif damage == "missing_step":
                        rows[250].pop("step")
                    elif damage == "segment_overlap":
                        write_rows(arm_path, [rows[-1]], "metrics_from_005119.jsonl")
                    elif damage == "out_of_order":
                        rows[250], rows[251] = rows[251], rows[250]
                    else:
                        rows[250]["tokens"] += HALF_MILLION // 2
                    if damage != "segment_overlap":
                        (arm_path / "metrics_from_005119.jsonl").unlink(missing_ok=True)
                    write_rows(arm_path, rows)
                    result = cli(root, "--json")
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, "")
            (arm_path / "metrics_from_005119.jsonl").unlink(missing_ok=True)
            (arm_path / "metrics_from_000000.jsonl").unlink()
            missing = cli(root, "--json")
            self.assertNotEqual(missing.returncode, 0)

    def test_evaluations_use_exact_shared_tokens_without_nearest_or_final_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = write_run(
                root,
                "wave/base",
                eval_deltas={HALF_MILLION: 0, 2000 * HALF_MILLION: 0, 3000 * HALF_MILLION: 0, BUDGET: 0},
            )
            arm_path = write_run(
                root,
                "wave/arm",
                2 * HALF_MILLION,
                eval_deltas={
                    2 * HALF_MILLION: 9,
                    2000 * HALF_MILLION: 2,
                    3000 * HALF_MILLION: 4,
                    BUDGET - 2 * HALF_MILLION: 8,
                },
            )
            item = compare({"base": history(base_path), "arm": history(arm_path)}, "base")["comparisons"]["arm"]
            evaluation = item["evaluations"]["c4_loss"]
            self.assertEqual(evaluation["shared_tokens"], [2000 * HALF_MILLION, 3000 * HALF_MILLION])
            self.assertEqual(evaluation["mean_late_delta"], 4.0)
            self.assertEqual(evaluation["last_shared_delta"], 4.0)
            self.assertIsNone(evaluation["final_tokens"])
            self.assertIsNone(evaluation["final_delta"])
            rows = read_rows(arm_path)
            for row in rows:
                row.pop("c4_loss", None)
            write_rows(arm_path, rows)
            empty = compare({"base": history(base_path), "arm": history(arm_path)}, "base")["comparisons"]["arm"]
            self.assertEqual(empty["evaluations"]["c4_loss"]["shared_tokens"], [])
            self.assertIsNone(empty["evaluations"]["c4_loss"]["mean_late_delta"])
            self.assertIsNone(empty["evaluations"]["c4_loss"]["last_shared_delta"])
            self.assertEqual(empty["evaluations"]["selection_loss"]["mean_late_delta"], 4.0)

    def test_incomplete_baseline_and_unequal_budgets_never_produce_final_deltas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = write_run(root, "wave/base", observed_steps=750, status="running")
            write_run(root, "wave/arm", 2 * HALF_MILLION)
            for scenario in ("incomplete_baseline", "complete_unequal_budgets"):
                with self.subTest(scenario=scenario):
                    if scenario == "complete_unequal_budgets":
                        manifest = json.loads((base_path / "run.json").read_text())
                        manifest.update(status="complete", completed_steps=750)
                        manifest["config"]["training"].update(steps=750, tokens=750 * HALF_MILLION)
                        (base_path / "run.json").write_text(json.dumps(manifest))
                    result = cli(root, "--json")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    item = json.loads(result.stdout)["comparisons"]["wave/arm"]
                    self.assertTrue(item["completion"]["complete"])
                    self.assertFalse(item["final_comparison_eligible"])
                    self.assertEqual(item["common_tokens"], 750 * HALF_MILLION)
                    self.assertEqual(item["training_windows"], [])
                    self.assertIsNone(item["evaluations"]["selection_loss"]["final_delta"])
                    self.assertIsNone(item["runtime"]["delta"]["final_total_seconds"])

    def test_missing_runtime_fields_are_null_and_nonfinite_loss_fails_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run(root, "wave/base")
            arm_path = write_run(root, "wave/arm")
            rows = read_rows(arm_path)
            for row in rows:
                for key in ("step_seconds", "elapsed_seconds", "end_to_end_seconds", "peak_allocated_bytes"):
                    row.pop(key)
            write_rows(arm_path, rows)
            result = cli(root, "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            timing = json.loads(result.stdout)["comparisons"]["wave/arm"]["runtime"]["arm"]
            self.assertIsNone(timing["median_step_seconds"])
            self.assertIsNone(timing["startup_seconds"])
            self.assertIsNone(timing["shared_step_seconds"])
            self.assertIsNone(timing["peak_allocated_gib"])
            self.assertIsNone(timing["eval_seconds"])
            self.assertEqual(timing["total_seconds"], 10270)
            manifest = json.loads((arm_path / "run.json").read_text())
            manifest.pop("elapsed_seconds")
            (arm_path / "run.json").write_text(json.dumps(manifest))
            unavailable = cli(root, "--json")
            self.assertEqual(unavailable.returncode, 0, unavailable.stderr)
            timing = json.loads(unavailable.stdout)["comparisons"]["wave/arm"]["runtime"]
            self.assertIsNone(timing["arm"]["total_seconds"])
            self.assertFalse(timing["arm"]["total_is_final"])
            self.assertIsNone(timing["delta"]["final_total_seconds"])
            rows[0]["train_loss"] = float("nan")
            write_rows(arm_path, rows)
            rejected = cli(root, "--json")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertEqual(rejected.stdout, "")

    def test_json_path_and_stdout_roundtrip_and_segment_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_run(root, "wave/base")
            arm_path = write_run(root, "wave/arm")
            rows = read_rows(arm_path)
            prefix, resumed = rows[:2560], rows[2560:]
            for index, row in enumerate(resumed, start=1):
                row["elapsed_seconds"] = 7 + index * 2
                row["end_to_end_seconds"] = 7 + index * 2
            write_rows(arm_path, prefix)
            write_rows(arm_path, resumed, "metrics_from_002560.jsonl")
            manifest = json.loads((arm_path / "run.json").read_text())
            manifest["elapsed_seconds"] = 5128
            (arm_path / "run.json").write_text(json.dumps(manifest))
            stdout = cli(root, "--json")
            self.assertEqual(stdout.returncode, 0, stdout.stderr)
            expected = json.loads(stdout.stdout)
            output_path = root / "comparison.json"
            persisted = cli(root, "--json", str(output_path))
            self.assertEqual(persisted.returncode, 0, persisted.stderr)
            self.assertEqual(persisted.stdout, "")
            self.assertEqual(json.loads(output_path.read_text()), expected)
            self.assertFalse(output_path.with_suffix(".json.tmp").exists())
            timing = expected["comparisons"]["wave/arm"]["runtime"]
            self.assertEqual(timing["arm"]["segments"], 2)
            self.assertEqual(timing["arm"]["startup_seconds"], 17)
            self.assertEqual(timing["arm"]["shared_step_seconds"], 10240)
            self.assertEqual(timing["arm"]["total_seconds"], 10268)
            self.assertEqual(timing["delta"]["final_total_seconds"], -2)
            self.assertEqual(cli(root).returncode, 0)
            manifest.pop("elapsed_seconds")
            (arm_path / "run.json").write_text(json.dumps(manifest))
            metric_only = cli(root, "--json")
            self.assertEqual(metric_only.returncode, 0, metric_only.stderr)
            self.assertEqual(
                json.loads(metric_only.stdout)["comparisons"]["wave/arm"]["runtime"]["arm"]["total_seconds"], 10267
            )


if __name__ == "__main__":
    unittest.main()
