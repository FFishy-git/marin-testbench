# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for post-warmup mean timings and incomplete evidence."""

import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from audit_8gpu_timings import METHODS, timing_window


def test_timing_window_retains_slow_steps_and_excludes_warmup():
    rows = [
        {"step": step, "step_seconds": seconds, "peak_allocated_bytes": 2**30}
        for step, seconds in enumerate([1000, 2, 2, 2, 10], start=1)
    ]
    result = timing_window(rows, 2, 5)
    assert result["measured_updates"] == 4
    assert result["mean_step_seconds"] == 4
    assert result["median_step_seconds"] == 2
    assert result["total_step_seconds"] == 16
    assert result["std_step_seconds"] == 4
    assert result["half_window_means"] == [2, 6]
    assert result["half_window_change_pct"] == 200


@pytest.mark.parametrize("steps", [[1, 2, 4, 5], [1, 2, 3, 3, 4, 5], [1, 2, 3, 4]])
def test_timing_window_rejects_missing_or_duplicate_updates(steps):
    rows = [{"step": step, "step_seconds": 2, "peak_allocated_bytes": 2**30} for step in steps]
    with pytest.raises(ValueError, match="Incomplete timing window"):
        timing_window(rows, 2, 5)


@pytest.mark.parametrize("mode", ["--table-only", "--results-only"])
def test_results_use_matched_mean_and_peak_for_every_quality_row(tmp_path, mode):
    paper = Path(__file__).resolve().parents[1] / "paper"
    results = tmp_path / "results"
    scripts = tmp_path / "scripts"
    results.mkdir()
    scripts.mkdir()
    script = scripts / "render_results_viz.py"
    shutil.copyfile(paper / "scripts" / script.name, script)
    for name in ("summary.json", "lr_summary.json", "mechanism_summary.json", "systems_summary.json"):
        shutil.copyfile(paper / "results" / name, results / name)
    runs = json.loads((results / "summary.json").read_text())
    for run in runs.values():
        run["step_overhead"] = 999
        run["peak_gib"] = 999
    (results / "summary.json").write_text(json.dumps(runs))
    measured = json.loads((paper / "results" / "measured_8gpu_timings.json").read_text())
    expected = {}
    for index, (arm, tax_id) in enumerate(METHODS.items()):
        costs = measured["methods"][arm]["measured_benchmark"]
        costs["mean_step_seconds"] = 2 + index / 10
        costs["median_step_seconds"] = 999
        costs["mean_overhead_vs_muont_pct"] = 999
        costs["peak_allocated_gib"] = 10 + index / 10
        percent = f"${index * 5:+.1f}$" if index else "$0.0$"
        expected[tax_id] = [percent, f"${10 + index / 10:.1f}$"]
    (results / "measured_8gpu_timings.json").write_text(json.dumps(measured))
    subprocess.run([sys.executable, "-I", str(script), mode], check=True, capture_output=True)
    table = (results / "table_main.tex").read_text()
    rows = [line for line in table.splitlines() if r"\taxid{" in line and " & " in line]
    assert len(rows) == 20
    observed = set()
    for row in rows:
        tax_id = row.split(r"\taxid{", 1)[1].split("}", 1)[0]
        columns = row.removesuffix(r" \\").split(" & ")
        assert columns[-2:] == expected[tax_id]
        observed.add(tax_id)
    assert observed == set(METHODS.values())
    assert sorted(path.name for path in results.glob("*.tex")) == ["table_main.tex"]
    if mode == "--table-only":
        assert not (tmp_path / "figs").exists()
        return

    plane = (tmp_path / "figs" / "plane_plots.tex").read_text()
    coordinates = []
    for line in plane.splitlines():
        if line.startswith(r"\addplot"):
            payload = line.split("coordinates {", 1)[1].removesuffix("};")
            coordinates.extend(
                (float(x), float(y)) for x, y in re.findall(r"\(([-\d.]+),([-\d.]+)\)", payload)
            )
    expected_coordinates = []
    for row in rows:
        arm = row.split(r"\taxid{", 1)[1].split("}", 1)[0]
        if arm == "Abl-BothDiag":
            continue
        x = float(expected[arm][0].strip("$"))
        # Quality precision in the plot is finer than the table's display.
        quality_key = {
            "MuonT": "wave0/muont",
            "Muon": "wave0/muon",
            "WT-MLP": "wave2/inner_outer_mlpdiag",
            "WT-MLP-Lag": (
                "wave_sym_fastlag/inner_outer_mlpdiag_lag" if r"\ddagger" in row
                else "wave4/inner_outer_mlpdiag_lag"
            ),
            "WT-Sym": "wave_sym_fastlag/inner_outer_sym_mlpdiag",
            "WT-Sym-Lag": "wave_sym_fastlag/inner_outer_sym_mlpdiag_lag",
            "OD-All": "wave2/inner_outer_leftdiag",
            "OD-Outer": "wave10/outer_leftdiag_upstream_fused",
            "DT-All": "wave1/inner_outer_all",
            "DT-All-Lag": "wave3/inner_outer_all_lag",
            "Null-WT": "wave3/null_mlpdiag",
            "Null-Dense": "wave1/null_all",
            "Null-OD": "wave10/null_leftdiag_upstream_fused",
            "Abl-OuterDense": "wave2/outer_left_all",
            "Abl-No2048": "wave2/inner_outer_hidden",
            "Abl-DownOnly": "wave0/outer_left_down",
            "Abl-DownSwap": "wave0/down_hybrid",
            "Abl-OutOnly": "wave1/left_all",
        }[arm]
        y = round(runs[quality_key]["d_sel_late"], 6)
        expected_coordinates.append((x, y))
    assert Counter(coordinates) == Counter(expected_coordinates)
    offscale = re.search(
        r"\\draw\[-Latex, famAbl, [^\n]+\(axis cs:([-\d.]+),0.0093\)", plane
    )
    assert offscale is not None
    assert float(offscale.group(1)) == float(expected["Abl-BothDiag"][0].strip("$"))
    for name in ("attribution.dat", "attribution_plots.tex"):
        assert (tmp_path / "figs" / name).read_bytes() == (paper / "figs" / name).read_bytes()
    assert sorted(path.name for path in (tmp_path / "figs").iterdir()) == [
        "attribution.dat", "attribution_plots.tex", "plane_plots.tex"
    ]
