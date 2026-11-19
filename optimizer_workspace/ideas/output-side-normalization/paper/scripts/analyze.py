"""Build the paper's numbers and figure data from the executed run histories.

Reads the per-update JSONL that each run wrote on the PVC share and emits

  results/*_numbers.tex   LaTeX macros for numbers quoted in the prose
  figs/*.dat              pgfplots tables for the figures

Every number in the manuscript that comes from a run comes from here, so the
prose cannot drift from the histories. Run from the paper directory:

    python -I scripts/analyze.py
"""

import json
import os
import statistics
from collections import OrderedDict

PVC = "/apdcephfs_tj32/share_303214315/hunyuan/kenanchen/toner/output-side-normalization"
RUNS = os.path.join(PVC, "runs")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The 11 evaluations in the second half of the 0.5M-token budget, and the 4
# late ones where the learning rate is below 15% of peak.
EVAL_STEPS_LATE11 = [2751, 3001, 3251, 3501, 3751, 4001, 4251, 4501, 4751, 5001, 5120]
EVAL_STEPS_LATE4 = [4501, 4751, 5001, 5120]
EVAL_STEPS_LATE11_1M = [1376, 1501, 1626, 1751, 1876, 2001, 2126, 2251, 2376, 2501, 2560]

# Step-time medians use this window so that they are comparable across waves
# despite root fallbacks accumulating over training.
STEP_WINDOW = (2049, 4096)
# Training-loss windows of 536,870,912 tokens each.
WINDOW_UPDATES = 1024


def load(run):
    path = os.path.join(RUNS, run, "metrics_from_000000.jsonl")
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["step"])
    return rows


def evals(rows):
    return {r["step"]: r for r in rows if r.get("c4_loss") is not None}


def mean_at(rows, steps, key):
    table = evals(rows)
    values = [table[s][key] for s in steps if s in table]
    if not values:
        return None
    return statistics.fmean(values)


def final(rows, key):
    table = evals(rows)
    return table[max(table)][key]


def median_step_seconds(rows, lo=STEP_WINDOW[0], hi=STEP_WINDOW[1]):
    values = [r["step_seconds"] for r in rows if lo <= r["step"] <= hi]
    return statistics.median(values)


def paired_train_delta(rows, base_rows, lo, hi):
    """Mean per-update paired training-loss difference over an update window.

    Pairing is exact: within one seed and batch every arm consumes the same
    tokens in the same order, so the runs are differenced update by update
    before averaging. Also returns the effective standard error, inflated by
    the lag-1 autocorrelation of the difference series when that is positive.
    """
    base = {r["step"]: r["train_loss"] for r in base_rows}
    diffs = [
        r["train_loss"] - base[r["step"]]
        for r in rows
        if lo <= r["step"] <= hi and r["step"] in base
    ]
    if len(diffs) < 8:
        return None, None, None
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    centered = [d - mean for d in diffs]
    denom = sum(c * c for c in centered)
    rho = sum(a * b for a, b in zip(centered, centered[1:])) / denom if denom else 0.0
    inflation = ((1 + rho) / (1 - rho)) ** 0.5 if 0 < rho < 0.98 else 1.0
    stderr = sd / (len(diffs) ** 0.5) * inflation
    return mean, stderr, rho


# ---------------------------------------------------------------------------
# Arm inventory. Each entry names its seed-matched, same-batch baseline.

SEED0 = "wave0/muont"
ARMS = OrderedDict([
    # Wave 0: down-projection-only arms, on the upstream batched update path.
    ("wave0/muon", dict(label="Muon", base=SEED0, seed=0, group="baseline")),
    ("wave0/muont", dict(label="MuonT (input only)", base=SEED0, seed=0, group="baseline")),
    ("wave0/outer_left_down", dict(label="outer-left, down only", base=SEED0, seed=0, group="wave0")),
    ("wave0/down_hybrid", dict(label="down hybrid (output replaces input)", base=SEED0, seed=0, group="wave0")),
    # Wave 1: all 224 block matrices, per-matrix update path.
    ("wave1/inner_outer_all", dict(label="two-sided, dense", base=SEED0, seed=0, group="wave1")),
    ("wave1/null_all", dict(label="matched null (identity left)", base=SEED0, seed=0, group="control")),
    ("wave1/left_all", dict(label="output-only (input removed)", base=SEED0, seed=0, group="wave1")),
    ("wave1/inner_outer_all_diag", dict(label="both sides diagonal", base=SEED0, seed=0, group="wave1")),
    # Wave 2: input side held at full MuonT geometry, output side cheapened.
    ("wave2/inner_outer_hidden", dict(label="no width-2048 output", base=SEED0, seed=0, group="wave2")),
    ("wave2/inner_outer_mlpdiag", dict(label="width-2048 output diagonal", base=SEED0, seed=0, group="wave2")),
    ("wave2/inner_outer_leftdiag", dict(label="all output diagonal", base=SEED0, seed=0, group="wave2")),
    ("wave2/outer_left_all", dict(label="outer output factor only", base=SEED0, seed=0, group="wave2")),
    # Wave 3: matched control for the leading arm, second seed, lagged factors.
    ("wave3/null_mlpdiag", dict(label="matched null for mlpdiag", base=SEED0, seed=0, group="control")),
    ("wave3/inner_outer_all_lag", dict(label="two-sided dense, lagged", base=SEED0, seed=0, group="wave3")),
    ("wave3/inner_outer_mlpdiag_seed1", dict(label="mlpdiag, seed 1", base="wave3/muont_seed1", seed=1, group="seed")),
    ("wave3/muont_seed1", dict(label="MuonT, seed 1", base="wave3/muont_seed1", seed=1, group="baseline")),
    # Wave 4: lagged leading arm, 1M batch pair, third seed.
    ("wave4/inner_outer_mlpdiag_lag", dict(label="mlpdiag, lagged", base=SEED0, seed=0, group="wave4")),
    ("wave4/inner_outer_mlpdiag_seed2", dict(label="mlpdiag, seed 2", base="wave4-baseline/muont_seed2", seed=2, group="seed")),
    ("wave4-baseline/muont_seed2", dict(label="MuonT, seed 2", base="wave4-baseline/muont_seed2", seed=2, group="baseline")),
    ("wave4/inner_outer_mlpdiag_1m", dict(label="mlpdiag, 1M batch", base="wave4/muont_1m", seed=0, group="batch", late=EVAL_STEPS_LATE11_1M)),
    ("wave4/muont_1m", dict(label="MuonT, 1M batch", base="wave4/muont_1m", seed=0, group="baseline", late=EVAL_STEPS_LATE11_1M)),
    # Wave 7: fixed-age input basis installation.
    ("wave7/muont", dict(label="MuonT, fixed age", base="wave7/muont", seed=0, group="baseline")),
    ("wave7/inner_outer_leftdiag", dict(label="all output diagonal, fixed age", base="wave7/muont", seed=0, group="wave7")),
    ("wave7/outer_leftdiag", dict(label="outer diagonal, fixed age", base="wave7/muont", seed=0, group="wave7")),
    ("wave7/outer_leftdiag_fused", dict(label="outer diagonal fused, fixed age", base="wave7/muont", seed=0, group="wave7")),
    # Wave 10 / 12: systems variants and the selected-LR repeat.
    ("wave10/muont", dict(label="MuonT, wave 10", base="wave10/muont", seed=0, group="baseline")),
    ("wave10/outer_leftdiag_fused", dict(label="outer diagonal fused", base="wave10/muont", seed=0, group="wave10")),
    ("wave10/outer_leftdiag_upstream_fused", dict(label="outer diagonal, upstream product", base="wave10/muont", seed=0, group="wave10")),
    ("wave10/null_leftdiag_upstream_fused", dict(label="matched null, upstream product", base="wave10/muont", seed=0, group="control")),
    ("wave12/muont", dict(label="MuonT, wave 12", base="wave12/muont", seed=0, group="baseline")),
    ("wave12/inner_outer_leftdiag", dict(label="all output diagonal, selected LR", base="wave12/muont", seed=0, group="wave12")),
    ("wave12/outer_leftdiag", dict(label="outer diagonal, selected LR", base="wave12/muont", seed=0, group="wave12")),
    ("wave12/outer_leftdiag_fused", dict(label="outer diagonal fused, selected LR", base="wave12/muont", seed=0, group="wave12")),
])

cache = {}
for run in ARMS:
    cache[run] = load(run)

summary = OrderedDict()
for run, meta in ARMS.items():
    rows = cache[run]
    base_rows = cache[meta["base"]]
    late = meta.get("late", EVAL_STEPS_LATE11)
    late4 = EVAL_STEPS_LATE4 if late is EVAL_STEPS_LATE11 else late[-4:]
    last = rows[-1]["step"]

    rec = dict(meta)
    rec["run"] = run
    rec["updates"] = last
    rec["c4_final"] = final(rows, "c4_loss")
    rec["sel_final"] = final(rows, "selection_loss")
    rec["c4_late"] = mean_at(rows, late, "c4_loss")
    rec["sel_late"] = mean_at(rows, late, "selection_loss")
    rec["sel_late4"] = mean_at(rows, late4, "selection_loss")
    rec["step_median"] = median_step_seconds(rows)
    rec["peak_gib"] = max(r["peak_allocated_bytes"] for r in rows) / 2**30
    rec["end_to_end"] = last_e2e = rows[-1].get("end_to_end_seconds")
    rec["output_fallbacks"] = max(r.get("output_fallbacks") or 0 for r in rows)
    rec["output_age_max"] = max(r.get("output_max_basis_age") or 0 for r in rows)
    rec["input_age_max"] = max(r.get("input_max_basis_age") or 0 for r in rows)
    rec["output_host_total"] = sum(r.get("output_host_seconds") or 0.0 for r in rows)

    if run != meta["base"]:
        rec["d_c4_final"] = rec["c4_final"] - final(base_rows, "c4_loss")
        rec["d_sel_final"] = rec["sel_final"] - final(base_rows, "selection_loss")
        rec["d_c4_late"] = rec["c4_late"] - mean_at(base_rows, late, "c4_loss")
        rec["d_sel_late"] = rec["sel_late"] - mean_at(base_rows, late, "selection_loss")
        rec["d_sel_late4"] = rec["sel_late4"] - mean_at(base_rows, late4, "selection_loss")
        mean, stderr, rho = paired_train_delta(rows, base_rows, last - WINDOW_UPDATES + 1, last)
        rec["d_train"], rec["d_train_se"], rec["d_train_rho"] = mean, stderr, rho
        rec["step_overhead"] = 100.0 * (rec["step_median"] / median_step_seconds(base_rows) - 1.0)
        base_e2e = base_rows[-1].get("end_to_end_seconds")
        rec["e2e_overhead"] = 100.0 * (last_e2e / base_e2e - 1.0) if base_e2e else None
        windows = []
        for w in range(last // WINDOW_UPDATES):
            lo, hi = w * WINDOW_UPDATES + 1, (w + 1) * WINDOW_UPDATES
            windows.append(paired_train_delta(rows, base_rows, lo, hi)[0])
        rec["windows"] = windows
    else:
        for key in ("d_c4_final", "d_sel_final", "d_c4_late", "d_sel_late",
                    "d_sel_late4", "d_train", "d_train_se", "d_train_rho",
                    "step_overhead", "e2e_overhead"):
            rec[key] = 0.0 if key.startswith(("d_", "step", "e2e")) else None
        rec["windows"] = [0.0] * (last // WINDOW_UPDATES)
    summary[run] = rec


# ---------------------------------------------------------------------------
# Figure data.

def write_dat(name, header, rows):
    path = os.path.join(HERE, "figs", name)
    with open(path, "w") as handle:
        handle.write(" ".join(header) + "\n")
        for row in rows:
            handle.write(" ".join(
                ("nan" if v is None else (v if isinstance(v, str) else f"{v:.8g}"))
                for v in row) + "\n")
    return path


# Figure 1: cost-quality plane for the seed-0 0.5M arms that share the wave-0
# reference frame. Emitted as ready-to-include pgfplots blocks, one per
# category, so the figure cannot drift from the histories.
PLANE_LABELS = {
    "inner_outer_mlpdiag": "mlpdiag",
    "inner_outer_all": "dense",
    "inner_outer_all_lag": "dense lag",
    "inner_outer_mlpdiag_lag": "mlpdiag lag",
    "inner_outer_leftdiag": "alldiag",
    "outer_left_all": "outer only",
    "inner_outer_hidden": "no 2048",
    "inner_outer_all_diag": "diag both sides",
    "left_all": "output only",
    "outer_left_down": "down only",
    "down_hybrid": "down replace",
    "null_all": "null (dense)",
    "null_mlpdiag": "null (mlpdiag)",
    "muon": "Muon",
    "muont": "MuonT",
}
CATEGORY = {
    "wave1": "candidate", "wave2": "candidate", "wave3": "candidate",
    "wave4": "candidate", "wave0": "candidate",
    "control": "null", "baseline": "baseline",
}
plane_groups = {"candidate": [], "null": [], "baseline": []}
for run, rec in summary.items():
    if rec["seed"] != 0 or rec["updates"] != 5120 or rec["base"] != SEED0:
        continue
    arm = run.split("/")[-1]
    cat = CATEGORY.get(rec["group"])
    if cat is None:
        continue
    plane_groups[cat].append(
        (rec["step_overhead"], rec["d_sel_late"], PLANE_LABELS.get(arm, arm)))
PLANE_STYLE = {
    "candidate": ("mark=*", "laBlueLine", "anchor=west, xshift=3pt",
                  "output-side candidates"),
    "null": ("mark=square*", "laOrangeLine", "anchor=west, xshift=3pt",
             "matched nulls (identity factors)"),
    "baseline": ("mark=triangle*, mark size=2.6pt", "black",
                 "anchor=east, xshift=-4pt", "references"),
}
with open(os.path.join(HERE, "figs", "plane_plots.tex"), "w") as handle:
    handle.write("% Generated by scripts/analyze.py. Do not edit.\n")
    for cat in ("candidate", "null", "baseline"):
        mark, colour, place, legend = PLANE_STYLE[cat]
        coords = " ".join(
            f"({x:.3f},{y:.6f}) [{label}]" for x, y, label in sorted(plane_groups[cat]))
        handle.write(
            f"\\addplot[only marks, {mark}, color={colour},\n"
            f"  point meta=explicit symbolic, nodes near coords={{\\pgfplotspointmeta}},\n"
            f"  every node near coord/.append style={{font=\\tiny, {place},\n"
            f"    text={colour}!80!black}}]\n"
            f"  coordinates {{{coords}}};\n"
            f"\\addlegendentry{{{legend}}}\n")

# Wave 12 transferred-learning-rate outcome, for the tuning figure's second
# panel: the same geometries at their screening-selected rates.
transfer = []
for run in ["wave12/inner_outer_leftdiag", "wave12/outer_leftdiag", "wave12/outer_leftdiag_fused"]:
    rec = summary[run]
    transfer.append((run.split("/")[-1], rec["d_sel_late"], rec["d_sel_final"], rec["d_c4_late"]))
write_dat("transfer.dat", ["arm", "dsel_late", "dsel_final", "dc4_late"], transfer)

# The same three, as pgfplots bar coordinates against symbolic arm labels.
TRANSFER_LABELS = {
    "wave12/inner_outer_leftdiag": "all diagonal",
    "wave12/outer_leftdiag": "outer diagonal",
    "wave12/outer_leftdiag_fused": "fused outer",
}
with open(os.path.join(HERE, "figs", "transfer_bars.tex"), "w") as handle:
    handle.write("% Generated by scripts/analyze.py. Do not edit.\n")
    coords = " ".join(
        f"({TRANSFER_LABELS[r]},{summary[r]['d_sel_late']:.6f})" for r in TRANSFER_LABELS)
    handle.write("\\addplot[fill=crossred!65, draw=crossred!85] coordinates {"
                 + coords + "};\n")

# Five-window paired training deltas, as an appendix table. Shows whether an
# arm's advantage is a stable offset or a decaying transient.
WINDOW_ROWS = [
    ("wave2/inner_outer_mlpdiag", r"mlpdiag"),
    ("wave1/inner_outer_all", r"dense, both widths"),
    ("wave3/inner_outer_all_lag", r"dense, lagged"),
    ("wave2/inner_outer_leftdiag", r"all diagonal"),
    ("wave2/outer_left_all", r"outer factor only"),
    ("wave2/inner_outer_hidden", r"no 2048 output"),
    ("wave0/outer_left_down", r"down only"),
    (None, None),
    ("wave1/null_all", r"\textbf{null (dense)}"),
    ("wave3/null_mlpdiag", r"\textbf{null (mlpdiag)}"),
    (None, None),
    ("wave3/inner_outer_mlpdiag_seed1", r"mlpdiag, seed 1"),
    ("wave4/inner_outer_mlpdiag_seed2", r"mlpdiag, seed 2"),
    (None, None),
    ("wave0/muon", r"Muon"),
    ("wave1/left_all", r"output only"),
    ("wave1/inner_outer_all_diag", r"both sides diagonal"),
]
wlines = ["% Generated by scripts/analyze.py. Do not edit.",
          r"\begin{tabular}{lrrrrr}",
          r"\toprule",
          r"Arm & Window 1 & Window 2 & Window 3 & Window 4 & Window 5 \\",
          r"\midrule"]
for wrun, wlabel in WINDOW_ROWS:
    if wrun is None:
        wlines.append(r"\addlinespace")
        continue
    wcells = " & ".join(f"${w:+.4f}$" for w in summary[wrun]["windows"])
    wlines.append(f"{wlabel} & {wcells} \\\\")
wlines += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(HERE, "results", "table_windows.tex"), "w") as handle:
    handle.write("\n".join(wlines) + "\n")

# Figure 2: paired training-loss delta against the seed-matched baseline,
# block-averaged over 64 updates so the curve is readable.
BLOCK = 64
series_runs = [
    "wave2/inner_outer_mlpdiag", "wave1/inner_outer_all", "wave2/inner_outer_leftdiag",
    "wave1/null_all", "wave3/null_mlpdiag", "wave1/left_all", "wave1/inner_outer_all_diag",
    "wave0/outer_left_down", "wave0/muon",
]
cols, table = ["tokens"], {}
for run in series_runs:
    rows, base_rows = cache[run], cache[ARMS[run]["base"]]
    base = {r["step"]: r["train_loss"] for r in base_rows}
    name = run.split("/")[-1]
    cols.append(name)
    for b in range(5120 // BLOCK):
        lo, hi = b * BLOCK + 1, (b + 1) * BLOCK
        diffs = [r["train_loss"] - base[r["step"]] for r in rows
                 if lo <= r["step"] <= hi and r["step"] in base]
        table.setdefault(b, {})["tokens"] = hi * 524288 / 1e9
        table[b][name] = statistics.fmean(diffs) if diffs else None
write_dat("train_delta.dat", cols,
          [[table[b].get(c) for c in cols] for b in sorted(table)])

# Figure 3: the freshness evidence. Output basis age and cumulative exact
# fallbacks for a dense-output arm against the diagonal-output arms.
fresh_runs = ["wave1/inner_outer_all", "wave2/inner_outer_mlpdiag", "wave2/inner_outer_leftdiag"]
cols = ["tokens"] + [f"{r.split('/')[-1]}_age" for r in fresh_runs] + \
       [f"{r.split('/')[-1]}_fb" for r in fresh_runs]
rows_out = []
for b in range(5120 // BLOCK):
    lo, hi = b * BLOCK + 1, (b + 1) * BLOCK
    row = [hi * 524288 / 1e9]
    for run in fresh_runs:
        seg = [r.get("output_max_basis_age") or 0 for r in cache[run] if lo <= r["step"] <= hi]
        row.append(statistics.fmean(seg) if seg else None)
    for run in fresh_runs:
        seg = [r.get("output_fallbacks") or 0 for r in cache[run] if lo <= r["step"] <= hi]
        row.append(max(seg) if seg else None)
    rows_out.append(row)
write_dat("freshness.dat", cols, rows_out)

# Figure 4: seed replication of the leading geometry on the fixed C4 set,
# with the matched null for reference.
seed_rows = []
for run, tag in [("wave2/inner_outer_mlpdiag", "0"),
                 ("wave3/inner_outer_mlpdiag_seed1", "1"),
                 ("wave4/inner_outer_mlpdiag_seed2", "2")]:
    rec = summary[run]
    seed_rows.append((tag, rec["d_c4_final"], rec["d_c4_late"], rec["d_sel_late"]))
write_dat("seeds.dat", ["seed", "dc4_final", "dc4_late", "dsel_late"], seed_rows)

# Five-window training deltas for the leading arms and controls.
win_runs = ["wave2/inner_outer_mlpdiag", "wave1/inner_outer_all",
            "wave2/inner_outer_leftdiag", "wave3/null_mlpdiag", "wave1/null_all"]
cols = ["window"] + [r.split("/")[-1] for r in win_runs]
write_dat("windows.dat", cols,
          [[w + 1] + [summary[r]["windows"][w] for r in win_runs] for w in range(5)])


# The main results table, generated so the manuscript cannot drift from the
# histories. Row order is by held-out selection delta; the null rows and the
# baseline are separated by rules because they are read differently.
TABLE_ROWS = [
    ("wave2/inner_outer_mlpdiag", r"mlpdiag (leading)", True),
    ("wave3/inner_outer_all_lag", r"dense, lagged", False),
    ("wave1/inner_outer_all", r"dense, both widths", False),
    ("wave4/inner_outer_mlpdiag_lag", r"mlpdiag, lagged", False),
    ("wave2/inner_outer_leftdiag", r"all diagonal", False),
    ("wave2/outer_left_all", r"outer factor only", False),
    ("wave2/inner_outer_hidden", r"no 2048 output", False),
    ("wave0/outer_left_down", r"down only", False),
    (None, None, None),
    ("wave1/null_all", r"\textbf{null (dense)}", False),
    ("wave3/null_mlpdiag", r"\textbf{null (mlpdiag)}", False),
    (None, None, None),
    ("wave0/muont", r"\textbf{MuonT} (base)", False),
    ("wave0/down_hybrid", r"down, output replaces", False),
    ("wave0/muon", r"Muon", False),
    ("wave1/left_all", r"output only", False),
    ("wave1/inner_outer_all_diag", r"both sides diagonal", False),
]
lines = ["% Generated by scripts/analyze.py. Do not edit.",
         r"\begin{tabular}{lrrrrrrr}",
         r"\toprule",
         r"Arm & $\Delta$ sel. & $\Delta$ C4 & $\Delta$ train & s.e. & "
         r"Step \% & E2E \% & Peak GiB \\",
         r"\midrule"]
for run, label, bold in TABLE_ROWS:
    if run is None:
        lines.append(r"\addlinespace")
        continue
    rec = summary[run]
    if run == SEED0:
        lines.append(f"{label} & $0$ & $0$ & $0$ & --- & $0$ & $0$ & "
                     f"{rec['peak_gib']:.1f} \\\\")
        continue
    sel = f"{rec['d_sel_late']:+.4f}"
    sel = rf"\textbf{{${sel}$}}" if bold else f"${sel}$"
    step = f"{rec['step_overhead']:+.1f}"
    step = rf"$\mathbf{{{step}}}$" if run.endswith("leftdiag") else f"${step}$"
    lines.append(
        f"{label} & {sel} & ${rec['d_c4_late']:+.4f}$ & ${rec['d_train']:+.4f}$ & "
        f"${rec['d_train_se']:.5f}$ & {step} & ${rec['e2e_overhead']:+.1f}$ & "
        f"{rec['peak_gib']:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
# The whole tabular is emitted, not just its rows: LaTeX's \input appends a
# \relax, which would start a cell and make a following \bottomrule a
# misplaced \noalign if the rules lived in the section source.
with open(os.path.join(HERE, "results", "table_main.tex"), "w") as handle:
    handle.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# LaTeX macros.

# TeX control-sequence names may not contain digits, so wave and seed numbers
# are spelled out: wave10/muont becomes \cselWaveOneZeroMuont.
DIGIT_WORDS = str.maketrans({d: w for d, w in zip(
    "0123456789",
    ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"])})


def tex_name(run, field):
    stem = "".join(part.capitalize() for part in
                   run.replace("/", "-").replace("_", "-").split("-"))
    return "\\" + field + stem.translate(DIGIT_WORDS)


def fmt(value, places=6, percent=False, signed=False):
    if value is None:
        return "n/a"
    if percent:
        return f"{value:+.2f}" if signed else f"{value:.2f}"
    return f"{value:+.{places}f}" if signed else f"{value:.{places}f}"


lines = ["% Generated by scripts/analyze.py from the executed run histories. Do not edit.\n"]
for run, rec in summary.items():
    lines.append(f"% {run}")
    lines.append(f"\\newcommand{{{tex_name(run, 'csel')}}}{{{fmt(rec['sel_late'])}}}")
    lines.append(f"\\newcommand{{{tex_name(run, 'cdsel')}}}{{{fmt(rec['d_sel_late'], signed=True)}}}")
    lines.append(f"\\newcommand{{{tex_name(run, 'cdselfinal')}}}{{{fmt(rec['d_sel_final'], signed=True)}}}")
    lines.append(f"\\newcommand{{{tex_name(run, 'cdcfour')}}}{{{fmt(rec['d_c4_late'], signed=True)}}}")
    lines.append(f"\\newcommand{{{tex_name(run, 'cdcfourfinal')}}}{{{fmt(rec['d_c4_final'], signed=True)}}}")
    lines.append(f"\\newcommand{{{tex_name(run, 'cstep')}}}{{{fmt(rec['step_median'], places=3)}}}")
    lines.append(f"\\newcommand{{{tex_name(run, 'coverhead')}}}{{{fmt(rec['step_overhead'], percent=True)}}}")
    lines.append(f"\\newcommand{{{tex_name(run, 'cetoe')}}}{{{fmt(rec['e2e_overhead'], percent=True)}}}")
    lines.append(f"\\newcommand{{{tex_name(run, 'cmem')}}}{{{fmt(rec['peak_gib'], places=2)}}}")
    lines.append(f"\\newcommand{{{tex_name(run, 'cdtrain')}}}{{{fmt(rec['d_train'], signed=True)}}}")
    lines.append(f"\\newcommand{{{tex_name(run, 'cdtrainse')}}}{{{fmt(rec['d_train_se'])}}}")
    lines.append(f"\\newcommand{{{tex_name(run, 'cfb')}}}{{{rec['output_fallbacks']}}}")
with open(os.path.join(HERE, "results", "run_numbers.tex"), "w") as handle:
    handle.write("\n".join(lines) + "\n")

# A compact machine-readable dump for cross-checking the prose.
dump = {run: {k: v for k, v in rec.items() if k not in {"late"}} for run, rec in summary.items()}
with open(os.path.join(HERE, "results", "summary.json"), "w") as handle:
    json.dump(dump, handle, indent=1, sort_keys=True)

# Console report, ordered by held-out selection delta within the seed-0 frame.
print(f"{'arm':44s} {'dsel_late':>10s} {'dc4_late':>10s} {'dtrain':>10s} "
      f"{'se':>9s} {'step%':>7s} {'e2e%':>7s} {'GiB':>6s} {'out_fb':>7s}")
for run, rec in sorted(summary.items(), key=lambda kv: (kv[1]["d_sel_late"] is None, kv[1]["d_sel_late"])):
    print(f"{run:44s} {fmt(rec['d_sel_late'], signed=True):>10s} "
          f"{fmt(rec['d_c4_late'], signed=True):>10s} "
          f"{fmt(rec['d_train'], signed=True):>10s} {fmt(rec['d_train_se']):>9s} "
          f"{fmt(rec['step_overhead'], percent=True, signed=True):>7s} "
          f"{fmt(rec['e2e_overhead'], percent=True, signed=True):>7s} "
          f"{rec['peak_gib']:6.2f} {rec['output_fallbacks']:7d}")
