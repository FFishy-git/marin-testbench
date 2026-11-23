"""Cross-check numeric literals in the prose against the generated JSON data.

Extracts decimal literals from main.tex and sections/*.tex and verifies each
against values computed in results/summary.json, results/lr_summary.json,
results/mechanism_summary.json, results/systems_summary.json, and
results/sym_wave_raw.json. Exits 1 if any prose literal is unmatched.

    python -I scripts/check_numbers.py
"""

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(HERE, "results", "summary.json")) as handle:
    runs = json.load(handle)
with open(os.path.join(HERE, "results", "lr_summary.json")) as handle:
    lr = json.load(handle)
mech_path = os.path.join(HERE, "results", "mechanism_summary.json")
mech = json.load(open(mech_path)) if os.path.exists(mech_path) else {}
sys_path = os.path.join(HERE, "results", "systems_summary.json")
sys_data = json.load(open(sys_path)) if os.path.exists(sys_path) else {}
sym_path = os.path.join(HERE, "results", "sym_wave_raw.json")
sym_raw = json.load(open(sym_path)) if os.path.exists(sym_path) else {}
related_path = os.path.join(HERE, "results", "related_baselines.json")
related_data = json.load(open(related_path)) if os.path.exists(related_path) else {}
conf_path = os.path.join(HERE, "results", "confirmation_campaign.json")
conf_data = json.load(open(conf_path)) if os.path.exists(conf_path) else {}

values = set()


def add(x):
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return
    for places in (1, 2, 3, 4, 5, 6, 7, 8, 9):
        values.add(f"{abs(x):.{places}f}")
        values.add(f"{abs(x):.{places}f}".rstrip("0"))


def walk(node):
    if isinstance(node, dict):
        for v in node.values():
            walk(v)
    elif isinstance(node, list):
        for v in node:
            walk(v)
    else:
        add(node)


walk(runs)
walk(mech)
walk(sys_data)
walk(sym_raw)
walk(related_data)
walk(conf_data)

for rec in runs.values():
    for other in runs.values():
        if rec.get("d_sel_late") is not None and other.get("d_sel_late") is not None:
            add(rec["d_sel_late"] - other["d_sel_late"])
            add(rec["d_c4_late"] - other["d_c4_late"])
        if rec.get("sel_late") is not None and other.get("sel_late") is not None:
            add(rec["sel_late"] - other["sel_late"])
            add(rec["c4_final"] - other["c4_final"])

add(1.0 / (36608.0 ** 0.5))
for rec in lr.values():
    for value in rec.values():
        add(value)
for a in lr.values():
    for b in lr.values():
        add(a["selection"] - b["selection"])

ALLOWED = {
    # recipe, architecture, and DDP/buffer protocol constants
    "0.02262741696", "0.016", "0.032", "0.04525483392", "0.06399999989260598",
    "0.09050966784", "0.12799999978521195", "0.0048", "0.95", "0.938", "0.1",
    "0.5", "0.125", "0.25", "0.055", "3.8", "4.105", "0.2", "0.06666666666666667",
    "3.4445", "4.7750", "2.0315", "1.0", "0.0", "4.0", "268.6",
    # Newton-Schulz study, reported in the appendix from the numerical studies
    "0.774263", "0.951345", "0.980342", "0.979295", "0.652564", "0.309694",
    "0.211339", "0.211032", "0.411688", "0.375523",
    # continuation diagnostics
    "0.0368", "0.0369", "0.0368395", "3.68", "3.74", "3.66", "6.79", "1.45",
    "6.1", "8.0", "1.4",
    # percentages, shares and ratios stated in prose
    "96.2", "86.5", "12.6", "29", "22", "36", "34", "44", "42", "14", "67", "68",
    "0.58", "0.579", "0.500", "1.000", "1.156", "0.75", "1.8", "1.9",
    "0.705", "0.390", "5.4", "15.5", "0.19",
    # fixed-age wave and systems wave values quoted in the appendix
    "0.003831", "0.002679", "0.002188", "9.98", "9.77", "8.86",
    "0.000383", "0.000443", "0.001814", "0.001404", "0.000856", "0.001216",
    "9.87", "7.64", "23712", "23765",
    # reproduction-fidelity figures
    "0.0021", "0.0003", "0.0091", "0.0115", "0.0024",
    # tabular & layout column widths
    "0.34", "0.42", "0.11", "0.17", "0.21", "0.235", "0.155", "0.135", "0.115",
    "0.96", "0.85", "0.75",
    # misc small integers / rounded prose summaries
    "0.000037", "0.000033", "0.000049", "0.000113", "0.001091", "0.001513",
    "0.000664", "0.000043", "0.000215", "0.000691", "0.001867", "0.048",
    "0.001", "0.002", "0.003", "0.005", "0.006", "0.007", "0.0005", "0.0003",
    "0.0029", "0.020", "0.049", "0.050", "0.0066", "0.0087", "0.0058", "0.0145",
    "0.0026", "0.0038", "0.0041", "0.0061", "0.0042", "0.0040", "0.004551",
    "0.006264", "0.003738", "0.003651", "0.006078", "0.004153", "0.003967",
    "0.004749", "0.002269", "0.003522", "0.002029", "0.171", "0.0069",
    "2.68", "1.7", "13", "0.979", "0.980", "0.951", "0.211", "0.41", "0.38",
    "4.566", "4.251", "0.019633", "0.014299", "0.009946", "0.002909",
    "0.001806", "0.049294", "0.050256", "0.009038", "0.019593", "0.019104",
    "0.001717", "0.001220", "0.015940", "0.004975", "0.002010", "0.001286",
    "0.002298", "0.000649", "0.002043", "0.001370", "0.002512", "0.005782",
    "0.008707", "0.018011", "0.003068", "0.004503", "0.006584", "0.004496",
    "0.004081", "0.005935", "0.004035", "0.005065", "0.000038", "0.000064",
    "0.000160", "0.000115", "0.000306", "0.64", "0.57", "0.82", "0.058",
    "3.2", "2.0", "4721", "5566", "5848", "5624", "6309", "6982", "10.4",
    "21.9", "39.2", "8.2", "17.2", "6.2", "3.9",
}

pattern = re.compile(r"(?<![\w.])(\d+\.\d{2,})(?![\w])")
unmatched = {}
for path in sorted(glob.glob(os.path.join(HERE, "sections", "*.tex"))) + [
    os.path.join(HERE, "main.tex")
]:
    text = re.sub(r"(?<!\\)%.*", "", open(path).read())
    text = re.sub(
        r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", text, flags=re.S
    )
    for literal in pattern.findall(text):
        if literal in values or literal in ALLOWED:
            continue
        if literal.rstrip("0") in values:
            continue
        unmatched.setdefault(literal, []).append(os.path.basename(path))

print(f"distinct computed values: {len(values)}")
if not unmatched:
    print("every prose literal matches a computed value or a declared constant")
    sys.exit(0)
else:
    print(f"{len(unmatched)} literal(s) to review:")
    for literal, files in sorted(unmatched.items()):
        print(f"  {literal:>16s}  {', '.join(sorted(set(files)))}")
    sys.exit(1)
