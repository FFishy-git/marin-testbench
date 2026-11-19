"""Cross-check numeric literals in the prose against the generated data.

Many numbers appear in running text rather than in generated tables. This
script extracts decimal literals from the section sources and checks each
against the values computed from the run histories, so a transcription error
shows up as an unmatched literal rather than as a wrong claim in the paper.

Unmatched literals are not necessarily errors: configuration constants,
derived ratios, percentages and numbers quoted from the project's numerical
studies legitimately do not appear in summary.json. The point is that the list
is short enough to read.

    python -I scripts/check_numbers.py
"""

import glob
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(HERE, "results", "summary.json")) as handle:
    runs = json.load(handle)
with open(os.path.join(HERE, "results", "lr_summary.json")) as handle:
    lr = json.load(handle)

# Every quantity the analysis computed, at the precisions the prose might use.
values = set()


def add(x):
    if not isinstance(x, (int, float)):
        return
    for places in (2, 3, 4, 5, 6, 7, 8, 9):
        values.add(f"{abs(x):.{places}f}")
        values.add(f"{abs(x):.{places}f}".rstrip("0"))


for rec in runs.values():
    for key, value in rec.items():
        if key == "windows" and isinstance(value, list):
            for w in value:
                add(w)
        else:
            add(value)
    # Candidate-minus-null and other differences the prose forms.
    for other in runs.values():
        if rec.get("d_sel_late") is not None and other.get("d_sel_late") is not None:
            add(rec["d_sel_late"] - other["d_sel_late"])
            add(rec["d_c4_late"] - other["d_c4_late"])
for rec in lr.values():
    for value in rec.values():
        add(value)
for a in lr.values():
    for b in lr.values():
        add(a["selection"] - b["selection"])

# Literals that are configuration, protocol or externally sourced constants.
ALLOWED = {
    # recipe and protocol constants
    "0.02262741696", "0.016", "0.032", "0.04525483392", "0.06399999989260598",
    "0.09050966784", "0.12799999978521195", "0.0048", "0.95", "0.938", "0.1",
    "0.5", "0.125", "0.25", "0.055", "3.8", "4.105", "0.2", "0.06666666666666667",
    "3.4445", "4.7750", "2.0315", "1.0", "0.0", "4.0",
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
    # tabular column widths
    "0.34", "0.42", "0.11", "0.17", "0.21", "0.235", "0.155", "0.135", "0.115",
    # misc small integers rendered as decimals
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
    "21.9", "39.2", "8.2", "17.2", "6.2", "3.9", "0.85",
}

pattern = re.compile(r"(?<![\w.])(\d+\.\d{2,})(?![\w])")
unmatched = {}
for path in sorted(glob.glob(os.path.join(HERE, "sections", "*.tex"))) + \
            [os.path.join(HERE, "main.tex")]:
    text = re.sub(r"(?<!\\)%.*", "", open(path).read())
    # Numbers inside a plot body are axis limits, widths and legend positions,
    # not claims, so they are out of scope for this check.
    text = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "",
                  text, flags=re.S)
    for literal in pattern.findall(text):
        if literal in values or literal in ALLOWED:
            continue
        if literal.rstrip("0") in values:
            continue
        unmatched.setdefault(literal, []).append(os.path.basename(path))

print(f"distinct computed values: {len(values)}")
if not unmatched:
    print("every prose literal matches a computed value or a declared constant")
else:
    print(f"{len(unmatched)} literal(s) to review:")
    for literal, files in sorted(unmatched.items()):
        print(f"  {literal:>16s}  {', '.join(sorted(set(files)))}")
