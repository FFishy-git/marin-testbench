# Theory Illustration Primer (`theory_illustration/`)

Self-contained visual tutorial paper explaining the **rotation** and **eigenvector / singular-vector mechanism** of output-side normalization (`WT-MLP-Lag`) from first principles using 2D geometric diagrams, hand-checkable $2\times 2$ numerical examples, and verified `130M`/`520M` Transformer measurements.

## Files
- `main.tex` / `main.pdf`: Standalone pedagogical primer (leaves `../paper_v3/main.pdf` untouched).
- `scripts/generate_illustrations.py`: CPU-only Matplotlib generator producing `figs/fig1_logic_roadmap.pdf` through `figs/fig5_canyon_and_payoff.pdf`.
- `paperstyle.sty` & `refs.bib`: Typography and bibliography definitions.

## Build (CPU-only)
```bash
uv run --offline --no-project --with matplotlib python scripts/generate_illustrations.py
PATH="/opt/texlive/2026/bin/x86_64-linux:$PATH" latexmk -pdf -interaction=nonstopmode main.tex
```
