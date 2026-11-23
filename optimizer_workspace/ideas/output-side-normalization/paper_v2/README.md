# Width-Thresholded Output-Cotangent Normalization — Clean Manuscript (`paper_v2`)

Self-contained rewrite organized along a deductive research logic (starting with **Methodology** and **Experimental Results**; Abstract and Introduction deferred for the next pass).

## Structure & Writing Conventions
- Every paragraph carries a descriptive `\paragraph{...}` title and opens with its **main claim** before supporting derivations or quantitative details.
- Every figure and table is self-contained with explicit metric, baseline, sample-size/averaging-window, and uncertainty definitions.
- Results are organized by seven sequential research questions (**Q1–Q7**) rather than chronological training waves.
- All prose numerals are verified deterministically against JSON summaries in `results/`.

## Build & Verification
```bash
python -I scripts/check_tex.py
python -I scripts/check_numbers.py
PATH=/opt/texlive/2026/bin/x86_64-linux:$PATH latexmk -pdf main.tex
```
