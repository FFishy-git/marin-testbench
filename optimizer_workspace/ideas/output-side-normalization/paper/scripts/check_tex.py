"""Static checks on the manuscript that substitute for a trial compile.

No TeX toolchain is available in the authoring environment, so this verifies
what can be verified without one: that every \\input target exists, that every
\\ref and \\cite resolves, that user-defined macros are defined before use, and
that environments and braces balance per file.

    python -I scripts/check_tex.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Control sequences provided by the document class, the loaded packages, the
# template's own macro.tex/paperstyle.sty, or plain TeX. Checking against a
# curated list keeps the report short enough to act on.
KNOWN = set("""
documentclass usepackage PassOptionsToPackage RequirePackage input include
begin end newcommand renewcommand providecommand NewDocumentCommand
DeclareMathOperator newtheorem newenvironment newif newtcolorbox def gdef
edef let relax expandafter csname endcsname makeatletter makeatother
title author date maketitle thanks footnote footnotemark footnotetext
section subsection subsubsection paragraph subparagraph appendix
label ref eqref pageref cite citep citet bibliography bibliographystyle
tableofcontents newpage clearpage pagebreak linebreak hfill vfill hspace vspace
textbf textit texttt textsc textrm emph underline textsuperscript
mathbb mathbf mathrm mathcal mathsf operatorname text
frac dfrac tfrac nicefrac sqrt sum prod int lim max min log exp
left right big Big bigg Bigg langle rangle lVert rVert lvert rvert
alpha beta gamma delta epsilon varepsilon zeta eta theta iota kappa lambda mu
nu xi pi rho sigma tau upsilon phi varphi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
in notin subset subseteq cup cap setminus emptyset forall exists neg
leq geq neq approx equiv sim simeq propto times cdot cdots ldots dots vdots
to mapsto rightarrow leftarrow Rightarrow Leftarrow leftrightarrow
partial nabla infty pm mp oplus otimes circ star ast
quad qquad enspace thinspace noindent indent par smallskip medskip bigskip
centering raggedright raggedleft small footnotesize scriptsize tiny
large Large LARGE huge Huge normalsize
toprule midrule bottomrule addlinespace cmidrule multicolumn multirow
caption captionsetup includegraphics graphicspath
color textcolor definecolor colorlet providecolor
item itemize enumerate description
setlength renewcommand topfraction hypersetup urlstyle url href nolinkurl
interfootnotelinepenalty suppressfloats FloatBarrier
tikz draw node fill path coordinate addplot addlegendentry nextgroupplot
pgfplotsset usepgfplotslibrary usetikzlibrary pgfplotspointmeta
thisrow ding phantom vcenter hbox overset underbrace boxed
ifdefined pdfoutput pdftexversion fi else ifx global
patchcmd l@section addvspace p@ plus minus
RNum braRNum eqi brai spn stle enc dec pre heaviside hermite highlight addref
AIbox olivebox olivebox2 lightbluebox bluebox violetbox githubquote finding
termhl eqcallout termtinttrue termtintfalse iftermtint
papershorttitle paperaffiliations papernote papermeta paperemail paperlogo
paperlogos paperlogoitem pagesf pagesfsb pagestyle
sectionblue checkgreen crossred
mathbf mathit boldsymbol
hat widehat tilde widetilde bar vec dot ddot acute grave check breve
Bigl Bigr bigl bigr Biggl Biggr middle
bfseries normalfont itshape ttfamily rmfamily sffamily mdseries
linewidth textwidth columnwidth textheight headsep parindent parskip baselineskip
abovedisplayskip abovedisplayshortskip topsep
today verb uppercase lowercase romannumeral number the
thesection thesubsection thesubsubsection thetcbcounter thefootnote
tabcolsep arraystretch extracolsep
titleformat titlespacing tcbset newmdenv ifblank notblank
@plus @minus
""".split())


def read(path):
    with open(path) as handle:
        return handle.read()


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def main():
    main_path = os.path.join(HERE, "main.tex")
    files = [("main.tex", strip_comments(read(main_path)))]

    # Resolve \input targets transitively.
    queue = list(files)
    seen = {"main.tex"}
    problems = []
    while queue:
        name, text = queue.pop()
        for target in re.findall(r"\\input\{([^}]+)\}", text):
            candidate = target if target.endswith(".tex") else target + ".tex"
            path = os.path.join(HERE, candidate)
            if not os.path.exists(path):
                problems.append(f"missing \\input target: {target} (from {name})")
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            entry = (candidate, strip_comments(read(path)))
            files.append(entry)
            queue.append(entry)

    whole = "\n".join(text for _, text in files)

    # Labels and references.
    labels = set(re.findall(r"\\label\{([^}]+)\}", whole))
    refs = set(re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", whole))
    for missing in sorted(refs - labels):
        problems.append(f"undefined reference: {missing}")
    duplicates = [l for l in labels
                  if len(re.findall(r"\\label\{" + re.escape(l) + r"\}", whole)) > 1]
    for dup in sorted(set(duplicates)):
        problems.append(f"duplicate label: {dup}")

    # Citations against refs.bib.
    bib = read(os.path.join(HERE, "refs.bib"))
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for group in re.findall(r"\\cite[tp]?\*?(?:\[[^]]*\])*\{([^}]+)\}", whole):
        cited.update(k.strip() for k in group.split(","))
    for missing in sorted(cited - keys):
        problems.append(f"citation with no bib entry: {missing}")
    for unused in sorted(keys - cited):
        problems.append(f"note: bib entry never cited: {unused}")

    # Macro definitions versus use.
    defined = set(re.findall(
        r"\\(?:newcommand|renewcommand|providecommand|def|NewDocumentCommand)"
        r"\s*\{?\\([A-Za-z@]+)", whole))
    for extra in ("macro.tex", "paperstyle.sty"):
        path = os.path.join(HERE, extra)
        if os.path.exists(path):
            text = strip_comments(read(path))
            defined |= set(re.findall(
                r"\\(?:newcommand|renewcommand|providecommand|def|NewDocumentCommand|"
                r"newtcolorbox|newenvironment|newmdenv|DeclareMathOperator)"
                r"\s*\*?\s*(?:\[[^]]*\])?\s*\{?\\?([A-Za-z@]+)", text))
            defined |= set(re.findall(r"\\newif\\if([A-Za-z@]+)", text))
    # Environment names are not control sequences.
    envs = set(re.findall(r"\\begin\{([A-Za-z*]+)\}", whole))
    used = set(re.findall(r"\\([A-Za-z@]+)", whole))
    unknown = sorted(used - defined - KNOWN - envs)
    # Filter obvious package-provided plotting/tabular keys.
    unknown = [u for u in unknown if not u.startswith(("pgf", "tikz", "axis", "group"))]
    for name in unknown:
        problems.append(f"possibly undefined macro: \\{name}")

    # Environment and brace balance per file.
    for name, text in files:
        opens = re.findall(r"\\begin\{([A-Za-z*]+)\}", text)
        closes = re.findall(r"\\end\{([A-Za-z*]+)\}", text)
        for env in set(opens) | set(closes):
            if opens.count(env) != closes.count(env):
                problems.append(
                    f"{name}: environment {env} opened {opens.count(env)}, "
                    f"closed {closes.count(env)}")
        depth = 0
        for match in re.finditer(r"(?<!\\)([{}])", text):
            depth += 1 if match.group(1) == "{" else -1
        if depth != 0:
            problems.append(f"{name}: brace imbalance {depth:+d}")

    print(f"checked {len(files)} files, {len(labels)} labels, {len(cited)} citations")
    notes = [p for p in problems if p.startswith("note:")]
    errors = [p for p in problems if not p.startswith("note:")]
    for problem in errors:
        print("  ERROR", problem)
    for note in notes:
        print("  ", note)
    if not errors:
        print("no structural problems found")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
