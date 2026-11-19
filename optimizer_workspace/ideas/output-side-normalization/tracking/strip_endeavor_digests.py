# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Move receipt digests out of the human-readable ENDEAVOR.md.

Rules are generic so the tool stays usable as new prose is written:
every 32+ hex digest, revision and marker is removed from ENDEAVOR.md
together with the clause that cites it, while the surrounding claim is
kept. Markdown link targets are preserved so paths keep resolving.
Removed blocks are returned verbatim for preservation in DETAILS.md.
"""

import argparse
import re
import textwrap
from pathlib import Path

WIDTH = 72

# A digest/marker/revision token: 32+ hex chars with at least one letter.
HEX = r"`?(?<![0-9a-zA-Z])(?=[0-9a-f]*[a-f])[0-9a-f]{32,}(?![0-9a-zA-Z])`?"
HEX_RE = re.compile(HEX)
LINK_RE = re.compile(r"\]\([^)]*\)")
LABEL = r"(?:SHA-?256|digest|marker)"

# Ordered clause patterns. Each removes a citation, never a claim.
CLAUSE = [
    re.compile(rf";\s*(?:its|the|their)\s+[^;.]{{0,60}}?{LABEL}\s+(?:is|remains)\s+{HEX}"),
    re.compile(rf"^[A-Z][\w` /-]{{0,40}}?{LABEL}\s+(?:is|remains)\s+{HEX};\s*"),
    re.compile(
        rf"[,;:]?\s*{LABEL}\s*(?:at\s+\d{{1,2}}:\d{{2}}(?::\d{{2}})?\s*)?"
        rf"(?:is\s+|remains\s+|matches[^:.;]*:\s*|:\s*)?{HEX}"
    ),
    re.compile(rf"[,;]?\s*marker\s*{HEX}"),
    re.compile(rf"\s*(?:and|;)\s*(?:archive|result|binding)\s+{LABEL}\s+{HEX}"),
    re.compile(rf"[,;:]?\s*{HEX}"),
]

PUNCT = [
    (re.compile(r"\s+([.;,:])"), r"\1"),
    (re.compile(r"([;,:])\s*\."), "."),
    (re.compile(r"\s{2,}"), " "),
    (re.compile(r",\s*,"), ","),
]

# A sentence that is only a receipt label once its digest is gone.
LABEL_ONLY = re.compile(
    r"^(?:the\s+|its\s+|a\s+)?[\w` /-]{0,48}?"
    r"(?:receipt|binding|archive|digest|release|log|hash|snapshot|state)s?\.$",
    re.IGNORECASE,
)

# A sentence left dangling on its copula because the digest was its only object,
# e.g. "Its preparation SHA-256 is <digest>." -> "Its preparation is."
# Such a sentence is dropped rather than completed with invented words.
DANGLING = re.compile(r"\b(?:is|are|was|were|remains?)\s*\.$", re.IGNORECASE)


def _words(text: str) -> int:
    return len(re.findall(r"[A-Za-z][A-Za-z0-9/_-]*", text))


def _wrap(text: str, initial: str = "", subsequent: str = "") -> str:
    return textwrap.fill(
        text,
        width=WIDTH,
        initial_indent=initial,
        subsequent_indent=subsequent,
        break_on_hyphens=False,
        break_long_words=False,
    )


def _clean(flat: str) -> str:
    """Strip digest citations from one unwrapped run of prose."""
    links: list[str] = []

    def hide(match: re.Match[str]) -> str:
        links.append(match.group(0))
        return f"\x00{len(links) - 1}\x00"

    flat = LINK_RE.sub(hide, flat)
    for pattern in CLAUSE:
        flat = pattern.sub("", flat)
    for pattern, repl in PUNCT:
        flat = pattern.sub(repl, flat)
    kept_sentences = []
    for sentence in re.split(r"(?<=[.])\s+", flat):
        sentence = sentence.strip()
        if not sentence:
            continue
        # Drop semicolon clauses that lost their content.
        clauses = [c.strip() for c in sentence.split(";")]
        clauses = [c for c in clauses if _words(c) >= 3 or "\x00" in c]
        sentence = "; ".join(clauses)
        sentence = re.sub(r";\s*\.", ".", sentence)
        if not sentence or sentence == ".":
            continue
        # A sentence naming a path or identifier stays: only digests go.
        substantive = "\x00" in sentence or "`" in sentence or "/" in sentence
        # Length alone never decides: short sentences carry findings
        # ("No release exists."). Only pure citation scaffolding goes.
        if not substantive and (LABEL_ONLY.match(sentence) or DANGLING.search(sentence)):
            continue
        if DANGLING.search(sentence) and _words(sentence) < 6:
            continue
        if not sentence.endswith((".", ":", "`")):
            sentence += "."
        kept_sentences.append(sentence)
    flat = " ".join(kept_sentences).strip()
    flat = re.sub(r"^([a-z])", lambda m: m.group(1).upper(), flat)
    for index, original in enumerate(links):
        flat = flat.replace(f"\x00{index}\x00", original)
    return flat


def strip_prose(block: str) -> str:
    """Clean a paragraph, or a bullet list item by item."""
    lines = block.splitlines()
    if any(line.lstrip().startswith(("- ", "* ")) for line in lines):
        items: list[str] = []
        for line in lines:
            if line.lstrip().startswith(("- ", "* ")) or not items:
                items.append(line)
            else:
                items[-1] += " " + line.strip()
        out = []
        for item in items:
            match = re.match(r"^(\s*[-*]\s+)(.*)$", item, re.S)
            prefix, body = (match.group(1), match.group(2)) if match else ("", item)
            cleaned = _clean(" ".join(body.split()))
            if cleaned:
                out.append(_wrap(cleaned, prefix, " " * len(prefix)) if prefix else _wrap(cleaned))
        return "\n".join(out)
    return _wrap(_clean(" ".join(block.split())))


DROP_ROW = re.compile(
    rf"^\|\s*(?:Revision|Plan {LABEL}|Receipt {LABEL}|`?tokenizer\.json`?\s*{LABEL})\s*\|",
    re.IGNORECASE,
)


def _is_divider(row: list[str]) -> bool:
    return all(set(cell) <= set("-: ") and cell for cell in row)


def strip_table(block: str) -> str:
    """Drop all-digest columns, then clean digests out of remaining cells."""
    all_lines = block.splitlines()
    # The source sometimes glues a following heading onto the last table row;
    # only the leading run of pipe lines is the table.
    cut = next((i for i, l in enumerate(all_lines) if not l.lstrip().startswith("|")), len(all_lines))
    lines, tail = all_lines[:cut], all_lines[cut:]
    lines = [line for line in lines if not DROP_ROW.match(line)]
    suffix = ("\n\n" + "\n".join(tail)) if tail else ""
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    if len({len(r) for r in rows}) != 1:
        # Ragged table: keep every column, clean digests out of the cells.
        return (
            "\n".join("| " + " | ".join(_clean(c) if HEX_RE.search(c) else c for c in row) + " |" for row in rows)
            + suffix
        )
    columns = len(rows[0])
    body = [r for r in rows if not _is_divider(r)]
    digest_cols = [i for i in range(columns) if body[1:] and all(HEX_RE.fullmatch(r[i]) for r in body[1:] if r[i])]
    keep = [i for i in range(columns) if i not in digest_cols]
    if len(keep) < 2:
        return suffix.lstrip("\n")  # the table existed only to carry digests
    rows = [[r[i] for i in keep] for r in rows]
    out = []
    for row in rows:
        if _is_divider(row):
            out.append("| " + " | ".join(row) + " |")
            continue
        cells = [_clean(c) if HEX_RE.search(c) else c for c in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + suffix


def strip_document(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return cleaned ENDEAVOR text and the (section, original block) pairs changed."""
    section = "(preamble)"
    out: list[str] = []
    removed: list[tuple[str, str]] = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("## "):
                section = line[3:].strip()
        if block.lstrip().startswith("#") or not HEX_RE.search(LINK_RE.sub("", block)):
            out.append(block)
            continue
        new = strip_table(block) if block.lstrip().startswith("|") else strip_prose(block)
        removed.append((section, block))
        if new.strip():
            out.append(new)
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(out)), removed


def residual_digests(text: str) -> list[str]:
    """Digest tokens still present outside markdown link targets."""
    return HEX_RE.findall(LINK_RE.sub("", text))


def archive_blocks(removed: list[tuple[str, str]]) -> str:
    """Render removed blocks as DETAILS.md sections, preserving them verbatim."""
    parts = []
    for section, block in removed:
        parts.append(f"\n### From ENDEAVOR.md section: {section}\n\n{block.strip()}\n")
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Move digests from ENDEAVOR.md into DETAILS.md in one atomic pass.

    Both files are read once and rewritten only if neither changed meanwhile,
    so a concurrent ledger refresh can never lose an archived block.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--check", action="store_true", help="report residual digests only")
    args = parser.parse_args(argv)

    research = args.directory / "ENDEAVOR.md"
    details = args.directory / "DETAILS.md"
    before = (research.stat().st_mtime_ns, details.stat().st_mtime_ns)
    research_text, details_text = research.read_text(), details.read_text()

    cleaned, removed = strip_document(research_text)
    residual = residual_digests(cleaned)
    if args.check:
        for digest in residual:
            print(f"residual digest: {digest}")
        print(f"{len(residual)} residual, {len(removed)} block(s) would move")
        return 1 if residual else 0
    if not removed:
        print("no digests to move")
        return 0

    archived = details_text.rstrip("\n") + "\n" + archive_blocks(removed)
    for digest in residual_digests(research_text):
        if digest not in archived:
            print(f"refusing: {digest} would be lost")
            return 1
    if (
        (research.stat().st_mtime_ns, details.stat().st_mtime_ns) != before
        or research.read_text() != research_text
        or details.read_text() != details_text
    ):
        print("refusing: a document changed during the pass; rerun")
        return 1
    details.write_text(archived)
    research.write_text(cleaned)
    print(f"moved {len(removed)} block(s); {len(residual_digests(cleaned))} residual")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
