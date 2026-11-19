# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Move operational milestone subsections out of the human-readable ENDEAVOR.md.

The research narrative regrows: each diagnostic round adds CPU-acceptance,
archive-acceptance, owner-preparation, queue and per-case decision
subsections. Those are receipts of how a gate was run, not findings about
output-side normalization, and ENDEAVOR.md keeps their aggregate in its
state table. This moves them to DETAILS.md verbatim.

Classification is by heading title, because the titles are systematic.
Moving a heading moves its whole subtree. Top-level `## ` headings, the
generator's managed blocks and its insertion anchors are never touched.
"""

import argparse
import re
from pathlib import Path

# Subsections of these top-level sections are eligible to move.
PARENTS = ("Remaining Experimental Program",)

# A heading whose title matches is a receipt of how a gate was run.
OPERATIONAL = re.compile(
    r"""(
      CPU\ (?:Acceptance|Check|Validation|Verified|Preparation)
    | Archive\ Acceptance
    | (?:Owner|Queue)\ (?:Preparation|Acceptance|Draft)
    | Execution\ Owner | Validator\ (?:Component|Preparation)
    | Handoff
    | Case\ (?:Decision|Rejection)
    | Trajector(?:y|ies)\ (?:Completion|Retained)
    | Command\ Completion
    | Running\ Progress | Recurring\ Reporting | Terminal\ Waiter
    | Wave\ Transition
    | Exit-Drain | Marker-Exit | Exit-Bracket
    | Ownership\ Failure\ Attribution
    | Execution\ Failure\ and\ Restoration
    | Recovery\ Launch | Failed\ Follow-Ups
    | Queue\ And\ Reporting | Validation\ Queue
    | Configuration\ Preparation
    | Evidence\ Preparation | Evidence-Helper | Numerical\ Protocol
    | Localization\ (?:Component|Execution|Handoff)
    | Rolling\ Journal\ Checkpoint
    )""",
    re.VERBOSE,
)

HEADING = re.compile(r"^(#{2,6})\s+(.*?)\s*$")
MARKER = re.compile(r"^<!--\s*(?:BEGIN|END)\s+AUTOMATED")


def classify(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return (start, end, parent) for each subtree that should move."""
    moves: list[tuple[int, int, str]] = []
    parent = None
    index = 0
    while index < len(lines):
        match = HEADING.match(lines[index])
        if not match:
            index += 1
            continue
        depth, title = len(match.group(1)), match.group(2)
        if depth == 2:
            parent = title if title in PARENTS else None
            index += 1
            continue
        if parent is None or not OPERATIONAL.search(title):
            index += 1
            continue
        # Move this heading and every deeper heading under it.
        end = index + 1
        while end < len(lines):
            inner = HEADING.match(lines[end])
            if inner and len(inner.group(1)) <= depth:
                break
            if MARKER.match(lines[end]):  # never swallow a managed block
                break
            end += 1
        moves.append((index, end, parent))
        index = end
    return moves


def apply_moves(lines: list[str], moves: list[tuple[int, int, str]]) -> tuple[list[str], str]:
    """Return the kept lines and the verbatim archive text for the moved ones."""
    cut = {i for start, end, _ in moves for i in range(start, end)}
    kept = [line for i, line in enumerate(lines) if i not in cut]
    parts = []
    for start, end, parent in moves:
        body = "\n".join(lines[start:end]).strip()
        parts.append(f"\n### From ENDEAVOR.md: {parent}\n\n{body}\n")
    return kept, "".join(parts)


def archive_section(archive: str, today: str) -> str:
    return (
        f"\n## Operational Milestones Moved Out of ENDEAVOR.md ({today})\n\n"
        "Verbatim subsections moved from ENDEAVOR.md's research narrative so it\n"
        "keeps findings, results and plans. Each entry names the section it came\n"
        "from. The aggregate outcome of these rounds stays in ENDEAVOR.md's\n"
        "Current Experiment State table.\n" + archive
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path.cwd())
    parser.add_argument("--check", action="store_true", help="list the classification only")
    parser.add_argument("--today", default=None, help="date stamp for the archive heading")
    args = parser.parse_args(argv)

    research = args.directory / "ENDEAVOR.md"
    details = args.directory / "DETAILS.md"
    research_text, details_text = research.read_text(), details.read_text()
    lines = research_text.split("\n")
    moves = classify(lines)

    if args.check:
        for start, end, parent in moves:
            size = sum(len(line) + 1 for line in lines[start:end])
            print(f"MOVE {size:7d} B  {lines[start][:84]}")
        kept_headings = [
            line
            for i, line in enumerate(lines)
            if HEADING.match(line) and not any(s <= i < e for s, e, _ in moves)
        ]
        moved = sum(sum(len(line) + 1 for line in lines[s:e]) for s, e, _ in moves)
        print(f"\n{len(moves)} subtree(s), {moved} B would move; {len(kept_headings)} headings stay")
        return 0
    if not moves:
        print("no operational subsections to move")
        return 0

    kept, archive = apply_moves(lines, moves)
    kept_text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept))
    archived = details_text.rstrip("\n") + "\n" + archive_section(archive, args.today or "")

    # Every original line must survive in one of the two documents.
    from collections import Counter

    before = Counter(lines)
    after = Counter(kept_text.split("\n")) + Counter(archive.split("\n"))
    lost = {line: count for line, count in before.items() if count > after.get(line, 0) and line.strip()}
    if lost:
        for line in list(lost)[:5]:
            print(f"refusing: would lose {line[:90]!r}")
        return 1
    if research.read_text() != research_text or details.read_text() != details_text:
        print("refusing: a document changed during the pass; rerun")
        return 1
    details.write_text(archived)
    research.write_text(kept_text)
    print(f"moved {len(moves)} subtree(s); ENDEAVOR.md now {len(kept_text)} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
