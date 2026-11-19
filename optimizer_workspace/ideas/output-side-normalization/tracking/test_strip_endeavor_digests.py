# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stripping digests from ENDEAVOR.md keeps every claim and archives the payload."""

import os

import pytest
import strip_endeavor_digests
from strip_endeavor_digests import main, residual_digests, strip_document

DIGEST = "a" * 64
OTHER = "b" * 64

RESEARCH = f"""# Endeavor

## Wave 12

Fused reaches validation loss 3.2814, which is 0.0025 below MuonT.
Its preparation is `prep.json`, SHA-256 `{DIGEST}`.
Receipt: [`logs/wave12.json`](logs/wave12.json).

## Wave 13

No release exists. The failed archive SHA-256 is `{OTHER}`.
"""


@pytest.fixture
def documents(tmp_path):
    (tmp_path / "ENDEAVOR.md").write_text(RESEARCH)
    (tmp_path / "DETAILS.md").write_text("# Details\n\n## Receipts\n")
    return tmp_path


def test_claims_and_numbers_survive_while_digests_move(documents):
    assert main([str(documents)]) == 0
    research = (documents / "ENDEAVOR.md").read_text()
    details = (documents / "DETAILS.md").read_text()

    assert "3.2814" in research and "0.0025 below MuonT" in research
    assert "No release exists." in research
    assert residual_digests(research) == []
    assert DIGEST in details and OTHER in details


def test_markdown_link_targets_keep_resolving(documents):
    main([str(documents)])
    assert "[`logs/wave12.json`](logs/wave12.json)" in (documents / "ENDEAVOR.md").read_text()


def test_check_reports_without_writing(documents):
    before = (documents / "ENDEAVOR.md").read_text()
    assert main([str(documents), "--check"]) == 0
    assert (documents / "ENDEAVOR.md").read_text() == before


def test_rerun_is_idempotent(documents):
    main([str(documents)])
    research = (documents / "ENDEAVOR.md").read_text()
    details = (documents / "DETAILS.md").read_text()

    assert main([str(documents)]) == 0
    assert (documents / "ENDEAVOR.md").read_text() == research
    assert (documents / "DETAILS.md").read_text() == details


@pytest.mark.parametrize("filename", ("ENDEAVOR.md", "DETAILS.md"))
def test_concurrent_refresh_aborts_instead_of_losing_an_archive(documents, monkeypatch, filename):
    original = strip_endeavor_digests.strip_document
    changed = documents / filename
    metadata = changed.stat()
    before = changed.read_text()

    def refresh_then_strip(text):
        changed.write_text(before + "\nA later refresh wrote this.\n")
        os.utime(changed, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        return original(text)

    monkeypatch.setattr(strip_endeavor_digests, "strip_document", refresh_then_strip)
    assert main([str(documents)]) == 1
    assert "A later refresh wrote this." in changed.read_text()
    assert DIGEST in (documents / "ENDEAVOR.md").read_text()
    assert DIGEST not in (documents / "DETAILS.md").read_text()


def test_a_digest_that_cannot_be_archived_is_not_removed(documents, monkeypatch):
    monkeypatch.setattr(strip_endeavor_digests, "archive_blocks", lambda removed: "")
    assert main([str(documents)]) == 1
    assert DIGEST in (documents / "ENDEAVOR.md").read_text()


def test_strip_document_returns_removed_blocks_verbatim():
    cleaned, removed = strip_document(RESEARCH)
    assert removed, "the citation blocks should be reported for archiving"
    assert all(block.strip() for _, block in removed)
    assert DIGEST in "".join(block for _, block in removed)
    assert DIGEST not in cleaned
