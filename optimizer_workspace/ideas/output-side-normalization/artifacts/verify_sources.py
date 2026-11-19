"""Validate the collected sources and write their structured provenance manifest."""

import hashlib
import json
import re
import tarfile
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
ARTICLE_COMMIT = "453badd98fe389f009884063ca88378db15bf582"
ARTICLE_BLOB = "1db4aae72ab2e0deb979ce5228f273f82a4bac74"


class CitationMetadata(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        name = attrs.get("name", "")
        if tag == "meta" and name.startswith("citation_"):
            self.values.setdefault(name, []).append(attrs.get("content", ""))


def file_record(path):
    content = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main():
    references = json.loads((ROOT / "reference_queue.json").read_text())["references"]
    metadata_by_id = {}
    for path in (ROOT / "retrieval").glob("*-abstract.html"):
        metadata = CitationMetadata()
        metadata.feed(path.read_text())
        if "citation_arxiv_id" in metadata.values:
            metadata_by_id[metadata.values["citation_arxiv_id"][0]] = (path, metadata.values)

    records = []
    for reference in references:
        for relative in reference["artifacts"]:
            path = ROOT / relative
            assert path.is_file(), f"Missing source: {relative}"
            record = file_record(path)
            record.update({"reference_id": reference["id"], "source_urls": reference["urls"]})
            if path.suffix == ".pdf":
                reader = PdfReader(path)
                texts = [page.extract_text() or "" for page in reader.pages]
                expected = reference["arxiv_id"] + reference["version"]
                identifiers = re.findall(r"\d{4}\.\d{4,5}v\d+", "\n".join(texts[:2]))
                assert expected in identifiers, (relative, expected, identifiers)
                assert texts and sum(map(len, texts)) > 1000, relative
                metadata_path, metadata = metadata_by_id[reference["arxiv_id"]]
                record.update(
                    {
                        "arxiv_id": reference["arxiv_id"],
                        "version": reference["version"],
                        "title": metadata["citation_title"][0],
                        "authors": metadata.get("citation_author", []),
                        "initial_date": metadata.get("citation_date", [None])[0],
                        "revision_date": metadata.get("citation_online_date", [None])[0],
                        "pages": len(texts),
                        "extracted_text_characters": sum(map(len, texts)),
                        "metadata": file_record(metadata_path),
                        "validation": "all_pages_parsed_and_embedded_arxiv_version_matched",
                    }
                )
            elif path.suffix == ".gz":
                with tarfile.open(path) as archive:
                    members = archive.getmembers()
                    for member in members:
                        member_path = Path(member.name)
                        assert not member_path.is_absolute() and ".." not in member_path.parts
                record.update(
                    {
                        "commit": reference["commit"],
                        "archive_members": len(members),
                        "validation": "gzip_tar_parsed",
                    }
                )
            elif path.name == "toner-blog-pinned.html":
                content = path.read_bytes()
                blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
                assert blob == ARTICLE_BLOB
                assert content == (ROOT / "external-sources/toner-blog.html").read_bytes()
                record.update(
                    {
                        "commit": ARTICLE_COMMIT,
                        "git_blob_sha1": blob,
                        "validation": "complete_article_matches_pinned_git_blob",
                    }
                )
            elif reference["id"] == "muon":
                record["commit"] = reference["commit"]
            records.append(record)

    manifest = {
        "retrieved_on": "2026-11-15",
        "timezone": "Asia/Shanghai",
        "proxy": "http://star-proxy.oa.com:3128",
        "training_or_checkpoint_experiments_performed": False,
        "original_proposal_sha256": "2ab5d88ea85e0531526b31a5e800d625d017227483564ccd4d671220adbeaad2",
        "sources": records,
        "pdf_count": sum(record["path"].endswith(".pdf") for record in records),
        "limitations": [
            "No GO-MUON author implementation was identified.",
            "The Freon README is third-party discovery evidence and not an author implementation.",
            "Paper results were inspected but not independently replicated.",
            "Search responses and alternate text captures do not replace the primary sources.",
        ],
    }
    (ROOT / "external_source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {manifest['pdf_count']} PDFs and {len(records)} source artifacts.")
    for record in records:
        if "pages" in record:
            print(f"{record['path']}: {record['version']}, {record['pages']} pages, {record['bytes']} bytes")


if __name__ == "__main__":
    main()
