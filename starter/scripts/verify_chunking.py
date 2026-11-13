"""Verify Module 1 chunking strategies (fixed_size vs by_paragraph) on normalized pages."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import sys

STARTER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STARTER_DIR.parent
sys.path.insert(0, str(STARTER_DIR))

from src.chunking import chunk_corpus  # noqa: E402
from src.models import DocumentPage  # noqa: E402


def _load_normalized_pages(path: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pages.append(DocumentPage(**row))
    return pages


def _summarize(name: str, chunks: list) -> None:
    lengths = [len(c.text) for c in chunks]
    print(
        f"[{name}] chunks={len(chunks)} | total_chars={sum(lengths)} | "
        f"min={min(lengths)} | median={int(statistics.median(lengths))} | max={max(lengths)}"
    )


def main() -> None:
    pages = _load_normalized_pages(STARTER_DIR / "results" / "normalized_pages.jsonl")
    assert len(pages) == 36, f"Expected 36 normalized pages, got {len(pages)}"

    chunks_a = chunk_corpus(pages, strategy="fixed_size", size=500, overlap=50)
    chunks_b = chunk_corpus(pages, strategy="by_paragraph", max_size=1050)

    for label, chunks in (("A:fixed_size", chunks_a), ("B:by_paragraph", chunks_b)):
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), f"{label}: duplicate chunk_ids"
        for c in chunks:
            assert c.text.strip(), f"{label}: empty chunk {c.chunk_id}"
            assert c.document_id and c.source and c.page > 0, f"{label}: missing provenance on {c.chunk_id}"
            assert "risk_id" in c.metadata, f"{label}: missing risk_id on {c.chunk_id}"

    # Verify by_paragraph section continuity across continuation pages
    for c in chunks_b:
        assert "section_type" in c.metadata, f"Missing section_type on {c.chunk_id}"
    p18_sections = {c.metadata["section_type"] for c in chunks_b if c.page == 18}
    p25_sections = {c.metadata["section_type"] for c in chunks_b if c.page == 25}
    p28_sections = {c.metadata["section_type"] for c in chunks_b if c.page == 28}
    assert p18_sections == {"scenarios"}, f"Expected p18 scenarios only, got {p18_sections}"
    assert p25_sections == {"scenarios", "references"}, f"Expected p25 scenarios+references, got {p25_sections}"
    assert p28_sections == {"mitigation"}, f"Expected p28 mitigation only, got {p28_sections}"

    # Check that all 10 pre-locked coherence positions appear in at least one chunk on their page
    coh = json.loads((REPO_ROOT / "benchmark" / "coherence_positions.json").read_text(encoding="utf-8"))
    for label, chunks in (("A", chunks_a), ("B", chunks_b)):
        for pos in coh["positions"]:
            pg = pos["page"]
            anchor = pos["anchor_quote"]
            # Check full anchor or first 35 chars in case fixed_size splits mid-anchor
            prefix = anchor[:35]
            matched = [c for c in chunks if c.page == pg and (anchor in c.text or prefix in c.text)]
            assert matched, f"Strategy {label}: anchor missing for {pos['position_id']} on p{pg}"

    _summarize("A fixed_size(500, 50)", chunks_a)
    _summarize("B by_paragraph(max=1050)", chunks_b)
    print("B section_type counts:", dict(Counter(c.metadata["section_type"] for c in chunks_b)))
    print(f"p18 sections={sorted(p18_sections)}, p25 sections={sorted(p25_sections)}, p28 sections={sorted(p28_sections)}")

    # Print one concrete comparison example on p28 (LLM06)
    print("\n--- Sample comparison on p28 (LLM06) ---")
    print("A chunk[0] preview:", repr(next(c.text for c in chunks_a if c.page == 28)[:180]))
    print("B chunk[0] preview:", repr(next(c.text for c in chunks_b if c.page == 28)[:180]))


if __name__ == "__main__":
    main()
