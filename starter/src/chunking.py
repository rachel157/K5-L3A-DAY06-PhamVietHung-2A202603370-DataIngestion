"""Implement fixed-size and structure-aware chunking."""
from __future__ import annotations

import re
from typing import Any

from .models import Chunk, DocumentPage

_SECTION_HEADING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("overview", re.compile(r"^(?:Letter from the Project Leads|What[’']s New in the 2025 Top 10)$", re.IGNORECASE)),
    ("description", re.compile(r"^Description$", re.IGNORECASE)),
    ("examples", re.compile(r"^Common Examples of (?:Vulnerability|Risks?)$", re.IGNORECASE)),
    ("mitigation", re.compile(r"^Prevention and Mitigation Strategies$", re.IGNORECASE)),
    ("scenarios", re.compile(r"^(?:Example|Sample) Attack Scenarios$", re.IGNORECASE)),
    ("references", re.compile(r"^Reference Links$", re.IGNORECASE)),
    ("frameworks", re.compile(r"^Related Frameworks and Taxonomies$", re.IGNORECASE)),
)

_RISK_HEADER_RE = re.compile(r"^LLM\d{2}(?::\s*\d{4})?\s*:?\s*.+$")


def _detect_section_heading(paragraph: str) -> str | None:
    """Return canonical section_type if a paragraph is a top-level OWASP section heading."""
    text = paragraph.strip()
    for section_type, pattern in _SECTION_HEADING_PATTERNS:
        if pattern.match(text):
            return section_type
    return None


def _base_metadata(page: DocumentPage, strategy_name: str) -> dict[str, Any]:
    """Copy stable traceability fields from a DocumentPage."""
    meta: dict[str, Any] = {
        "chunking": strategy_name,
        "risk_id": page.metadata.get("risk_id"),
        "risk_title": page.metadata.get("risk_title"),
        "printed_page": page.metadata.get("printed_page"),
    }
    return {k: v for k, v in meta.items() if v is not None}


def _split_oversized_paragraph(paragraph: str, max_size: int) -> list[str]:
    """Split an oversized paragraph at sentence boundaries, falling back to words."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if s.strip()]
    if not sentences:
        return []

    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_size:
            if current:
                pieces.append(current)
                current = ""
            # Fallback to word-level splitting only when a single sentence exceeds max_size.
            word_buf = ""
            for word in sentence.split():
                candidate = f"{word_buf} {word}".strip()
                if len(candidate) <= max_size:
                    word_buf = candidate
                else:
                    if word_buf:
                        pieces.append(word_buf)
                    word_buf = word
            if word_buf:
                pieces.append(word_buf)
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_size:
            current = candidate
        else:
            pieces.append(current)
            current = sentence

    if current:
        pieces.append(current)
    return pieces


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split page text into overlapping fixed-size windows and retain page metadata."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got {overlap} (size={size})")

    text = page.text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + size, len(text))
        window = text[start:end].strip()
        if window:
            idx += 1
            meta = {
                **_base_metadata(page, "fixed_size"),
                "char_len": len(window),
                "start_offset": start,
                "end_offset": end,
            }
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}:fixed:p{page.page:02d}:c{idx:03d}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=window,
                    metadata=meta,
                )
            )
        if end >= len(text):
            break
        start += step

    return chunks


def by_paragraph(page: DocumentPage, max_size: int = 1050) -> list[Chunk]:
    """Group paragraphs within the same section up to max_size; split oversized paragraphs."""
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")

    paragraphs = [p.strip() for p in page.text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    default_section = "overview" if page.metadata.get("risk_id") == "OVERVIEW" else "description"
    current_section: str = page.metadata.get("initial_section_type", default_section)

    grouped: list[tuple[str, str]] = []
    buffer: list[str] = []
    buffer_section = current_section

    def flush() -> None:
        if not buffer:
            return
        grouped.append(("\n\n".join(buffer), buffer_section))
        buffer.clear()

    def current_len() -> int:
        if not buffer:
            return 0
        return sum(len(p) for p in buffer) + 2 * (len(buffer) - 1)

    for para in paragraphs:
        detected = _detect_section_heading(para)
        if detected is not None and detected != current_section:
            # Keep an isolated top risk header (e.g., "LLM01:2025 ...") attached to Description.
            if buffer and len(buffer) == 1 and _RISK_HEADER_RE.match(buffer[0]):
                current_section = detected
                buffer_section = detected
            else:
                flush()
                current_section = detected
                buffer_section = detected

        sub_paras = (
            _split_oversized_paragraph(para, max_size)
            if len(para) > max_size
            else [para]
        )
        for sub_para in sub_paras:
            projected = len(sub_para) if not buffer else current_len() + 2 + len(sub_para)
            if projected <= max_size:
                buffer.append(sub_para)
                buffer_section = current_section
            else:
                flush()
                buffer.append(sub_para)
                buffer_section = current_section

    flush()

    chunks: list[Chunk] = []
    for idx, (chunk_text, section_type) in enumerate(grouped, start=1):
        meta = {
            **_base_metadata(page, "by_paragraph"),
            "section_type": section_type,
            "char_len": len(chunk_text),
        }
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}:para:p{page.page:02d}:c{idx:03d}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=chunk_text,
                metadata=meta,
            )
        )
    return chunks


def chunk_corpus(
    pages: list[DocumentPage],
    strategy: str,
    size: int = 500,
    overlap: int = 50,
    max_size: int = 1050,
) -> list[Chunk]:
    """Chunk an ordered sequence of pages while preserving cross-page section continuity."""
    all_chunks: list[Chunk] = []
    prev_risk_id: str | None = None
    active_section = "overview"

    for page in pages:
        if strategy == "fixed_size":
            all_chunks.extend(fixed_size(page, size=size, overlap=overlap))
            continue

        if strategy != "by_paragraph":
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        risk_id = page.metadata.get("risk_id")
        if risk_id != prev_risk_id:
            active_section = "overview" if risk_id == "OVERVIEW" else "description"
        prev_risk_id = risk_id

        page_with_ctx = DocumentPage(
            document_id=page.document_id,
            source=page.source,
            page=page.page,
            text=page.text,
            metadata={**page.metadata, "initial_section_type": active_section},
        )
        page_chunks = by_paragraph(page_with_ctx, max_size=max_size)
        if page_chunks:
            active_section = page_chunks[-1].metadata["section_type"]
            all_chunks.extend(page_chunks)

    return all_chunks
