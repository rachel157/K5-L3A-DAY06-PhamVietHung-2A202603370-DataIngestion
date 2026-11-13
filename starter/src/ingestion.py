"""Read and normalize source data for Day 06 ingestion."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage

DEFAULT_EXCLUDED_PAGES: dict[int, str] = {
    1: "cover_page",
    2: "license_and_revision_history",
    3: "table_of_contents",
    4: "table_of_contents",
    6: "acknowledgments_and_project_leads_signoff",
    14: "reference_overflow_only_two_bullets",
    43: "image_only_appendix_no_extractable_text",
    44: "sponsors_page_garbled_encoding",
    45: "supporters_page_garbled_encoding",
}

_HEADER_FOOTER_PATTERNS = (
    re.compile(r"^OWASP Top 10 for LLM Applications v\d+\.\d+$", re.IGNORECASE),
    re.compile(r"^\d*genai\.owasp\.org(?:\s+\d+)?$", re.IGNORECASE),
    re.compile(r"^\d{1,2}\s+genai\.owasp\.org$", re.IGNORECASE),
)

_HEADING_EXACT = {
    "Letter from the Project Leads",
    "What's New in the 2025 Top 10",
    "What’s New in the 2025 Top 10",
    "Moving Forward",
    "Description",
    "Types of Prompt Injection Vulnerabilities",
    "Common Examples of Vulnerability",
    "Common Examples of Risks",
    "Common Examples of Risk",
    "Prevention and Mitigation Strategies",
    "Example Attack Scenarios",
    "Sample Attack Scenarios",
    "Reference Links",
    "Related Frameworks and Taxonomies",
}

_RISK_HEADING_RE = re.compile(r"^(LLM\d{2})(?::\s*\d{4})?\s*:?\s*(.+)$")
_LIST_ITEM_RE = re.compile(r"^(?:[•\-\*]|\d+\.)\s+")
_SCENARIO_RE = re.compile(r"^Scenario\s*#\s*\d+", re.IGNORECASE)
_SUBHEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9\s,&/\-:'()]{2,80}:?$")

_SECTION_MAP = (
    ("description", re.compile(r"^Description$", re.IGNORECASE | re.MULTILINE)),
    ("examples", re.compile(r"^Common Examples of (?:Vulnerability|Risks?)$", re.IGNORECASE | re.MULTILINE)),
    ("mitigation", re.compile(r"^Prevention and Mitigation Strategies$", re.IGNORECASE | re.MULTILINE)),
    ("scenarios", re.compile(r"^(?:Example|Sample) Attack Scenarios$", re.IGNORECASE | re.MULTILINE)),
    ("references", re.compile(r"^Reference Links$", re.IGNORECASE | re.MULTILINE)),
    ("frameworks", re.compile(r"^Related Frameworks and Taxonomies$", re.IGNORECASE | re.MULTILINE)),
)


def _diagnose_raw_page(raw_text: str) -> tuple[str, dict[str, Any]]:
    """Inspect extracted raw text to detect empty or garbled-encoding pages."""
    stripped = raw_text.strip()
    raw_chars = len(stripped)
    if raw_chars < 30:
        return "empty_or_image_only", {"raw_chars": raw_chars, "suspicious_ratio": 0.0}

    non_space = [ch for ch in stripped if not ch.isspace()]
    if not non_space:
        return "empty_or_image_only", {"raw_chars": 0, "suspicious_ratio": 0.0}

    # Count unusual Latin-1/control/replacement characters that signal broken font maps.
    suspicious = sum(
        1
        for ch in non_space
        if ch == "\ufffd" or (0x0080 <= ord(ch) <= 0x024F and ch not in "°±²³")
    )
    suspicious_ratio = round(suspicious / len(non_space), 4)
    if suspicious_ratio > 0.10:
        return "garbled_encoding", {"raw_chars": raw_chars, "suspicious_ratio": suspicious_ratio}

    return "extracted", {"raw_chars": raw_chars, "suspicious_ratio": suspicious_ratio}


def load_pdf_with_log(
    path: Path, document_id: str, source: str
) -> tuple[list[DocumentPage], list[dict[str, Any]]]:
    """Extract every physical PDF page with page numbers and extraction diagnostics."""
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    logs: list[dict[str, Any]] = []

    for page_num, pdf_page in enumerate(reader.pages, start=1):
        raw_text = pdf_page.extract_text() or ""
        status, stats = _diagnose_raw_page(raw_text)
        metadata = {
            "extraction_status": status,
            "raw_chars": stats["raw_chars"],
            "suspicious_ratio": stats["suspicious_ratio"],
        }
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_num,
                text=raw_text,
                metadata=metadata,
            )
        )
        logs.append(
            {
                "page": page_num,
                "extraction_status": status,
                "raw_chars": stats["raw_chars"],
                "suspicious_ratio": stats["suspicious_ratio"],
            }
        )

    return pages, logs


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical page; keep its page number, source, and extraction status."""
    pages, _ = load_pdf_with_log(path, document_id=document_id, source=source)
    return pages


def _is_structural_line(line: str) -> bool:
    """Identify standalone headings and scenario headers that occupy their own paragraph."""
    if line in _HEADING_EXACT:
        return True
    if _RISK_HEADING_RE.match(line) or _SCENARIO_RE.match(line):
        return True
    if _LIST_ITEM_RE.match(line):
        # Treat short numbered titles (e.g., "1. Constrain model behavior") as headings,
        # while wrapped list sentences continue in the paragraph buffer.
        words = line.split()
        return len(words) <= 6 and not line.endswith((",", ".", ";"))
    if len(line) <= 75 and not line.endswith((".", ",", ";", ")")) and _SUBHEADING_RE.match(line):
        words = line.split()
        if len(words) <= 9 and not line[0].islower():
            return True
    return False


def normalize_text(text: str) -> str:
    """Normalize Unicode, remove repeating headers/footers, and preserve paragraph boundaries."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ")
    # Repair PDF soft-hyphens and replacement glyphs joining split words.
    normalized = re.sub(r"(?<=\w)\u00ad\s*(?=\w)", "", normalized)
    normalized = re.sub(r"(?<=\w)\ufffd(?=\w)", "-", normalized)
    normalized = normalized.replace("\ufffd", "")

    cleaned_lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        if any(pat.match(line) for pat in _HEADER_FOOTER_PATTERNS):
            continue
        if line.isdigit() and len(line) <= 2:
            continue
        cleaned_lines.append(line)

    # Reconstruct paragraphs while preserving headings and list items.
    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            joined = " ".join(buffer)
            joined = re.sub(r"(\w)-\s+(\w)", r"\1-\2", joined)
            paragraphs.append(joined)
            buffer.clear()

    for line in cleaned_lines:
        if not line:
            flush_buffer()
            continue
        if _is_structural_line(line):
            flush_buffer()
            paragraphs.append(line)
            continue
        if _LIST_ITEM_RE.match(line):
            flush_buffer()
            buffer.append(line)
            continue
        if buffer and buffer[-1].endswith("-") and line[0].islower():
            buffer[-1] = buffer[-1][:-1] + line
        else:
            buffer.append(line)

    flush_buffer()
    return "\n\n".join(paragraphs).strip()


def filter_content_pages(
    pages: list[DocumentPage],
    excluded_pages: dict[int, str] | None = None,
) -> tuple[list[DocumentPage], list[dict[str, Any]]]:
    """Keep substantive content pages and return an explicit audit log for all pages."""
    excluded = DEFAULT_EXCLUDED_PAGES if excluded_pages is None else excluded_pages
    kept_pages: list[DocumentPage] = []
    audit_log: list[dict[str, Any]] = []

    for page in pages:
        status = page.metadata.get("extraction_status", "extracted")
        reason = excluded.get(page.page)

        if status != "extracted":
            decision = "excluded_error"
            final_reason = reason or status
        elif reason is not None:
            decision = "excluded_non_content"
            final_reason = reason
        else:
            decision = "kept"
            final_reason = "substantive_content"

        norm_text = normalize_text(page.text) if decision == "kept" else ""
        if decision == "kept" and len(norm_text) < 50:
            decision = "excluded_empty_after_normalization"
            final_reason = "insufficient_text_after_normalization"

        audit_entry = {
            "page": page.page,
            "decision": decision,
            "reason": final_reason,
            "extraction_status": status,
            "raw_chars": page.metadata.get("raw_chars", len(page.text.strip())),
            "normalized_chars": len(norm_text),
        }
        audit_log.append(audit_entry)

        if decision == "kept":
            kept_pages.append(
                DocumentPage(
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=norm_text,
                    metadata={**page.metadata, "normalized_chars": len(norm_text)},
                )
            )

    return kept_pages, audit_log


def annotate_owasp_metadata(pages: list[DocumentPage]) -> list[DocumentPage]:
    """Attach risk_id, risk_title, printed_page, and section_types across sequential pages."""
    current_risk_id = "OVERVIEW"
    current_risk_title = "What's New in the 2025 Top 10"
    annotated: list[DocumentPage] = []

    for page in pages:
        for line in page.text.splitlines():
            match = _RISK_HEADING_RE.match(line.strip())
            if match:
                current_risk_id = match.group(1)
                current_risk_title = match.group(2).strip()
                break

        section_types = [
            name for name, pattern in _SECTION_MAP if pattern.search(page.text)
        ]
        if not section_types and current_risk_id == "OVERVIEW":
            section_types = ["overview"]

        enriched_meta = {
            **page.metadata,
            "risk_id": current_risk_id,
            "risk_title": current_risk_title,
            "printed_page": page.page - 4 if page.page >= 5 else page.page,
            "section_types": section_types,
        }
        annotated.append(
            DocumentPage(
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=page.text,
                metadata=enriched_meta,
            )
        )

    return annotated


def export_ingestion_artifacts(
    pages: list[DocumentPage],
    audit_log: list[dict[str, Any]],
    pages_out: Path,
    log_out: Path,
) -> None:
    """Persist normalized pages (JSONL) and full page audit log (JSON)."""
    pages_out.parent.mkdir(parents=True, exist_ok=True)
    log_out.parent.mkdir(parents=True, exist_ok=True)

    with pages_out.open("w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")

    summary = {
        "total_physical_pages": len(audit_log),
        "kept_pages": sum(1 for row in audit_log if row["decision"] == "kept"),
        "excluded_pages": sum(1 for row in audit_log if row["decision"] != "kept"),
        "total_normalized_chars": sum(len(p.text) for p in pages),
        "pages": audit_log,
    }
    log_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
