"""Run and verify initial PDF ingestion and update data/manifest.json statistics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

STARTER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STARTER_DIR.parent
sys.path.insert(0, str(STARTER_DIR))

from src.ingestion import (  # noqa: E402
    annotate_owasp_metadata,
    export_ingestion_artifacts,
    filter_content_pages,
    load_pdf,
)


def main() -> None:
    manifest_path = REPO_ROOT / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    pdf_path = REPO_ROOT / manifest["source_path"]
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing source PDF: {pdf_path}")

    raw_pages = load_pdf(
        pdf_path,
        document_id=manifest["source_id"],
        source=manifest["source_path"],
    )
    excluded_map = {item["page"]: item["reason"] for item in manifest["excluded_pages"]}
    kept_pages, audit_log = filter_content_pages(raw_pages, excluded_pages=excluded_map)
    annotated_pages = annotate_owasp_metadata(kept_pages)

    pages_out = STARTER_DIR / "results" / "normalized_pages.jsonl"
    log_out = STARTER_DIR / "results" / "ingestion_log.json"
    export_ingestion_artifacts(annotated_pages, audit_log, pages_out=pages_out, log_out=log_out)

    manifest["total_physical_pages"] = len(raw_pages)
    manifest["included_pages"] = [p.page for p in annotated_pages]
    manifest["included_page_count"] = len(annotated_pages)
    manifest["total_normalized_chars"] = sum(len(p.text) for p in annotated_pages)
    manifest["per_page_chars"] = {str(p.page): len(p.text) for p in annotated_pages}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Assertions required by Day 06 lab rules
    assert len(annotated_pages) >= 30, f"Expected >= 30 pages, got {len(annotated_pages)}"
    log_by_page = {row["page"]: row for row in audit_log}
    assert log_by_page[43]["decision"] == "excluded_non_content", (
        f"Page 43 should be excluded_non_content, got {log_by_page[43]}"
    )
    for err_page in (44, 45):
        assert log_by_page[err_page]["decision"] == "excluded_error", (
            f"Page {err_page} should be marked excluded_error, got {log_by_page[err_page]}"
        )
        assert log_by_page[err_page]["extraction_status"] == "garbled_encoding"
    for page in annotated_pages:
        assert "OWASP Top 10 for LLM Applications v2.0" not in page.text, (
            f"Header leaked into page {page.page}"
        )
        assert "genai.owasp.org" not in page.text, (
            f"Footer leaked into page {page.page}"
        )

    print(f"Physical pages: {len(raw_pages)}")
    print(f"Kept content pages: {len(annotated_pages)}")
    print(f"Total normalized chars: {manifest['total_normalized_chars']}")
    print("Excluded / error pages:")
    for row in audit_log:
        if row["decision"] != "kept":
            print(
                f"  - p{row['page']:02d}: decision={row['decision']} "
                f"status={row['extraction_status']} reason={row['reason']} "
                f"raw_chars={row['raw_chars']}"
            )
    print("\nSample annotated metadata (first 5 kept pages):")
    for p in annotated_pages[:5]:
        print(
            f"  - p{p.page:02d} (printed={p.metadata['printed_page']}): "
            f"risk={p.metadata['risk_id']} sections={p.metadata['section_types']} "
            f"chars={len(p.text)}"
        )


if __name__ == "__main__":
    main()
