"""Evaluate retrieved evidence and preserve per-query traces."""
from __future__ import annotations

import re
from typing import Any
import unicodedata


def _normalize_for_match(text: str) -> str:
    """Normalize Unicode, typographic quotes, and whitespace for robust evidence matching."""
    norm = unicodedata.normalize("NFKC", text)
    norm = (
        norm.replace("\u00a0", " ")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return re.sub(r"\s+", " ", norm).strip()


def _valid_gold_pages(question: dict[str, Any]) -> set[int]:
    """Return all accepted gold page numbers for a benchmark question."""
    pages: set[int] = set()
    if question.get("gold_page") is not None:
        pages.add(int(question["gold_page"]))
    for alt_page in question.get("alternative_gold_pages") or []:
        pages.add(int(alt_page))
    return pages


def _valid_evidence_quotes(question: dict[str, Any]) -> list[str]:
    """Return normalized primary and alternative gold evidence quotes."""
    raw_quotes: list[str] = []
    if question.get("evidence_quote"):
        raw_quotes.append(str(question["evidence_quote"]))
    for alt in question.get("alternative_evidence_quotes") or []:
        if alt:
            raw_quotes.append(str(alt))
    return [_normalize_for_match(q) for q in raw_quotes if _normalize_for_match(q)]


def _hit_matches_filter(meta: dict[str, Any], expected_filter: dict[str, Any]) -> bool:
    """Return True if hit metadata satisfies every non-null key/value in expected_filter."""
    for key, expected_val in expected_filter.items():
        if expected_val is None:
            continue
        if meta.get(key) != expected_val:
            return False
    return True


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
    strategy: str | None = None,
    filter_applied: dict[str, Any] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    """Check source/page and enough gold evidence; keep hit@1 and hit@3 trace.

    Accept any documented alternative gold evidence. Do not infer human grounding
    or chunk coherence labels from exact string match.
    """
    gold_source_id = question.get("gold_source_id")
    valid_pages = _valid_gold_pages(question)
    valid_quotes_norm = _valid_evidence_quotes(question)
    requested_filter = question.get("filter") or None

    top_hits = hits[:top_k]
    evaluated_hits: list[dict[str, Any]] = []

    for rank, hit in enumerate(top_hits, start=1):
        meta = dict(hit.get("metadata") or {})
        hit_doc_id = meta.get("document_id", hit.get("document_id"))
        hit_page = meta.get("page", hit.get("page"))

        page_match = (hit_doc_id == gold_source_id) and (hit_page in valid_pages)
        hit_text_norm = _normalize_for_match(str(hit.get("text", "")))
        evidence_match = page_match and any(
            quote_norm in hit_text_norm for quote_norm in valid_quotes_norm
        )

        filter_compliant = (
            _hit_matches_filter(meta, requested_filter)
            if requested_filter
            else None
        )

        evaluated_hits.append(
            {
                "rank": rank,
                "chunk_id": hit.get("chunk_id"),
                "score": float(hit.get("score", 0.0)),
                "text": hit.get("text", ""),
                "metadata": meta,
                "page_match": page_match,
                "evidence_match": evidence_match,
                "filter_compliant": filter_compliant,
            }
        )

    hit_at_1 = bool(evaluated_hits and evaluated_hits[0]["evidence_match"])
    evidence_hit_at_3 = any(h["evidence_match"] for h in evaluated_hits)
    gold_page_hit_at_3 = any(h["page_match"] for h in evaluated_hits)

    # Track whether full gold evidence only appears when combining multiple gold-page chunks.
    combined_gold_text = " ".join(
        _normalize_for_match(h["text"]) for h in evaluated_hits if h["page_match"]
    )
    multi_chunk_evidence_hit_at_3 = (
        gold_page_hit_at_3
        and bool(valid_quotes_norm)
        and any(q_norm in combined_gold_text for q_norm in valid_quotes_norm)
    )
    evidence_split_across_chunks = (not evidence_hit_at_3) and multi_chunk_evidence_hit_at_3

    if requested_filter:
        all_hits_filter_compliant: bool | None = bool(evaluated_hits) and all(
            bool(h["filter_compliant"]) for h in evaluated_hits
        )
    else:
        all_hits_filter_compliant = None

    resolved_strategy = strategy or (
        evaluated_hits[0]["metadata"].get("strategy") if evaluated_hits else None
    )

    return {
        "question_id": question.get("id"),
        "split": question.get("split"),
        "question_type": question.get("question_type"),
        "question": question.get("question"),
        "expected_answer": question.get("expected_answer"),
        "strategy": resolved_strategy,
        "filter_requested": requested_filter,
        "filter_applied": filter_applied,
        "gold_source_id": gold_source_id,
        "gold_page": question.get("gold_page"),
        "alternative_gold_pages": question.get("alternative_gold_pages") or [],
        "gold_risk_id": question.get("gold_risk_id"),
        "gold_section_type": question.get("gold_section_type"),
        "evidence_quote": question.get("evidence_quote"),
        "alternative_evidence_quotes": question.get("alternative_evidence_quotes") or [],
        "hit_at_1": hit_at_1,
        "evidence_hit_at_3": evidence_hit_at_3,
        "gold_page_hit_at_3": gold_page_hit_at_3,
        "multi_chunk_evidence_hit_at_3": multi_chunk_evidence_hit_at_3,
        "evidence_split_across_chunks": evidence_split_across_chunks,
        "all_hits_filter_compliant": all_hits_filter_compliant,
        "hits": evaluated_hits,
        "human_grounding_score": None,
        "human_grounding_notes": "",
    }


def _aggregate_bucket(bucket: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Hit@1, Evidence Hit@3, and filter compliance counts for one trace bucket."""
    total = len(bucket)
    if total == 0:
        return {
            "total": 0,
            "hit_at_1_count": 0,
            "hit_at_1": 0.0,
            "evidence_hit_at_3_count": 0,
            "evidence_hit_at_3": 0.0,
            "gold_page_hit_at_3_count": 0,
            "gold_page_hit_at_3": 0.0,
            "multi_chunk_hit_at_3_count": 0,
            "multi_chunk_hit_at_3": 0.0,
            "split_evidence_count": 0,
            "filter_question_count": 0,
            "filter_compliant_query_count": 0,
            "filter_compliance_rate": None,
            "filter_compliant_hit_count": 0,
            "total_filter_checked_hits": 0,
        }

    hit1 = sum(1 for t in bucket if t.get("hit_at_1"))
    hit3 = sum(1 for t in bucket if t.get("evidence_hit_at_3"))
    page_hit3 = sum(1 for t in bucket if t.get("gold_page_hit_at_3"))
    multi_hit3 = sum(1 for t in bucket if t.get("multi_chunk_evidence_hit_at_3"))
    split_count = sum(1 for t in bucket if t.get("evidence_split_across_chunks"))

    filter_traces = [t for t in bucket if t.get("filter_requested")]
    filter_q_count = len(filter_traces)
    filter_q_pass = sum(1 for t in filter_traces if t.get("all_hits_filter_compliant") is True)
    filter_hit_pass = sum(
        1
        for t in filter_traces
        for h in t.get("hits", [])
        if h.get("filter_compliant") is True
    )
    filter_hit_total = sum(len(t.get("hits", [])) for t in filter_traces)

    return {
        "total": total,
        "hit_at_1_count": hit1,
        "hit_at_1": round(hit1 / total, 4),
        "evidence_hit_at_3_count": hit3,
        "evidence_hit_at_3": round(hit3 / total, 4),
        "gold_page_hit_at_3_count": page_hit3,
        "gold_page_hit_at_3": round(page_hit3 / total, 4),
        "multi_chunk_hit_at_3_count": multi_hit3,
        "multi_chunk_hit_at_3": round(multi_hit3 / total, 4),
        "split_evidence_count": split_count,
        "filter_question_count": filter_q_count,
        "filter_compliant_query_count": filter_q_pass,
        "filter_compliance_rate": (
            round(filter_q_pass / filter_q_count, 4) if filter_q_count > 0 else None
        ),
        "filter_compliant_hit_count": filter_hit_pass,
        "total_filter_checked_hits": filter_hit_total,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        strategy = str(trace.get("strategy") or "unknown")
        grouped.setdefault(strategy, []).append(trace)

    by_strategy: dict[str, Any] = {}
    for strategy, strat_traces in grouped.items():
        dev_traces = [t for t in strat_traces if t.get("split") == "dev"]
        test_traces = [t for t in strat_traces if t.get("split") == "test"]
        by_strategy[strategy] = {
            "dev": _aggregate_bucket(dev_traces),
            "test": _aggregate_bucket(test_traces),
            "overall": _aggregate_bucket(strat_traces),
        }

    return {"by_strategy": by_strategy}
