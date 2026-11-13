"""End-to-end Day 06 ingestion, embedding, pgvector search, and benchmark evaluation pipeline."""
from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any

from dotenv import load_dotenv
import psycopg

from .chunking import chunk_corpus
from .embeddings import MyEmbedder
from .evaluation import _normalize_for_match, evaluate_query, summarize
from .ingestion import (
    annotate_owasp_metadata,
    export_ingestion_artifacts,
    filter_content_pages,
    load_pdf_with_log,
)
from .models import Chunk
from .store import PgVectorStore

STARTER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STARTER_DIR.parent

PDF_PATH = STARTER_DIR / "data" / "OWASP-Top-10-for-LLMs-v2025.pdf"
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.json"
BENCHMARK_PATH = REPO_ROOT / "benchmark" / "questions.json"
COHERENCE_PATH = REPO_ROOT / "benchmark" / "coherence_positions.json"
EXPERIMENT_LOG_PATH = REPO_ROOT / "artifacts" / "experiment_log.csv"

RESULTS_DIR = STARTER_DIR / "results"
PAGES_OUT_PATH = RESULTS_DIR / "normalized_pages.jsonl"
INGESTION_LOG_PATH = RESULTS_DIR / "ingestion_log.json"


def _chunk_length_stats(chunks: list[Chunk]) -> dict[str, Any]:
    """Compute character-length distribution statistics for a chunk list."""
    lengths = [len(c.text) for c in chunks]
    if not lengths:
        return {
            "chunk_count": 0,
            "min_chars": 0,
            "median_chars": 0.0,
            "mean_chars": 0.0,
            "max_chars": 0,
            "total_chars": 0,
        }
    return {
        "chunk_count": len(lengths),
        "min_chars": min(lengths),
        "median_chars": round(float(statistics.median(lengths)), 2),
        "mean_chars": round(float(statistics.mean(lengths)), 2),
        "max_chars": max(lengths),
        "total_chars": sum(lengths),
    }


def _count_strategy_rows(database_url: str, strategy: str) -> int:
    """Count rows currently persisted in document_chunks for a given strategy."""
    with psycopg.connect(database_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks WHERE strategy = %s;", (strategy,))
            row = cur.fetchone()
            return int(row[0]) if row else 0


def _db_environment_versions(database_url: str) -> dict[str, str]:
    """Read PostgreSQL server version and installed pgvector extension version."""
    with psycopg.connect(database_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT version(), (SELECT extversion FROM pg_extension WHERE extname = 'vector');"
            )
            row = cur.fetchone()
            return {
                "postgresql": str(row[0]) if row and row[0] else "unknown",
                "pgvector": str(row[1]) if row and row[1] else "unknown",
            }


def _package_version(dist_name: str) -> str:
    """Return installed Python distribution version or 'unknown'."""
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _find_covering_chunks(chunks: list[Chunk], page: int, anchor_quote: str) -> list[dict[str, Any]]:
    """Find chunks on the target page covering anchor_quote (with page fallback if split)."""
    page_chunks = [c for c in chunks if c.page == page]
    norm_anchor = _normalize_for_match(anchor_quote)
    matched = [c for c in page_chunks if norm_anchor in _normalize_for_match(c.text)]
    if matched:
        exact_match = True
        selected = matched
    else:
        exact_match = False
        selected = page_chunks

    return [
        {
            "chunk_id": c.chunk_id,
            "page": c.page,
            "char_len": len(c.text),
            "section_type": c.metadata.get("section_type"),
            "risk_id": c.metadata.get("risk_id"),
            "anchor_contained": exact_match,
            "text": c.text,
        }
        for c in selected
    ]


def _build_coherence_samples(
    positions_doc: dict[str, Any],
    chunks_a: list[Chunk],
    chunks_b: list[Chunk],
) -> dict[str, Any]:
    """Prepare the 10 pre-locked coherence samples side-by-side for manual grading."""
    samples: list[dict[str, Any]] = []
    for pos in positions_doc.get("positions", []):
        page = int(pos["page"])
        anchor = str(pos["anchor_quote"])
        samples.append(
            {
                "position_id": pos["position_id"],
                "page": page,
                "risk_id": pos["risk_id"],
                "section_type": pos["section_type"],
                "anchor_quote": anchor,
                "strategies": {
                    "A_fixed_size": {
                        "covering_chunks": _find_covering_chunks(chunks_a, page, anchor),
                        "human_score": None,
                        "human_rationale": "",
                    },
                    "B_C_by_paragraph": {
                        "covering_chunks": _find_covering_chunks(chunks_b, page, anchor),
                        "human_score": None,
                        "human_rationale": "",
                    },
                },
            }
        )
    return {
        "description": positions_doc.get("description", ""),
        "grading_scale": "0 = cut mid-idea/missing subject; 1 = partially understandable/missing context; 2 = self-contained and preserves idea",
        "note": "Strategies B and C share identical by_paragraph chunks; C adds retrieval-time section_type filtering.",
        "samples": samples,
    }


def _build_test_audit_samples(all_traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Prepare the 5 test questions across A/B/C for manual metadata and grounding audit."""
    by_question: dict[str, dict[str, Any]] = {}
    for trace in all_traces:
        if trace.get("split") != "test":
            continue
        qid = str(trace["question_id"])
        entry = by_question.setdefault(
            qid,
            {
                "question_id": qid,
                "question_type": trace.get("question_type"),
                "question": trace.get("question"),
                "expected_answer": trace.get("expected_answer"),
                "gold_source_id": trace.get("gold_source_id"),
                "gold_page": trace.get("gold_page"),
                "alternative_gold_pages": trace.get("alternative_gold_pages"),
                "gold_risk_id": trace.get("gold_risk_id"),
                "gold_section_type": trace.get("gold_section_type"),
                "filter_requested": trace.get("filter_requested"),
                "evidence_quote": trace.get("evidence_quote"),
                "alternative_evidence_quotes": trace.get("alternative_evidence_quotes"),
                "strategies": {},
            },
        )
        entry["strategies"][str(trace["strategy"])] = {
            "filter_applied": trace.get("filter_applied"),
            "hit_at_1": trace.get("hit_at_1"),
            "evidence_hit_at_3": trace.get("evidence_hit_at_3"),
            "evidence_split_across_chunks": trace.get("evidence_split_across_chunks"),
            "all_hits_filter_compliant": trace.get("all_hits_filter_compliant"),
            "hits": trace.get("hits"),
            "human_grounding_score": None,
            "human_grounding_rationale": "",
            "metadata_sufficient_and_correct_count": None,
            "metadata_errors": [],
        }

    return {
        "grounding_scale": "0 = no evidence; 1 = partial/missing condition or context; 2 = sufficient to answer and trace source",
        "test_questions": list(by_question.values()),
    }


def _write_experiment_log(summary: dict[str, Any], embed_model: str) -> None:
    """Record controlled A/B/C configurations and dev/test retrieval metrics to artifacts/experiment_log.csv."""
    EXPERIMENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    by_strat = summary["by_strategy"]

    def fmt_metric(stat_bucket: dict[str, Any], key: str, count_key: str) -> str:
        total = stat_bucket["total"]
        count = stat_bucket[count_key]
        val = stat_bucket[key]
        return f"{count}/{total} ({val:.4f})" if total else "0/0 (0.0000)"

    rows = [
        {
            "strategy": "A",
            "chunking": "fixed_size",
            "chunk_size": 500,
            "overlap": 50,
            "metadata": "document_id,source,page,risk_id,risk_title,printed_page",
            "filter": "none",
            "vectorizer": f"openai/{embed_model} (1536d)",
            "hypothesis": "Baseline fixed windows may split sentences, headings, or multi-bullet evidence across boundaries.",
            "dev_hit1": fmt_metric(by_strat["A"]["dev"], "hit_at_1", "hit_at_1_count"),
            "dev_hit3": fmt_metric(by_strat["A"]["dev"], "evidence_hit_at_3", "evidence_hit_at_3_count"),
            "test_hit1": fmt_metric(by_strat["A"]["test"], "hit_at_1", "hit_at_1_count"),
            "test_hit3": fmt_metric(by_strat["A"]["test"], "evidence_hit_at_3", "evidence_hit_at_3_count"),
            "notes": "Controlled baseline: fixed_size chunking without section_type filter.",
        },
        {
            "strategy": "B",
            "chunking": "by_paragraph",
            "chunk_size": 1050,
            "overlap": 0,
            "metadata": "document_id,source,page,risk_id,risk_title,printed_page,section_type",
            "filter": "none",
            "vectorizer": f"openai/{embed_model} (1536d)",
            "hypothesis": "Paragraph- and section-aware grouping preserves complete ideas and multi-bullet evidence better than A.",
            "dev_hit1": fmt_metric(by_strat["B"]["dev"], "hit_at_1", "hit_at_1_count"),
            "dev_hit3": fmt_metric(by_strat["B"]["dev"], "evidence_hit_at_3", "evidence_hit_at_3_count"),
            "test_hit1": fmt_metric(by_strat["B"]["test"], "hit_at_1", "hit_at_1_count"),
            "test_hit3": fmt_metric(by_strat["B"]["test"], "evidence_hit_at_3", "evidence_hit_at_3_count"),
            "notes": "Changes only chunking vs A; section_type stored in metadata but not filtered at search time.",
        },
        {
            "strategy": "C",
            "chunking": "by_paragraph",
            "chunk_size": 1050,
            "overlap": 0,
            "metadata": "document_id,source,page,risk_id,risk_title,printed_page,section_type",
            "filter": "section_type when specified in benchmark question",
            "vectorizer": f"openai/{embed_model} (1536d)",
            "hypothesis": "Applying section_type SQL filter on top of B improves scope compliance and separates mitigation/scenario hits from description/references.",
            "dev_hit1": fmt_metric(by_strat["C"]["dev"], "hit_at_1", "hit_at_1_count"),
            "dev_hit3": fmt_metric(by_strat["C"]["dev"], "evidence_hit_at_3", "evidence_hit_at_3_count"),
            "test_hit1": fmt_metric(by_strat["C"]["test"], "hit_at_1", "hit_at_1_count"),
            "test_hit3": fmt_metric(by_strat["C"]["test"], "evidence_hit_at_3", "evidence_hit_at_3_count"),
            "notes": "Changes only retrieval filter vs B; runs unfiltered for questions with filter=null.",
        },
    ]

    fieldnames = [
        "strategy",
        "chunking",
        "chunk_size",
        "overlap",
        "metadata",
        "filter",
        "vectorizer",
        "hypothesis",
        "dev_hit1",
        "dev_hit3",
        "test_hit1",
        "test_hit3",
        "notes",
    ]
    with EXPERIMENT_LOG_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the full Day 06 ingestion, indexing, retrieval, and benchmark pipeline."""
    load_dotenv(STARTER_DIR / ".env")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_started_at = datetime.now(timezone.utc).isoformat()
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # 1. Load & normalize corpus; validate >= 30 content pages.
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source_id = manifest["source_id"]
    raw_pages, page_logs = load_pdf_with_log(
        PDF_PATH,
        document_id=source_id,
        source=manifest["source_path"],
    )
    kept_pages, audit_log = filter_content_pages(raw_pages)
    if len(kept_pages) < 30:
        raise RuntimeError(f"Expected >= 30 content pages, got {len(kept_pages)}")
    annotated_pages = annotate_owasp_metadata(kept_pages)
    export_ingestion_artifacts(annotated_pages, audit_log, PAGES_OUT_PATH, INGESTION_LOG_PATH)
    print(f"[Step 1] Corpus ready: {len(annotated_pages)} kept content pages (from {len(page_logs)} physical pages).")

    # Load benchmark & coherence positions.
    benchmark_doc = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    questions: list[dict[str, Any]] = benchmark_doc["questions"]
    positions_doc = json.loads(COHERENCE_PATH.read_text(encoding="utf-8"))
    if len(questions) < 20:
        raise RuntimeError(f"Expected >= 20 benchmark questions, got {len(questions)}")
    print(f"[Step 1] Benchmark loaded: {len(questions)} questions, {len(positions_doc['positions'])} coherence positions.")

    # 2. Build chunks for Strategy A (fixed_size) and Strategy B/C (by_paragraph).
    chunks_a = chunk_corpus(annotated_pages, strategy="fixed_size", size=500, overlap=50)
    chunks_b = chunk_corpus(annotated_pages, strategy="by_paragraph", max_size=1050)
    stats_a = _chunk_length_stats(chunks_a)
    stats_b = _chunk_length_stats(chunks_b)
    print(f"[Step 2] Chunks built: A={stats_a['chunk_count']} (fixed_size), B/C={stats_b['chunk_count']} (by_paragraph).")

    # 3. Embed with same model and store vectors/metadata per strategy in PostgreSQL + pgvector.
    embedder = MyEmbedder(model_name=embedding_model)
    store = PgVectorStore(dimensions=embedder.dimensions)
    store.create_schema()

    strategy_metrics: dict[str, Any] = {}

    # Strategy A: embed + upsert + idempotent re-ingestion check
    t0 = time.perf_counter()
    tok_before = embedder.total_tokens
    cost_before = embedder.total_cost
    vectors_a = embedder.embed_documents([c.text for c in chunks_a])
    store.upsert("A", chunks_a, vectors_a)
    t_a = time.perf_counter() - t0
    tokens_a = embedder.total_tokens - tok_before
    cost_a = embedder.total_cost - cost_before

    count_before_reingest = _count_strategy_rows(store.database_url, "A")
    store.upsert("A", chunks_a, vectors_a)
    count_after_reingest = _count_strategy_rows(store.database_url, "A")
    if count_before_reingest != count_after_reingest or count_after_reingest != len(chunks_a):
        raise RuntimeError(
            f"Idempotent re-ingest failed for A: before={count_before_reingest}, after={count_after_reingest}, expected={len(chunks_a)}"
        )

    strategy_metrics["A"] = {
        **stats_a,
        "chunking": "fixed_size(size=500, overlap=50)",
        "filter_enabled": False,
        "ingest_time_s": round(t_a, 3),
        "embed_tokens": tokens_a,
        "embed_cost_usd": round(cost_a, 8),
        "stored_rows": count_after_reingest,
        "idempotent_reingest_verified": True,
    }

    # Strategy B: embed + upsert
    t0 = time.perf_counter()
    tok_before = embedder.total_tokens
    cost_before = embedder.total_cost
    vectors_b = embedder.embed_documents([c.text for c in chunks_b])
    store.upsert("B", chunks_b, vectors_b)
    t_b = time.perf_counter() - t0
    tokens_b = embedder.total_tokens - tok_before
    cost_b = embedder.total_cost - cost_before

    strategy_metrics["B"] = {
        **stats_b,
        "chunking": "by_paragraph(max_size=1050)",
        "filter_enabled": False,
        "ingest_time_s": round(t_b, 3),
        "embed_tokens": tokens_b,
        "embed_cost_usd": round(cost_b, 8),
        "stored_rows": _count_strategy_rows(store.database_url, "B"),
    }

    # Strategy C: shares exact same by_paragraph chunks & vectors as B; only enables SQL metadata filter at retrieval.
    t0 = time.perf_counter()
    store.upsert("C", chunks_b, vectors_b)
    t_c = time.perf_counter() - t0
    strategy_metrics["C"] = {
        **stats_b,
        "chunking": "by_paragraph(max_size=1050)",
        "filter_enabled": True,
        "ingest_time_s": round(t_c, 3),
        "embed_tokens_reused_from_B": tokens_b,
        "incremental_embed_tokens": 0,
        "incremental_embed_cost_usd": 0.0,
        "stored_rows": _count_strategy_rows(store.database_url, "C"),
        "notes": "Shares B chunks and OpenAI vectors so B vs C isolates metadata filtering only.",
    }
    print("[Step 3] Embeddings and pgvector storage complete for A, B, C (idempotency verified on A).")

    # 4. Embed all benchmark questions once and evaluate A, B, C on the identical vectors.
    q_tok_before = embedder.total_tokens
    q_cost_before = embedder.total_cost
    t0 = time.perf_counter()
    query_texts = [str(q["question"]) for q in questions]
    query_vec_list = embedder.embed_documents(query_texts)
    query_vectors = {q["id"]: vec for q, vec in zip(questions, query_vec_list)}
    query_embed_time_s = time.perf_counter() - t0
    query_tokens = embedder.total_tokens - q_tok_before
    query_cost = embedder.total_cost - q_cost_before

    all_traces: list[dict[str, Any]] = []
    strategy_configs = [
        ("A", False),
        ("B", False),
        ("C", True),
    ]

    for strat_name, use_filter in strategy_configs:
        strat_traces: list[dict[str, Any]] = []
        t_search_start = time.perf_counter()
        for q in questions:
            applied_filter = q.get("filter") if use_filter else None
            hits = store.search(
                strategy=strat_name,
                query_vector=query_vectors[q["id"]],
                top_k=3,
                filters=applied_filter,
            )
            trace = evaluate_query(
                question=q,
                hits=hits,
                strategy=strat_name,
                filter_applied=applied_filter,
                top_k=3,
            )
            strat_traces.append(trace)
            all_traces.append(trace)

        strategy_metrics[strat_name]["search_time_s"] = round(time.perf_counter() - t_search_start, 3)
        (RESULTS_DIR / f"traces_{strat_name}.json").write_text(
            json.dumps(strat_traces, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (RESULTS_DIR / "traces_all.json").write_text(
        json.dumps(all_traces, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("[Step 4] Saved full top-3 traces for A, B, C.")

    # 5. Summarize retrieval metrics and prepare samples for manual assessment.
    summary = summarize(all_traces)
    db_versions = _db_environment_versions(store.database_url)
    metrics_payload = {
        "run_started_at": run_started_at,
        "run_completed_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "openai_sdk": _package_version("openai"),
            "psycopg": _package_version("psycopg"),
            "pypdf": _package_version("pypdf"),
            **db_versions,
            "embedding_model": embedder.model_name,
            "embedding_dimensions": embedder.dimensions,
            "distance_metric": "cosine (<=>)",
            "search_mode": "exact sequential scan (no approximate ANN index)",
        },
        "usage": {
            "total_openai_tokens": embedder.total_tokens,
            "total_openai_cost_usd": round(embedder.total_cost, 8),
            "query_embed_tokens": query_tokens,
            "query_embed_cost_usd": round(query_cost, 8),
            "query_embed_time_s": round(query_embed_time_s, 3),
        },
        "strategies": strategy_metrics,
        "retrieval_summary": summary["by_strategy"],
    }
    (RESULTS_DIR / "metrics_summary.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    coherence_samples = _build_coherence_samples(positions_doc, chunks_a, chunks_b)
    (RESULTS_DIR / "coherence_samples.json").write_text(
        json.dumps(coherence_samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    test_audit_samples = _build_test_audit_samples(all_traces)
    (RESULTS_DIR / "test_audit_samples.json").write_text(
        json.dumps(test_audit_samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _write_experiment_log(summary, embedder.model_name)

    print("\n=== Retrieval Summary ===")
    for strat_name in ("A", "B", "C"):
        for split in ("dev", "test"):
            bucket = summary["by_strategy"][strat_name][split]
            fc = (
                f"{bucket['filter_compliant_query_count']}/{bucket['filter_question_count']}"
                if bucket["filter_question_count"] > 0
                else "N/A"
            )
            print(
                f"Strategy {strat_name} [{split:4s}] | "
                f"Hit@1={bucket['hit_at_1_count']}/{bucket['total']} ({bucket['hit_at_1']:.2%}) | "
                f"Evidence Hit@3={bucket['evidence_hit_at_3_count']}/{bucket['total']} ({bucket['evidence_hit_at_3']:.2%}) | "
                f"Page Hit@3={bucket['gold_page_hit_at_3_count']}/{bucket['total']} | "
                f"Filter Compliant={fc}"
            )

    print(
        f"\nTotal OpenAI usage: {embedder.total_tokens} tokens (${embedder.total_cost:.6f} USD). "
        f"Artifacts saved to {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
