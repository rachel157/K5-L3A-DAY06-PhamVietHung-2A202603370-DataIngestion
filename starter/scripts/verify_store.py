"""Standalone verification script for Module 3: PgVectorStore.

Tests:
1. Idempotent schema creation (vector extension, document_chunks table, indices).
2. Upsert and duplicate prevention on re-ingestion.
3. Stale chunk cleanup during document re-ingestion.
4. Cosine similarity search accuracy (top-k nearest neighbors).
5. Parameterized SQL metadata filtering by section_type (including SQL injection safety).
6. Input dimension validation.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env")

import psycopg
from src.models import Chunk
from src.store import PgVectorStore

TEST_STRATEGY = "test_verification_strategy"
DIMENSIONS = 1536


def make_unit_vector(active_index: int, total_dims: int = DIMENSIONS) -> list[float]:
    """Create an orthogonal unit vector with 1.0 at active_index and 0.0 elsewhere."""
    vec = [0.0] * total_dims
    vec[active_index % total_dims] = 1.0
    return vec


def count_chunks_for_strategy(database_url: str, strategy: str) -> int:
    with psycopg.connect(database_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks WHERE strategy = %s;", (strategy,))
            row = cur.fetchone()
            return int(row[0]) if row else 0


def get_chunk_ids_for_strategy(database_url: str, strategy: str) -> set[str]:
    with psycopg.connect(database_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id FROM document_chunks WHERE strategy = %s;", (strategy,))
            rows = cur.fetchall()
            return {r[0] for r in rows}


def test_schema_creation_idempotency(store: PgVectorStore) -> None:
    print("[1/6] Testing idempotent schema creation...")
    # Call once
    store.create_schema()
    # Call again to verify idempotency
    store.create_schema()

    # Verify table and extension in database
    with psycopg.connect(store.database_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            ext = cur.fetchone()
            assert ext and ext[0] == "vector", "vector extension missing from pg_extension"

            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_name = 'document_chunks';
            """)
            tbl = cur.fetchone()
            assert tbl and tbl[0] == "document_chunks", "document_chunks table missing from database"

    print("  PASS: create_schema is fully idempotent and verifies vector extension & table.")


def test_upsert_and_duplicate_prevention(store: PgVectorStore) -> None:
    print("[2/6] Testing batch upsert and duplicate prevention on re-ingestion...")
    store.delete_strategy(TEST_STRATEGY)

    # Create 5 synthetic chunks
    chunks = [
        Chunk(
            chunk_id=f"doc1:test:c{i:03d}",
            document_id="doc1",
            source="test.pdf",
            page=i,
            text=f"Sample text content for chunk {i}",
            metadata={"section_type": sec, "index": i},
        )
        for i, sec in enumerate(["overview", "mitigation", "description", "mitigation", "examples"], start=1)
    ]
    vectors = [make_unit_vector(i) for i in range(1, 6)]

    # Initial upsert
    store.upsert(TEST_STRATEGY, chunks, vectors)
    count1 = count_chunks_for_strategy(store.database_url, TEST_STRATEGY)
    assert count1 == 5, f"Expected 5 chunks after initial upsert, found {count1}"

    # Re-ingest the exact same chunks with modified text
    modified_chunks = [
        Chunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            source=c.source,
            page=c.page,
            text=f"Updated text content for chunk {c.chunk_id}",
            metadata=c.metadata,
        )
        for c in chunks
    ]
    store.upsert(TEST_STRATEGY, modified_chunks, vectors)
    count2 = count_chunks_for_strategy(store.database_url, TEST_STRATEGY)
    assert count2 == 5, f"Expected exactly 5 chunks on re-ingest (no duplicates), found {count2}"

    # Verify that text was updated in place
    with psycopg.connect(store.database_url, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT text FROM document_chunks WHERE strategy = %s AND chunk_id = %s;",
                (TEST_STRATEGY, chunks[0].chunk_id),
            )
            row = cur.fetchone()
            assert row and "Updated text" in row[0], "Updated chunk text was not persisted"

    print("  PASS: Ingesting chunks repeatedly prevents duplicate rows and updates in place.")


def test_stale_chunk_cleanup(store: PgVectorStore) -> None:
    print("[3/6] Testing stale chunk cleanup on re-ingestion...")
    # Original chunks were c001, c002, c003, c004, c005
    # New batch retains c001, c002, c003, omits c004 and c005, and adds c006
    retained_and_new = [
        Chunk(
            chunk_id=f"doc1:test:c{i:03d}",
            document_id="doc1",
            source="test.pdf",
            page=i,
            text=f"Text for chunk {i}",
            metadata={"section_type": sec},
        )
        for i, sec in [(1, "overview"), (2, "mitigation"), (3, "description"), (6, "scenarios")]
    ]
    vectors = [make_unit_vector(i) for i in [1, 2, 3, 6]]

    store.upsert(TEST_STRATEGY, retained_and_new, vectors)

    current_ids = get_chunk_ids_for_strategy(store.database_url, TEST_STRATEGY)
    assert len(current_ids) == 4, f"Expected 4 chunks after stale removal, got {len(current_ids)}: {current_ids}"
    assert "doc1:test:c004" not in current_ids, "Stale chunk c004 was not removed"
    assert "doc1:test:c005" not in current_ids, "Stale chunk c005 was not removed"
    assert "doc1:test:c006" in current_ids, "New chunk c006 was not added"
    assert {"doc1:test:c001", "doc1:test:c002", "doc1:test:c003"}.issubset(current_ids)

    print("  PASS: Obsolete chunks removed cleanly while new/retained chunks are preserved.")


def test_cosine_search_unfiltered(store: PgVectorStore) -> None:
    print("[4/6] Testing cosine search without filter...")
    # Search with vector identical to chunk c002 (active_index=2)
    query_vector = make_unit_vector(2)
    hits = store.search(TEST_STRATEGY, query_vector, top_k=3)

    assert len(hits) == 3, f"Expected 3 hits, got {len(hits)}"
    top_hit = hits[0]
    assert top_hit["chunk_id"] == "doc1:test:c002", f"Expected c002 as top hit, got {top_hit['chunk_id']}"
    assert top_hit["score"] > 0.999, f"Expected score ~ 1.0, got {top_hit['score']}"
    assert "text" in top_hit
    assert "metadata" in top_hit
    meta = top_hit["metadata"]
    assert meta["strategy"] == TEST_STRATEGY
    assert meta["document_id"] == "doc1"
    assert meta["source"] == "test.pdf"
    assert meta["page"] == 2
    assert meta["section_type"] == "mitigation"

    print(f"  Top hit: {top_hit['chunk_id']} | score: {top_hit['score']:.4f} | section: {meta['section_type']}")
    print("  PASS: Cosine search correctly ranks nearest neighbors and returns enriched metadata.")


def test_cosine_search_filtered_by_section_type(store: PgVectorStore) -> None:
    print("[5/6] Testing parameterized SQL search with section_type filter...")
    # Query vector matches chunk c001 (active_index=1, section_type="overview")
    query_vector = make_unit_vector(1)

    # 1. Without filter: c001 should be #1
    unfiltered = store.search(TEST_STRATEGY, query_vector, top_k=2)
    assert unfiltered[0]["chunk_id"] == "doc1:test:c001"

    # 2. With filter: section_type="mitigation"
    # Even though c001 has highest vector similarity, it must be excluded!
    filtered = store.search(
        TEST_STRATEGY,
        query_vector,
        top_k=3,
        filters={"section_type": "mitigation"},
    )
    assert len(filtered) > 0, "Expected at least 1 hit with section_type='mitigation'"
    for hit in filtered:
        sec = hit["metadata"].get("section_type")
        assert sec == "mitigation", f"Returned chunk {hit['chunk_id']} has section_type '{sec}', expected 'mitigation'"
        assert hit["chunk_id"] != "doc1:test:c001", "c001 must not be in filtered results"

    # 3. Filter with 0 matches
    empty_hits = store.search(
        TEST_STRATEGY,
        query_vector,
        top_k=3,
        filters={"section_type": "nonexistent_section_xyz"},
    )
    assert len(empty_hits) == 0, f"Expected 0 hits for non-existent section, got {len(empty_hits)}"

    # 4. SQL Injection safety test
    sql_injection_filter = {"section_type": "mitigation' OR '1'='1"}
    safe_hits = store.search(TEST_STRATEGY, query_vector, top_k=3, filters=sql_injection_filter)
    assert len(safe_hits) == 0, f"SQL injection attempt returned unexpected hits: {safe_hits}"

    print("  PASS: section_type SQL filtering works correctly and protects against injection.")


def test_dimension_validation(store: PgVectorStore) -> None:
    print("[6/6] Testing dimension mismatch validations...")
    bad_vector = [0.1] * 100  # wrong dimensions

    dummy_chunk = Chunk(
        chunk_id="bad:dim:001",
        document_id="doc1",
        source="test.pdf",
        page=1,
        text="Sample",
    )

    try:
        store.upsert(TEST_STRATEGY, [dummy_chunk], [bad_vector])
        raise AssertionError("FAILURE: upsert did not raise ValueError on dimension mismatch")
    except ValueError as exc:
        print(f"  Upsert dimension mismatch caught as expected: {exc}")

    try:
        store.search(TEST_STRATEGY, bad_vector, top_k=3)
        raise AssertionError("FAILURE: search did not raise ValueError on dimension mismatch")
    except ValueError as exc:
        print(f"  Search dimension mismatch caught as expected: {exc}")

    print("  PASS: Dimension mismatches rejected with explicit ValueError.")


def main() -> None:
    print("==================================================")
    print("Starting Module 3 PgVectorStore Verification")
    print("==================================================")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        sys.exit("Error: DATABASE_URL is not set.")

    store = PgVectorStore(database_url=database_url, dimensions=DIMENSIONS)

    try:
        test_schema_creation_idempotency(store)
        test_upsert_and_duplicate_prevention(store)
        test_stale_chunk_cleanup(store)
        test_cosine_search_unfiltered(store)
        test_cosine_search_filtered_by_section_type(store)
        test_dimension_validation(store)
    finally:
        # Clean up test rows
        store.delete_strategy(TEST_STRATEGY)
        print("Test data cleaned up successfully.")

    print("==================================================")
    print("ALL MODULE 3 VECTOR STORE VERIFICATION CHECKS PASSED")
    print("==================================================")


if __name__ == "__main__":
    main()
