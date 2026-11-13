"""Store and search vectors using PostgreSQL + pgvector."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

from .models import Chunk

_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV_PATH)

logger = logging.getLogger("store")


class PgVectorStore:
    def __init__(self, database_url: str | None = None, dimensions: int = 1536) -> None:
        """Retain connection settings; validate model dimensions."""
        resolved_url = database_url or os.getenv("DATABASE_URL")
        if not resolved_url or not resolved_url.strip():
            raise ValueError("DATABASE_URL is not configured or empty.")

        if dimensions <= 0:
            raise ValueError(f"dimensions must be a positive integer, got {dimensions}")

        self.database_url = resolved_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        """Create vector extension, document_chunks table, and indices idempotently."""
        with psycopg.connect(self.database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                # Idempotent extension creation
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                # Idempotent table creation with PRIMARY KEY (strategy, chunk_id)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        strategy VARCHAR(64) NOT NULL,
                        chunk_id VARCHAR(255) NOT NULL,
                        document_id VARCHAR(255) NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        section_type VARCHAR(64),
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.dimensions}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (strategy, chunk_id)
                    );
                """)

                # Helpful lookup indices
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_document_chunks_strategy_doc
                    ON document_chunks (strategy, document_id);
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_document_chunks_strategy_section
                    ON document_chunks (strategy, section_type);
                """)
            conn.commit()
        logger.info(f"Schema created / verified on database: dimensions={self.dimensions}")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Validate lengths/dimensions, deduplicate in-memory, remove stale chunks, and upsert batch."""
        if not strategy or not strategy.strip():
            raise ValueError("strategy cannot be empty.")

        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatched lengths: {len(chunks)} chunks vs {len(vectors)} vectors.")

        if not chunks:
            return

        # Validate vector dimensions
        for i, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise ValueError(
                    f"Vector at index {i} has dimension {len(vec)}, expected {self.dimensions}."
                )

        # In-memory deduplication by (strategy, chunk_id) to prevent duplicate row conflict in single batch
        deduped: dict[str, tuple[Chunk, list[float]]] = {}
        for chunk, vec in zip(chunks, vectors):
            cid = chunk.chunk_id or f"{chunk.document_id}:{strategy}:p{chunk.page}:{hash(chunk.text)}"
            deduped[cid] = (chunk, vec)
        unique_items = list(deduped.values())

        # Collect distinct document IDs and incoming chunk IDs for stale chunk cleanup
        doc_ids = list({chunk.document_id for chunk, _ in unique_items})
        incoming_chunk_ids = [chunk.chunk_id for chunk, _ in unique_items]

        with psycopg.connect(self.database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                # 1. Stale chunk cleanup inside the same transaction:
                # Delete obsolete chunks previously stored for the given strategy and document_ids
                # that are not present in the current batch.
                cur.execute(
                    """
                    DELETE FROM document_chunks
                    WHERE strategy = %(strategy)s
                      AND document_id = ANY(%(doc_ids)s)
                      AND chunk_id != ALL(%(incoming_chunk_ids)s);
                    """,
                    {
                        "strategy": strategy,
                        "doc_ids": doc_ids,
                        "incoming_chunk_ids": incoming_chunk_ids,
                    },
                )

                # 2. Batch upsert with ON CONFLICT (strategy, chunk_id) DO UPDATE
                upsert_query = """
                    INSERT INTO document_chunks (
                        strategy, chunk_id, document_id, source, page, section_type, text, metadata, embedding, updated_at
                    ) VALUES (
                        %(strategy)s, %(chunk_id)s, %(document_id)s, %(source)s, %(page)s,
                        %(section_type)s, %(text)s, %(metadata)s::jsonb, %(embedding)s::vector, NOW()
                    )
                    ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        source = EXCLUDED.source,
                        page = EXCLUDED.page,
                        section_type = EXCLUDED.section_type,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW();
                """

                params_list = []
                for chunk, vec in unique_items:
                    cid = chunk.chunk_id or f"{chunk.document_id}:{strategy}:p{chunk.page}:{hash(chunk.text)}"
                    section_type = chunk.metadata.get("section_type") if chunk.metadata else None
                    meta_json = json.dumps(chunk.metadata or {})
                    vec_str = json.dumps(vec)

                    params_list.append({
                        "strategy": strategy,
                        "chunk_id": cid,
                        "document_id": chunk.document_id,
                        "source": chunk.source,
                        "page": chunk.page,
                        "section_type": section_type,
                        "text": chunk.text,
                        "metadata": meta_json,
                        "embedding": vec_str,
                    })

                cur.executemany(upsert_query, params_list)
            conn.commit()

        logger.info(f"Upserted {len(unique_items)} chunks for strategy='{strategy}'.")

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter permitted metadata in SQL, rank by cosine distance, return top-k."""
        if not strategy or not strategy.strip():
            raise ValueError("strategy cannot be empty.")

        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} does not match store dimension {self.dimensions}."
            )

        if top_k <= 0:
            return []

        # Prepare filter clause and parameters
        filter_sql = ""
        params: dict[str, Any] = {
            "strategy": strategy,
            "query_vec": json.dumps(query_vector),
            "top_k": top_k,
        }

        if filters:
            if "section_type" in filters and filters["section_type"] is not None:
                filter_sql += " AND section_type = %(section_type)s"
                params["section_type"] = filters["section_type"]
            if "document_id" in filters and filters["document_id"] is not None:
                filter_sql += " AND document_id = %(document_id)s"
                params["document_id"] = filters["document_id"]

        query = f"""
            SELECT
                chunk_id,
                text,
                document_id,
                source,
                page,
                section_type,
                metadata,
                (1.0 - (embedding <=> %(query_vec)s::vector)) AS score
            FROM document_chunks
            WHERE strategy = %(strategy)s
            {filter_sql}
            ORDER BY embedding <=> %(query_vec)s::vector ASC
            LIMIT %(top_k)s;
        """

        with psycopg.connect(self.database_url, connect_timeout=5, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            meta_dict = dict(meta) if isinstance(meta, dict) else {}
            # Overlay first-class columns into metadata for traceability
            meta_dict["strategy"] = strategy
            meta_dict["document_id"] = row["document_id"]
            meta_dict["source"] = row["source"]
            meta_dict["page"] = row["page"]
            meta_dict["section_type"] = row["section_type"]

            results.append({
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "score": float(row["score"]),
                "metadata": meta_dict,
            })

        return results

    def delete_strategy(self, strategy: str) -> int:
        """Delete all chunks for a strategy (useful for cleanup)."""
        with psycopg.connect(self.database_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE strategy = %s;", (strategy,))
                deleted = cur.rowcount
            conn.commit()
        return deleted
