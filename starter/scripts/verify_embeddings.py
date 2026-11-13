"""Standalone verification script for Module 2: MyEmbedder.

Tests:
1. Fail fast on missing/empty credentials (no silent mock fallback).
2. Initialization and dimensions with valid credentials (1536 dims).
3. Batched document embedding preserving exact index order.
4. Single query embedding (1536 dims).
5. Token usage accumulation and cost calculation ($0.02 / 1M tokens).
6. Edge case handling (empty list).
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from dotenv import load_dotenv
load_dotenv(PROJECT_DIR / ".env")

from src.embeddings import MyEmbedder


def test_missing_credentials_fails_fast() -> None:
    print("[1/5] Testing strict failure on missing/empty credentials...")
    try:
        MyEmbedder(model_name="text-embedding-3-small", api_key="")
        raise AssertionError("FAILURE: MyEmbedder initialized with empty API key without raising an error!")
    except ValueError as exc:
        print(f"  Expected ValueError caught: {exc}")

    # Also test with whitespace-only key
    try:
        MyEmbedder(model_name="text-embedding-3-small", api_key="   ")
        raise AssertionError("FAILURE: MyEmbedder initialized with whitespace API key without raising an error!")
    except ValueError as exc:
        print(f"  Expected ValueError caught: {exc}")

    print("  PASS: Missing/empty credentials fail fast; no mock fallback.")


def test_embedding_dimensions_and_types() -> None:
    print("[2/5] Testing embedding dimensions and float types...")
    embedder = MyEmbedder(model_name="text-embedding-3-small")
    assert embedder.dimensions == 1536, f"Expected 1536 dimensions, got {embedder.dimensions}"
    assert embedder.model_name == "text-embedding-3-small"

    sample_docs = [
        "LLM01: Prompt Injection vulnerabilities occur when an attacker manipulates prompt inputs.",
        "LLM02: Sensitive Information Disclosure occurs when confidential information is leaked.",
    ]
    doc_vectors = embedder.embed_documents(sample_docs)
    assert len(doc_vectors) == 2, f"Expected 2 document vectors, got {len(doc_vectors)}"
    for idx, vec in enumerate(doc_vectors):
        assert len(vec) == 1536, f"Vector {idx} dimension mismatch: expected 1536, got {len(vec)}"
        assert all(isinstance(val, float) for val in vec), f"Vector {idx} contains non-float values"
        # Check normalized vector length is close to 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 0.02, f"Vector {idx} L2 norm {norm} is not ~1.0"

    # Vectors for different texts should not be identical
    assert doc_vectors[0] != doc_vectors[1], "Vectors for distinct texts should not be identical"
    print("  PASS: Document embeddings produce 1536-dim normalized float vectors.")


def test_query_embedding() -> None:
    print("[3/5] Testing query embedding...")
    embedder = MyEmbedder(model_name="text-embedding-3-small")
    query = "How to prevent prompt injection?"
    q_vec = embedder.embed_query(query)

    assert isinstance(q_vec, list), f"Expected list[float], got {type(q_vec)}"
    assert len(q_vec) == 1536, f"Expected 1536 dimensions, got {len(q_vec)}"
    assert all(isinstance(val, float) for val in q_vec), "Query vector contains non-float values"
    norm = math.sqrt(sum(v * v for v in q_vec))
    assert abs(norm - 1.0) < 0.02, f"Query vector L2 norm {norm} is not ~1.0"
    print("  PASS: Query embedding produces 1536-dim normalized float vector.")


def test_token_tracking_and_cost_calculation() -> None:
    print("[4/5] Testing cumulative token and cost tracking...")
    embedder = MyEmbedder(model_name="text-embedding-3-small")
    initial_tokens = embedder.total_tokens
    initial_cost = embedder.total_cost
    initial_calls = embedder.total_calls
    assert initial_tokens == 0
    assert initial_cost == 0.0
    assert initial_calls == 0

    embedder.embed_documents(["First sample batch text.", "Second sample batch text."])
    tokens_after_first = embedder.total_tokens
    cost_after_first = embedder.total_cost
    calls_after_first = embedder.total_calls

    assert tokens_after_first > 0, "total_tokens must be > 0 after embedding"
    assert cost_after_first > 0.0, "total_cost must be > 0.0 after embedding"
    assert calls_after_first == 1

    expected_cost = (tokens_after_first / 1_000_000.0) * 0.02
    assert abs(cost_after_first - expected_cost) < 1e-9, (
        f"Cost mismatch: expected {expected_cost}, got {cost_after_first}"
    )

    # Make second call (query) and check accumulation
    embedder.embed_query("A short follow-up query text.")
    assert embedder.total_tokens > tokens_after_first, "Tokens must accumulate across calls"
    assert embedder.total_cost > cost_after_first, "Cost must accumulate across calls"
    assert embedder.total_calls == 2, "Call count must accumulate"

    cumulative_expected = (embedder.total_tokens / 1_000_000.0) * 0.02
    assert abs(embedder.total_cost - cumulative_expected) < 1e-9

    print(f"  Recorded tokens: {embedder.total_tokens} tokens across {embedder.total_calls} calls")
    print(f"  Calculated cost: ${embedder.total_cost:.8f} USD")
    print("  PASS: Token consumption and costs correctly accumulated and verified.")


def test_edge_cases() -> None:
    print("[5/5] Testing edge cases...")
    embedder = MyEmbedder(model_name="text-embedding-3-small")

    # Empty list
    empty_result = embedder.embed_documents([])
    assert empty_result == [], f"Expected [], got {empty_result}"

    # Empty query raises ValueError
    try:
        embedder.embed_query("")
        raise AssertionError("FAILURE: embed_query('') did not raise ValueError")
    except ValueError:
        pass

    try:
        embedder.embed_query("   ")
        raise AssertionError("FAILURE: embed_query('   ') did not raise ValueError")
    except ValueError:
        pass

    print("  PASS: Edge cases handled cleanly.")


def main() -> None:
    print("==================================================")
    print("Starting Module 2 Embeddings Verification")
    print("==================================================")
    test_missing_credentials_fails_fast()
    test_embedding_dimensions_and_types()
    test_query_embedding()
    test_token_tracking_and_cost_calculation()
    test_edge_cases()
    print("==================================================")
    print("ALL MODULE 2 EMBEDDINGS VERIFICATION CHECKS PASSED")
    print("==================================================")


if __name__ == "__main__":
    main()
