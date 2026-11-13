"""Standalone verification script for Module 4: src/evaluation.py."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation import evaluate_query, summarize


def test_hit_at_1_and_whitespace_normalization() -> None:
    print("[1/5] Testing Hit@1 with Unicode/quote/whitespace normalization...")
    question = {
        "id": "q_test_1",
        "split": "dev",
        "question_type": "direct",
        "question": "Why was Vector and Embeddings added?",
        "expected_answer": "Community guidance on RAG.",
        "gold_source_id": "owasp-llm-top10-2025",
        "gold_page": 5,
        "alternative_gold_pages": [],
        "evidence_quote": "The Vector and Embeddings entry responds to the community’s requests for guidance.",
        "alternative_evidence_quotes": [],
        "filter": None,
    }
    hits = [
        {
            "chunk_id": "c1",
            "text": "The Vector and Embeddings entry responds to   the community's\nrequests for guidance.",
            "score": 0.91,
            "metadata": {"strategy": "A", "document_id": "owasp-llm-top10-2025", "page": 5},
        }
    ]
    trace = evaluate_query(question, hits, strategy="A")
    assert trace["hit_at_1"] is True
    assert trace["evidence_hit_at_3"] is True
    assert trace["gold_page_hit_at_3"] is True
    assert trace["evidence_split_across_chunks"] is False
    assert trace["human_grounding_score"] is None
    print("  PASS: Hit@1 and quote normalization verified.")


def test_alternative_gold_pages_and_quotes() -> None:
    print("[2/5] Testing alternative gold pages and alternative quotes...")
    question = {
        "id": "q_test_2",
        "split": "test",
        "question_type": "paraphrase",
        "question": "How does RAG affect emotional intelligence?",
        "expected_answer": "Reduces empathy.",
        "gold_source_id": "owasp-llm-top10-2025",
        "gold_page": 34,
        "alternative_gold_pages": [35],
        "evidence_quote": "Primary quote not retrieved.",
        "alternative_evidence_quotes": [
            "reducing emotional intelligence or empathy in responses."
        ],
        "filter": None,
    }
    hits = [
        {
            "chunk_id": "c1",
            "text": "Unrelated chunk.",
            "score": 0.85,
            "metadata": {"strategy": "B", "document_id": "owasp-llm-top10-2025", "page": 10},
        },
        {
            "chunk_id": "c2",
            "text": "In practice, reducing emotional intelligence or empathy in responses.",
            "score": 0.82,
            "metadata": {"strategy": "B", "document_id": "owasp-llm-top10-2025", "page": 35},
        },
    ]
    trace = evaluate_query(question, hits, strategy="B")
    assert trace["hit_at_1"] is False
    assert trace["evidence_hit_at_3"] is True
    assert trace["hits"][1]["evidence_match"] is True
    print("  PASS: Alternative gold pages and quotes accepted.")


def test_split_evidence_flag() -> None:
    print("[3/5] Testing split-across-chunks detection...")
    question = {
        "id": "q_test_3",
        "split": "test",
        "question_type": "conditional_exception",
        "question": "Long multi-bullet quote?",
        "expected_answer": "Two bullets.",
        "gold_source_id": "owasp-llm-top10-2025",
        "gold_page": 28,
        "alternative_gold_pages": [],
        "evidence_quote": "First bullet item appears here. Second bullet item appears after it.",
        "alternative_evidence_quotes": [],
        "filter": None,
    }
    hits = [
        {
            "chunk_id": "c1",
            "text": "First bullet item appears here.",
            "score": 0.88,
            "metadata": {"strategy": "A", "document_id": "owasp-llm-top10-2025", "page": 28},
        },
        {
            "chunk_id": "c2",
            "text": "Second bullet item appears after it.",
            "score": 0.84,
            "metadata": {"strategy": "A", "document_id": "owasp-llm-top10-2025", "page": 28},
        },
    ]
    trace = evaluate_query(question, hits, strategy="A")
    assert trace["hit_at_1"] is False
    assert trace["evidence_hit_at_3"] is False
    assert trace["gold_page_hit_at_3"] is True
    assert trace["multi_chunk_evidence_hit_at_3"] is True
    assert trace["evidence_split_across_chunks"] is True
    print("  PASS: Primary strict Hit@3 stays False while split-chunk diagnostic flags True.")


def test_filter_compliance_and_summary() -> None:
    print("[4/5] Testing metadata filter compliance and summarize aggregation...")
    q_filter = {
        "id": "q_f",
        "split": "test",
        "question_type": "metadata_filter",
        "question": "Filtered query?",
        "expected_answer": "Mitigation detail.",
        "gold_source_id": "owasp-llm-top10-2025",
        "gold_page": 34,
        "alternative_gold_pages": [],
        "evidence_quote": "Use permission-aware vector stores.",
        "alternative_evidence_quotes": [],
        "filter": {"section_type": "mitigation"},
    }
    hits_b = [
        {
            "chunk_id": "b1",
            "text": "Use permission-aware vector stores.",
            "score": 0.90,
            "metadata": {
                "strategy": "B",
                "document_id": "owasp-llm-top10-2025",
                "page": 34,
                "section_type": "examples",
            },
        }
    ]
    hits_c = [
        {
            "chunk_id": "c1",
            "text": "Use permission-aware vector stores.",
            "score": 0.89,
            "metadata": {
                "strategy": "C",
                "document_id": "owasp-llm-top10-2025",
                "page": 34,
                "section_type": "mitigation",
            },
        }
    ]
    t_b = evaluate_query(q_filter, hits_b, strategy="B", filter_applied=None)
    t_c = evaluate_query(q_filter, hits_c, strategy="C", filter_applied={"section_type": "mitigation"})
    assert t_b["all_hits_filter_compliant"] is False
    assert t_c["all_hits_filter_compliant"] is True

    summary = summarize([t_b, t_c])
    assert summary["by_strategy"]["B"]["test"]["filter_compliance_rate"] == 0.0
    assert summary["by_strategy"]["C"]["test"]["filter_compliance_rate"] == 1.0
    assert summary["by_strategy"]["C"]["dev"]["total"] == 0
    print("  PASS: Filter compliance and dev/test split aggregation verified.")


def test_wrong_page_rejected() -> None:
    print("[5/5] Testing source/page mismatch rejection...")
    question = {
        "id": "q_wrong_page",
        "split": "dev",
        "question_type": "direct",
        "question": "Q",
        "expected_answer": "A",
        "gold_source_id": "owasp-llm-top10-2025",
        "gold_page": 8,
        "alternative_gold_pages": [],
        "evidence_quote": "What is the RAG Triad?",
        "alternative_evidence_quotes": [],
        "filter": None,
    }
    hits = [
        {
            "chunk_id": "ref35",
            "text": "Reference link: What is the RAG Triad?",
            "score": 0.95,
            "metadata": {"strategy": "A", "document_id": "owasp-llm-top10-2025", "page": 35},
        }
    ]
    trace = evaluate_query(question, hits, strategy="A")
    assert trace["hit_at_1"] is False
    assert trace["evidence_hit_at_3"] is False
    assert trace["gold_page_hit_at_3"] is False
    print("  PASS: Quote on wrong page rejected.")


def main() -> None:
    print("==================================================")
    print("Starting Module 4 Evaluation Verification")
    print("==================================================")
    test_hit_at_1_and_whitespace_normalization()
    test_alternative_gold_pages_and_quotes()
    test_split_evidence_flag()
    test_filter_compliance_and_summary()
    test_wrong_page_rejected()
    print("==================================================")
    print("ALL MODULE 4 EVALUATION VERIFICATION CHECKS PASSED")
    print("==================================================")


if __name__ == "__main__":
    main()
