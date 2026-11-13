"""Verify benchmark questions and pre-locked coherence positions against normalized corpus."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

STARTER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STARTER_DIR.parent

ALLOWED_FILTER_SECTIONS = {"description", "examples", "mitigation", "scenarios"}


def main() -> None:
    manifest = json.loads((REPO_ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    included_pages = set(manifest["included_pages"])

    pages_path = STARTER_DIR / "results" / "normalized_pages.jsonl"
    pages_by_num: dict[int, dict] = {}
    for line in pages_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pages_by_num[row["page"]] = row

    bench = json.loads((REPO_ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    questions = bench["questions"]

    assert len(questions) >= 20, f"Expected >= 20 questions, got {len(questions)}"
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "Duplicate question IDs found"
    texts = [q["question"].strip().lower() for q in questions]
    assert len(texts) == len(set(texts)), "Duplicate question texts found"

    dev_qs = [q for q in questions if q["split"] == "dev"]
    test_qs = [q for q in questions if q["split"] == "test"]
    assert len(dev_qs) >= 15, f"Expected >= 15 dev questions, got {len(dev_qs)}"
    assert len(test_qs) >= 5, f"Expected >= 5 test questions, got {len(test_qs)}"

    dev_risks = {q["gold_risk_id"] for q in dev_qs}
    test_risks = {q["gold_risk_id"] for q in test_qs}
    overlap_risks = dev_risks & test_risks
    assert not overlap_risks, f"Topic leakage between dev and test: {overlap_risks}"

    filter_qs = [q for q in questions if q.get("filter") is not None]
    test_filter_qs = [q for q in test_qs if q.get("filter") is not None]
    assert len(filter_qs) >= 3, f"Expected >= 3 filter questions, got {len(filter_qs)}"
    assert len(test_filter_qs) >= 1, f"Expected >= 1 test filter question, got {len(test_filter_qs)}"

    for q in questions:
        assert q["gold_source_id"] == manifest["source_id"], f"{q['id']}: bad source_id"
        gp = q["gold_page"]
        assert gp in included_pages, f"{q['id']}: gold_page {gp} not in manifest included_pages"
        page_text = pages_by_num[gp]["text"]
        assert q["evidence_quote"] in page_text, (
            f"{q['id']}: primary evidence_quote not found verbatim on page {gp}"
        )

        alt_pages = q.get("alternative_gold_pages", [])
        alt_quotes = q.get("alternative_evidence_quotes", [])
        assert len(alt_pages) == len(alt_quotes), f"{q['id']}: alt pages/quotes length mismatch"
        for ap, aq in zip(alt_pages, alt_quotes):
            assert ap in included_pages, f"{q['id']}: alt page {ap} not in manifest"
            assert aq in pages_by_num[ap]["text"], (
                f"{q['id']}: alt evidence_quote not found verbatim on page {ap}"
            )

        if q.get("filter") is not None:
            flt = q["filter"]
            assert set(flt.keys()) == {"section_type"}, f"{q['id']}: unexpected filter keys {flt}"
            assert flt["section_type"] in ALLOWED_FILTER_SECTIONS, f"{q['id']}: invalid section_type"
            assert flt["section_type"] == q["gold_section_type"], (
                f"{q['id']}: filter section_type mismatches gold_section_type"
            )

    # Verify 10 pre-locked Chunk Coherence positions
    coh = json.loads((REPO_ROOT / "benchmark" / "coherence_positions.json").read_text(encoding="utf-8"))
    positions = coh["positions"]
    assert len(positions) == 10, f"Expected 10 coherence positions, got {len(positions)}"
    pos_ids = [p["position_id"] for p in positions]
    assert len(pos_ids) == len(set(pos_ids)), "Duplicate position_ids"
    for pos in positions:
        pg = pos["page"]
        assert pg in included_pages, f"{pos['position_id']}: page {pg} not in included_pages"
        assert pos["anchor_quote"] in pages_by_num[pg]["text"], (
            f"{pos['position_id']}: anchor_quote not found verbatim on page {pg}"
        )

    print(f"Benchmark verified: {len(questions)} questions ({len(dev_qs)} dev / {len(test_qs)} test)")
    print(f"Dev risk_ids: {sorted(dev_risks)}")
    print(f"Test risk_ids (holdout): {sorted(test_risks)}")
    print(f"Filter questions: {len(filter_qs)} total ({len(test_filter_qs)} in test)")
    print("Question types:", dict(Counter(q["question_type"] for q in questions)))
    print("Gold section types:", dict(Counter(q["gold_section_type"] for q in questions)))
    print(f"Coherence positions verified: {len(positions)}/10")


if __name__ == "__main__":
    main()
