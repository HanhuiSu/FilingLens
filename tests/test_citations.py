"""Unit tests for citation construction and policy boundaries."""

from __future__ import annotations

from src.agent.citations import (
    apply_text_citation_policy,
    collect_citations_from_claims,
    numeric_citation_from_evidence,
    text_citation_from_evidence,
)


def test_numeric_and_text_citation_shapes_are_distinct():
    numeric = numeric_citation_from_evidence(
        {
            "evidence_id": "N1",
            "ticker": "AAPL",
            "metric": "revenue",
            "period_type": "quarterly",
            "period_end": "2025-09-27",
        }
    )
    text = text_citation_from_evidence(
        {
            "evidence_id": "T1",
            "ticker": "AAPL",
            "filing_id": "f1",
            "form_type": "10-Q",
            "fiscal_period": "2025Q3",
            "section": "ITEM_2",
            "text_snippet": "Revenue discussion.",
            "supporting_snippet": "Revenue discussion.",
            "supporting_terms": ["revenue"],
            "score_breakdown": {"lexical_bonus": 0.04},
        }
    )

    assert numeric["source_kind"] == "structured"
    assert numeric["section"] == "STRUCTURED"
    assert text["source_kind"] == "document"
    assert text["section"] == "ITEM_2"
    assert text["supporting_snippet"] == "Revenue discussion."
    assert text["supporting_terms"] == ["revenue"]
    assert text["score_breakdown"]["lexical_bonus"] == 0.04


def test_collect_citations_from_claims_uses_claim_evidence_ids():
    numeric_citations, text_citations = collect_citations_from_claims(
        numeric_claims=[{"sentence": "AAPL revenue was 100.", "evidence_ids": ["N1"]}],
        text_claims=[{"sentence": "Demand was strong.", "evidence_ids": ["T1"]}],
        numeric_evidence=[{"evidence_id": "N1", "ticker": "AAPL", "metric": "revenue"}],
        text_evidence=[{"evidence_id": "T1", "ticker": "AAPL", "section": "ITEM_2"}],
    )

    assert [c["evidence_id"] for c in numeric_citations] == ["N1"]
    assert [c["evidence_id"] for c in text_citations] == ["T1"]


def test_apply_text_citation_policy_demotes_mixed_sections_and_caps():
    selected, reasons, unbalanced = apply_text_citation_policy(
        {"task_type": "fact_qa"},
        [
            {"ticker": "AAPL", "filing_id": "f1", "fiscal_period": "2025", "section": "MIXED", "chunk_order": 1},
            {"ticker": "AAPL", "filing_id": "f1", "fiscal_period": "2025", "section": "ITEM_1A", "chunk_order": 2},
            {"ticker": "AAPL", "filing_id": "f1", "fiscal_period": "2025", "section": "ITEM_7", "chunk_order": 3},
        ],
    )

    assert unbalanced is False
    assert reasons == []
    assert [c["section"] for c in selected] == ["ITEM_1A", "ITEM_7"]
