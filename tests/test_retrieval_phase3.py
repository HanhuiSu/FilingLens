"""Phase 3 retrieval policy and citation-governance unit tests."""

from __future__ import annotations

import numpy as np


def test_risk_profile_prefers_item_1a_over_item_7():
    from src.tools.search_filings import _select_with_profile

    candidates = [
        {"filing_id": "f1", "section": "ITEM_7", "quality": "high", "fiscal_period": "2025-12-31"},
        {"filing_id": "f2", "section": "ITEM_1A", "quality": "high", "fiscal_period": "2025-12-31"},
        {"filing_id": "f3", "section": "ITEM_2", "quality": "high", "fiscal_period": "2025-12-31"},
    ]
    embs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=float)
    q = np.array([1.0, 0.0], dtype=float)
    picked = _select_with_profile(
        candidates=candidates,
        candidate_embs=embs,
        query_emb=q,
        top_k=3,
        lambda_mult=0.8,
        retrieval_profile="risk_summary",
        target_periods=[],
        max_per_filing=0,
        max_per_section=0,
    )
    assert picked
    assert picked[0]["section"] == "ITEM_1A"
    assert any(p["section"] == "ITEM_7" for p in picked[1:])


def test_summary_profile_balances_multiple_section_groups():
    from src.tools.search_filings import _section_group, _select_with_profile

    candidates = [
        {"filing_id": "f1", "section": "ITEM_2", "quality": "high", "fiscal_period": "2025-12-31"},
        {"filing_id": "f2", "section": "ITEM_2", "quality": "high", "fiscal_period": "2025-12-31"},
        {"filing_id": "f3", "section": "ITEM_1A", "quality": "high", "fiscal_period": "2025-12-31"},
        {"filing_id": "f4", "section": "ITEM_8", "quality": "high", "fiscal_period": "2025-12-31"},
    ]
    embs = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.98, 0.01]], dtype=float)
    q = np.array([1.0, 0.0], dtype=float)
    picked = _select_with_profile(
        candidates=candidates,
        candidate_embs=embs,
        query_emb=q,
        top_k=3,
        lambda_mult=0.8,
        retrieval_profile="summary",
        target_periods=[],
        max_per_filing=0,
        max_per_section=2,
    )
    groups = {_section_group(str(p.get("section", ""))) for p in picked}
    assert len(groups) >= 2


def test_repetition_caps_limit_single_filing_and_section():
    from src.tools.search_filings import _select_with_profile

    candidates = [
        {"filing_id": "f1", "section": "ITEM_2", "quality": "high", "fiscal_period": "2025-12-31"},
        {"filing_id": "f1", "section": "ITEM_2", "quality": "high", "fiscal_period": "2025-09-30"},
        {"filing_id": "f1", "section": "ITEM_2", "quality": "high", "fiscal_period": "2025-06-30"},
        {"filing_id": "f2", "section": "ITEM_7", "quality": "high", "fiscal_period": "2025-12-31"},
        {"filing_id": "f3", "section": "ITEM_1A", "quality": "high", "fiscal_period": "2025-12-31"},
    ]
    embs = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03], [0.96, 0.04]], dtype=float)
    q = np.array([1.0, 0.0], dtype=float)
    picked = _select_with_profile(
        candidates=candidates,
        candidate_embs=embs,
        query_emb=q,
        top_k=4,
        lambda_mult=0.8,
        retrieval_profile="summary",
        target_periods=[],
        max_per_filing=1,
        max_per_section=1,
    )
    filing_counts: dict[str, int] = {}
    section_counts: dict[str, int] = {}
    for row in picked:
        filing = str(row.get("filing_id", ""))
        section = str(row.get("section", ""))
        filing_counts[filing] = filing_counts.get(filing, 0) + 1
        section_counts[section] = section_counts.get(section, 0) + 1
    assert all(v <= 1 for v in filing_counts.values())
    assert all(v <= 1 for v in section_counts.values())


def test_mixed_sections_are_not_primary_when_alternatives_exist():
    from src.tools.search_filings import _select_with_profile

    candidates = [
        {"filing_id": "f1", "section": "MIXED", "quality": "high", "fiscal_period": "2025-12-31"},
        {"filing_id": "f2", "section": "ITEM_2", "quality": "high", "fiscal_period": "2025-12-31"},
        {"filing_id": "f3", "section": "ITEM_1A", "quality": "high", "fiscal_period": "2025-12-31"},
    ]
    embs = np.array([[1.0, 0.0], [0.96, 0.04], [0.95, 0.05]], dtype=float)
    q = np.array([1.0, 0.0], dtype=float)
    picked = _select_with_profile(
        candidates=candidates,
        candidate_embs=embs,
        query_emb=q,
        top_k=3,
        lambda_mult=0.85,
        retrieval_profile="fact_support",
        target_periods=[],
        max_per_filing=0,
        max_per_section=0,
    )
    assert len(picked) >= 2
    assert picked[0]["section"] != "MIXED"
    assert picked[1]["section"] != "MIXED"


def test_lexical_bonus_and_score_breakdown_are_exposed():
    from src.tools.search_filings import _select_with_profile

    candidates = [
        {
            "filing_id": "f1",
            "section": "ITEM_7",
            "quality": "high",
            "fiscal_period": "2025-12-31",
            "text": "Cloud revenue increased as enterprise demand improved.",
        },
        {
            "filing_id": "f2",
            "section": "ITEM_7",
            "quality": "high",
            "fiscal_period": "2025-12-31",
            "text": "General business discussion without the requested terms.",
        },
    ]
    embs = np.array([[0.99, 0.01], [1.0, 0.0]], dtype=float)
    q = np.array([1.0, 0.0], dtype=float)
    picked = _select_with_profile(
        candidates=candidates,
        candidate_embs=embs,
        query_emb=q,
        top_k=2,
        lambda_mult=0.8,
        retrieval_profile="trend_support",
        target_periods=[],
        max_per_filing=0,
        max_per_section=0,
        query="cloud revenue demand",
    )

    assert picked[0]["filing_id"] == "f1"
    assert picked[0]["score_breakdown"]["lexical_bonus"] > 0
    assert "vector_score" in picked[0]["score_breakdown"]
    assert picked[0]["score_breakdown"]["final_score"] == picked[0]["final_score"]


def test_supporting_snippet_prefers_sentence_with_query_terms():
    from src.tools.search_filings import _supporting_snippet

    snippet, terms = _supporting_snippet(
        "Generic opening sentence. Competition and regulation are material risk factors for the business.",
        query="competition regulation risk",
        ticker="AAPL",
    )

    assert "Competition" in snippet
    assert {"competition", "regulation", "risk"} <= set(terms)


def test_comparison_text_policy_requires_balanced_company_coverage():
    from src.agent.nodes import _apply_text_citation_policy

    state = {
        "task_type": "company_comparison",
        "companies": ["AAPL"],
        "comparison_target": "MSFT",
        "retrieval_policy": {
            "comparison_text_cap_per_company": 2,
            "require_balanced_comparison_text": True,
        },
    }
    text_citations = [
        {"ticker": "AAPL", "section": "ITEM_2", "fiscal_period": "2025-12-31", "chunk_order": 1, "quality": "high", "score": 0.8, "evidence_id": "T1"},
        {"ticker": "AAPL", "section": "ITEM_1A", "fiscal_period": "2025-12-31", "chunk_order": 2, "quality": "high", "score": 0.7, "evidence_id": "T2"},
    ]
    filtered, reasons, unbalanced = _apply_text_citation_policy(state, text_citations=text_citations)
    assert filtered == []
    assert unbalanced is True
    assert any("comparison_text_evidence_unbalanced" in r.get("reason", "") for r in reasons)


def test_comparison_text_policy_caps_each_company_to_two_citations():
    from src.agent.nodes import _apply_text_citation_policy

    state = {
        "task_type": "company_comparison",
        "companies": ["AAPL"],
        "comparison_target": "MSFT",
        "retrieval_policy": {
            "comparison_text_cap_per_company": 2,
            "require_balanced_comparison_text": True,
        },
    }
    text_citations = [
        {"ticker": "AAPL", "section": "ITEM_2", "fiscal_period": "2025-12-31", "chunk_order": 1, "quality": "high", "score": 0.9, "evidence_id": "T1"},
        {"ticker": "AAPL", "section": "ITEM_7", "fiscal_period": "2025-12-31", "chunk_order": 2, "quality": "high", "score": 0.8, "evidence_id": "T2"},
        {"ticker": "AAPL", "section": "ITEM_1A", "fiscal_period": "2025-12-31", "chunk_order": 3, "quality": "high", "score": 0.7, "evidence_id": "T3"},
        {"ticker": "MSFT", "section": "ITEM_2", "fiscal_period": "2025-12-31", "chunk_order": 1, "quality": "high", "score": 0.9, "evidence_id": "T4"},
        {"ticker": "MSFT", "section": "ITEM_7", "fiscal_period": "2025-12-31", "chunk_order": 2, "quality": "high", "score": 0.8, "evidence_id": "T5"},
        {"ticker": "MSFT", "section": "ITEM_1A", "fiscal_period": "2025-12-31", "chunk_order": 3, "quality": "high", "score": 0.7, "evidence_id": "T6"},
    ]
    filtered, reasons, unbalanced = _apply_text_citation_policy(state, text_citations=text_citations)
    assert unbalanced is False
    assert reasons == []
    assert len(filtered) <= 4
    aapl = [c for c in filtered if c.get("ticker") == "AAPL"]
    msft = [c for c in filtered if c.get("ticker") == "MSFT"]
    assert len(aapl) <= 2
    assert len(msft) <= 2
