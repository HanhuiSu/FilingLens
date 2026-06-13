"""Unit tests for corpus-cleaning and retrieval-quality helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_select_primary_document_prefers_target_type():
    from scripts.parse_filings import _select_primary_document

    raw = """
<DOCUMENT>
<TYPE>EX-31.1
<TEXT>attachment text</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>10-Q
<TEXT>main filing body</TEXT>
</DOCUMENT>
"""
    selected = _select_primary_document(raw, form_type="10-Q")
    assert "main filing body" in selected
    assert "attachment text" not in selected


def test_split_sections_tracks_part_and_instance():
    from scripts.parse_filings import _split_sections

    text = """
PART I
Item 1. Financial Statements
Alpha
Item 2. MD&A
Beta
PART II
Item 1. Legal Proceedings
Gamma
Item 1A. Risk Factors
Delta
"""
    sections = _split_sections(text)
    item1 = [s for s in sections if s["section"] == "ITEM_1"]
    assert len(item1) == 2
    assert item1[0]["part"] == "PART_I"
    assert item1[0]["instance"] == 1
    assert item1[1]["part"] == "PART_II"
    assert item1[1]["instance"] == 2


def test_html_to_text_handles_tag_attrs_none():
    from scripts.parse_filings import _html_to_text

    html = """
<html><body>
<div style="display:none">hidden</div>
<div>Visible content</div>
</body></html>
"""
    text = _html_to_text(html)
    assert "Visible content" in text
    assert "hidden" not in text


def test_parse_processed_sections_v2_headers():
    from scripts.chunk_filings import _parse_processed_sections

    text = """
=== SECTION:ITEM_1A | PART:PART_II | INSTANCE:2 | QUALITY:high ===
Risk text one.

=== SECTION:UNKNOWN | PART:UNKNOWN_PART | INSTANCE:1 | QUALITY:low ===
Fallback text.
"""
    sections = _parse_processed_sections(text)
    assert len(sections) == 2
    assert sections[0]["section"] == "ITEM_1A"
    assert sections[0]["part"] == "PART_II"
    assert sections[0]["section_instance"] == 2
    assert sections[0]["quality"] == "high"


def test_splitter_fallback_when_tiktoken_unavailable(monkeypatch):
    import scripts.chunk_filings as mod

    def _raise(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(mod.RecursiveCharacterTextSplitter, "from_tiktoken_encoder", _raise)
    splitter = mod._splitter()
    chunks = splitter.split_text("alpha beta gamma delta")
    assert isinstance(chunks, list)
    assert chunks


def test_is_mixed_like_logic():
    from src.tools.search_filings import _is_mixed_like

    assert _is_mixed_like("MIXED", "medium")
    assert _is_mixed_like("UNKNOWN", "medium")
    assert _is_mixed_like("ITEM_1A", "low")
    assert not _is_mixed_like("ITEM_1A", "high")


def test_dedupe_candidates_removes_same_payload():
    from src.tools.search_filings import _dedupe_candidates

    candidates = [
        {"filing_id": "f1", "section": "ITEM_1A", "text": "same text", "chunk_order": 1},
        {"filing_id": "f1", "section": "ITEM_1A", "text": "same text", "chunk_order": 2},
        {"filing_id": "f1", "section": "ITEM_1A", "text": "different", "chunk_order": 3},
    ]
    out = _dedupe_candidates(candidates)
    assert len(out) == 2


def test_select_with_profile_prefers_relevance_and_diversity():
    from src.tools.search_filings import _select_with_profile

    candidates = [
        {"filing_id": "f1", "section": "ITEM_1A"},
        {"filing_id": "f1", "section": "ITEM_1A"},
        {"filing_id": "f2", "section": "ITEM_2"},
    ]
    # c0 close to query, c1 almost same as c0, c2 moderately close but different.
    embs = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.70, 0.71],
        ],
        dtype=float,
    )
    q = np.array([1.0, 0.0], dtype=float)
    picked = _select_with_profile(
        candidates=candidates,
        candidate_embs=embs,
        query_emb=q,
        top_k=2,
        lambda_mult=0.75,
        retrieval_profile="summary",
        target_periods=[],
        max_per_filing=1,
        max_per_section=0,
    )
    assert len(picked) == 2
    assert picked[0]["filing_id"] == "f1"
    # second pick should diversify away from near-duplicate f1 chunk
    assert any(p["filing_id"] == "f2" for p in picked[1:])


def test_apply_section_allowlist_prefers_whitelist():
    from src.tools.search_filings import _apply_section_allowlist

    candidates = [
        {"section": "ITEM_1A", "filing_id": "f1"},
        {"section": "ITEM_7", "filing_id": "f2"},
        {"section": "ITEM_2", "filing_id": "f3"},
    ]
    out, allow, fallback = _apply_section_allowlist(
        candidates=candidates,
        section_allowlist=["item_1a", "item_7"],
        strict_sections=False,
        min_section_hits=2,
    )
    assert allow == {"ITEM_1A", "ITEM_7"}
    assert not fallback
    assert len(out) == 2
    assert all(c["section"] in {"ITEM_1A", "ITEM_7"} for c in out)


def test_apply_section_allowlist_fallback_when_hits_insufficient():
    from src.tools.search_filings import _apply_section_allowlist

    candidates = [
        {"section": "ITEM_2", "filing_id": "f1"},
        {"section": "ITEM_3", "filing_id": "f2"},
    ]
    out, allow, fallback = _apply_section_allowlist(
        candidates=candidates,
        section_allowlist=["ITEM_1A"],
        strict_sections=False,
        min_section_hits=3,
    )
    assert allow == {"ITEM_1A"}
    assert fallback
    assert out == candidates


def test_apply_section_allowlist_strict_mode():
    from src.tools.search_filings import _apply_section_allowlist

    candidates = [
        {"section": "ITEM_1A", "filing_id": "f1"},
        {"section": "ITEM_7", "filing_id": "f2"},
    ]
    out, _, fallback = _apply_section_allowlist(
        candidates=candidates,
        section_allowlist=["ITEM_1A"],
        strict_sections=True,
        min_section_hits=3,
    )
    assert not fallback
    assert len(out) == 1
    assert out[0]["section"] == "ITEM_1A"
