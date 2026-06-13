from __future__ import annotations

from src.agent.answer_contract import check_answer_contract


def _state(answer: str, claims: list[dict]) -> dict:
    return {
        "final_answer": answer,
        "analytical_claims": claims,
        "numeric_evidence": [{"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "value": 68_127_000_000}],
        "text_evidence": [{"evidence_id": "T1", "ticker": "NVDA", "claim": "Revenue growth was driven by demand."}],
        "evidence_packet": {
            "numeric_table": [{"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "value": 68_127_000_000}],
            "text_snippets": [{"evidence_id": "T1", "ticker": "NVDA", "claim": "Revenue growth was driven by demand."}],
        },
    }


def test_evidence_backed_claim_requires_citation_refs():
    answer = "NVDA revenue was $68.127B."
    result = check_answer_contract(
        answer,
        _state(answer, [{"id": "eb1", "text": answer, "tier": "evidence_backed", "citation_refs": []}]),
    )

    assert result.route == "blocked"
    assert any(item.code == "evidence_backed_claim_without_citation" for item in result.violations)


def test_inferred_claim_with_strong_causal_wording_is_repairable():
    answer = "合理推断：主要原因就是 AI 需求。"
    result = check_answer_contract(
        answer,
        _state(
            answer,
            [
                {
                    "id": "ei1",
                    "text": answer,
                    "tier": "evidence_inferred",
                    "citation_refs": ["T1"],
                    "supporting_claim_ids": ["eb1"],
                }
            ],
        ),
    )

    assert result.route == "repair_answer"
    assert any(item.code == "strong_causal_in_inferred_claim" for item in result.violations)


def test_labeled_inferred_claim_may_use_supporting_claim_without_direct_citation():
    answer = "合理推断：AI 需求可能解释增长。"
    result = check_answer_contract(
        answer,
        _state(
            answer,
            [
                {
                    "id": "eb1",
                    "text": "Revenue growth was driven by demand.",
                    "tier": "evidence_backed",
                    "citation_refs": ["T1"],
                },
                {
                    "id": "ei1",
                    "text": answer,
                    "tier": "evidence_inferred",
                    "supporting_claim_ids": ["eb1"],
                },
            ],
        ),
    )

    assert result.route == "pass"


def test_unlabeled_inferred_claim_is_repairable_even_with_support():
    answer = "AI 需求可能解释增长。"
    result = check_answer_contract(
        answer,
        _state(
            answer,
            [
                {
                    "id": "eb1",
                    "text": "Revenue growth was driven by demand.",
                    "tier": "evidence_backed",
                    "citation_refs": ["T1"],
                },
                {
                    "id": "ei1",
                    "text": answer,
                    "tier": "evidence_inferred",
                    "supporting_claim_ids": ["eb1"],
                },
            ],
        ),
    )

    assert result.route == "repair_answer"
    assert any(item.code == "inferred_claim_missing_marker" for item in result.violations)


def test_hypothesis_claim_must_be_marked_as_hypothesis():
    answer = "云厂商资本开支继续扩张。"
    result = check_answer_contract(
        answer,
        _state(answer, [{"id": "hv1", "text": answer, "tier": "hypothesis_to_verify"}]),
    )

    assert result.route == "repair_answer"
    assert any(item.code == "hypothesis_missing_marker" for item in result.violations)
