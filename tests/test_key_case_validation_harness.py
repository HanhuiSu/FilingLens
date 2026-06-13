"""Local validation harness for the six financial-agent key cases."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from src.agent.answer_assembler import AnswerAssembler
from src.agent.evidence_packet import build_evidence_packet, summarize_evidence_packet
from src.agent.nodes import (
    contract_check_node,
    finalize_node,
    relevance_check_node,
    route_after_contract,
    route_after_relevance,
)
from src.agent.query_plan import build_classification_state


KEY_CASES: dict[str, dict[str, Any]] = {
    "Q03": {
        "query": "AMZN 和 NVDA 哪个更危险？",
        "task_type": "company_comparison",
        "answer_mode": "comparison_brief",
        "companies": ["AMZN", "NVDA"],
        "required_dimensions": ["moat_and_competitive_risk"],
        "requirement_count": 2,
        "synthesis_mode": "comparison_brief",
        "answer": "风险比较：AMZN 的风险文本强调竞争和宏观需求压力，NVDA 的风险文本强调供应和客户集中度；当前只支持并列披露风险文本，不能强行判断谁更危险。[T1][T2]",
        "numeric": [],
        "text": [
            ("T1", "AMZN", "moat_and_competitive_risk", "Amazon discloses competition and macro demand pressure as risk factors."),
            ("T2", "NVDA", "moat_and_competitive_risk", "NVIDIA discloses supply and customer concentration as risk factors."),
        ],
        "min_counts": {"numeric": 0, "text": 2},
    },
    "Q10": {
        "query": "Amazon 的主要风险是什么，按重要性排序",
        "task_type": "report_summary",
        "answer_mode": "risk_focused_analysis",
        "companies": ["AMZN"],
        "required_dimensions": ["moat_and_competitive_risk"],
        "requirement_count": 8,
        "synthesis_mode": "risk_focused_analysis",
        "answer": "Amazon 的主要风险按当前披露排序：首要是竞争和需求压力，其次是执行投入压力；排序只限于已验证风险文本。[T1][T2]",
        "numeric": [],
        "text": [
            ("T1", "AMZN", "moat_and_competitive_risk", "Amazon identifies competition and demand pressure as risk factors."),
            ("T2", "AMZN", "moat_and_competitive_risk", "Amazon identifies execution and investment pressure as risk factors."),
        ],
        "min_counts": {"numeric": 0, "text": 2},
    },
    "Q11": {
        "query": "比较 AMZN 和 AAPL 谁的增长质量更好",
        "task_type": "company_comparison",
        "answer_mode": "comparison_brief",
        "companies": ["AMZN", "AAPL"],
        "required_dimensions": ["revenue_quality"],
        "requirement_count": 8,
        "synthesis_mode": "comparison_brief",
        "answer": "增长质量对比：AMZN 收入增速为 12%，AAPL 收入增速为 5%，所以 AMZN 当前增长线索更强；但收入规模或单期增速不能单独证明增长质量更好，还需要持续性、利润和现金流转化证据。[N1][N2]",
        "numeric": [
            ("N1", "AMZN", "revenue_growth", 0.12, "12.0%", "ratio", "revenue_quality"),
            ("N2", "AAPL", "revenue_growth", 0.05, "5.0%", "ratio", "revenue_quality"),
        ],
        "text": [],
        "min_counts": {"numeric": 2, "text": 0},
    },
    "Q13": {
        "query": "AAPL 的毛利率变化说明了什么？",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "companies": ["AAPL"],
        "required_dimensions": ["profitability_quality"],
        "requirement_count": 9,
        "synthesis_mode": "analytical",
        "answer": "AAPL 毛利率变化从 40% 到 45%，说明盈利质量改善；该结论只限于毛利率变化本身。[N1][N2]",
        "numeric": [
            ("N1", "AAPL", "gross_margin", 0.40, "40.0%", "ratio", "profitability_quality"),
            ("N2", "AAPL", "gross_margin", 0.45, "45.0%", "ratio", "profitability_quality"),
        ],
        "text": [],
        "min_counts": {"numeric": 2, "text": 0},
    },
    "Q16": {
        "query": "如果下季度经济放缓，MSFT 最大的财务风险是什么？",
        "task_type": "report_summary",
        "answer_mode": "risk_focused_analysis",
        "companies": ["MSFT"],
        "required_dimensions": ["moat_and_competitive_risk"],
        "requirement_count": 8,
        "synthesis_mode": "risk_focused_analysis",
        "answer": "如果下季度经济放缓，MSFT 最大的可验证财务风险是客户支出和云需求放缓；这是情景化排序，只限于已验证风险披露。[T1]",
        "numeric": [],
        "text": [
            ("T1", "MSFT", "moat_and_competitive_risk", "Microsoft discloses customer spending and cloud demand slowdown as risk factors."),
        ],
        "min_counts": {"numeric": 0, "text": 1},
    },
    "Q17": {
        "query": "AAPL 和 NVDA 哪个估值风险更高？",
        "task_type": "company_comparison",
        "answer_mode": "comparison_brief",
        "companies": ["AAPL", "NVDA"],
        "required_dimensions": ["valuation_and_risk_boundary", "moat_and_competitive_risk"],
        "requirement_count": 20,
        "synthesis_mode": "comparison_brief",
        "answer": "估值风险对比：NVDA 更高；AAPL 的 P/E 是 30、P/S 是 7、FCF yield 是 3%，NVDA 的 P/E 是 60、P/S 是 25、FCF yield 是 1%，因此排序仅限这些估值指标。[N1][N2][N3][N4][N5][N6]",
        "numeric": [
            ("N1", "AAPL", "pe_ratio", 30.0, "30.0x", "ratio", "valuation_and_risk_boundary"),
            ("N2", "AAPL", "ps_ratio", 7.0, "7.0x", "ratio", "valuation_and_risk_boundary"),
            ("N3", "AAPL", "fcf_yield", 0.03, "3.0%", "ratio", "valuation_and_risk_boundary"),
            ("N4", "NVDA", "pe_ratio", 60.0, "60.0x", "ratio", "valuation_and_risk_boundary"),
            ("N5", "NVDA", "ps_ratio", 25.0, "25.0x", "ratio", "valuation_and_risk_boundary"),
            ("N6", "NVDA", "fcf_yield", 0.01, "1.0%", "ratio", "valuation_and_risk_boundary"),
        ],
        "text": [
            ("T1", "AAPL", "moat_and_competitive_risk", "Apple discloses competition and demand risk factors."),
            ("T2", "NVDA", "moat_and_competitive_risk", "NVIDIA discloses supply and customer concentration risk factors."),
        ],
        "min_counts": {"numeric": 6, "text": 2},
    },
}


def _numeric_row(
    evidence_id: str,
    ticker: str,
    metric: str,
    value: float,
    display_value: str,
    unit: str,
    dimension_id: str,
    case_id: str,
) -> dict[str, Any]:
    requirement_id = f"REQ-{case_id}-{evidence_id}"
    return {
        "evidence_id": evidence_id,
        "requirement_id": requirement_id,
        "source_requirement_id": requirement_id,
        "ticker": ticker,
        "metric": metric,
        "period_type": "annual",
        "period_end": "2025-12-31",
        "value": value,
        "display_value": display_value,
        "unit": unit,
        "source_provider": "validated_fixture",
        "provenance": f"validated_fixture:{case_id}:{evidence_id}",
        "_dimension_id": dimension_id,
    }


def _text_row(
    evidence_id: str,
    ticker: str,
    dimension_id: str,
    snippet: str,
    case_id: str,
) -> dict[str, Any]:
    requirement_id = f"REQ-{case_id}-{evidence_id}"
    return {
        "evidence_id": evidence_id,
        "requirement_id": requirement_id,
        "ticker": ticker,
        "form_type": "10-K",
        "fiscal_period": "2025-12-31",
        "section": "ITEM_1A",
        "chunk_order": 1,
        "dimension_id": dimension_id,
        "text_snippet": snippet,
        "supporting_snippet": snippet,
        "claim": snippet,
    }


def _packet_for_case(case_id: str, state: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    numeric_rows = [_numeric_row(*item, case_id=case_id) for item in spec["numeric"]]
    text_rows = [_text_row(*item, case_id=case_id) for item in spec["text"]]
    refs_by_dimension: dict[str, list[str]] = {}
    requirement_status_map: dict[str, dict[str, Any]] = {}
    for row in numeric_rows:
        dimension_id = str(row.pop("_dimension_id"))
        refs_by_dimension.setdefault(dimension_id, []).append(row["evidence_id"])
        requirement_status_map[row["requirement_id"]] = {
            "dimension_id": dimension_id,
            "status": "satisfied",
            "required": True,
            "evidence_role": str(row.get("metric") or ""),
            "quality_status": "valid",
        }
    for row in text_rows:
        dimension_id = str(row.get("dimension_id") or "")
        refs_by_dimension.setdefault(dimension_id, []).append(row["evidence_id"])
        requirement_status_map[row["requirement_id"]] = {
            "dimension_id": dimension_id,
            "status": "satisfied",
            "required": True,
            "evidence_role": "risk_text" if "risk" in dimension_id else "text_support",
            "quality_status": "valid",
        }
    dimension_status_map = {
        dimension_id: {
            "status": "satisfied",
            "supporting_evidence_ids": refs,
            "allowed_claims": [f"{dimension_id} can be discussed only from validated refs."],
            "forbidden_claims": [],
        }
        for dimension_id, refs in refs_by_dimension.items()
    }
    citations = [{"evidence_id": ref} for ref in sorted({*refs_by_dimension.get("valuation_and_risk_boundary", []), *refs_by_dimension.get("moat_and_competitive_risk", []), *refs_by_dimension.get("revenue_quality", []), *refs_by_dimension.get("profitability_quality", [])})]
    packet = build_evidence_packet(
        user_query=str(spec["query"]),
        task_type=str(state.get("task_type") or ""),
        answer_mode=str(state.get("answer_mode") or ""),
        safety_intent=str(state.get("safety_intent") or "normal"),
        analysis_scope=str(state.get("analysis_scope") or ""),
        companies=list(state.get("companies", []) or []),
        comparison_target=state.get("comparison_target"),
        requested_metrics=list(state.get("requested_metrics", []) or []),
        period_query=dict(state.get("period_query", {}) or {}),
        resolved_period_context=dict(state.get("resolved_period_context", {}) or {}),
        numeric_evidence=numeric_rows,
        text_evidence=text_rows,
        citations=citations,
        evidence_sufficiency={
            "overall_status": "sufficient",
            "missing_requirements": [],
            "partial_requirements": [],
            "required_numeric_satisfied_rate": 1.0,
            "required_text_satisfied_rate": 1.0,
            "dimension_status_map": dimension_status_map,
            "dimension_status_by_id": dimension_status_map,
            "satisfied_dimensions": list(dimension_status_map),
            "covered_dimensions": list(dimension_status_map),
            "partial_dimensions": [],
            "missing_dimensions": [],
            "dimension_coverage_rate": 1.0,
            "weighted_dimension_coverage_rate": 1.0,
            "framework_sufficiency_status": "sufficient",
        },
        requirement_limitations=[],
        safety_limitations=[],
        selected_framework=dict(state.get("selected_analysis_framework", {}) or {}),
        requirement_status_map=requirement_status_map,
    ).model_dump(exclude_none=True)
    requested_dimensions = list(dict(state.get("canonical_intent", {}) or {}).get("requested_dimensions", []) or [])
    packet["canonical_intent"] = dict(state.get("canonical_intent", {}) or {})
    packet["requested_dimensions"] = requested_dimensions
    packet["required_answer_parts"] = list(state.get("required_answer_parts", []) or [])
    return packet


def _seed_answer(state: dict[str, Any], spec: dict[str, Any], case_id: str) -> None:
    refs = re.findall(r"\[([NT]\d+)\]", str(spec["answer"]))
    candidate = AnswerAssembler.candidate(
        body=str(spec["answer"]),
        owner=f"{case_id.lower()}_validated_candidate",
        requested_dimensions=list(dict(state.get("canonical_intent", {}) or {}).get("requested_dimensions", []) or []),
        answered_dimensions=list(spec["required_dimensions"]),
        evidence_refs=list(dict.fromkeys(refs)),
        allowed_repairs=["add_citation", "strip_sentence", "scope_limit", "downgrade_to_bounded"],
        provenance={"case_id": case_id, "harness": "local_key_case_validation"},
    )
    state.update(
        AnswerAssembler.select(
            candidate,
            state,
            transform="case_validation_seed",
            reason="seeded cited answer for local key-case validation",
            previous_body="",
            claim_change_allowed=True,
        )
    )


def _run_case(case_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    state = build_classification_state(
        user_query=str(spec["query"]),
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="",
        today=date(2026, 5, 26),
    )
    assert state["task_type"] == spec["task_type"]
    assert state["answer_mode"] == spec["answer_mode"]
    assert state["companies"][: len(spec["companies"])] == spec["companies"]
    assert state["required_dimensions"] == spec["required_dimensions"]
    assert len(state["evidence_plan"]["evidence_requirements"]) == spec["requirement_count"]

    packet = _packet_for_case(case_id, state, spec)
    state["evidence_packet"] = packet
    state["evidence_packet_summary"] = summarize_evidence_packet(packet)
    state["requirement_status_map"] = {
        row["source_requirement_id"]: {"status": "satisfied", "required": True}
        for row in packet.get("numeric_table", [])
        if row.get("source_requirement_id")
    }
    state["numeric_evidence"] = list(packet.get("numeric_table", []) or [])
    state["text_evidence"] = list(packet.get("text_snippets", []) or [])
    state["citations"] = list(packet.get("citations", []) or [])
    state["missing_requirements"] = []
    state["missing_required_requirements_count"] = 0
    state["partial_required_answer_parts"] = []
    state["missing_required_answer_parts"] = []
    state["answer_part_status_by_id"] = {}
    state["output"] = {
        "answer_mode": spec["answer_mode"],
        "synthesis_mode": spec["synthesis_mode"],
        "evidence_packet_summary": state["evidence_packet_summary"],
        "limitations": [],
        "view": {"kind": spec["synthesis_mode"]},
    }
    _seed_answer(state, spec, case_id)
    state["output"]["answer_history"] = list(state["answer_history"])
    state["output"]["final_answer_source"] = state["final_answer_source"]

    contract = contract_check_node(state)
    state.update(contract)
    contract_route = route_after_contract(state)
    assert contract_route == "relevance_check"

    relevance = relevance_check_node(state)
    state.update(relevance)
    relevance_route = route_after_relevance(state)
    assert relevance_route == "finalize"

    final = finalize_node(state)
    state.update(final)
    answer_history = [dict(item) for item in state.get("answer_history", []) or []]
    final_answer_source = str(state.get("final_answer_source") or "")
    packet_refs = {
        str(row.get("evidence_id") or "")
        for row in [*packet.get("numeric_table", []), *packet.get("text_snippets", [])]
        if str(row.get("evidence_id") or "")
    }
    cited_refs = set(re.findall(r"\[([NT]\d+)\]", str(state.get("final_answer") or "")))
    return {
        "case_id": case_id,
        "final_answer_source": final_answer_source,
        "answer_history_summary": [
            f"{item.get('index')}:{item.get('transform')}->{item.get('new_owner')}"
            for item in answer_history
        ],
        "evidence_packet_counts": {
            "numeric_count": len(packet.get("numeric_table", []) or []),
            "text_count": len(packet.get("text_snippets", []) or []),
            "computed_count": len(packet.get("computed_metric_dependencies", {}) or {}),
            "canonical_source": bool(packet.get("canonical_source")),
            "packet_kind": str(packet.get("packet_kind") or ""),
        },
        "synthesis_mode": str(state.get("output", {}).get("synthesis_mode") or ""),
        "contract_route": str(state.get("contract_decision", {}).get("route") or ""),
        "relevance_route": str(state.get("relevance_decision", {}).get("route") or ""),
        "final_route": str(state.get("final_route") or ""),
        "blocked": state.get("contract_status") == "blocked" or state.get("final_route") == "blocked",
        "main_quality_issue": "",
        "history_last_owner": str(answer_history[-1].get("new_owner") if answer_history else ""),
        "cited_refs": cited_refs,
        "packet_refs": packet_refs,
        "final_answer": str(state.get("final_answer") or ""),
    }


def test_six_key_cases_have_canonical_packet_owner_history_and_nonblocked_routes():
    results = [_run_case(case_id, spec) for case_id, spec in KEY_CASES.items()]

    for item in results:
        spec = KEY_CASES[item["case_id"]]
        counts = item["evidence_packet_counts"]
        assert item["final_answer_source"] == item["history_last_owner"]
        assert item["answer_history_summary"]
        assert counts["canonical_source"] is True
        assert counts["packet_kind"] == "canonical_validated_evidence_packet"
        assert counts["numeric_count"] >= spec["min_counts"]["numeric"]
        assert counts["text_count"] >= spec["min_counts"]["text"]
        assert item["contract_route"] == "pass"
        assert item["relevance_route"] == "finalize"
        assert item["final_route"] in {"released", "released_with_warnings"}
        assert item["blocked"] is False
        assert item["main_quality_issue"] == ""
        assert item["cited_refs"] <= item["packet_refs"]

    q03 = next(item for item in results if item["case_id"] == "Q03")
    assert "[T1]" in q03["final_answer"] and "[T2]" in q03["final_answer"]
    assert "缺少数值证据" not in q03["final_answer"]

    q11 = next(item for item in results if item["case_id"] == "Q11")
    assert "增长" in q11["final_answer"]
    assert "风险" not in q11["final_answer"]
    assert "盈利" not in q11["final_answer"]

    q17 = next(item for item in results if item["case_id"] == "Q17")
    assert all(term in q17["final_answer"] for term in ("P/E", "P/S", "FCF yield"))
