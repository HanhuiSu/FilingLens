"""Validation and deterministic fallback for Research Planner V1."""

from __future__ import annotations

import re
from typing import Any, Mapping

from config import settings
from src.agent.constants import ALLOWED_ANALYSIS_METRICS, ALLOWED_ANALYSIS_TOOLS, KNOWN_SEC_SECTIONS
from src.agent.entity_resolution import resolve_companies
from src.agent.types import EvidenceRequest, PlanValidationResult, ReasoningPolicy, RequiredAnswerPart, ResearchPlan


QUESTION_TYPES = {
    "direct_fact",
    "trend_analysis",
    "causal_explanation",
    "overview",
    "risk_analysis",
    "valuation_boundary",
    "cash_flow_quality",
    "comparison",
    "cautious_outlook",
    "investment_boundary",
    "unknown",
}

REQUEST_TYPES = {"numeric", "text", "calculation", "event"}
REQUEST_SCOPES = {"core", "optional_context", "diagnostic"}
EVIDENCE_ROLES = {
    "",
    "current_revenue",
    "comparator_revenue",
    "revenue_growth_calculation",
    "revenue_growth_text",
    "driver_text",
}
CAUSAL_FALLBACK_ANSWER_POLICY = (
    "If direct growth-driver text is unavailable, do not assert a definitive cause. "
    "Still provide a bounded analytical answer that separates verified facts, evidence-based inferences, "
    "hypotheses to verify, counterpoints, observation metrics, and evidence boundaries."
)
CAUSAL_MINIMUM_ANSWER_POLICY = (
    "For causal revenue-growth questions, the minimum answer must include direct_answer, verified_evidence, "
    "inferred_drivers, hypotheses_to_verify, counterpoints, key observation metrics, and evidence_boundary. "
    "Missing driver evidence is not a silent fallback to a revenue summary."
)

_SECTION_ALIASES = {
    "mda": "MD&A",
    "md&a": "MD&A",
    "management discussion": "ITEM_7",
    "management discussion and analysis": "ITEM_7",
    "item 7": "ITEM_7",
    "item_7": "ITEM_7",
    "item 2": "ITEM_2",
    "item_2": "ITEM_2",
    "item 1": "ITEM_1",
    "item_1": "ITEM_1",
    "segment discussion": "ITEM_7",
    "segments": "ITEM_7",
    "risk factors": "ITEM_1A",
    "item 1a": "ITEM_1A",
    "item_1a": "ITEM_1A",
}

_METRIC_ALIASES = {
    "revenue_numeric": "revenue",
    "latest_revenue": "revenue",
    "total_revenue": "revenue",
    "sales": "revenue",
    "net_sales": "revenue",
    "revenue_history": "revenue",
    "net_income_numeric": "net_income",
    "profit": "net_income",
    "earnings": "net_income",
    "rev_growth": "revenue_growth",
    "growth": "revenue_growth",
    "operating_cashflow": "operating_cash_flow",
    "cash_from_operations": "operating_cash_flow",
    "cfo": "operating_cash_flow",
    "fcf": "free_cash_flow",
    "capex": "capital_expenditure",
    "capital_expenditures": "capital_expenditure",
    "cash": "cash_and_equivalents",
    "cash_equivalents": "cash_and_equivalents",
    "cash_and_cash_equivalents": "cash_and_equivalents",
    "debt": "total_debt",
    "shares": "shares_outstanding",
    "latest_price": "adjusted_close",
    "share_price": "adjusted_close",
    "pe": "pe_ratio",
    "p/e": "pe_ratio",
    "ps": "ps_ratio",
    "p/s": "ps_ratio",
}


def _ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        clean = str(item or "").strip()
        if clean and clean not in out:
            out.append(clean)
    return out


def is_causal_explanation_query(user_query: str) -> bool:
    q = str(user_query or "").lower()
    causal_terms = (
        "为什么",
        "为何",
        "原因",
        "驱动",
        "由什么",
        "why",
        "what drove",
        "what drives",
        "driver",
        "drivers",
        "because",
    )
    growth_terms = ("增长", "增加", "提升", "growth", "increase", "increased", "grew")
    metric_terms = ("营收", "收入", "revenue", "sales", "net sales")
    return any(term in q for term in causal_terms) and (
        any(term in q for term in growth_terms) or any(term in q for term in metric_terms)
    )


def _normalize_company(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    resolved = resolve_companies(raw, parsed_companies=[raw])
    if resolved.resolved_companies:
        ticker = resolved.resolved_companies[0].ticker.upper()
    else:
        ticker = raw.upper()
    return ticker if ticker in {str(t).upper() for t in settings.target_tickers} else None


def _normalize_companies(raw: Any, fallback: list[str], rejected: list[dict[str, Any]]) -> list[str]:
    values = raw if isinstance(raw, list) else []
    companies: list[str] = []
    for item in values:
        ticker = _normalize_company(item)
        if ticker and ticker not in companies:
            companies.append(ticker)
        elif item:
            rejected.append({"type": "company", "value": item, "reason": "unsupported_company"})
    for item in fallback:
        ticker = _normalize_company(item)
        if ticker and ticker not in companies:
            companies.append(ticker)
    return companies


def _normalize_metric(value: Any) -> str | None:
    metric = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    metric = _METRIC_ALIASES.get(metric, metric)
    return metric if metric in ALLOWED_ANALYSIS_METRICS else None


def _normalize_section(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.upper().replace(" ", "_").replace("-", "_")
    normalized = normalized.replace("ITEM__", "ITEM_")
    if normalized in KNOWN_SEC_SECTIONS:
        return normalized
    alias = _SECTION_ALIASES.get(raw.lower())
    if alias and alias in KNOWN_SEC_SECTIONS:
        return alias
    return None


def _clean_id(value: Any, fallback: str) -> str:
    raw = str(value or "").strip().lower()
    raw = re.sub(r"[^a-z0-9_]+", "_", raw).strip("_")
    return raw or fallback


def _default_causal_parts() -> list[RequiredAnswerPart]:
    return [
        RequiredAnswerPart(
            id="direct_answer",
            description="Answer the why question directly while preserving evidence boundaries.",
            required=True,
            evidence_request_ids=[],
            success_criteria=["clear answer that distinguishes verified facts, inferences, and hypotheses"],
        ),
        RequiredAnswerPart(
            id="verified_evidence",
            description="Present directly verified revenue and filing evidence.",
            required=True,
            evidence_request_ids=[
                "current_revenue",
                "comparator_revenue",
                "revenue_growth_calculation",
                "revenue_growth_text",
                "growth_driver_text",
            ],
            success_criteria=["all verified facts have citations"],
        ),
        RequiredAnswerPart(
            id="quantify_growth",
            description="Quantify the revenue growth or change.",
            required=True,
            evidence_request_ids=[
                "current_revenue",
                "comparator_revenue",
                "revenue_growth_calculation",
                "revenue_growth_text",
            ],
            success_criteria=[
                "current revenue plus comparator revenue plus growth calculation, or filing text with total/company revenue growth"
            ],
        ),
        RequiredAnswerPart(
            id="identify_growth_drivers",
            description="Explain disclosed drivers when available; otherwise provide bounded analytical hypotheses.",
            required=True,
            evidence_request_ids=["growth_driver_text"],
            success_criteria=[
                "validated MD&A, operating-results, or segment text evidence for definitive causes",
                "bounded analytical framework when direct driver text is unavailable",
            ],
            forbidden_without_evidence=["demand drove growth", "segment strength caused growth", "product mix caused growth"],
        ),
        RequiredAnswerPart(
            id="inferred_drivers",
            description="Separate evidence-based inferences from verified facts.",
            required=True,
            evidence_request_ids=["growth_driver_text"],
            success_criteria=["uses weak inference wording and cites supporting premises when available"],
            forbidden_without_evidence=["proven cause", "definitive cause"],
        ),
        RequiredAnswerPart(
            id="hypotheses_to_verify",
            description="List reasonable hypotheses that still require verification.",
            required=True,
            evidence_request_ids=[],
            success_criteria=["hypotheses are explicitly marked as待验证 or to verify"],
        ),
        RequiredAnswerPart(
            id="counterpoints",
            description="List counterpoints or alternative explanations.",
            required=True,
            evidence_request_ids=[],
            success_criteria=["alternative explanations are framed as possible factors, not facts"],
        ),
        RequiredAnswerPart(
            id="evidence_boundary",
            description="State what cannot be concluded from the available evidence.",
            required=True,
            evidence_request_ids=[],
            success_criteria=["visible evidence boundary when driver text is missing"],
        ),
        RequiredAnswerPart(
            id="state_evidence_boundary",
            description="Compatibility alias for evidence_boundary.",
            required=True,
            evidence_request_ids=[],
            success_criteria=["visible evidence boundary when driver text is missing"],
        ),
    ]


def deterministic_causal_research_plan(
    *,
    user_query: str,
    companies: list[str],
    source: str = "deterministic_causal_fallback",
) -> ResearchPlan:
    company = companies[0] if companies else None
    return ResearchPlan(
        question_type="causal_explanation",
        user_goal=f"Explain why revenue growth changed for {company or 'the company'}.".strip(),
        companies=list(companies),
        time_scope="latest",
        required_answer_parts=_default_causal_parts(),
        evidence_requests=[
            EvidenceRequest(
                id="current_revenue",
                type="numeric",
                scope="core",
                company=company,
                metrics=["revenue"],
                tool="query_financial_data",
                required=True,
                min_results=1,
                answer_part_ids=["quantify_growth"],
                purpose="Collect current-period total revenue before discussing growth causes.",
                evidence_role="current_revenue",
                alternative_group="revenue_growth_numeric",
            ),
            EvidenceRequest(
                id="comparator_revenue",
                type="numeric",
                scope="core",
                company=company,
                metrics=["revenue"],
                tool="query_financial_data",
                required=True,
                min_results=1,
                answer_part_ids=["quantify_growth"],
                purpose="Collect prior or comparable-period total revenue so growth can be quantified.",
                fallback_strategy=["prior_period", "same_period_last_year"],
                evidence_role="comparator_revenue",
                alternative_group="revenue_growth_numeric",
            ),
            EvidenceRequest(
                id="revenue_growth_calculation",
                type="calculation",
                scope="core",
                company=company,
                metrics=["revenue_growth"],
                tool="compute_metrics",
                required=True,
                min_results=1,
                answer_part_ids=["quantify_growth"],
                purpose="Compute total revenue growth from current and comparator revenue.",
                evidence_role="revenue_growth_calculation",
                alternative_group="revenue_growth_numeric",
            ),
            EvidenceRequest(
                id="revenue_growth_text",
                type="text",
                scope="core",
                company=company,
                sections=["ITEM_7", "ITEM_2", "MD&A"],
                queries=[
                    "total revenue increased grew year over year revenue growth",
                    "overall revenue year-on-year sequential growth",
                    "revenue increased percentage total revenue",
                    user_query,
                ],
                tool="search_filings",
                required=True,
                min_results=1,
                answer_part_ids=["quantify_growth"],
                purpose="Find filing text that explicitly states total/company revenue growth.",
                fallback_strategy=["broaden_to_operating_results", "broaden_to_segment_discussion"],
                evidence_role="revenue_growth_text",
                alternative_group="revenue_growth_text",
            ),
            EvidenceRequest(
                id="growth_driver_text",
                type="text",
                scope="core",
                company=company,
                sections=["ITEM_7", "ITEM_2", "MD&A"],
                queries=[
                    "total revenue increased because growth driven by demand",
                    "year-on-year and sequential growth was driven by demand",
                    "operating results revenue growth drivers",
                    "segment revenue data center demand growth",
                    user_query,
                ],
                tool="search_filings",
                required=True,
                min_results=1,
                answer_part_ids=["identify_growth_drivers"],
                purpose="Find validated filing text that explains growth drivers.",
                fallback_strategy=["broaden_to_operating_results", "broaden_to_segment_discussion"],
                evidence_role="driver_text",
                alternative_group="growth_driver_text",
            ),
        ],
        fallback_answer_policy=CAUSAL_FALLBACK_ANSWER_POLICY,
        minimum_answer_policy=CAUSAL_MINIMUM_ANSWER_POLICY,
        reasoning_policy=ReasoningPolicy(),
        forbidden_claims=[
            "Do not infer demand, product mix, segment strength, customer behavior, or management intent without driver text.",
            "Do not present numeric growth as a causal explanation.",
        ],
        planner_source=source,
    )


def _infer_question_type(raw: Mapping[str, Any], *, user_query: str, answer_mode: str, safety_intent: str) -> str:
    candidate = str(raw.get("question_type") or "").strip().lower()
    if candidate in QUESTION_TYPES:
        if candidate == "causal_explanation" and not is_causal_explanation_query(user_query):
            candidate = ""
        else:
            return candidate
    if is_causal_explanation_query(user_query):
        return "causal_explanation"
    if safety_intent == "investment_advice_like":
        return "investment_boundary"
    if answer_mode == "comparison_brief":
        return "comparison"
    if answer_mode == "risk_focused_analysis":
        return "risk_analysis"
    if answer_mode == "cautious_outlook":
        return "cautious_outlook"
    if answer_mode == "analytical":
        return "overview"
    return "direct_fact"


def _validate_parts(raw_parts: Any, question_type: str, warnings: list[dict[str, Any]]) -> list[RequiredAnswerPart]:
    parts: list[RequiredAnswerPart] = []
    if isinstance(raw_parts, list):
        for idx, item in enumerate(raw_parts):
            if not isinstance(item, Mapping):
                warnings.append({"field": "required_answer_parts", "value": item, "reason": "part_not_object"})
                continue
            part_id = _clean_id(item.get("id"), f"part_{idx + 1}")
            parts.append(
                RequiredAnswerPart(
                    id=part_id,
                    description=str(item.get("description") or item.get("goal") or part_id).strip(),
                    required=bool(item.get("required", True)),
                    evidence_request_ids=_ordered_unique([str(x) for x in item.get("evidence_request_ids", []) or [] if str(x)]),
                    success_criteria=_ordered_unique([str(x) for x in item.get("success_criteria", []) or [] if str(x)]),
                    forbidden_without_evidence=_ordered_unique(
                        [str(x) for x in item.get("forbidden_without_evidence", []) or [] if str(x)]
                    ),
                )
            )
    if question_type == "causal_explanation":
        existing = {part.id for part in parts}
        for default in _default_causal_parts():
            if default.id not in existing:
                parts.append(default)
                warnings.append({"field": "required_answer_parts", "value": default.id, "reason": "injected_causal_required_part"})
    if not parts:
        parts.append(
            RequiredAnswerPart(
                id="answer_user_question",
                description="Answer the user's question with available validated evidence.",
                required=True,
            )
        )
    return parts


def _validate_requests(
    raw_requests: Any,
    *,
    question_type: str,
    companies: list[str],
    part_ids: set[str],
    warnings: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> list[EvidenceRequest]:
    requests: list[EvidenceRequest] = []
    if isinstance(raw_requests, list):
        for idx, item in enumerate(raw_requests):
            if not isinstance(item, Mapping):
                rejected.append({"type": "evidence_request", "value": item, "reason": "request_not_object"})
                continue
            request_id = _clean_id(item.get("id"), f"request_{idx + 1}")
            req_type = str(item.get("type") or item.get("requirement_type") or "").strip().lower()
            if req_type not in REQUEST_TYPES:
                rejected.append({"type": "evidence_request_type", "value": req_type, "reason": "request_type_not_allowed"})
                continue
            scope = str(item.get("scope") or item.get("requirement_scope") or "core").strip()
            if scope not in REQUEST_SCOPES:
                warnings.append({"field": "scope", "value": scope, "reason": "scope_normalized_to_core"})
                scope = "core"
            company = _normalize_company(item.get("company")) or (companies[0] if companies else None)
            metrics: list[str] = []
            for raw_metric in item.get("metrics", []) or ([] if not item.get("metric") else [item.get("metric")]):
                metric = _normalize_metric(raw_metric)
                if metric and metric not in metrics:
                    metrics.append(metric)
                elif raw_metric:
                    rejected.append({"type": "metric", "value": raw_metric, "reason": "metric_not_allowed", "request_id": request_id})
            sections: list[str] = []
            for raw_section in item.get("sections", []) or item.get("section_preferences", []) or []:
                section = _normalize_section(raw_section)
                if section and section not in sections:
                    sections.append(section)
                elif raw_section:
                    rejected.append({"type": "section", "value": raw_section, "reason": "section_not_allowed", "request_id": request_id})
            tool = str(item.get("tool") or "").strip()
            if tool and tool not in ALLOWED_ANALYSIS_TOOLS:
                rejected.append({"type": "tool", "value": tool, "reason": "tool_not_allowed", "request_id": request_id})
                tool = ""
            answer_part_ids = [
                _clean_id(x, "")
                for x in item.get("answer_part_ids", []) or item.get("required_answer_part_ids", []) or []
                if str(x).strip()
            ]
            answer_part_ids = [x for x in _ordered_unique(answer_part_ids) if x in part_ids]
            if question_type == "causal_explanation" and not answer_part_ids:
                answer_part_ids = ["identify_growth_drivers"] if req_type == "text" else ["quantify_growth"]
            queries = _ordered_unique([str(x).strip() for x in item.get("queries", []) or [] if str(x).strip()])
            if req_type == "text" and not queries and str(item.get("query") or item.get("retrieval_query") or "").strip():
                queries = [str(item.get("query") or item.get("retrieval_query")).strip()]
            if req_type == "text" and not queries:
                rejected.append({"type": "query", "value": request_id, "reason": "text_request_requires_query"})
                continue
            evidence_role = _clean_id(item.get("evidence_role") or item.get("role"), "")
            if question_type == "causal_explanation" and not evidence_role:
                if req_type == "text" and "identify_growth_drivers" in answer_part_ids:
                    evidence_role = "driver_text"
                elif req_type == "text" and "quantify_growth" in answer_part_ids:
                    evidence_role = "revenue_growth_text"
                elif req_type == "calculation" and "revenue_growth" in metrics:
                    evidence_role = "revenue_growth_calculation"
            if evidence_role not in EVIDENCE_ROLES:
                warnings.append({"field": "evidence_role", "value": evidence_role, "reason": "unsupported_evidence_role_dropped"})
                evidence_role = ""
            try:
                min_results = max(1, int(item.get("min_results", 1) or 1))
            except (TypeError, ValueError):
                min_results = 1
            requests.append(
                EvidenceRequest(
                    id=request_id,
                    type=req_type,  # type: ignore[arg-type]
                    scope=scope,  # type: ignore[arg-type]
                    company=company,
                    metrics=metrics,
                    sections=sections,
                    queries=queries,
                    tool=tool or None,
                    required=bool(item.get("required", scope == "core")),
                    min_results=min_results,
                    answer_part_ids=answer_part_ids,
                    purpose=str(item.get("purpose") or item.get("description") or "").strip(),
                    fallback_strategy=_ordered_unique([str(x) for x in item.get("fallback_strategy", []) or [] if str(x)]),
                    evidence_role=evidence_role,
                    alternative_group=_clean_id(item.get("alternative_group"), ""),
                )
            )
    if question_type == "causal_explanation":
        existing_roles = {request.evidence_role for request in requests if request.evidence_role}
        has_driver_text = any(
            "identify_growth_drivers" in request.answer_part_ids and request.type == "text"
            for request in requests
        )
        has_growth_text = any(
            "quantify_growth" in request.answer_part_ids and request.type == "text"
            for request in requests
        )
        fallback = deterministic_causal_research_plan(user_query="", companies=companies)
        for request in fallback.evidence_requests:
            if request.evidence_role and request.evidence_role in existing_roles:
                continue
            if request.evidence_role == "driver_text" and has_driver_text:
                continue
            if request.evidence_role == "revenue_growth_text" and has_growth_text:
                continue
            if request.id not in {existing.id for existing in requests}:
                requests.append(request)
                warnings.append({"field": "evidence_requests", "value": request.id, "reason": "injected_causal_evidence_request"})
    return requests


def _fallback_policy(raw_value: Any, question_type: str) -> str:
    raw_policy = str(raw_value or "").strip()
    if question_type != "causal_explanation":
        return raw_policy or "Use only validated evidence and state missing evidence boundaries."
    lowered = raw_policy.lower()
    if (
        ("cannot explain" in lowered or "不能解释" in raw_policy or "无法解释" in raw_policy)
        and ("quantif" in lowered or "量化" in raw_policy)
    ):
        return raw_policy
    return CAUSAL_FALLBACK_ANSWER_POLICY


def _minimum_answer_policy(raw_value: Any, question_type: str) -> str:
    raw_policy = str(raw_value or "").strip()
    if raw_policy:
        return raw_policy
    if question_type == "causal_explanation":
        return CAUSAL_MINIMUM_ANSWER_POLICY
    return "Answer the user's required parts with verified facts first, then clearly labeled inferences, hypotheses, and evidence boundaries when needed."


def _reasoning_policy(raw_value: Any, question_type: str, safety_intent: str) -> ReasoningPolicy:
    raw = dict(raw_value or {}) if isinstance(raw_value, Mapping) else {}
    policy = ReasoningPolicy(
        allow_inference=bool(raw.get("allow_inference", True)),
        allow_hypotheses=bool(raw.get("allow_hypotheses", True)),
        must_separate_claim_tiers=bool(raw.get("must_separate_claim_tiers", True)),
        forbid_uncited_definitive_causal_claims=bool(raw.get("forbid_uncited_definitive_causal_claims", True)),
    )
    if question_type in {"direct_fact", "unknown"}:
        policy.allow_hypotheses = False
    if safety_intent == "investment_advice_like":
        policy.allow_inference = True
        policy.allow_hypotheses = True
        policy.must_separate_claim_tiers = True
        policy.forbid_uncited_definitive_causal_claims = True
    return policy


def validate_research_plan(
    raw_plan: Any,
    *,
    user_query: str,
    companies: list[str],
    answer_mode: str,
    safety_intent: str,
) -> PlanValidationResult:
    """Return a program-approved research plan or an invalid fallback marker."""
    raw = dict(raw_plan) if isinstance(raw_plan, Mapping) else {}
    warnings: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    corrected: list[dict[str, Any]] = []
    causal_query = is_causal_explanation_query(user_query)

    if (not raw or set(raw.keys()) <= {"intent", "methodology_intent", "question_type"}) and causal_query:
        plan = deterministic_causal_research_plan(user_query=user_query, companies=companies, source="deterministic_planner_failure")
        return PlanValidationResult(
            valid=True,
            used_fallback=True,
            fallback_reason="planner_missing_or_intent_only_for_causal_query",
            warnings=[{"field": "research_plan_raw", "reason": "deterministic_causal_plan_used"}],
            plan=plan,
        )
    if not raw:
        return PlanValidationResult(
            valid=False,
            used_fallback=True,
            fallback_reason="planner_missing",
            warnings=[{"field": "research_plan_raw", "reason": "planner_missing"}],
            plan=ResearchPlan(question_type="unknown", user_goal=user_query, companies=list(companies), planner_source="legacy_fallback"),
        )

    question_type = _infer_question_type(raw, user_query=user_query, answer_mode=answer_mode, safety_intent=safety_intent)
    normalized_companies = _normalize_companies(raw.get("companies"), companies, rejected)
    if not normalized_companies and companies:
        normalized_companies = list(companies)
        corrected.append({"field": "companies", "reason": "used_classification_companies"})
    if not normalized_companies:
        rejected.append({"type": "company", "value": raw.get("companies"), "reason": "empty_company_scope"})

    parts = _validate_parts(raw.get("required_answer_parts"), question_type, warnings)
    part_ids = {part.id for part in parts}
    requests = _validate_requests(
        raw.get("evidence_requests"),
        question_type=question_type,
        companies=normalized_companies,
        part_ids=part_ids,
        warnings=warnings,
        rejected=rejected,
    )
    if question_type == "causal_explanation":
        request_ids_by_part: dict[str, list[str]] = {}
        for req in requests:
            for part_id in req.answer_part_ids:
                request_ids_by_part.setdefault(part_id, [])
                if req.id not in request_ids_by_part[part_id]:
                    request_ids_by_part[part_id].append(req.id)
        for part in parts:
            if part.id in {"quantify_growth", "identify_growth_drivers"}:
                part.evidence_request_ids = _ordered_unique(
                    list(part.evidence_request_ids or []) + request_ids_by_part.get(part.id, [])
                )

    if question_type == "causal_explanation" and not any("identify_growth_drivers" in req.answer_part_ids for req in requests):
        requests.extend(deterministic_causal_research_plan(user_query=user_query, companies=normalized_companies).evidence_requests)
        warnings.append({"field": "evidence_requests", "reason": "causal_driver_request_missing_injected"})

    valid = bool(normalized_companies) and bool(parts) and bool(requests)
    if question_type == "causal_explanation":
        valid = valid and any(part.id == "identify_growth_drivers" for part in parts)
    forbidden_claims = _ordered_unique([str(x) for x in raw.get("forbidden_claims", []) or [] if str(x)])
    if question_type == "causal_explanation":
        forbidden_claims.extend(["Do not infer causes from numeric growth alone."])
    if safety_intent == "investment_advice_like" or question_type == "investment_boundary":
        forbidden_claims.extend(
            [
                "Do not provide buy, sell, or hold recommendations.",
                "Do not provide target prices or deterministic stock forecasts.",
            ]
        )
    plan = ResearchPlan(
        question_type=question_type,  # type: ignore[arg-type]
        user_goal=str(raw.get("user_goal") or raw.get("goal") or user_query).strip()[:500],
        companies=normalized_companies,
        time_scope=str(raw.get("time_scope") or "latest").strip()[:120],
        required_answer_parts=parts,
        evidence_requests=requests,
        fallback_answer_policy=_fallback_policy(raw.get("fallback_answer_policy"), question_type),
        minimum_answer_policy=_minimum_answer_policy(raw.get("minimum_answer_policy"), question_type),
        reasoning_policy=_reasoning_policy(raw.get("reasoning_policy"), question_type, safety_intent),
        forbidden_claims=_ordered_unique(forbidden_claims),
        planner_source="llm_validated",
        plan_warnings=warnings,
    )
    return PlanValidationResult(
        valid=valid,
        used_fallback=False,
        fallback_reason="" if valid else "plan_validation_failed",
        warnings=warnings,
        rejected_items=rejected,
        corrected_fields=corrected,
        plan=plan,
    )
