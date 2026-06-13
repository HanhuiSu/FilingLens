"""Typed internal domain objects for the financial-analysis agent.

These Pydantic models are intentionally internal. LangGraph state and FastAPI
wire payloads remain plain dictionaries for compatibility.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentDomainModel(BaseModel):
    """Base model with permissive input for incremental migration."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class PeriodQuery(AgentDomainModel):
    period_type: str | None = None
    year: int | None = None
    quarter: int | None = None
    trailing_n: int | None = None
    year_basis: str = "fiscal"
    comparison_basis: str = "same_period"
    is_explicit: bool = False
    needs_clarification: bool = False
    clarification_reason: str | None = None


class ResolvedPeriodContext(AgentDomainModel):
    target_period_type: str | None = None
    year_basis: str = "fiscal"
    comparison_basis: str = "same_period"
    comparison_basis_label: str = "same_period"
    needs_clarification: bool = False
    clarification_reason: str | None = None
    common_periods: list[str] = Field(default_factory=list)


class EventQuery(AgentDomainModel):
    event_type: str = "any"
    fiscal_period: str | None = None
    event_date: str | None = None
    latest_n: int = 4
    window_days: list[int] = Field(default_factory=lambda: [1, 5, 10])
    sort_by: str = "event_date"
    sort_order: str = "desc"


class RetrievalPolicy(AgentDomainModel):
    retrieval_profile: str = "summary"
    text_top_k: int = 3
    max_per_filing: int = 2
    max_per_section: int = 1
    comparison_text_cap_per_company: int = 2
    require_balanced_comparison_text: bool = False
    skip_fact_text_when_structured_sufficient: bool = False
    section_allowlist: list[str] | None = None
    strict_sections: bool = False
    event_intent: str = "none"
    market_reaction_requested: bool = False


RequirementScope = Literal["core", "optional_context", "diagnostic"]
AnswerQualityTier = Literal[
    "true_answer",
    "bounded_analysis",
    "bounded_answer",
    "scope_limit",
    "safe_refusal",
    "invalid_fallback",
]


class CanonicalIntent(AgentDomainModel):
    intent_family: str = "overview"
    analysis_scope: str = "unknown"
    output_language: str = "zh"
    companies: list[str] = Field(default_factory=list)
    requested_dimensions: list[str] = Field(default_factory=list)
    segment_focus: str = ""
    segment_or_product_scope: str = ""
    time_focus: str = "latest"
    user_expectation: str = "quick_answer"
    safety_intent: str = "normal"
    confidence: float = 0.0
    source_signals: list[dict[str, Any]] = Field(default_factory=list)
    legacy_methodology_intent: str = ""
    answer_mode: str = "direct_fact"
    intent_merge_decision: dict[str, Any] = Field(default_factory=dict)
    time_scope: dict[str, Any] = Field(default_factory=dict)


class EvidencePolicy(AgentDomainModel):
    policy_id: str = "default"
    intent_family: str = "overview"
    answer_mode: str = "analytical"
    primary_dimension: str = ""
    required_dimensions: list[str] = Field(default_factory=list)
    optional_dimensions: list[str] = Field(default_factory=list)
    core_requirements: list[str] = Field(default_factory=list)
    optional_context_requirements: list[str] = Field(default_factory=list)
    diagnostic_requirements: list[str] = Field(default_factory=list)
    sufficiency_rule: str = "all_core_requirements"
    allowed_degradation: list[str] = Field(default_factory=list)


ResearchQuestionType = Literal[
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
]

EvidenceRequestType = Literal["numeric", "text", "calculation", "event"]
DriverLevel = Literal[
    "company_level_driver",
    "segment_level_driver",
    "product_level_driver",
    "market_context",
    "unknown",
]
ClaimScope = Literal["company", "segment", "product", "market_context", "unknown"]
AllowedClaimStrength = Literal["definitive", "bounded_inference", "hypothesis_only"]


class ClaimTier(str, Enum):
    EVIDENCE_BACKED = "evidence_backed"
    EVIDENCE_INFERRED = "evidence_inferred"
    HYPOTHESIS_TO_VERIFY = "hypothesis_to_verify"


class AnalyticalClaim(AgentDomainModel):
    id: str = ""
    text: str = ""
    tier: ClaimTier = ClaimTier.EVIDENCE_BACKED
    citation_refs: list[str] = Field(default_factory=list)
    supporting_claim_ids: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    caveat: str | None = None


class ReasoningPolicy(AgentDomainModel):
    allow_inference: bool = True
    allow_hypotheses: bool = True
    must_separate_claim_tiers: bool = True
    forbid_uncited_definitive_causal_claims: bool = True


class PlanExecutionStrategy(str, Enum):
    REPLACE = "replace"
    MERGE = "merge"
    AUGMENT_ONLY = "augment_only"
    LEGACY_ONLY = "legacy_only"


class CoverageDecision(AgentDomainModel):
    strategy: PlanExecutionStrategy = PlanExecutionStrategy.LEGACY_ONLY
    legacy_core_count: int = 0
    research_core_count: int = 0
    retained_legacy_core_count: int = 0
    dropped_legacy_core_ids: list[str] = Field(default_factory=list)
    added_research_requirement_ids: list[str] = Field(default_factory=list)
    coverage_ratio: float = 1.0
    warnings: list[str] = Field(default_factory=list)
    reason: str = ""


class RequirementMergeSummary(AgentDomainModel):
    strategy: PlanExecutionStrategy = PlanExecutionStrategy.LEGACY_ONLY
    merged_total_requirements: int = 0
    deduped_requirements: int = 0
    legacy_only_count: int = 0
    research_only_count: int = 0
    legacy_research_count: int = 0
    retained_legacy_core_count: int = 0
    added_research_requirement_ids: list[str] = Field(default_factory=list)
    dropped_legacy_core_ids: list[str] = Field(default_factory=list)


class TextEvidenceQuality(AgentDomainModel):
    is_valid: bool = True
    reason: str = ""
    snippet_length: int = 0
    semantic_term_count: int = 0
    is_section_header_only: bool = False
    specificity_score: float = 0.0


class EvidenceValidationRecord(AgentDomainModel):
    requirement_id: str = ""
    evidence_type: str = ""
    tool: str = ""
    tool_returned_count: int = 0
    validated_evidence_count: int = 0
    rejected_evidence_reason: str = ""
    status: str = ""


class RequiredAnswerPart(AgentDomainModel):
    id: str
    description: str = ""
    required: bool = True
    evidence_request_ids: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    forbidden_without_evidence: list[str] = Field(default_factory=list)


class EvidenceRequest(AgentDomainModel):
    id: str
    type: EvidenceRequestType = "numeric"
    scope: RequirementScope = "core"
    company: str | None = None
    metrics: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    tool: str | None = None
    required: bool = True
    min_results: int = 1
    answer_part_ids: list[str] = Field(default_factory=list)
    purpose: str = ""
    fallback_strategy: list[str] = Field(default_factory=list)
    evidence_role: str = ""
    alternative_group: str = ""


class ResearchPlan(AgentDomainModel):
    question_type: ResearchQuestionType = "unknown"
    user_goal: str = ""
    companies: list[str] = Field(default_factory=list)
    time_scope: str = "latest"
    required_answer_parts: list[RequiredAnswerPart] = Field(default_factory=list)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)
    fallback_answer_policy: str = ""
    minimum_answer_policy: str = ""
    reasoning_policy: ReasoningPolicy = Field(default_factory=ReasoningPolicy)
    forbidden_claims: list[str] = Field(default_factory=list)
    planner_source: str = ""
    plan_warnings: list[dict[str, Any]] = Field(default_factory=list)


class PlanValidationResult(AgentDomainModel):
    valid: bool = False
    used_fallback: bool = False
    fallback_reason: str = ""
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    rejected_items: list[dict[str, Any]] = Field(default_factory=list)
    corrected_fields: list[dict[str, Any]] = Field(default_factory=list)
    plan: ResearchPlan = Field(default_factory=ResearchPlan)


class AnswerRelevanceDecision(AgentDomainModel):
    decision: str = "not_run"
    status: str = "not_run"
    route: str = "finalize"
    action: Literal["pass", "add_citation", "strip_sentence", "scope_limit", "downgrade_to_bounded", "retry_evidence", "block"] = "pass"
    covered_answer_parts: list[str] = Field(default_factory=list)
    missing_answer_parts: list[str] = Field(default_factory=list)
    missing_required_answer_parts: list[str] = Field(default_factory=list)
    partial_answer_parts: list[str] = Field(default_factory=list)
    partial_required_answer_parts: list[str] = Field(default_factory=list)
    missing_but_analyzable_answer_parts: list[str] = Field(default_factory=list)
    missing_and_unanswerable_answer_parts: list[str] = Field(default_factory=list)
    deterministic_relevance_failures: list[dict[str, Any]] = Field(default_factory=list)
    llm_relevance_notes: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    repair_instructions: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    public_summary: str = ""
    final_route: str = ""


class ContractDecision(AgentDomainModel):
    decision: str = "passed"
    route: str = "pass"
    hard_blockers: list[dict[str, Any]] = Field(default_factory=list)
    repairable: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    optional_gaps: list[dict[str, Any]] = Field(default_factory=list)


class DraftReleaseDecision(AgentDomainModel):
    decision: str = "not_run"
    released: bool = False
    source: str = ""
    reason: str = ""
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class AnswerCandidate(AgentDomainModel):
    body: str = ""
    owner: str = ""
    requested_dimensions: list[str] = Field(default_factory=list)
    answered_dimensions: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    allowed_repairs: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class AnalysisPlan(AgentDomainModel):
    user_intent: str = ""
    task_type: str = "fact_qa"
    answer_mode: str = "direct_fact"
    safety_intent: str = "normal"
    methodology_intent: str = ""
    analysis_scope: str = ""
    primary_dimension: str = ""
    required_dimensions: list[str] = Field(default_factory=list)
    optional_dimensions: list[str] = Field(default_factory=list)
    requested_dimensions: list[str] = Field(default_factory=list)
    supporting_context_dimensions: list[str] = Field(default_factory=list)
    time_policy: str = ""
    period_scope: str = ""
    companies: list[str] = Field(default_factory=list)
    time_range: dict[str, Any] | None = None
    analysis_dimensions: list[str] = Field(default_factory=list)
    needed_evidence: list[str] = Field(default_factory=list)
    proposed_tools: list[str] = Field(default_factory=list)
    validated_tools: list[str] = Field(default_factory=list)
    section_preferences: list[str] = Field(default_factory=list)
    metric_requirements: list[str] = Field(default_factory=list)
    answer_policy: dict[str, Any] = Field(default_factory=dict)
    rejected_plan_items: list[dict[str, Any]] = Field(default_factory=list)
    canonical_intent: dict[str, Any] = Field(default_factory=dict)
    evidence_policy: dict[str, Any] = Field(default_factory=dict)
    evidence_policy_id: str = ""


class SafetyDecision(AgentDomainModel):
    answer_mode: str = "direct_fact"
    safety_intent: str = "normal"
    needs_tools: bool = True
    requires_non_advisory_framing: bool = False
    requires_forward_looking_caution: bool = False
    disallows_realtime_claims: bool = False
    policy_reasons: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceRequirement(AgentDomainModel):
    requirement_id: str
    requirement_type: str = "numeric"
    company: str | None = None
    framework_id: str | None = None
    dimension_id: str | None = None
    dimension_name: str | None = None
    analysis_purpose: str | None = None
    metric: str | None = None
    metrics: list[str] = Field(default_factory=list)
    period_type: str | None = None
    period_end: str | None = None
    section_preferences: list[str] = Field(default_factory=list)
    retrieval_query: str | None = None
    purpose: str = ""
    required: bool = True
    requirement_scope: RequirementScope = "core"
    min_results: int = 1
    fallback_strategy: list[str] = Field(default_factory=list)
    retrieval_intent: str = ""
    retrieval_profile: str = ""
    primary_sections: list[str] = Field(default_factory=list)
    fallback_sections: list[str] = Field(default_factory=list)
    broadened_queries: list[str] = Field(default_factory=list)
    answer_part_ids: list[str] = Field(default_factory=list)
    evidence_request_id: str | None = None
    evidence_role: str = ""
    alternative_group: str = ""
    segment_or_product_scope: str = ""


class EvidencePlan(AgentDomainModel):
    user_query: str = ""
    task_type: str = "fact_qa"
    answer_mode: str = "direct_fact"
    safety_intent: str = "normal"
    methodology_intent: str = ""
    analysis_scope: str = ""
    primary_dimension: str = ""
    required_dimensions: list[str] = Field(default_factory=list)
    optional_dimensions: list[str] = Field(default_factory=list)
    supporting_context_dimensions: list[str] = Field(default_factory=list)
    evidence_policy_id: str = ""
    evidence_policy: dict[str, Any] = Field(default_factory=dict)
    canonical_intent: dict[str, Any] = Field(default_factory=dict)
    time_policy: str = ""
    period_scope: str = ""
    analysis_goal: str = ""
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)
    sufficiency_criteria: dict[str, Any] = Field(default_factory=dict)
    core_requirement_ids: list[str] = Field(default_factory=list)
    optional_context_requirement_ids: list[str] = Field(default_factory=list)
    diagnostic_requirement_ids: list[str] = Field(default_factory=list)
    expected_synthesis_style: str = "analytical_brief"
    rejected_requirements: list[dict[str, Any]] = Field(default_factory=list)
    research_plan: dict[str, Any] = Field(default_factory=dict)
    required_answer_parts: list[dict[str, Any]] = Field(default_factory=list)
    evidence_request_map: dict[str, dict[str, Any]] = Field(default_factory=dict)
    plan_source: str = ""
    plan_coverage_decision: dict[str, Any] = Field(default_factory=dict)
    requirement_merge_summary: dict[str, Any] = Field(default_factory=dict)


class EvidenceCollectionResult(AgentDomainModel):
    requirement_id: str = ""
    status: str = "missing"
    evidence_type: str = "numeric"
    items: list[dict[str, Any]] = Field(default_factory=list)
    failure_reason: str | None = None
    retry_count: int = 0
    framework_id: str = ""
    dimension_id: str = ""
    dimension_name: str = ""
    analysis_purpose: str = ""
    company: str = ""
    retrieval_query: str = ""
    section_preferences: list[str] = Field(default_factory=list)
    fallback_queries: list[str] = Field(default_factory=list)
    fallback_sections: list[str] = Field(default_factory=list)
    raw_hit_count: int = 0
    section_filtered_hit_count: int = 0
    usable_hit_count: int = 0
    snippet_support_passed_count: int = 0
    raw_text_claim_count: int = 0
    candidate_text_claim_count: int = 0
    validated_text_claim_count: int = 0
    text_claim_validated_count: int = 0
    rejected_text_claims: list[dict[str, Any]] = Field(default_factory=list)
    text_claim_validation_warnings: list[dict[str, Any]] = Field(default_factory=list)
    claim_generation_error: str = ""
    text_citation_kept_count: int = 0
    final_validated_text_count: int = 0
    drop_stage: str | None = None
    top_raw_snippets: list[dict[str, Any]] = Field(default_factory=list)
    top_rejected_snippets: list[dict[str, Any]] = Field(default_factory=list)
    rejection_reasons: dict[str, int] = Field(default_factory=dict)


class EvidenceSufficiencyResult(AgentDomainModel):
    overall_status: str = "insufficient"
    satisfied_requirements: list[str] = Field(default_factory=list)
    partial_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    rejected_requirements: list[dict[str, Any]] = Field(default_factory=list)
    degradation_reason: str | None = None
    can_synthesize: bool = False
    requirement_limitations: list[dict[str, Any]] = Field(default_factory=list)
    required_numeric_satisfied_rate: float | None = None
    required_text_satisfied_rate: float | None = None
    missing_required_requirements: list[str] = Field(default_factory=list)
    missing_optional_requirements: list[str] = Field(default_factory=list)
    missing_required_requirements_count: int = 0
    missing_optional_requirements_count: int = 0
    company_evidence_balance: float | None = None
    # Canonical DimensionStatus field is dimension_status_by_id; dimension_status_map is a legacy alias.
    dimension_status_by_id: dict[str, dict[str, Any]] = Field(default_factory=dict)
    dimension_status_map: dict[str, dict[str, Any]] = Field(default_factory=dict)
    covered_dimensions: list[str] = Field(default_factory=list)
    satisfied_dimensions: list[str] = Field(default_factory=list)
    partial_dimensions: list[str] = Field(default_factory=list)
    missing_dimensions: list[str] = Field(default_factory=list)
    dimension_coverage_rate: float = 1.0
    weighted_dimension_coverage_rate: float = 1.0
    framework_sufficiency_status: str = "sufficient"
    answer_part_status_by_id: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evidence_gap_by_answer_part: dict[str, dict[str, Any]] = Field(default_factory=dict)
    missing_required_answer_parts: list[str] = Field(default_factory=list)
    partial_required_answer_parts: list[str] = Field(default_factory=list)
    missing_but_analyzable_answer_parts: list[str] = Field(default_factory=list)
    missing_and_unanswerable_answer_parts: list[str] = Field(default_factory=list)
    answer_parts_fully_satisfied: bool = True
    answer_parts_clean_pass: bool = True
    evidence_health: str = "complete"
    tool_error_context: list[dict[str, Any]] = Field(default_factory=list)


class QueryPlan(AgentDomainModel):
    task_type: str = "fact_qa"
    answer_mode: str = "direct_fact"
    safety_intent: str = "normal"
    methodology_intent: str = ""
    analysis_scope: str = ""
    primary_dimension: str = ""
    required_dimensions: list[str] = Field(default_factory=list)
    optional_dimensions: list[str] = Field(default_factory=list)
    time_policy: str = ""
    period_scope: str = ""
    format_constraints: dict[str, Any] = Field(default_factory=dict)
    needs_clarification: bool = False
    clarification_question: str | None = None
    needs_tools: bool = True
    period_query: PeriodQuery = Field(default_factory=PeriodQuery)
    resolved_period_context: ResolvedPeriodContext = Field(default_factory=ResolvedPeriodContext)
    comparison_basis_label: str = "same_period"
    selected_tools: list[str] = Field(default_factory=list)
    retrieval_policy: RetrievalPolicy = Field(default_factory=RetrievalPolicy)
    event_intent: str = "none"
    market_reaction_requested: bool = False
    event_query: EventQuery = Field(default_factory=EventQuery)
    analysis_plan: AnalysisPlan = Field(default_factory=AnalysisPlan)
    evidence_plan: EvidencePlan = Field(default_factory=EvidencePlan)


class NumericEvidence(AgentDomainModel):
    evidence_id: str
    requirement_id: str = ""
    ticker: str = ""
    metric: str = ""
    period_type: str = ""
    period_end: str = ""
    value: Any = None
    unit: str = ""
    source_tool: str = ""
    provenance: str = ""
    source_provider: str = ""
    source_url: str = ""
    source_filing_id: str = ""
    confidence: str = ""
    extraction_method: str = ""
    source_tag: str = ""
    reconciliation_warning: str = ""


class TextEvidence(AgentDomainModel):
    evidence_id: str
    requirement_id: str = ""
    ticker: str = ""
    filing_id: str = ""
    form_type: str = ""
    fiscal_period: str = ""
    section: str = ""
    chunk_order: int = 0
    text_snippet: str = ""
    supporting_snippet: str = ""
    supporting_terms: list[str] = Field(default_factory=list)
    score_breakdown: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(AgentDomainModel):
    numeric_evidence: list[dict[str, Any]] = Field(default_factory=list)
    text_evidence: list[dict[str, Any]] = Field(default_factory=list)
    numeric_map: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evidence_map: dict[str, dict[str, Any]] = Field(default_factory=dict)


class EvidencePacketTheme(AgentDomainModel):
    theme_code: str = ""
    label: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    snippet_count: int = 0


class EvidencePacket(AgentDomainModel):
    user_query: str = ""
    task_type: str = "fact_qa"
    answer_mode: str = "direct_fact"
    safety_intent: str = "normal"
    analysis_scope: str = ""
    time_policy: str = ""
    period_scope: str = ""
    selected_framework: dict[str, Any] = Field(default_factory=dict)
    active_dimensions: list[str] = Field(default_factory=list)
    numeric_table: list[dict[str, Any]] = Field(default_factory=list)
    comparison_table: list[dict[str, Any]] = Field(default_factory=list)
    text_snippets: list[dict[str, Any]] = Field(default_factory=list)
    numeric_evidence_by_dimension: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    text_evidence_by_dimension: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    available_metrics_by_dimension: dict[str, list[str]] = Field(default_factory=dict)
    missing_metrics_by_dimension: dict[str, list[str]] = Field(default_factory=dict)
    text_evidence_by_dimension_refs: dict[str, list[str]] = Field(default_factory=dict)
    computed_metric_dependencies: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    valuation_period_basis: dict[str, str] = Field(default_factory=dict)
    grouped_risk_themes: list[EvidencePacketTheme] = Field(default_factory=list)
    grouped_business_themes: list[EvidencePacketTheme] = Field(default_factory=list)
    provenance_notes: list[str] = Field(default_factory=list)
    missing_evidence_summary: dict[str, Any] = Field(default_factory=dict)
    dimension_sufficiency: dict[str, Any] = Field(default_factory=dict)
    # Canonical DimensionStatus field is dimension_status_by_id; dimension_status_map is a legacy alias.
    dimension_status_by_id: dict[str, dict[str, Any]] = Field(default_factory=dict)
    dimension_status_map: dict[str, dict[str, Any]] = Field(default_factory=dict)
    dimension_summary: list[dict[str, Any]] = Field(default_factory=list)
    red_flags: list[dict[str, Any]] = Field(default_factory=list)
    missing_evidence_flags: list[dict[str, Any]] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    allowed_claims: list[str] = Field(default_factory=list)
    limitations: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    comparison_judgment_frame: dict[str, Any] = Field(default_factory=dict)


class AnalystDraftItem(AgentDomainModel):
    statement: str = ""
    citation_refs: list[str] = Field(default_factory=list)


class AnalystDraftConclusion(AgentDomainModel):
    statement: str = ""
    stance: str = ""
    preferred_company: str = ""
    citation_refs: list[str] = Field(default_factory=list)


class AnalystDraftDimensionAnalysis(AgentDomainModel):
    dimension_id: str = ""
    status: str = ""
    claim: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class AnalystDraft(AgentDomainModel):
    framework_summary: str = ""
    dimension_analyses: list[AnalystDraftDimensionAnalysis] = Field(default_factory=list)
    overall_judgment: str = ""
    methodology_counterpoints: list[str] = Field(default_factory=list)
    methodology_limitations: list[str] = Field(default_factory=list)
    follow_up_metrics: list[str] = Field(default_factory=list)
    tentative_conclusion: AnalystDraftConclusion = Field(default_factory=AnalystDraftConclusion)
    decision_basis: list[AnalystDraftItem] = Field(default_factory=list)
    supporting_points: list[AnalystDraftItem] = Field(default_factory=list)
    counterpoints: list[AnalystDraftItem] = Field(default_factory=list)
    risk_tradeoffs: list[AnalystDraftItem] = Field(default_factory=list)
    uncertainty_notes: list[AnalystDraftItem] = Field(default_factory=list)
    citation_refs: list[str] = Field(default_factory=list)
    safety_notes: list[AnalystDraftItem] = Field(default_factory=list)


class DraftViolation(AgentDomainModel):
    field: str = ""
    reason: str = ""
    item: dict[str, Any] = Field(default_factory=dict)
    unknown_refs: list[str] = Field(default_factory=list)


class DraftValidationResult(AgentDomainModel):
    passed: bool = False
    status: str = "invalid"
    final_status: str = "invalid"
    accepted_draft: dict[str, Any] = Field(default_factory=dict)
    violations: list[DraftViolation] = Field(default_factory=list)
    rejected_items: list[dict[str, Any]] = Field(default_factory=list)
    repair_instructions: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None


class AnalystDraftValidationResult(DraftValidationResult):
    pass


class ComparisonJudgmentFrame(AgentDomainModel):
    companies: list[str] = Field(default_factory=list)
    preferred_company: str = ""
    preference_type: str = ""
    profitability_winner: str = ""
    profitability_reason: str = ""
    scale_winner: str = ""
    scale_reason: str = ""
    margin_winner: str = ""
    margin_reason: str = ""
    growth_winner: str = "unavailable"
    growth_reason: str = ""
    risk_tradeoff: str = ""
    counterpoint: str = ""
    risk_considerations: list[dict[str, Any]] = Field(default_factory=list)
    confidence_level: str = ""
    evidence_basis: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    limitations: list[str] = Field(default_factory=list)


class ValidatedClaim(AgentDomainModel):
    sentence: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    claim_type: str = ""


class CitationPolicyResult(AgentDomainModel):
    text_citations: list[dict[str, Any]] = Field(default_factory=list)
    policy_reasons: list[dict[str, Any]] = Field(default_factory=list)
    comparison_text_unbalanced: bool = False


class MethodologyDimensionSection(AgentDomainModel):
    dimension_id: str = ""
    title: str = ""
    status: Literal["satisfied", "partial", "missing"] = "missing"
    summary: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    limitation: str | None = None


class MethodologyAnswer(AgentDomainModel):
    framework_id: str = ""
    task_type: str = ""
    answer_mode: str = ""
    analysis_scope: str = ""
    judgment: str = ""
    overview_profile: dict[str, Any] = Field(default_factory=dict)
    data_sanity_flags: list[dict[str, Any]] = Field(default_factory=list)
    overview_risk_ranking: list[dict[str, Any]] = Field(default_factory=list)
    metric_table: list[dict[str, Any]] = Field(default_factory=list)
    single_company_metric_table: list[dict[str, Any]] = Field(default_factory=list)
    dimension_sections: list[MethodologyDimensionSection] = Field(default_factory=list)
    counterpoint: str | None = None
    limitations: list[str] = Field(default_factory=list)
    follow_up_metrics: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class RiskTheme(AgentDomainModel):
    theme_key: str = ""
    theme_id: str = ""
    company: str = ""
    theme_name: str = ""
    severity_hint: str = ""
    why_it_matters: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    supporting_quote: str = ""
    supporting_snippet: str = ""
    related_financial_context: list[str] = Field(default_factory=list)
    direct_support_score: float = 0.0
    mechanism_support_level: Literal["direct", "inferred", "template_only"] = "inferred"
    score: float = 0.0


class RiskFocusedAnswer(AgentDomainModel):
    company: str = ""
    answer_mode: str = "risk_focused_analysis"
    analysis_scope: str = "single_company"
    top_risk: RiskTheme | None = None
    secondary_risks: list[RiskTheme] = Field(default_factory=list)
    direct_judgment: str = ""
    risk_ranking: list[dict[str, Any]] = Field(default_factory=list)
    supplemental_risk_signals: list[dict[str, Any]] = Field(default_factory=list)
    template_only_risk_items: list[dict[str, Any]] = Field(default_factory=list)
    why_core_issue: list[str] = Field(default_factory=list)
    transmission_path: list[str] = Field(default_factory=list)
    filing_evidence: list[dict[str, Any]] = Field(default_factory=list)
    financial_context: list[str] = Field(default_factory=list)
    financial_impact_analysis: list[str] = Field(default_factory=list)
    key_observation_metrics: list[str] = Field(default_factory=list)
    evidence_boundaries: list[str] = Field(default_factory=list)
    scenario_anchor: dict[str, Any] = Field(default_factory=dict)
    scenario_caveated: bool = False
    evidence_refs: list[str] = Field(default_factory=list)


class AnalyticalSynthesis(AgentDomainModel):
    short_answer: str = ""
    key_facts: list[dict[str, Any]] = Field(default_factory=list)
    analysis: list[dict[str, Any]] = Field(default_factory=list)
    risks_or_uncertainties: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    synthesis_strategy: str = "synthesis_degraded"
    synthesis_mode: str = "synthesis_degraded"
    accepted_draft: dict[str, Any] = Field(default_factory=dict)
    comparison_judgment_frame: dict[str, Any] = Field(default_factory=dict)
    methodology_answer: dict[str, Any] = Field(default_factory=dict)
    risk_focused_answer: dict[str, Any] = Field(default_factory=dict)
    methodology_report: dict[str, Any] = Field(default_factory=dict)
    final_answer_source: str = ""
    draft_status: str = ""
    draft_final_status: str = ""
    unsupported_synthesis_items: list[dict[str, Any]] = Field(default_factory=list)
    requirement_summary: dict[str, Any] = Field(default_factory=dict)
    missing_requirements: list[str] = Field(default_factory=list)
    degradation_reason: str | None = None
    research_plan: dict[str, Any] = Field(default_factory=dict)
    numeric_evidence_cards: list[dict[str, Any]] = Field(default_factory=list)
    text_evidence_cards: list[dict[str, Any]] = Field(default_factory=list)
    analytical_claims: list[AnalyticalClaim] = Field(default_factory=list)
    claim_tiers: dict[str, int] = Field(default_factory=dict)
    analytical_reasoning_status: str = ""
    evidence_health: str = "complete"
    tool_error_context: list[dict[str, Any]] = Field(default_factory=list)
    segment_or_product_scope: str = ""
    user_query: str = ""
    analysis_plan: dict[str, Any] = Field(default_factory=dict)
    answer_policy: dict[str, Any] = Field(default_factory=dict)
    task_type: str = ""
    answer_mode: str = ""
    safety_intent: str = ""


class RenderedAnswer(AgentDomainModel):
    final_answer: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
