"""Request / response Pydantic models for the FastAPI service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.agent.progress import validate_trace_id


# ── Requests ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question about financial filings or data",
    )
    client_trace_id: str | None = Field(
        default=None,
        description="Optional client-generated trace id used for live progress polling",
    )

    @field_validator("client_trace_id")
    @classmethod
    def _validate_client_trace_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return validate_trace_id(text)


# ── Responses ─────────────────────────────────────────────────────

class NumericEvidenceCard(BaseModel):
    evidence_id: str = ""
    requirement_id: str = ""
    ticker: str = ""
    metric: str = ""
    value: float | int | str | None = None
    display_value: str = ""
    unit: str = ""
    period_type: str = ""
    period_end: str = ""
    source_tool: str = ""
    provenance: str = ""
    source_provider: str = ""
    source_url: str = ""
    source_filing_id: str = ""
    confidence: str = ""
    extraction_method: str = ""
    source_tag: str = ""
    reconciliation_warning: str = ""


class TextEvidenceCard(BaseModel):
    evidence_id: str = ""
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
    dimension_id: str = ""
    claim: str = ""
    claim_source: str = ""
    original_claim: str = ""
    citation_ref: str = ""
    risk_theme: str = ""
    theme_name: str = ""
    evidence_summary: str = ""
    driver_level: str = ""
    driver_levels: list[str] = Field(default_factory=list)
    claim_scope: str = ""
    allowed_claim_strength: str = ""
    scope_reason: str = ""
    evidence_summary_scope_overclaim: bool = False
    summary_scope_warning: str = ""


class LimitationItem(BaseModel):
    code: str = ""
    severity: str = ""
    message: str = ""


class MarketReactionEvent(BaseModel):
    ticker: str = ""
    filing_id: str = ""
    form_type: str = ""
    fiscal_period: str = ""
    event_date: str = ""
    trading_anchor_date: str = ""
    return_1d: float | None = None
    return_3d: float | None = None
    return_5d: float | None = None
    return_10d: float | None = None
    coverage_flag: str = ""


class MarketReactionBlock(BaseModel):
    title: str = ""
    anchor_rule: str = ""
    events: list[MarketReactionEvent] = []
    highlights: list[str] = []
    limitations: list[str] = []


class TaskTable(BaseModel):
    columns: list[str] = []
    rows: list[dict[str, Any]] = []


class TaskView(BaseModel):
    kind: str = ""
    short_answer: str = ""
    key_facts: list[dict[str, Any]] = []
    analysis: list[dict[str, Any]] = []
    risks_or_uncertainties: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    limitations: list[dict[str, Any]] = []

    # fact_qa
    headline_metric: dict[str, Any] = {}
    period_note: str = ""
    supporting_points: list[str] = []

    # trend_analysis
    trend_conclusion: str = ""
    change_points: list[str] = []
    trend_table: TaskTable | None = None

    # company_comparison
    comparison_basis_line: str = ""
    comparison_table: TaskTable | None = None
    delta_summary: str = ""

    # report_summary
    executive_summary: str = ""
    key_data_points: list[str] = []
    text_findings: list[str] = []
    risk_and_limits: str = ""

    # conversational / synthesis views
    capabilities: list[str] = []
    boundaries: list[str] = []
    clarification_question: str = ""
    example_questions: list[str] = []
    redirect_reason: str = ""
    supported_questions: list[str] = []
    non_advisory_statement: str = ""
    comparison_dimensions: list[str] = []
    key_data: list[dict[str, Any]] = []
    analysis_judgment: list[dict[str, Any]] = []
    risks_and_limits: list[dict[str, Any]] = []
    disclosed_trends: list[dict[str, Any]] = []
    supporting_factors: list[dict[str, Any]] = []
    risk_factors: list[dict[str, Any]] = []
    cannot_predict: str = ""
    methodology_answer: dict[str, Any] = {}
    risk_focused_answer: dict[str, Any] = {}
    metric_table: list[dict[str, Any]] = []
    single_company_metric_table: list[dict[str, Any]] = []
    dimension_sections: list[dict[str, Any]] = []
    methodology_counterpoints: list[str] = []
    methodology_limitations: list[str] = []
    follow_up_metrics: list[str] = []
    risk_themes: list[dict[str, Any]] = []
    top_risk_theme: dict[str, Any] = {}
    financial_context: list[str] = []
    evidence_boundaries: list[str] = []


class OutputPayload(BaseModel):
    output_language: str = "zh"
    task_type: str
    answer_mode: str = "direct_fact"
    safety_intent: str = "normal"
    methodology_intent: str = ""
    analysis_scope: str = ""
    time_policy: str = ""
    period_scope: str = ""
    needs_tools: bool = True
    needs_clarification: bool = False
    clarification_question: str | None = None
    title: str
    summary: str
    comparison_basis: str = ""
    key_points: list[str]
    numeric_evidence: list[NumericEvidenceCard]
    text_evidence: list[TextEvidenceCard]
    limitations: list[LimitationItem]
    used_tools: list[str]
    trace_id: str
    view: TaskView
    protocol_version: str = "phase4.v1"
    market_reaction: MarketReactionBlock | dict[str, Any] | None = None
    analysis_plan_summary: dict[str, Any] = {}
    analysis_framework: dict[str, Any] = {}
    dimension_sufficiency: dict[str, Any] = {}
    red_flags: list[dict[str, Any]] = []
    evidence_plan_summary: dict[str, Any] = {}
    evidence_packet_summary: dict[str, Any] = {}
    comparison_judgment_frame: dict[str, Any] = {}
    analyst_draft_summary: dict[str, Any] = {}
    draft_validation: dict[str, Any] = {}
    draft_attempts: list[dict[str, Any]] = []
    draft_revision_attempts: list[dict[str, Any]] = []
    draft_violations: list[dict[str, Any]] = []
    draft_final_status: str = ""
    draft_status: str = ""
    final_answer_source: str = ""
    answer_history: list[dict[str, Any]] = []
    answer_candidate: dict[str, Any] = {}
    answer_candidates: list[dict[str, Any]] = []
    answer_quality_tier: str = ""
    quality_tier_reason: str = ""
    main_question_covered: bool = True
    fallback_intent_match: bool = True
    answered_dimensions: list[str] = []
    unresolved_relevance_failures: list[str] = []
    format_constraints_satisfied: bool = True
    format_constraints: dict[str, Any] = {}
    repair_applied: bool = False
    repair_owner: str = ""
    source_before_repair: str = ""
    repair_types: list[str] = []
    material_claim_uncited_count: int = 0
    core_missing_parts: list[str] = []
    optional_missing_parts: list[str] = []
    risk_items_directly_supported_count: int = 0
    risk_items_template_only_count: int = 0
    company_specific_token_leakage: int = 0
    language_leakage: int = 0
    language_leakage_unresolved: bool = False
    segment_or_product_scope: str = ""
    canonical_intent: dict[str, Any] = {}
    answer_status: str = ""
    contract_decision: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    sufficiency_status: str = ""
    degradation_reason: str | None = None
    synthesis: dict[str, Any] = {}
    synthesis_strategy: str = ""
    synthesis_mode: str = ""
    analytical_claims: list[dict[str, Any]] = []
    claim_tiers: dict[str, int] = {}
    analytical_reasoning_status: str = ""
    evidence_health: str = ""
    tool_error_context: list[dict[str, Any]] = []
    contract: dict[str, Any] = {}
    report: dict[str, Any] = {}


class Citation(BaseModel):
    source: str = Field(description="Ticker symbol")
    filing_type: str = Field(description="10-K or 10-Q")
    period: str = Field(description="Fiscal period label")
    section: str = Field(description="Filing section (e.g. ITEM_1A)")
    part: str = Field(default="", description="Part marker (e.g. PART_I)")
    quality: str = Field(default="", description="Chunk quality label (high|medium|low)")
    text_snippet: str = Field(default="", description="Short excerpt from the cited chunk")
    supporting_snippet: str = Field(default="", description="Specific sentence/span supporting the cited claim")
    supporting_terms: list[str] = Field(default_factory=list, description="Query terms found in the supporting snippet")
    score_breakdown: dict[str, Any] = Field(default_factory=dict, description="Retrieval score components when available")
    source_kind: str = Field(default="document", description="Source type: structured or document")
    metric: str = Field(default="", description="Structured metric name when source_kind=structured")
    period_type: str = Field(default="", description="Structured period type when available")
    period_end: str = Field(default="", description="Structured period end date when available")
    filing_date: str = Field(default="", description="Optional filing date for structured source rows")
    source_provider: str = Field(default="", description="Structured data provider, e.g. sec_companyfacts or yfinance")
    source_url: str = Field(default="", description="Provider URL for structured source rows when available")
    source_filing_id: str = Field(default="", description="SEC accession or provider record id when available")
    confidence: str = Field(default="", description="Source confidence label")
    extraction_method: str = Field(default="", description="How the structured fact was extracted")
    source_tag: str = Field(default="", description="Provider/XBRL/yfinance source tag")
    reconciliation_warning: str = Field(default="", description="Data-source reconciliation warning, if any")
    section_fallback: bool = Field(default=False, description="Whether citation came from section fallback path")


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    used_tools: list[str]
    task_type: str
    trace_id: str
    output: OutputPayload | dict[str, Any] = {}
    contract_status: str = "not_checked"
    canonical_intent: dict[str, Any] = {}
    answer_status: str = ""
    contract_decision: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    repair_attempts: int = 0
    limitations: list[str] = []
    final_answer_source: str = ""
    answer_history: list[dict[str, Any]] = []
    answer_quality_tier: str = ""
    quality_tier_reason: str = ""
    main_question_covered: bool = True
    fallback_intent_match: bool = True
    answered_dimensions: list[str] = []
    unresolved_relevance_failures: list[str] = []
    format_constraints_satisfied: bool = True
    repair_applied: bool = False
    repair_owner: str = ""
    source_before_repair: str = ""
    repair_types: list[str] = []
    material_claim_uncited_count: int = 0
    core_missing_parts: list[str] = []
    optional_missing_parts: list[str] = []
    risk_items_directly_supported_count: int = 0
    risk_items_template_only_count: int = 0
    company_specific_token_leakage: int = 0
    output_language: str = ""
    language_leakage: int = 0
    language_leakage_unresolved: bool = False
    segment_or_product_scope: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    llm_provider: str = ""
    llm_base_url: str = ""
    llm_reasoning_model: str = ""
    analyst_draft_enabled: bool = True
    analyst_draft_max_attempts: int = 0
    analyst_draft_max_tokens: int = 0


class TraceResponse(BaseModel):
    trace_id: str
    user_query: str
    output_language: str = ""
    task_type: str
    answer_mode: str = "direct_fact"
    safety_intent: str = "normal"
    methodology_intent: str = ""
    canonical_intent: dict[str, Any] = {}
    intent_merge_decision: dict[str, Any] = {}
    evidence_policy_id: str = ""
    evidence_policy: dict[str, Any] = {}
    query_understanding_summary: dict[str, Any] = {}
    analysis_scope: str = ""
    primary_dimension: str = ""
    required_dimensions: list[str] = []
    optional_dimensions: list[str] = []
    supporting_context_dimensions: list[str] = []
    time_policy: str = ""
    period_scope: str = ""
    needs_clarification: bool = False
    clarification_question: str | None = None
    needs_tools: bool = True
    data_route: str
    analysis_plan_raw: dict[str, Any] = {}
    analysis_plan: dict[str, Any] = {}
    selected_analysis_framework: dict[str, Any] = {}
    research_plan_raw: dict[str, Any] = {}
    research_plan_validated: dict[str, Any] = {}
    research_plan_used: dict[str, Any] = {}
    research_plan_validation: dict[str, Any] = {}
    research_plan_source: str = ""
    research_plan_fallback_reason: str = ""
    research_plan_duration_ms: int = 0
    required_answer_parts: list[dict[str, Any]] = []
    legacy_evidence_plan: dict[str, Any] = {}
    plan_coverage_decision: dict[str, Any] = {}
    requirement_merge_summary: dict[str, Any] = {}
    evidence_plan_used: dict[str, Any] = {}
    rejected_plan_items: list[dict[str, Any]] = []
    validated_tools: list[str] = []
    safety_decision: dict[str, Any] = {}
    safety_policy_reasons: list[dict[str, Any]] = []
    safety_limitations: list[dict[str, Any]] = []
    evidence_plan: dict[str, Any] = {}
    evidence_plan_summary: dict[str, Any] = {}
    evidence_requirements: list[dict[str, Any]] = []
    evidence_collection_results: list[dict[str, Any]] = []
    evidence_sufficiency: dict[str, Any] = {}
    evidence_sufficiency_summary: dict[str, Any] = {}
    answer_part_status_by_id: dict[str, Any] = {}
    evidence_gap_by_answer_part: dict[str, Any] = {}
    missing_required_answer_parts: list[str] = []
    partial_required_answer_parts: list[str] = []
    missing_but_analyzable_answer_parts: list[str] = []
    missing_and_unanswerable_answer_parts: list[str] = []
    evidence_health: str = ""
    tool_error_context: list[dict[str, Any]] = []
    collection_evidence_collection_results: list[dict[str, Any]] = []
    collection_evidence_sufficiency: dict[str, Any] = {}
    collection_evidence_sufficiency_summary: dict[str, Any] = {}
    evidence_retry_history: list[dict[str, Any]] = []
    retry_history: list[dict[str, Any]] = []
    requirement_limitations: list[dict[str, Any]] = []
    collected_evidence_by_requirement: dict[str, Any] = {}
    requirement_status_map: dict[str, Any] = {}
    selected_framework: str = ""
    active_dimensions: list[str] = []
    dimension_status_map: dict[str, Any] = {}
    dimension_status_by_id: dict[str, Any] = {}
    covered_dimensions: list[str] = []
    satisfied_dimensions: list[str] = []
    partial_dimensions: list[str] = []
    missing_dimensions: list[str] = []
    dimension_coverage_rate: float | None = None
    weighted_dimension_coverage_rate: float | None = None
    framework_sufficiency_status: str = ""
    red_flags: list[dict[str, Any]] = []
    missing_evidence_flags: list[dict[str, Any]] = []
    forbidden_claims: list[str] = []
    allowed_claims: list[str] = []
    available_metrics_by_dimension: dict[str, Any] = {}
    missing_metrics_by_dimension: dict[str, Any] = {}
    text_evidence_by_dimension: dict[str, Any] = {}
    final_methodology_coverage_rate: float | None = None
    final_requirement_status_map: dict[str, Any] = {}
    collection_requirement_status_map: dict[str, Any] = {}
    evidence_validation_records: list[dict[str, Any]] = []
    trace_summary: dict[str, Any] = {}
    collection_trace_summary: dict[str, Any] = {}
    missing_requirements: list[str] = []
    degradation_reason: str | None = None
    validated_requirement_ids: list[str] = []
    validated_numeric_evidence_count: int = 0
    validated_text_evidence_count: int = 0
    raw_retrieval_hits_by_requirement: dict[str, int] = {}
    text_requirement_diagnostics: dict[str, Any] = {}
    rejected_requirements: list[dict[str, Any]] = []
    synthesis: dict[str, Any] = {}
    evidence_packet: dict[str, Any] = {}
    evidence_packet_summary: dict[str, Any] = {}
    comparison_judgment_frame: dict[str, Any] = {}
    analyst_draft: dict[str, Any] = {}
    analyst_draft_validation: dict[str, Any] = {}
    draft_validation: dict[str, Any] = {}
    draft_attempts: list[dict[str, Any]] = []
    draft_revision_attempts: list[dict[str, Any]] = []
    draft_violations: list[dict[str, Any]] = []
    draft_final_status: str = ""
    draft_status: str = ""
    final_answer_source: str = ""
    answer_history: list[dict[str, Any]] = []
    answer_candidate: dict[str, Any] = {}
    answer_candidates: list[dict[str, Any]] = []
    draft_release_decision: dict[str, Any] = {}
    synthesis_strategy: str = ""
    synthesis_mode: str = ""
    analytical_claims: list[dict[str, Any]] = []
    claim_tiers: dict[str, int] = {}
    analytical_reasoning_status: str = ""
    unsupported_synthesis_items: list[dict[str, Any]] = []
    why_tools_skipped: list[dict[str, Any]] = []
    companies: list[str]
    comparison_target: str | None = None
    time_range: dict[str, str] | None = None
    period_query: dict[str, Any] = {}
    resolved_period_context: dict[str, Any] = {}
    comparison_basis_label: str = ""
    requested_metrics: list[str]
    selected_tools: list[str]
    retrieval_policy: dict[str, Any] = {}
    retrieval_debug: dict[str, Any] = {}
    event_intent: str = "none"
    market_reaction_requested: bool = False
    event_query: dict[str, Any] = {}
    event_results: list[dict[str, Any]] = []
    market_reaction_evidence: list[dict[str, Any]] = []
    market_reaction_limitations: list[str] = []
    tool_results: list[dict[str, Any]]
    numeric_evidence: list[dict[str, Any]] = []
    text_evidence: list[dict[str, Any]] = []
    unsupported_claims: list[dict[str, Any]] = []
    numeric_citations: list[dict[str, Any]] = []
    text_citations: list[dict[str, Any]] = []
    citations: list[dict[str, Any]]
    output: OutputPayload | dict[str, Any] = {}
    structured_sources: list[dict[str, Any]] = []
    document_citations: list[dict[str, Any]] = []
    contract_result: dict[str, Any] = {}
    contract_decision: dict[str, Any] = {}
    contract_status: str = "not_checked"
    contract_attempts: int = 0
    repair_actions: list[dict[str, Any]] = []
    final_contract_status: str = "not_checked"
    contract_public_summary: str = ""
    contract_evidence_retry_count: int = 0
    relevance_decision: dict[str, Any] = {}
    relevance_status: str = "not_run"
    relevance_repair_attempts: int = 0
    final_route: str = ""
    answer_quality_tier: str = ""
    quality_tier_reason: str = ""
    main_question_covered: bool = True
    fallback_intent_match: bool = True
    answered_dimensions: list[str] = []
    unresolved_relevance_failures: list[str] = []
    format_constraints_satisfied: bool = True
    format_constraints: dict[str, Any] = {}
    repair_applied: bool = False
    repair_owner: str = ""
    source_before_repair: str = ""
    repair_types: list[str] = []
    material_claim_uncited_count: int = 0
    core_missing_parts: list[str] = []
    optional_missing_parts: list[str] = []
    risk_items_directly_supported_count: int = 0
    risk_items_template_only_count: int = 0
    company_specific_token_leakage: int = 0
    language_leakage: int = 0
    language_leakage_unresolved: bool = False
    segment_or_product_scope: str = ""
    report: dict[str, Any] = {}
    report_contract_result: dict[str, Any] = {}
    report_contract_status: str = ""
    evidence_loop_count: int
    final_answer: str


class TraceUiResponse(BaseModel):
    trace_id: str
    query: str = ""
    user_query: str = ""
    final_answer: str = ""
    local_trace_path: str = ""
    companies: list[dict[str, Any]] = []
    task_type: str = ""
    answer_mode: str = ""
    canonical_intent: dict[str, Any] = {}
    intent_merge_decision: dict[str, Any] = {}
    evidence_policy_id: str = ""
    evidence_policy: dict[str, Any] = {}
    research_plan_raw: dict[str, Any] = {}
    research_plan_validated: dict[str, Any] = {}
    research_plan_used: dict[str, Any] = {}
    research_plan_validation: dict[str, Any] = {}
    required_answer_parts: list[dict[str, Any]] = []
    legacy_evidence_plan: dict[str, Any] = {}
    plan_coverage_decision: dict[str, Any] = {}
    requirement_merge_summary: dict[str, Any] = {}
    evidence_plan_used: dict[str, Any] = {}
    semantic_parser_mode: str = ""
    semantic_parser: dict[str, Any] = {}
    semantic_proposal: dict[str, Any] = {}
    rule_methodology_intent: str = ""
    proposed_methodology_intent: str = ""
    proposal_validation_warnings: list[dict[str, Any]] = []
    intent_conflict: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    contract_status: str = "not_run"
    contract_decision: dict[str, Any] = {}
    relevance_decision: dict[str, Any] = {}
    relevance_status: str = "not_run"
    draft_release_decision: dict[str, Any] = {}
    final_answer_source: str = ""
    answer_history: list[dict[str, Any]] = []
    answer_quality_tier: str = ""
    quality_tier_reason: str = ""
    main_question_covered: bool = True
    fallback_intent_match: bool = True
    answered_dimensions: list[str] = []
    unresolved_relevance_failures: list[str] = []
    format_constraints_satisfied: bool = True
    format_constraints: dict[str, Any] = {}
    repair_applied: bool = False
    repair_owner: str = ""
    source_before_repair: str = ""
    repair_types: list[str] = []
    material_claim_uncited_count: int = 0
    core_missing_parts: list[str] = []
    optional_missing_parts: list[str] = []
    risk_items_directly_supported_count: int = 0
    risk_items_template_only_count: int = 0
    company_specific_token_leakage: int = 0
    output_language: str = ""
    language_leakage: int = 0
    language_leakage_unresolved: bool = False
    segment_or_product_scope: str = ""
    repair_attempts: int = 0
    evidence_retry_count: int = 0
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    research_plan: dict[str, Any] = {}
    analytical_reasoning: dict[str, Any] = {}
    evidence_plan: dict[str, Any] = {}
    answer_part_status_by_id: dict[str, Any] = {}
    evidence_gap_by_answer_part: dict[str, Any] = {}
    missing_required_answer_parts: list[str] = []
    partial_required_answer_parts: list[str] = []
    missing_but_analyzable_answer_parts: list[str] = []
    missing_and_unanswerable_answer_parts: list[str] = []
    analytical_claims: list[dict[str, Any]] = []
    claim_tiers: dict[str, int] = {}
    analytical_reasoning_status: str = ""
    evidence_health: str = ""
    tool_error_context: list[dict[str, Any]] = []
    evidence_validation_records: list[dict[str, Any]] = []
    evidence_scope: dict[str, Any] = {}
    evidence_scope_by_ref: dict[str, Any] = {}
    scope_overclaim_check: dict[str, Any] = {}
    scope_overclaim_violations: list[dict[str, Any]] = []
    driver_scope_counts: dict[str, int] = {}
    evidence_packet: dict[str, Any] = {}
    dimensions: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    contract: dict[str, Any] = {}
    report: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = []
    progress_events: list[dict[str, Any]] = []


class ErrorResponse(BaseModel):
    detail: str
