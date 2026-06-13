"""AgentState — the shared state flowing through the LangGraph workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


class AgentState(TypedDict, total=False):
    """Mutable state dict passed between LangGraph nodes.

    Fields use *reducer* annotations where needed so that LangGraph
    merges partial updates correctly (e.g. ``messages`` uses the
    built-in ``add_messages`` reducer).
    """

    # ---- User input ----
    user_query: str
    output_language: str  # zh | en

    # ---- Classification / extraction ----
    task_type: str  # fact_qa | trend_analysis | company_comparison | report_summary
    answer_mode: str  # direct_fact | analytical | cautious_outlook | comparison_brief | clarification | meta | refusal_or_redirect
    safety_intent: str  # normal | investment_advice_like | unsupported_or_out_of_scope
    methodology_intent: str
    analysis_scope: str  # single_company | comparison | ""
    primary_dimension: str
    required_dimensions: list[str]
    optional_dimensions: list[str]
    supporting_context_dimensions: list[str]
    time_policy: str
    period_scope: str
    needs_clarification: bool
    clarification_question: str | None
    needs_tools: bool
    companies: list[str]  # primary ticker(s)
    comparison_target: str | None  # second ticker for comparison tasks
    time_range: dict[str, str] | None  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    period_query: dict[str, Any]
    resolved_period_context: dict[str, Any]
    comparison_basis_label: str
    requested_metrics: list[str]
    data_route: str  # documents_only | structured_only | hybrid
    query_understanding: dict[str, Any]
    query_understanding_summary: dict[str, Any]
    canonical_intent: dict[str, Any]
    intent_merge_decision: dict[str, Any]
    semantic_parser_mode: str
    semantic_parser: dict[str, Any]
    semantic_proposal: dict[str, Any]
    rule_methodology_intent: str
    proposed_methodology_intent: str
    proposal_validation_warnings: list[dict[str, Any]]
    intent_conflict: bool
    analysis_plan_raw: dict[str, Any]
    analysis_plan: dict[str, Any]
    selected_analysis_framework: dict[str, Any]
    research_plan_raw: dict[str, Any]
    research_plan_validated: dict[str, Any]
    research_plan_used: dict[str, Any]
    research_plan_validation: dict[str, Any]
    research_plan_source: str
    research_plan_fallback_reason: str
    research_plan_duration_ms: int
    required_answer_parts: list[dict[str, Any]]
    legacy_evidence_plan: dict[str, Any]
    plan_coverage_decision: dict[str, Any]
    requirement_merge_summary: dict[str, Any]
    evidence_plan_used: dict[str, Any]
    evidence_policy: dict[str, Any]
    evidence_policy_id: str
    rejected_plan_items: list[dict[str, Any]]
    validated_tools: list[str]
    safety_decision: dict[str, Any]
    safety_policy_reasons: list[dict[str, Any]]
    safety_limitations: list[dict[str, Any]]
    evidence_plan: dict[str, Any]
    evidence_requirements: list[dict[str, Any]]
    rejected_requirements: list[dict[str, Any]]

    # ---- Tool orchestration ----
    selected_tools: list[str]
    tool_results: list[dict[str, Any]]
    retrieved_docs: list[dict[str, Any]]
    evidence_collection_results: list[dict[str, Any]]
    evidence_sufficiency: dict[str, Any]
    evidence_sufficiency_summary: dict[str, Any]
    evidence_gap_by_answer_part: dict[str, dict[str, Any]]
    answer_part_status_by_id: dict[str, dict[str, Any]]
    missing_required_answer_parts: list[str]
    partial_required_answer_parts: list[str]
    missing_but_analyzable_answer_parts: list[str]
    missing_and_unanswerable_answer_parts: list[str]
    evidence_health: str
    tool_error_context: list[dict[str, Any]]
    collection_evidence_collection_results: list[dict[str, Any]]
    collection_evidence_sufficiency: dict[str, Any]
    collection_evidence_sufficiency_summary: dict[str, Any]
    evidence_retry_history: list[dict[str, Any]]
    requirement_limitations: list[dict[str, Any]]
    collected_evidence_by_requirement: dict[str, dict[str, Any]]
    requirement_status_map: dict[str, dict[str, Any]]
    dimension_status_map: dict[str, dict[str, Any]]
    dimension_status_by_id: dict[str, dict[str, Any]]
    covered_dimensions: list[str]
    satisfied_dimensions: list[str]
    partial_dimensions: list[str]
    missing_dimensions: list[str]
    dimension_coverage_rate: float
    weighted_dimension_coverage_rate: float
    framework_sufficiency_status: str
    red_flags: list[dict[str, Any]]
    missing_evidence_flags: list[dict[str, Any]]
    forbidden_claims: list[str]
    allowed_claims: list[str]
    final_requirement_status_map: dict[str, dict[str, Any]]
    collection_requirement_status_map: dict[str, dict[str, Any]]
    evidence_validation_records: list[dict[str, Any]]
    trace_summary: dict[str, Any]
    collection_trace_summary: dict[str, Any]
    missing_requirements: list[str]
    retry_history: list[dict[str, Any]]
    degradation_reason: str | None
    validated_requirement_ids: list[str]
    validated_numeric_evidence_count: int
    validated_text_evidence_count: int
    raw_retrieval_hits_by_requirement: dict[str, int]
    text_requirement_diagnostics: dict[str, dict[str, Any]]
    retrieval_policy: dict[str, Any]
    retrieval_debug: dict[str, Any]
    event_intent: str  # none | optional | required
    market_reaction_requested: bool
    event_query: dict[str, Any]
    event_results: list[dict[str, Any]]
    market_reaction_evidence: list[dict[str, Any]]
    market_reaction_limitations: list[str]

    # ---- Evidence & answer ----
    numeric_evidence: list[dict[str, Any]]
    text_evidence: list[dict[str, Any]]
    unsupported_claims: list[dict[str, Any]]
    numeric_citations: list[dict[str, Any]]
    text_citations: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    output: dict[str, Any]
    structured_sources: list[dict[str, Any]]
    document_citations: list[dict[str, Any]]
    synthesis: dict[str, Any]
    synthesis_strategy: str
    synthesis_mode: str
    analytical_claims: list[dict[str, Any]]
    claim_tiers: dict[str, int]
    analytical_reasoning_status: str
    unsupported_synthesis_items: list[dict[str, Any]]
    evidence_packet: dict[str, Any]
    evidence_packet_summary: dict[str, Any]
    comparison_judgment_frame: dict[str, Any]
    analyst_draft: dict[str, Any]
    analyst_draft_validation: dict[str, Any]
    draft_validation: dict[str, Any]
    draft_attempts: list[dict[str, Any]]
    draft_revision_attempts: list[dict[str, Any]]
    draft_violations: list[dict[str, Any]]
    draft_final_status: str
    draft_status: str
    draft_release_decision: dict[str, Any]
    answer_history: list[dict[str, Any]]
    answer_candidate: dict[str, Any]
    answer_candidates: list[dict[str, Any]]
    final_answer_source: str
    why_tools_skipped: list[dict[str, Any]]
    evidence_sufficient: bool
    evidence_loop_count: int  # guard against infinite loops
    final_answer: str

    # ---- Runtime answer contract guard ----
    draft_answer: str
    contract_result: dict[str, Any]
    contract_decision: dict[str, Any]
    contract_attempts: int
    contract_status: str  # not_checked | passed | repaired | failed | blocked
    contract_failure_reasons: list[str]
    contract_trace: dict[str, Any]
    repair_actions: list[dict[str, Any]]
    contract_evidence_retry_count: int
    max_contract_repairs: int
    max_contract_evidence_retries: int
    final_contract_status: str
    contract_public_summary: str
    limitations: list[str]
    relevance_decision: dict[str, Any]
    relevance_status: str
    relevance_attempts: int
    relevance_repair_attempts: int
    final_route: str
    answer_quality_tier: str
    quality_tier_reason: str
    main_question_covered: bool
    fallback_intent_match: bool
    answered_dimensions: list[str]
    unresolved_relevance_failures: list[str]
    format_constraints_satisfied: bool
    format_constraints: dict[str, Any]
    repair_applied: bool
    repair_owner: str
    source_before_repair: str
    repair_types: list[str]
    repair_attempts: int
    material_claim_uncited_count: int
    primary_generation_contract_debt: bool
    core_missing_parts: list[str]
    optional_missing_parts: list[str]
    risk_items_directly_supported_count: int
    risk_items_template_only_count: int
    company_specific_token_leakage: int
    segment_or_product_scope: str

    # ---- Company analysis report output ----
    report: dict[str, Any]
    report_sections: list[dict[str, Any]]
    report_contract_result: dict[str, Any]
    report_contract_status: str

    # ---- Tracing ----
    trace_id: str
    run_started_at: str
    progress_events: list[dict[str, Any]]

    # ---- LangGraph message history ----
    messages: Annotated[list[BaseMessage], add_messages]
