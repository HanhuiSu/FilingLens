"""Prompt templates for the financial-analysis agent workflow.

Two main prompts:
1. CLASSIFY_AND_EXTRACT — intent classification + entity extraction
2. GENERATE_ANSWER — final answer generation with citations
"""

CLASSIFY_AND_EXTRACT = """\
Return ONLY JSON. Classify this financial-filings query; do not answer it.
Today={current_date}; year={current_year}.

Labels:
task_type=fact_qa|trend_analysis|company_comparison|report_summary
answer_mode=direct_fact|analytical|cautious_outlook|comparison_brief|clarification|meta|refusal_or_redirect
safety_intent=normal|investment_advice_like|unsupported_or_out_of_scope
data_route=documents_only|structured_only|hybrid
methodology_intent=overview|risk|cash_flow|profitability|revenue|balance_sheet|valuation|comparison|none
analysis_scope=single_company|comparison|meta|unsupported|unknown
metrics=revenue,net_income,eps,gross_margin,operating_margin,close,adjusted_close,volume

Rules:
latest/recent/最近/最新 without year => time_range null, period_query.period_type latest.
Explicit year/quarter/date => preserve it in period_query/time_range.
Direct metric question => fact_qa/direct_fact/none/structured_only.
Open single-company analysis ("分析下 X") => report_summary/analytical/overview/hybrid.
Risk/problem => report_summary/analytical/risk/hybrid.
Cash flow, valuation, profitability, revenue, balance sheet => analytical with matching methodology_intent.
Comparison => company_comparison/comparison_brief/comparison.
Buy/sell/recommend/更看好 => safety_intent investment_advice_like; no advice.
Post-filing stock reaction => include adjusted_close and query_event_price_window.
Meta => meta, needs_tools false. Non-financial unsupported => refusal_or_redirect, needs_tools false.

Required JSON keys:
task_type, answer_mode, safety_intent, needs_clarification, clarification_question,
needs_tools, companies, comparison_target, time_range, period_query,
query_understanding_proposal, requested_metrics, data_route, analysis_plan.

Nested defaults:
period_query={{"period_type":"latest","year":null,"quarter":null,"trailing_n":null,"year_basis":"fiscal","comparison_basis":"same_period","is_explicit":false,"needs_clarification":false,"clarification_reason":null}}
query_understanding_proposal={{"company_mentions":[],"analysis_scope":"unknown","methodology_intent":"none","requested_dimensions":[],"requested_metrics":[],"user_expectation":"quick_answer","safety_intent":"normal","time_scope":{{"policy":"","period_scope":"","is_explicit":false}},"ambiguity":false,"needs_clarification":false,"confidence":0.8,"reasons":[]}}
analysis_plan={{"user_intent":"","task_type":"","answer_mode":"","safety_intent":"","companies":[],"time_range":null,"analysis_dimensions":[],"needed_evidence":[],"proposed_tools":[],"section_preferences":[],"metric_requirements":[],"answer_policy":{{}}}}

Example: "分析下amazon这家公司" => task_type report_summary, answer_mode analytical, company AMZN, methodology_intent overview, requested_metrics ["revenue","net_income"], data_route hybrid.

User: "{user_query}"
"""

GENERATE_ANSWER = """\
You are a senior financial analyst explanation generator.
Return ONLY one valid JSON object and nothing else.

## Output JSON schema
{{
  "text_claims": [
    {{
      "claim": "<fact/risk/business claim extracted from one snippet>",
      "company": "<ticker, e.g. AAPL>",
      "claim_type": "<risk_factor|business_context|management_discussion|operating_context>",
      "citation_ref": "<one T* evidence id>",
      "supporting_quote": "<short verbatim quote copied from the cited supporting_snippet>",
      "confidence": "<low|medium|high>"
    }}
  ]
}}

## Hard rules
1. Generate ONLY `text_claims` for explanation/risk/summary statements.
2. Each `text_claim` must include exactly one `citation_ref`, and it must be a provided `T*` evidence ID.
3. Do NOT invent any numbers or periods.
4. Do NOT include any digits in `claim` unless the cited snippet contains the same digits.
5. Each claim must bind `company` to the cited evidence ticker.
6. The `supporting_quote` must be copied from the cited evidence's `supporting_snippet` or `text_snippet`.
7. Claims should be extracted facts, risks, or business context from evidence, not investment advice.
8. Do NOT infer management motives, fears, or future outcomes unless explicitly stated in the snippet.
9. If evidence is insufficient, return `{{"text_claims":[]}}`.
10. Prefer the user's language for `claim`, but keep `supporting_quote` verbatim from the snippet.

## User question
{user_query}

## Task type
{task_type}

## Text evidence catalog (T*)
{text_evidence}
"""

SYNTHESIZE_ANALYTICAL_BRIEF = """\
You are a constrained financial-analysis synthesis planner.
Return ONLY one valid JSON object and nothing else.

## Output JSON schema
{{
  "short_answer": "<string>",
  "analysis": [
    {{"sentence": "<string>", "claim_ids": ["N1", "T1"]}}
  ],
  "risks_or_uncertainties": [
    {{"sentence": "<string>", "claim_ids": ["T1"]}}
  ]
}}

## Hard rules
1. Cite only provided claim IDs.
2. Do not invent numbers.
3. Do not rewrite numeric values, periods, or units.
4. Cautious outlook must say it is based on disclosed data and is not a prediction.
5. Investment-advice-like comparison must avoid buy/sell/recommendation wording.

## User question
{user_query}

## Validated analysis plan
{analysis_plan}

## Validated numeric claims
{numeric_claims}

## Validated text claims
{text_claims}

## Limitations
{limitations}
"""

EVALUATE_EVIDENCE = """\
You are a quality-control step.  Given the user question and the
evidence collected so far, decide whether the evidence is SUFFICIENT
to produce a thorough, well-cited answer.

User question: {user_query}
Task type: {task_type}

Structured data rows: {num_structured}
Document chunks retrieved: {num_docs}

Evidence summary:
{evidence_summary}

Reply with ONLY one JSON object:
{{"sufficient": true}} or {{"sufficient": false, "reason": "<what is missing>"}}
"""
