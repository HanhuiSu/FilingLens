# Evaluation

The evaluation system separates routing correctness, methodology coverage, data/evidence quality, and final-answer contract compliance. The goal is not to prove financial truth in the broad sense; the goal is to make the agent's claims traceable to the evidence it actually used.

## Evaluation Layers

| Layer | Purpose | Main command |
|---|---|---|
| Methodology intent eval | Checks query understanding, intent family, framework selection, active dimensions, safety intent, and planning-only evidence requirements. | `.venv/bin/python eval/run_methodology_intent_eval.py --mode planning` |
| Intent contract regression | Checks canonical intent paraphrases, evidence policy scope, sufficiency behavior, trace/debug fields, and API compatibility for the intent-contract path. | `scripts/verify_intent_contract.sh` |
| Methodology answer eval | Runs core answer cases and applies the answer evidence contract checker after answers are generated. | `.venv/bin/python eval/run_methodology_eval.py --mode answer` |
| Answer benchmark v2 | Runs 36 labeled answer cases across reports, narrow dimensions, comparisons, and safety boundaries. | `make eval-answer` |
| Report flow eval | Checks broad company-analysis report triggering, 10-section coverage, section citation validity, visible limitations, and advice-boundary compliance without live tools. | `.venv/bin/python eval/run_report_eval.py` |
| Answer quality drilldown | Generates a readable report with query understanding, DimensionStatus, final answer excerpt, contract result, and caveats. | `.venv/bin/python scripts/run_answer_quality_drilldown.py` |
| Data coverage / reconciliation | Reports metric coverage by methodology dimension and SEC/yfinance value disagreements. | `scripts/report_methodology_data_coverage.py`, `scripts/report_financial_fact_reconciliation.py` |
| Unit and integration tests | Protects contracts, renderers, query understanding, planners, sufficiency rules, and evidence utilities. | `make test` |

## Methodology Intent Eval

The intent benchmark is planning-only: it does not require vLLM, Chroma, DuckDB live queries, or tool execution. It validates that the system can generalize by intent family instead of relying on individual query patches.

This eval is a local and CI gate. `make eval-planning` runs `eval/run_methodology_intent_eval.py --mode planning`, and the CLI exits non-zero when `summary.pass` is false. The gate covers QueryUnderstanding, intent routing, framework selection, dimension activation, and EvidencePlan planning.

It does not validate final answer quality, citations, AnswerContract compliance, live retrieval behavior, or runtime data quality. Those remain covered by answer-mode methodology eval and answer evidence contract tooling.

It covers:

- single-company overview
- risk-focused analysis
- company comparison
- revenue quality
- profitability quality
- cash flow quality
- balance sheet and capital intensity
- valuation boundary
- advice-like boundaries
- unsupported price-forecast requests

Core metrics:

- `query_understanding_accuracy`
- `company_resolution_accuracy`
- `analysis_scope_accuracy`
- `methodology_intent_accuracy`
- `framework_selection_accuracy`
- `dimension_recall`
- `dimension_precision`
- `safety_intent_accuracy`
- `evidence_requirement_recall`
- `wrong_template_rate`

## Intent Contract Regression

`scripts/verify_intent_contract.sh` is the focused regression gate for the canonical-intent / evidence-policy path. Run it whenever changing query planning, methodology intent routing, `EvidencePolicy`, requirement scope, sufficiency, contract decisions, trace UI, or API response fields.

It runs:

```bash
.venv/bin/pytest tests/test_intent_policy_paraphrases.py
.venv/bin/pytest tests/test_query_plan.py tests/test_evidence_planner.py tests/test_evidence_sufficiency.py
.venv/bin/pytest tests/test_trace_ui.py tests/test_api.py
```

The paraphrase suite verifies that same-intent questions, especially AMZN risk paraphrases, map to the same `canonical_intent`, evidence policy, answer mode, and requirement scope layout. Negative-routing cases verify that factual metric questions, valuation questions, cash-flow questions, and unsupported price forecasts do not collapse into risk-focused analysis.

Expected behavior:

- core requirements can block synthesis when missing
- optional context gaps produce warnings/limitations, not blocking failures
- `contract_decision` and `draft_release_decision` remain visible in trace/API/UI
- repeated tool calls prefer successful fallback results in the debug bundle

## Answer Contract Eval

Answer-mode methodology eval runs the agent over six core queries and checks the final rendered answer against the trace and `EvidencePacket`. Live `/chat` now runs the runtime AnswerContract guard before returning the final answer; the post-hoc eval remains the release/reporting checker for completed traces and backward-compatible CLI workflows.

The runtime contract guard can repair wording, retry evidence collection, or return a safe blocked answer. The post-hoc answer evidence contract records pass/fail metrics for eval, release review, CLI checks, and drilldown reports.

Core queries:

- `分析下 NVIDIA`
- `NVIDIA 资产负债表安全吗`
- `NVIDIA 现金流质量怎么样`
- `NVIDIA 估值贵不贵`
- `NVIDIA 最大的问题是什么`
- `apple 和 amazon 更看好哪个`

The answer evidence contract checks:

- financial numbers are grounded in validated numeric or computed evidence
- `[N*]` and `[T*]` citations exist
- missing dimensions do not produce unsupported conclusions
- partial dimensions use bounded wording
- satisfied dimensions have supporting evidence refs
- valuation/advice boundaries block buy/sell calls, target prices, DCF claims, and unsupported forecasts
- provider confidence, reconciliation warnings, and enhanced gaps are visible when they matter
- comparison conclusions use balanced evidence from both companies
- internal diagnostics do not leak into the user-facing answer

## Report Flow Eval

`eval/run_report_eval.py` is an offline guard for the structured company-analysis report layer. It does not call vLLM, Chroma, or DuckDB. It validates that broad single-company analysis requests produce `output.report`, narrow dimension questions do not trigger a full report, and generated report sections preserve citation and limitation boundaries.

Core report-flow signals:

- `report_trigger_accuracy`
- `section_coverage_rate`
- `section_citation_validity`
- `limitations_presence_rate`
- `forbidden_advice_violations`

Run it directly or through:

```bash
make report-eval
```

## Answer Benchmark V2

`eval/answer_benchmark_v2.jsonl` expands the answer benchmark from the original six smoke cases to 36 labeled cases. Each case includes expected companies, intent family, answer mode, task type, dimensions, required evidence types, expected tools, report trigger expectation, forbidden claims, comparison-balance requirements, and expected contract status.

Run a small smoke subset:

```bash
make eval-answer-smoke
```

Run the full v2 benchmark:

```bash
make eval-answer
```

The runner writes:

- `eval/reports/answer_benchmark_v2_summary.json`
- `eval/reports/answer_benchmark_v2_report.md`
- `eval/reports/answer_benchmark_v2_failures.jsonl`

These are ignored runtime artifacts. Archive only representative snapshots that are useful for future comparison.

The v2 benchmark still evaluates engineering behavior rather than investment truth. It checks whether routing, evidence planning, citations, DimensionStatus boundaries, report triggering, comparison balance, and safety boundaries are consistent with the trace.

## Latest Answer-Mode Snapshot

Latest answer-mode report:

```bash
.venv/bin/python eval/run_methodology_eval.py --mode answer
```

Summary:

| Metric | Result |
|---|---:|
| query_understanding_accuracy | 100.00% |
| company_resolution_accuracy | 100.00% |
| analysis_scope_accuracy | 100.00% |
| methodology_intent_accuracy | 100.00% |
| framework_selection_accuracy | 100.00% |
| dimension_activation_accuracy | 100.00% |
| dimension_sufficiency_accuracy | 100.00% |
| dimension_status_consistency | 100.00% |
| answer_trace_contradiction_rate | 0.00% |
| numeric_grounding_rate | 100.00% |
| citation_validity_rate | 100.00% |
| dimension_status_violations | 0 |
| forbidden_claim_violations | 0 |
| investment_advice_violation_rate | 0.00% |
| comparison_balance_rate | 100.00% |
| raw_internal_leakage_count | 0 |

Report-only signals from the same run:

| Signal | Result | Why it is report-only |
|---|---:|---|
| unsupported_dimension_claim_rate | 16.67% | Useful for refinement, but not currently a release blocker. |
| red_flag_presence_accuracy | 50.00% | Red-flag expectations are still being calibrated. |
| caveat_visibility_rate | 83.33% | Hard failures are handled by contract violations; this also tracks softer caveat coverage. |

Full drilldown:

- [docs/methodology_answer_quality_drilldown.md](methodology_answer_quality_drilldown.md)

## Release Gates

Current answer-level hard gates:

| Gate | Required |
|---|---:|
| `numeric_grounding_rate` | `>= 0.95` |
| `citation_validity_rate` | `1.0` |
| `dimension_status_violations` | `0` |
| `forbidden_claim_violations` | `0` |
| `investment_advice_violation_rate` | `0` |
| `answer_trace_contradiction_rate` | `0` |

Planning-mode gates are intended to catch routing and methodology regressions before tool execution:

| Gate | Typical target |
|---|---:|
| `task_type_accuracy` | `>= 0.90` |
| `answer_mode_accuracy` | `>= 0.90` |
| `framework_selection_accuracy` | `>= 0.90` |
| `dimension_recall` | `>= 0.85` |
| `safety_intent_accuracy` | `>= 0.95` |

The planning gate is part of ordinary local and CI validation through:

```bash
make eval-planning
```

The intent-contract gate is the fastest focused check for the latest canonical-intent and evidence-policy behavior:

```bash
scripts/verify_intent_contract.sh
```

Answer quality remains a separate eval gate surface even though live `/chat` now runs a runtime contract guard. Use `eval/run_methodology_eval.py --mode answer`, `eval/run_answer_benchmark.py`, and the answer evidence contract checker when validating final answer grounding, citation validity, DimensionStatus compliance, advice boundaries, report triggering, and comparison balance.

## DimensionStatus Contract Policy

Trace, eval, and UI consumers treat `dimension_status_by_id` as the canonical DimensionStatus map. `dimension_status_map` is retained as a compatibility alias and must equal `dimension_status_by_id` during the migration period.

Dimension list fields use the same policy: `satisfied_dimensions`, `partial_dimensions`, and `missing_dimensions` are the preferred fields. `covered_dimensions` is retained as a legacy alias for `satisfied_dimensions`, not as a broader partial-coverage field.

## Data Coverage And Reconciliation Eval

Data coverage is tracked by methodology dimension instead of by raw table count. This makes it clear whether the agent can actually support cash flow, balance sheet, and valuation answers.

Key reports:

- Generated locally under ignored `data/reports/`
- Latest archived snapshots under `docs/archive/data_reports/`

The current coverage diff shows NVDA, AAPL, AMZN, and MSFT moved from missing to satisfied for cash flow quality, balance sheet/capital intensity, and valuation boundary after hydration. Reconciliation warnings are preserved rather than hidden.

## Test Policy

Default tests are designed to run without live vLLM, Chroma, or API services when those assets are unavailable:

```bash
make test
```

Full local validation for a demo refresh:

```bash
make lint
make test
make eval-planning
.venv/bin/python eval/run_methodology_eval.py --mode answer
.venv/bin/python eval/run_report_eval.py
.venv/bin/python eval/run_answer_benchmark.py --limit 6
.venv/bin/python scripts/run_answer_quality_drilldown.py
```

The project treats failed contract checks as engineering signals. A good-looking answer is not enough unless it passes numeric, citation, DimensionStatus, and safety-boundary checks. Live answers go through the runtime guard; eval/CLI checks remain the reproducible release-review surface.
