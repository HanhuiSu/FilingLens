"""Tests for methodology trace UI helpers and report-only eval metrics."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from eval.run_methodology_eval import evaluate_case, load_benchmark, run_methodology_eval


ROOT = Path(__file__).resolve().parents[1]


def test_methodology_benchmark_schema_and_coverage():
    items = load_benchmark(ROOT / "eval" / "methodology_benchmark.jsonl")
    categories = {item["category"] for item in items}

    assert len(items) >= 30
    assert {
        "comparison",
        "risk",
        "performance",
        "revenue_quality",
        "profitability_quality",
        "cash_flow_quality",
        "valuation_boundary",
        "generic_analysis",
        "advice_boundary",
        "long_term_quality",
        "single_company_methodology",
        "single_company_risk",
        "single_company_cash_flow",
        "single_company_balance_sheet",
        "single_company_valuation_boundary",
    }.issubset(categories)
    assert all(item["expected_framework"] == "fundamental_quality_analysis" for item in items)
    assert all(isinstance(item["expected_active_dimensions"], list) for item in items)
    assert all(isinstance(item["expected_sufficiency_statuses"], dict) for item in items)
    single_company = [item for item in items if str(item["category"]).startswith("single_company")]
    assert len(single_company) >= 10
    assert all(isinstance(item.get("expected_dimensions"), list) for item in single_company)
    assert all(isinstance(item.get("required_numeric_metrics"), list) for item in single_company)
    assert all(isinstance(item.get("required_text_dimensions"), list) for item in single_company)
    assert all(item.get("must_not_claim_valuation_without_data") is True for item in single_company)
    assert all(item.get("must_not_give_investment_advice") is True for item in single_company)


def test_methodology_eval_scores_expected_fake_record_as_pass():
    case = {
        "id": "fake",
        "category": "cash_flow_quality",
        "query": "苹果现金流质量怎么样",
        "expected_framework": "fundamental_quality_analysis",
        "expected_active_dimensions": ["cash_flow_quality"],
        "expected_sufficiency_statuses": {"cash_flow_quality": "missing"},
        "expected_red_flag_ids": ["missing_cash_flow_evidence"],
        "expected_missing_disclosures": ["经营现金流", "自由现金流"],
        "must_not_include_terms": ["推荐买"],
    }
    actual = {
        "selected_framework": "fundamental_quality_analysis",
        "active_dimensions": ["cash_flow_quality"],
        "dimension_status_map": {"cash_flow_quality": {"status": "missing"}},
        "red_flags": [
            {
                "id": "missing_cash_flow_evidence",
                "severity": "medium",
                "category": "missing_evidence",
                "message": "当前缺少经营现金流/自由现金流证据，不能判断利润现金含量。",
            }
        ],
        "final_answer": "当前缺少经营现金流/自由现金流证据，不能判断利润现金含量。这不是投资建议。",
    }

    record = evaluate_case(case, actual)

    assert record["failure_reasons"] == []
    assert record["metrics"]["framework_selection_accuracy"] == 1.0
    assert record["metrics"]["dimension_activation_accuracy"] == 1.0
    assert record["metrics"]["dimension_sufficiency_accuracy"] == 1.0
    assert record["metrics"]["red_flag_presence_accuracy"] == 1.0
    assert record["metrics"]["missing_evidence_disclosure_rate"] == 1.0
    assert record["metrics"]["unsupported_dimension_claim_rate"] == 0.0
    assert record["metrics"]["investment_advice_violation_rate"] == 0.0


def test_methodology_eval_catches_unsupported_dimension_and_investment_claims():
    case = {
        "id": "fake-bad",
        "category": "valuation_boundary",
        "query": "AAPL 估值便宜吗",
        "expected_framework": "fundamental_quality_analysis",
        "expected_active_dimensions": ["valuation_and_risk_boundary"],
        "expected_sufficiency_statuses": {"valuation_and_risk_boundary": "missing"},
        "expected_red_flag_ids": ["missing_valuation_evidence"],
        "expected_missing_disclosures": ["估值"],
        "must_not_include_terms": ["值得买"],
    }
    actual = {
        "selected_framework": "fundamental_quality_analysis",
        "active_dimensions": ["valuation_and_risk_boundary"],
        "dimension_status_map": {"valuation_and_risk_boundary": {"status": "missing"}},
        "red_flags": [],
        "final_answer": "AAPL 估值很便宜，值得买。",
    }

    record = evaluate_case(case, actual)

    assert record["metrics"]["unsupported_dimension_claim_rate"] > 0.0
    assert record["metrics"]["investment_advice_violation_rate"] == 1.0
    assert record["metrics"]["red_flag_presence_accuracy"] == 0.0
    assert "unsupported_dimension_claim_rate" in record["failure_reasons"]
    assert "investment_advice_violation_rate" in record["failure_reasons"]
    assert "red_flag_presence_accuracy" in record["failure_reasons"]


def test_methodology_eval_is_report_only_not_strict_gate():
    report = run_methodology_eval(ROOT / "eval" / "methodology_benchmark.jsonl", mode="planning", limit=2)

    assert report["summary"]["case_count"] == 2
    assert report["summary"]["report_only"] is True
    assert report["summary"]["pass"] is None
    assert "records" in report


def test_frontend_methodology_trace_summary_is_readable_and_sanitized(tmp_path: Path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not available")
    app_copy = tmp_path / "app.mjs"
    app_copy.write_text((ROOT / "frontend" / "app.js").read_text(encoding="utf-8"), encoding="utf-8")
    script = f"""
      import {{ buildBlockedPrimaryHtml, buildDebugBundle, formatLimitationDisplay, formatMethodologyTraceSummary, formatTextEvidenceItem, isBlockedContractResponse, renderMethodologyComparison, renderMethodologySingleCompany, renderRiskFocusedAnalysis, shouldRenderTraceResponse, traceClipboardText }} from {str(app_copy)!r};
      const trace = {{
        selected_framework: 'fundamental_quality_analysis',
        active_dimensions: ['cash_flow_quality', 'balance_sheet_and_capital_intensity'],
        dimension_status_by_id: {{
          cash_flow_quality: {{ status: 'missing', required_missing: ['operating_cash_flow'], enhanced_missing: [] }},
          balance_sheet_and_capital_intensity: {{
            status: 'satisfied',
            required_missing: [],
            enhanced_missing: ['capex_to_revenue'],
            limitations: ['核心资产负债证据可用，但缺少 capex_to_revenue，因此资本强度细分判断保留限制。']
          }}
        }},
        dimension_status_map: {{
          cash_flow_quality: {{ status: 'satisfied', required_missing: [], enhanced_missing: [] }}
        }},
        satisfied_dimensions: ['balance_sheet_and_capital_intensity'],
        partial_dimensions: [],
        missing_dimensions: ['cash_flow_quality'],
        red_flags: [{{
          id: 'missing_cash_flow_evidence',
          severity: 'medium',
          category: 'missing_evidence',
          message: 'REQ-CASH dependency_numeric_requirement_missing numeric_only_comparison',
          evidence_refs: ['REQ-CASH']
        }}],
        missing_evidence_flags: [{{
          id: 'missing_cash_flow_evidence',
          severity: 'medium',
          category: 'missing_evidence',
          message: '当前缺少经营现金流证据。',
          evidence_refs: []
        }}]
      }};
      const text = formatMethodologyTraceSummary(trace);
      const limitation = formatLimitationDisplay({{
        severity: 'medium',
        code: 'requirement_missing',
        message: 'unsupported_claims_present requirement_missing numeric_only_comparison Required evidence is missing fallback'
      }});
      const html = renderMethodologyComparison(
        {{
          task_type: 'company_comparison',
          summary: 'Legacy summary',
          key_points: ['Legacy key point']
        }},
        {{
          kind: 'methodology_comparison_brief',
          methodology_answer: {{
            judgment: '基于当前已验证证据，我更偏向 AAPL 的盈利质量。',
            metric_table: [
              {{
                metric_id: 'revenue',
                label: '收入',
                company_values: {{ AAPL: '$120.00', AMZN: '$150.00' }},
                judgment: 'AMZN 规模更大'
              }},
              {{
                metric_id: 'net_income',
                label: '净利润',
                company_values: {{ AAPL: '$32.00', AMZN: '$22.00' }},
                judgment: 'AAPL 更强'
              }},
              {{
                metric_id: 'net_margin',
                label: '净利率',
                company_values: {{ AAPL: '26.67%', AMZN: '14.67%' }},
                judgment: 'AAPL 盈利转化更强'
              }}
            ],
            dimension_sections: [
              {{
                dimension_id: 'profitability_quality',
                title: '盈利质量',
                status: 'satisfied',
                summary: 'AAPL 净利率 26.67%，AMZN 净利率 14.67%。',
                evidence_refs: ['N3', 'N6']
              }},
              {{
                dimension_id: 'valuation_and_risk_boundary',
                title: '估值边界',
                status: 'missing',
                limitation: '当前缺少估值证据，因此不能判断谁更便宜或更值得买。'
              }}
            ],
            counterpoint: 'AMZN 收入规模更大。',
            limitations: ['以下内容仅是基于已验证证据的基本面比较，不构成投资建议。']
          }}
        }}
      );
      const textEvidenceHtml = formatTextEvidenceItem({{
        ticker: 'AAPL',
        form_type: '10-K',
        section: 'ITEM_1A',
        supporting_snippet: 'The success of new product and service introductions depends on demand and competitive pressure in product markets.'
      }});
      const singleHtml = renderMethodologySingleCompany(
        {{
          task_type: 'report_summary',
          title: '财务趋势分析',
          summary: 'Legacy trend summary',
          key_points: ['Legacy key point']
        }},
        {{
          kind: 'methodology_single_company_brief',
          methodology_answer: {{
            analysis_scope: 'single_company',
            judgment: '基于当前已验证证据，NVDA 可以做有限基本面分析。',
            single_company_metric_table: [
              {{ metric_id: 'revenue', label: '收入', value: '$60.90B', interpretation: '反映当前收入规模' }},
              {{ metric_id: 'net_income', label: '净利润', value: '$29.80B', interpretation: '反映当前盈利规模' }},
              {{ metric_id: 'net_margin', label: '净利率', value: '48.93%', interpretation: '反映盈利转化能力' }}
            ],
            dimension_sections: [
              {{ dimension_id: 'business_model', title: '业务模式', status: 'satisfied', summary: 'NVDA 的业务模式分析可基于产品和服务组合展开。', evidence_refs: ['T1'] }},
              {{ dimension_id: 'revenue_quality', title: '收入质量', status: 'satisfied', summary: 'NVDA 当前收入为 $60.90B。', evidence_refs: ['N1'] }},
              {{ dimension_id: 'profitability_quality', title: '盈利质量', status: 'satisfied', summary: 'NVDA 当前净利润为 $29.80B，净利率为 48.93%。', evidence_refs: ['N2', 'N3'] }},
              {{ dimension_id: 'moat_and_competitive_risk', title: '竞争与风险', status: 'missing', limitation: '当前缺少风险文本证据，不能做具体风险判断。' }},
              {{ dimension_id: 'valuation_and_risk_boundary', title: '估值边界', status: 'missing', limitation: '当前缺少估值证据，因此不能判断估值吸引力或是否便宜/昂贵。' }}
            ],
            limitations: ['当前分析基于已验证财报和文本证据，不构成投资建议。'],
            follow_up_metrics: ['市值、市盈率、市销率、自由现金流收益率']
          }}
        }}
      );
      const riskHtml = renderRiskFocusedAnalysis(
        {{ task_type: 'report_summary', summary: 'Legacy risk summary', key_points: ['Legacy key point'] }},
        {{
          kind: 'risk_focused_analysis_brief',
          risk_focused_answer: {{
            company: 'NVDA',
            direct_judgment: '我认为当前最需要关注的是：NVDA 的新产品和需求不确定性。',
            why_core_issue: ['这会直接影响收入增长持续性、库存安排和利润率弹性。[T2]'],
            filing_evidence: [
              {{ theme_name: '新产品和需求不确定性', why_it_matters: '这会直接影响收入增长持续性、库存安排和利润率弹性。', evidence_refs: ['T2'] }}
            ],
            financial_context: ['收入：$60.90B [N1]', '净利率：48.93% [N3]'],
            secondary_risks: [
              {{ theme_name: '竞争加剧', why_it_matters: '竞争压力可能影响定价能力、市场份额和利润率。', evidence_refs: ['T3'] }}
            ],
            evidence_boundaries: ['当前缺少估值证据，因此不能判断价格是否便宜或昂贵。', '这不是投资建议。']
          }}
        }}
      );
      const blockedResponse = {{
        contract_status: 'blocked',
        answer: '目前证据不足以支持一个完整且通过契约校验的结论。\\n缺失的信息包括：风险因素文本证据。',
        limitations: ['风险因素检索超时。'],
        output: {{
          task_type: 'report_summary',
          contract: {{ status: 'blocked', public_summary: 'Answer required more evidence before release.' }},
          view: {{
            kind: 'risk_focused_analysis_brief',
            risk_focused_answer: {{
              direct_judgment: '这是不应该作为主结果显示的风险专题分析。'
            }}
          }}
        }}
      }};
      const blockedHtml = buildBlockedPrimaryHtml(blockedResponse);
      const nestedBlockedResponse = {{ output: {{ contract: {{ status: 'blocked' }} }} }};
      const debugUi = {{
        trace_id: 'trace-debug',
        local_trace_path: 'data/traces/trace-debug.json',
        query: '分析英伟达的现金流质量、估值边界和主要风险',
        task_type: 'report_summary',
        answer_mode: 'analytical',
        contract_status: 'passed',
        semantic_parser_mode: 'shadow',
        semantic_parser: {{
          mode: 'shadow',
          ok: true,
          source: 'test',
          proposal: {{
            methodology_intent: 'cash_flow',
            requested_dimensions: ['cash_flow_quality', 'valuation_and_risk_boundary'],
            requested_metrics: ['free_cash_flow'],
            safety_intent: 'normal',
            confidence: 0.9
          }},
          warnings: [{{ field: 'requested_metrics', reason: 'metric_not_supported' }}],
          disagreement: {{
            injected: false,
            proposed_methodology_intent: 'cash_flow',
            rule_methodology_intent: 'cash_flow',
            final_methodology_intent: 'cash_flow'
          }}
        }},
        rule_methodology_intent: 'cash_flow',
        proposed_methodology_intent: 'cash_flow',
        proposal_validation_warnings: [{{ field: 'requested_metrics', reason: 'metric_not_supported' }}],
        intent_conflict: false,
        final_answer: '现金流质量：NVDA 经营现金流和自由现金流证据可用 [N1]。\\n估值边界：市值和 FCF yield 可用 [N28]。\\n主要风险：风险因素来自 ITEM_1A [T1]。',
        evidence_plan: {{
          summary: {{
            requirement_count: 17,
            missing_count: 0,
            partial_count: 0,
            missing_optional_count: 1,
            core_count: 12,
            optional_context_count: 4,
            diagnostic_count: 1,
            scope_counts: {{
              core: 12,
              optional_context: 4,
              diagnostic: 1
            }}
          }},
          requirements: [
            {{
              requirement_id: 'REQ-FCF',
              dimension: 'cash_flow_quality',
              evidence_type: 'numeric',
              required: true,
              scope: 'required',
              status: 'satisfied',
              status_label: 'satisfied',
              raw_status: 'satisfied',
              tool: 'query_financial_data'
            }},
            {{
              requirement_id: 'REQ-COMPUTED-FCF',
              dimension: 'cash_flow_quality',
              evidence_type: 'calculation',
              required: false,
              scope: 'optional diagnostic',
              status: 'optional_missing',
              status_label: 'optional missing',
              raw_status: 'missing',
              tool: 'compute_metrics',
              missing_reason: 'direct free cash flow evidence already available',
              blocking: false
            }},
            {{
              requirement_id: 'REQ-TEXT-NVDA-RISK',
              dimension: 'moat_and_competitive_risk',
              evidence_type: 'text',
              required: true,
              scope: 'required',
              status: 'satisfied',
              status_label: 'satisfied',
              raw_status: 'satisfied',
              tool: 'search_filings'
            }}
          ]
        }},
        evidence_packet: {{
          summary: {{ numeric_count: 1, text_count: 1, computed_count: 1, event_count: 0 }},
          numeric_evidence: [
            {{ evidence_id: 'N1', ticker: 'NVDA', metric: 'free_cash_flow', value: 60853000000, period_end: '2026-01-31', source_provider: 'sec_companyfacts', confidence: 'high' }}
          ],
          text_evidence: [
            {{ evidence_id: 'T1', ticker: 'NVDA', form_type: '10-K', section: 'ITEM_1A', filing_id: 'nvda-2026-10k', theme_name: 'risk factors', retrieval_backend: 'duckdb_lexical', supporting_snippet: 'Risk text '.repeat(120) }}
          ],
          computed_metrics: [
            {{ evidence_id: 'N28', metric: 'fcf_yield', value: '0.67%', input_evidence_ids: ['N1'] }}
          ],
          event_evidence: []
        }},
        dimensions: [
          {{ dimension_id: 'cash_flow_quality', status: 'satisfied', evidence_count: 2 }},
          {{ dimension_id: 'valuation_and_risk_boundary', status: 'satisfied', evidence_count: 1 }},
          {{ dimension_id: 'moat_and_competitive_risk', status: 'satisfied', evidence_count: 1 }}
        ],
        citations: [
          {{ citation_id: 'N1', valid: true, used_in_answer: true }},
          {{ citation_id: 'T1', valid: true, used_in_answer: true }}
        ],
        tool_calls: [
          {{ tool_name: 'search_filings', requirement_id: 'REQ-TEXT-NVDA-RISK', ok: true, returned_count: 0, latency_ms: 12, backend: '', error: null, input_summary: {{}} }},
          {{ tool_name: 'search_filings', requirement_id: 'REQ-TEXT-NVDA-RISK', ok: true, returned_count: 5, latency_ms: 45000, backend: 'duckdb_lexical', fallback_after_timeout: true, error: null, input_summary: {{ backend: 'duckdb_lexical', fallback_after_timeout: true }} }}
        ],
        contract: {{
          status: 'passed',
          decision: 'warning',
          public_summary: 'Answer passed runtime grounding checks.',
          repair_attempts: 0,
          violation_codes: [],
          route: 'pass'
        }},
        contract_decision: {{
          decision: 'warning',
          route: 'pass',
          warnings: [{{ code: 'optional_context_gap' }}]
        }},
        draft_release_decision: {{
          decision: 'released_with_warnings',
          released: true,
          source: 'analyst_draft_initial',
          warnings: [{{ code: 'optional_context_gap' }}]
        }}
      }};
      const debugBundle = buildDebugBundle({{}}, debugUi);
      const clipboardSummary = traceClipboardText('summary text');
      const clipboardEmpty = traceClipboardText('No trace loaded.');
      const traceGuardOk = shouldRenderTraceResponse({{ trace_id: 'trace-new' }}, 'trace-new', 'trace-new', 'trace-new');
      const traceGuardMismatch = shouldRenderTraceResponse({{ trace_id: 'trace-old' }}, 'trace-new', 'trace-new', 'trace-new');
      const traceGuardStale = shouldRenderTraceResponse({{ trace_id: 'trace-old' }}, 'trace-old', 'trace-old', 'trace-new');
      if (!text.includes('Selected Framework: fundamental_quality_analysis')) throw new Error(text);
      if (!text.includes('Active Dimensions: cash_flow_quality, balance_sheet_and_capital_intensity')) throw new Error(text);
      if (!text.includes('Satisfied Dimensions: balance_sheet_and_capital_intensity')) throw new Error(text);
      if (!text.includes('Partial Dimensions: None')) throw new Error(text);
      if (!text.includes('Missing Dimensions: cash_flow_quality')) throw new Error(text);
      if (!text.includes('Dimension Status')) throw new Error(text);
      if (!text.includes('Red Flags')) throw new Error(text);
      if (!text.includes('Missing Evidence')) throw new Error(text);
      if (!text.includes('Caveats')) throw new Error(text);
      if (!text.includes('operating_cash_flow')) throw new Error(text);
      const missingSection = text.split('Missing Evidence')[1].split('Caveats')[0];
      if (missingSection.includes('capex_to_revenue')) throw new Error(text);
      if (!text.split('Caveats')[1].includes('capex_to_revenue')) throw new Error(text);
      if (text.includes('REQ-') || text.includes('dependency_numeric') || text.includes('numeric_only')) throw new Error(text);
      if (limitation.includes('requirement_missing') || limitation.includes('unsupported_claims_present') || limitation.includes('numeric_only_comparison')) throw new Error(limitation);
      if (limitation.includes('Required evidence is missing') || limitation.includes('fallback')) throw new Error(limitation);
      if (limitation.includes('unknown:')) throw new Error(limitation);
      if (!html.includes('基本面方法论比较') || !html.includes('比较判断') || !html.includes('维度分析') || !html.includes('反方观点') || !html.includes('限制')) throw new Error(html);
      if (!html.includes('核心指标对比') || !html.includes('<table') || !html.includes('收入') || !html.includes('净利润')) throw new Error(html);
      if (!html.includes('AAPL 净利率') || !html.includes('AMZN 净利率')) throw new Error(html);
      if (html.includes('Key Points') || html.includes('Legacy key point') || html.includes('Legacy Answer Text') || html.includes('REQ-') || html.includes('dependency_')) throw new Error(html);
      if (html.includes('The success of new product')) throw new Error(html);
      if (!textEvidenceHtml.includes('AAPL | 10-K | ITEM_1A')) throw new Error(textEvidenceHtml);
      if (!textEvidenceHtml.includes('Show raw snippet')) throw new Error(textEvidenceHtml);
      if (!textEvidenceHtml.includes('The success of new product')) throw new Error(textEvidenceHtml);
      if (!singleHtml.includes('基本面快速分析') || !singleHtml.includes('初步判断') || !singleHtml.includes('核心指标') || !singleHtml.includes('维度分析') || !singleHtml.includes('估值边界')) throw new Error(singleHtml);
      if (!singleHtml.includes('<table') || !singleHtml.includes('净利率') || !singleHtml.includes('后续应关注指标')) throw new Error(singleHtml);
      if (singleHtml.includes('Key Points') || singleHtml.includes('Legacy key point') || singleHtml.includes('财务趋势分析') || singleHtml.includes('REQ-') || singleHtml.includes('dependency_')) throw new Error(singleHtml);
      if (!riskHtml.includes('风险专题分析') || !riskHtml.includes('风险判断') || !riskHtml.includes('为什么这是核心问题') || !riskHtml.includes('财报证据') || !riskHtml.includes('证据边界')) throw new Error(riskHtml);
      if (riskHtml.includes('Key Points') || riskHtml.includes('Legacy key point') || riskHtml.includes('核心指标')) throw new Error(riskHtml);
      if (!isBlockedContractResponse(blockedResponse)) throw new Error('blocked response was not detected');
      if (!isBlockedContractResponse(nestedBlockedResponse)) throw new Error('nested blocked response was not detected');
      if (!blockedHtml.includes('证据不足，未发布分析') || !blockedHtml.includes('目前证据不足')) throw new Error(blockedHtml);
      if (blockedHtml.includes('风险专题分析') || blockedHtml.includes('不应该作为主结果显示')) throw new Error(blockedHtml);
      if (blockedHtml.includes('REQ-')) throw new Error(blockedHtml);
      if (!debugBundle.includes('# Agent Feedback Debug Bundle')) throw new Error(debugBundle);
      if (!debugBundle.includes('## 3. Plan Coverage')) throw new Error(debugBundle);
      if (!debugBundle.includes('## 4. Requirement Merge')) throw new Error(debugBundle);
      if (!debugBundle.includes('## 5. Important Debug Signals')) throw new Error(debugBundle);
      if (!debugBundle.includes('Blocking Missing:\\n- N/A')) throw new Error(debugBundle);
      if (!debugBundle.includes('Optional Missing:\\n- REQ-COMPUTED-FCF')) throw new Error(debugBundle);
      if (!debugBundle.includes('non-blocking diagnostic; direct free cash flow evidence already available')) throw new Error(debugBundle);
      if (!debugBundle.includes('status: optional missing')) throw new Error(debugBundle);
      if (!debugBundle.includes('fallback_after_timeout: yes')) throw new Error(debugBundle);
      if (!debugBundle.includes('backend: duckdb_lexical')) throw new Error(debugBundle);
      if (!debugBundle.includes('Contract Decision: warning')) throw new Error(debugBundle);
      if (!debugBundle.includes('Draft Release: released_with_warnings')) throw new Error(debugBundle);
      if (!debugBundle.includes('Requirement Scope Counts: core=12, optional_context=4, diagnostic=1')) throw new Error(debugBundle);
      if (!debugBundle.includes('Draft release decision: released_with_warnings')) throw new Error(debugBundle);
      if (!debugBundle.includes('Draft release warnings: code=optional_context_gap')) throw new Error(debugBundle);
      if (debugBundle.includes('[object Object]')) throw new Error(debugBundle);
      if (debugBundle.includes('requirement: REQ-TEXT-NVDA-RISK; status: passed; returned: 0; latency: 12')) throw new Error(debugBundle);
      if (!debugBundle.includes('### Semantic Parser') || !debugBundle.includes('- mode: shadow') || !debugBundle.includes('- injected: no')) throw new Error(debugBundle);
      if (!debugBundle.includes('proposal warnings: requested_metrics:metric_not_supported')) throw new Error(debugBundle);
      if (!debugBundle.includes('现金流质量') || !debugBundle.includes('主要风险')) throw new Error(debugBundle);
      if (debugBundle.includes('raw prompt') || debugBundle.includes('system prompt')) throw new Error(debugBundle);
      if (clipboardSummary !== 'summary text') throw new Error(clipboardSummary);
      if (clipboardEmpty !== '') throw new Error(clipboardEmpty);
      if (!traceGuardOk.ok) throw new Error(traceGuardOk.reason);
      if (traceGuardMismatch.ok || !traceGuardMismatch.reason.includes('requested trace-new')) throw new Error(JSON.stringify(traceGuardMismatch));
      if (traceGuardStale.ok || !traceGuardStale.reason.includes('current answer trace is trace-new')) throw new Error(JSON.stringify(traceGuardStale));
    """
    result = subprocess.run([node, "--input-type=module", "-e", script], cwd=ROOT, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
