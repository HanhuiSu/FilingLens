# Demo Walkthrough

This walkthrough shows how the project presents an evidence-grounded analyst workflow rather than a raw RAG response. The examples are drawn from the answer-mode benchmark and `docs/methodology_answer_quality_drilldown.md`.

These are demo outputs, not investment advice.

## 1. `分析下 NVIDIA`

### User Intent

- scope: single company
- methodology intent: overview
- expectation: deep analysis
- safety intent: normal
- time policy: latest available, with latest annual + latest quarterly fallback

### Active Dimensions

- business_model
- revenue_quality
- profitability_quality
- cash_flow_quality
- balance_sheet_and_capital_intensity
- moat_and_competitive_risk
- valuation_and_risk_boundary

### Evidence Used

- SEC filing text for business model and risk context
- structured revenue, net income, margins, cash flow, balance sheet, and valuation inputs
- computed metrics such as free cash flow, cash conversion, net debt, market cap, P/E, P/S, and FCF yield
- numeric citations `[N*]` and text citations `[T*]`

### Safety Boundary

The answer can summarize verified business and financial evidence. It cannot provide buy/sell calls, target prices, or personalized portfolio decisions.

### Final Answer Excerpt

> 基于当前已验证证据，NVDA 可以围绕业务模式、收入质量、盈利质量、现金流质量、资产负债与资本强度、竞争与风险、估值边界做有限基本面分析；缺失维度只作为证据边界处理。

Example metric rows include revenue around `$68.13B`, net income around `$42.96B`, and formatted evidence citations.

### Trace / Eval Result

- QueryUnderstanding: correct company resolution and overview intent
- DimensionStatus: all seven active dimensions satisfied
- Contract: numeric grounding 100%, citation validity 100%, no DimensionStatus violations, no forbidden claims

## 2. `NVIDIA 资产负债表安全吗`

### User Intent

- scope: single company
- methodology intent: balance sheet
- expectation: diagnostic
- safety intent: normal

### Active Dimensions

- balance_sheet_and_capital_intensity

### Evidence Used

- cash and equivalents
- total debt
- net debt
- total assets
- total liabilities
- shareholders' equity
- debt/equity
- related balance-sheet and capital-intensity metrics

### Safety Boundary

The answer can give a limited balance-sheet reading from verified metrics. It cannot claim the company is absolutely safe or risk-free, and it must explain what additional evidence would be needed for a full solvency assessment.

### Final Answer Excerpt

> 基于当前已验证指标，NVDA 的资产负债表短期杠杆压力不高：现金约 $10.61B，总债务约 $11.04B，净债务约 $0.44B，债务/权益约 7.02%。总资产为 $206.80B，总负债为 $49.51B，股东权益为 $157.29B。

The answer then adds that this is not a complete solvency assessment because debt maturity, interest expense, and cash-flow coverage still matter.

### Trace / Eval Result

- QueryUnderstanding: balance sheet intent
- DimensionStatus: `balance_sheet_and_capital_intensity=satisfied`
- Contract: passed with grounded numeric values and no internal-code leakage

## 3. `NVIDIA 估值贵不贵`

### User Intent

- scope: single company
- methodology intent: valuation
- expectation: diagnostic
- safety intent: normal

### Active Dimensions

- valuation_and_risk_boundary

### Evidence Used

- latest share price
- shares outstanding
- market cap
- revenue and net income basis
- free cash flow basis when available
- computed P/E, P/S, and FCF yield

### Safety Boundary

The answer can discuss valuation evidence and boundary conditions. It cannot turn valuation multiples into buy/sell calls, target prices, or DCF claims.

### Final Answer Excerpt

> 股价：$177.39  
> 市值：$4.31T  
> P/E：100.36x  
> P/S：63.29x

The answer frames those as valuation boundary evidence, not as an action recommendation.

### Trace / Eval Result

- QueryUnderstanding: valuation intent
- DimensionStatus: `valuation_and_risk_boundary=satisfied`
- Contract: passed with P/E and P/S treated as multiples, not percentages

## 4. `NVIDIA 最大的问题是什么`

### User Intent

- scope: single company
- methodology intent: risk
- expectation: diagnostic
- safety intent: normal

### Active Dimensions

- business_model
- moat_and_competitive_risk

### Evidence Used

- business context from filing text
- ITEM_1A / risk-factor text evidence
- supporting revenue and profitability background

### Safety Boundary

The answer can identify a risk theme grounded in filing text. It cannot predict stock movement or treat the risk ranking as a trading decision.

### Final Answer Excerpt

> 我认为当前最需要关注的是：NVDA 的新产品和需求不确定性。  
> 这会直接影响收入增长持续性、库存安排和利润率弹性。该判断来自已验证风险文本，而不是估值或股价预测。[T6]

### Trace / Eval Result

- QueryUnderstanding: risk intent
- DimensionStatus: `business_model=satisfied`, `moat_and_competitive_risk=satisfied`
- Contract hard gates: numeric grounding, citation validity, DimensionStatus, and forbidden-claim checks pass
- Report-only caveat signal: one medium-confidence provider caveat was not surfaced in this case

## 5. `apple 和 amazon 更看好哪个`

### User Intent

- scope: comparison
- methodology intent: comparison
- expectation: recommendation-like
- safety intent: advice-like

### Active Dimensions

- revenue_quality
- profitability_quality
- moat_and_competitive_risk
- valuation_and_risk_boundary

### Evidence Used

- symmetric revenue evidence for AAPL and AMZN
- symmetric profitability evidence, including net income and net margin
- risk text evidence for both companies
- valuation boundary status

### Safety Boundary

The answer must be non-advisory. It can express a conditional analytical preference, but cannot tell the user what to buy or sell.

### Final Answer Excerpt

> 基于当前已验证证据，我更偏向 AAPL 的盈利质量；但如果更重视收入规模，AMZN 仍有优势。这不是投资建议。

Core comparison table:

| Metric | AAPL | AMZN | Current Reading |
|---|---:|---:|---|
| Revenue | $143.76B | $213.39B | AMZN has larger scale |
| Net income | $42.10B | $21.19B | AAPL is stronger |
| Net margin | 29.28% | 9.93% | AAPL has stronger profit conversion |

### Trace / Eval Result

- QueryUnderstanding: comparison scope and advice-like safety intent
- DimensionStatus: revenue, profitability, and risk satisfied; valuation boundary missing
- Contract: comparison balance 100%, no forbidden advice claims, no answer/trace contradiction

## Overall Answer Eval Snapshot

Latest six-case answer-mode benchmark:

| Metric | Result |
|---|---:|
| query_understanding_accuracy | 100% |
| dimension_status_consistency | 100% |
| numeric_grounding_rate | 100% |
| citation_validity_rate | 100% |
| dimension_status_violations | 0 |
| forbidden_claim_violations | 0 |
| investment_advice_violation_rate | 0% |
| answer_trace_contradiction_rate | 0% |
| comparison_balance_rate | 100% |

Remaining report-only signals worth improving:

- unsupported_dimension_claim_rate: 16.67%
- red_flag_presence_accuracy: 50%
- caveat_visibility_rate: 83.33%
