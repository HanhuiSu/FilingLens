"""Run manual API probe suites through POST /chat.

This runner intentionally mirrors the frontend path: it calls the FastAPI
HTTP API and then fetches the sanitized trace UI bundle. It does not import or
invoke the agent graph directly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_BASE = "http://127.0.0.1:8080"
TIMEOUT_S = int(os.environ.get("MANUAL_PROBE_TIMEOUT_S", "420"))

CASES_20: list[dict[str, str]] = [
    {"case_id": "Q01", "query": "为什么 AMZN 的利润下降了？"},
    {"case_id": "Q02", "query": "AMZN 现在值得买吗？"},
    {"case_id": "Q03", "query": "AMZN 和 NVDA 哪个更危险？"},
    {"case_id": "Q04", "query": "NVDA 的营收增长主要是数据中心驱动的吗？"},
    {"case_id": "Q05", "query": "为什么 Amazon 的自由现金流变差了？"},
    {"case_id": "Q06", "query": "比较一下 MSFT 和 AAPL 谁的现金流质量更好"},
    {"case_id": "Q07", "query": "AAPL overview"},
    {"case_id": "Q08", "query": "分析一下 MSFT 的盈利质量"},
    {"case_id": "Q09", "query": "NVDA 的估值贵不贵？"},
    {"case_id": "Q10", "query": "Amazon 的主要风险是什么，按重要性排序"},
    {"case_id": "Q11", "query": "比较 AMZN 和 AAPL 谁的增长质量更好"},
    {"case_id": "Q12", "query": "MSFT 的资产负债表稳不稳？"},
    {"case_id": "Q13", "query": "AAPL 的毛利率变化说明了什么？"},
    {"case_id": "Q14", "query": "NVDA 最近现金流怎么样？"},
    {"case_id": "Q15", "query": "AMZN 的 AWS 对整体利润有多重要？"},
    {"case_id": "Q16", "query": "如果下季度经济放缓，MSFT 最大的财务风险是什么？"},
    {"case_id": "Q17", "query": "AAPL 和 NVDA 哪个估值风险更高？"},
    {"case_id": "Q18", "query": "为什么 NVDA 的网络业务增长这么快？"},
    {"case_id": "Q19", "query": "AMZN 的收入质量怎么样？"},
    {"case_id": "Q20", "query": "请用一句话回答：NVDA 是不是可以买？"},
]

CASES_60: list[dict[str, str]] = [
    {"case_id": "Q01", "query": "How do Apple's current fundamentals look across revenue, profitability, cash flow, risk, and valuation?"},
    {"case_id": "Q02", "query": "Give a Chinese overview of NVIDIA covering business, profitability, cash flow, and risks."},
    {"case_id": "Q03", "query": "How should we understand Amazon's business structure and financial quality?"},
    {"case_id": "Q04", "query": "What kind of company is Microsoft today, and where are its fundamental strengths?"},
    {"case_id": "Q05", "query": "What is Google's business model, and what are its main risks?"},
    {"case_id": "Q06", "query": "What key indicators should we watch to understand Meta's current fundamentals?"},
    {"case_id": "Q07", "query": "What is driving NVIDIA's revenue growth?"},
    {"case_id": "Q08", "query": "Is NVIDIA's Data Center business the main source of revenue growth?"},
    {"case_id": "Q09", "query": "How is Amazon's revenue quality?"},
    {"case_id": "Q10", "query": "Is Apple's revenue growth quality strong?"},
    {"case_id": "Q11", "query": "Is Microsoft's revenue growth sustainable?"},
    {"case_id": "Q12", "query": "Is Google's revenue growth mainly driven by advertising or cloud?"},
    {"case_id": "Q13", "query": "How should we assess Meta's revenue growth quality?"},
    {"case_id": "Q14", "query": "Does Tesla have any revenue growth quality issues?"},
    {"case_id": "Q15", "query": "How resilient is Costco's revenue quality in a weaker consumer environment?"},
    {"case_id": "Q16", "query": "How stable is Walmart's revenue base?"},
    {"case_id": "Q17", "query": "How strong is Microsoft's profitability quality?"},
    {"case_id": "Q18", "query": "What does Apple's gross margin tell us?"},
    {"case_id": "Q19", "query": "Has Amazon's margin profile improved?"},
    {"case_id": "Q20", "query": "Is NVIDIA's high margin profile sustainable?"},
    {"case_id": "Q21", "query": "How should we evaluate Google's profitability quality?"},
    {"case_id": "Q22", "query": "Has Meta's profitability recovered?"},
    {"case_id": "Q23", "query": "What is Tesla's biggest profitability issue?"},
    {"case_id": "Q24", "query": "How stable is Johnson & Johnson's profitability quality?"},
    {"case_id": "Q25", "query": "How is NVIDIA's recent cash-flow quality?"},
    {"case_id": "Q26", "query": "Why is Amazon's free cash flow under pressure?"},
    {"case_id": "Q27", "query": "Does Apple convert earnings into cash flow effectively?"},
    {"case_id": "Q28", "query": "Is Microsoft's free cash flow strong?"},
    {"case_id": "Q29", "query": "How is Google's cash-flow quality?"},
    {"case_id": "Q30", "query": "How does Meta's capital expenditure affect free cash flow?"},
    {"case_id": "Q31", "query": "How should we assess Tesla's cash-flow quality and capex pressure?"},
    {"case_id": "Q32", "query": "Which has better cash-flow quality, Apple or Microsoft?"},
    {"case_id": "Q33", "query": "How stable is Microsoft's balance sheet?"},
    {"case_id": "Q34", "query": "Is Apple's leverage high?"},
    {"case_id": "Q35", "query": "How safe is NVIDIA's balance sheet?"},
    {"case_id": "Q36", "query": "Does Amazon face pressure from debt and free cash flow?"},
    {"case_id": "Q37", "query": "How strong is Tesla's cash and debt position?"},
    {"case_id": "Q38", "query": "How should we assess JPMorgan's balance-sheet risk?"},
    {"case_id": "Q39", "query": "How could the interest-rate environment affect Bank of America's balance sheet?"},
    {"case_id": "Q40", "query": "How should we assess Goldman Sachs' balance sheet and capital pressure?"},
    {"case_id": "Q41", "query": "How expensive is NVIDIA's valuation based on available metrics?"},
    {"case_id": "Q42", "query": "How high is Apple's valuation risk?"},
    {"case_id": "Q43", "query": "Is Amazon worth buying now? Give a bounded view, not investment advice."},
    {"case_id": "Q44", "query": "How should we assess Microsoft's valuation risk?"},
    {"case_id": "Q45", "query": "Is Google's valuation more reasonable than other large tech companies?"},
    {"case_id": "Q46", "query": "Does Meta's valuation risk match its growth expectations?"},
    {"case_id": "Q47", "query": "Where does Tesla's valuation risk mainly come from?"},
    {"case_id": "Q48", "query": "Compare Apple and NVIDIA valuation risk based on P/E, P/S, and FCF yield."},
    {"case_id": "Q49", "query": "What are Amazon's main risks, ranked by importance?"},
    {"case_id": "Q50", "query": "What is NVIDIA's biggest business risk?"},
    {"case_id": "Q51", "query": "If AI demand slows, what is NVIDIA's biggest financial risk?"},
    {"case_id": "Q52", "query": "If the economy slows next quarter, what is Microsoft's biggest financial risk?"},
    {"case_id": "Q53", "query": "What regulatory risks does Google face?"},
    {"case_id": "Q54", "query": "What are Meta's main operating risks?"},
    {"case_id": "Q55", "query": "For Tesla, is the biggest risk demand, margin pressure, or competition?"},
    {"case_id": "Q56", "query": "What are Johnson & Johnson's main disclosed risk factors?"},
    {"case_id": "Q57", "query": "Which has better growth quality, Amazon or Apple?"},
    {"case_id": "Q58", "query": "Which company has more stable fundamentals, Microsoft or Google?"},
    {"case_id": "Q59", "query": "Which has higher growth risk, NVIDIA or AMD?"},
    {"case_id": "Q60", "query": "Among Apple, Microsoft, and NVIDIA, which has the highest valuation risk based on P/E, P/S, and FCF yield?"},
]

CASES_5: list[dict[str, str]] = [
    {"case_id": "Q01", "query": "What is driving NVIDIA's data center revenue growth?"},
    {"case_id": "Q02", "query": "Why is Amazon's free cash flow under pressure?"},
    {"case_id": "Q03", "query": "How risky is NVIDIA's valuation based on P/E, P/S, and FCF yield?"},
    {"case_id": "Q04", "query": "What is Microsoft's biggest financial risk if the economy slows next quarter?"},
    {"case_id": "Q05", "query": "Compare Apple and NVIDIA valuation risk."},
]

DEMO_300_BANK_PATH = ROOT / "eval" / "demo_question_bank_300.json"


def _load_demo_300_cases() -> list[dict[str, str]]:
    if not DEMO_300_BANK_PATH.exists():
        return []
    data = json.loads(DEMO_300_BANK_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {DEMO_300_BANK_PATH}")
    cases: list[dict[str, str]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"Expected object at demo case #{index}")
        case_id = str(item.get("case_id") or "").strip()
        query = str(item.get("query") or "").strip()
        category = str(item.get("category") or "").strip()
        language = str(item.get("language") or "").strip()
        expected_case_id = f"D{index:03d}"
        if case_id != expected_case_id:
            raise ValueError(f"Expected case_id {expected_case_id}, got {case_id!r}")
        if not query or not category or not language:
            raise ValueError(f"Missing query/category/language for {case_id}")
        cases.append({"case_id": case_id, "query": query, "category": category, "language": language})
    if len(cases) != 300:
        raise ValueError(f"Expected 300 demo cases, got {len(cases)}")
    return cases


CASES_300_DEMO: list[dict[str, str]] = _load_demo_300_cases()

DEMO_SELECTED_STRONG_IDS = {
    "D077", "D056", "D114", "D137", "D139", "D162", "D170", "D171", "D207",
    "D210", "D211", "D218", "D236", "D239", "D240", "D251", "D275",
}

DEMO_SELECTED_CASES: list[dict[str, str]] = [
    {"case_id": "D077", "query": "Why is NVIDIA's networking business growing so fast?", "language": "en", "category": "business-drivers-revenue-quality", "company": "NVIDIA", "ticker": "NVDA", "source_tier": "strong"},
    {"case_id": "D056", "query": "Is AMD's revenue growth mainly driven by AI demand?", "language": "en", "category": "business-drivers-revenue-quality", "company": "AMD", "ticker": "AMD", "source_tier": "strong"},
    {"case_id": "D114", "query": "Is Costco's low margin profile normal for its business model?", "language": "en", "category": "margins-business-model", "company": "Costco", "ticker": "COST", "source_tier": "strong"},
    {"case_id": "D137", "query": "Why is Amazon's free cash flow under pressure?", "language": "en", "category": "cash-flow-capex", "company": "Amazon", "ticker": "AMZN", "source_tier": "strong"},
    {"case_id": "D139", "query": "Is Microsoft's free cash flow strong?", "language": "en", "category": "cash-flow-capex", "company": "Microsoft", "ticker": "MSFT", "source_tier": "strong"},
    {"case_id": "D162", "query": "BAC 的利率环境风险会不会影响资产负债表？", "language": "zh", "category": "balance-sheet-valuation-risk", "company": "Bank of America", "ticker": "BAC", "source_tier": "strong"},
    {"case_id": "D170", "query": "How strong is Tesla's cash and debt position?", "language": "en", "category": "balance-sheet-valuation-risk", "company": "Tesla", "ticker": "TSLA", "source_tier": "strong"},
    {"case_id": "D171", "query": "Is Google's balance sheet financially strong?", "language": "en", "category": "balance-sheet-valuation-risk", "company": "Google", "ticker": "GOOGL", "source_tier": "strong"},
    {"case_id": "D207", "query": "Does Oracle's valuation risk match its cash-flow profile?", "language": "en", "category": "balance-sheet-valuation-risk", "company": "Oracle", "ticker": "ORCL", "source_tier": "strong"},
    {"case_id": "D210", "query": "Is Intel cheap for a reason?", "language": "en", "category": "balance-sheet-valuation-risk", "company": "Intel", "ticker": "INTC", "source_tier": "strong"},
    {"case_id": "D211", "query": "AMZN 的主要风险是什么，按重要性排序。", "language": "zh", "category": "risk-ranking-disclosure", "company": "Amazon", "ticker": "AMZN", "source_tier": "strong"},
    {"case_id": "D218", "query": "JNJ 的主要披露风险有哪些？", "language": "zh", "category": "risk-ranking-disclosure", "company": "Johnson & Johnson", "ticker": "JNJ", "source_tier": "strong"},
    {"case_id": "D236", "query": "What are AMD's main competitive risks?", "language": "en", "category": "risk-ranking-disclosure", "company": "AMD", "ticker": "AMD", "source_tier": "strong"},
    {"case_id": "D239", "query": "How serious is Salesforce's growth slowdown risk?", "language": "en", "category": "risk-ranking-disclosure", "company": "Salesforce", "ticker": "CRM", "source_tier": "strong"},
    {"case_id": "D240", "query": "Where is Intel's turnaround risk most concentrated?", "language": "en", "category": "risk-ranking-disclosure", "company": "Intel", "ticker": "INTC", "source_tier": "strong"},
    {"case_id": "D251", "query": "如果 iPhone 需求放缓，AAPL 的利润率会有什么风险？", "language": "zh", "category": "scenario-comparison", "company": "Apple", "ticker": "AAPL", "source_tier": "strong"},
    {"case_id": "D275", "query": "AAPL 和 MSFT 谁的现金流质量更好？", "language": "zh", "category": "scenario-comparison", "company": "Apple / Microsoft", "ticker": "AAPL/MSFT", "source_tier": "strong"},
    {"case_id": "D018", "query": "How should I understand Amazon's business model and financial quality?", "language": "en", "category": "company_overview_fundamentals", "company": "Amazon", "ticker": "AMZN", "source_tier": "maybe"},
    {"case_id": "D019", "query": "What kind of company is Microsoft today, and where are its fundamental strengths?", "language": "en", "category": "company_overview_fundamentals", "company": "Microsoft", "ticker": "MSFT", "source_tier": "maybe"},
    {"case_id": "D023", "query": "How should we evaluate Costco's business model and financial quality?", "language": "en", "category": "company_overview_fundamentals", "company": "Costco", "ticker": "COST", "source_tier": "maybe"},
    {"case_id": "D024", "query": "Is Walmart fundamentally stable?", "language": "en", "category": "company_overview_fundamentals", "company": "Walmart", "ticker": "WMT", "source_tier": "maybe"},
    {"case_id": "D061", "query": "NVDA 的数据中心业务是不是主要增长来源？", "language": "zh", "category": "business-drivers-revenue-quality", "company": "NVIDIA", "ticker": "NVDA", "source_tier": "maybe"},
    {"case_id": "D066", "query": "MSFT 的云业务是不是基本面核心？", "language": "zh", "category": "business-drivers-revenue-quality", "company": "Microsoft", "ticker": "MSFT", "source_tier": "maybe"},
    {"case_id": "D067", "query": "GOOGL Cloud 对 Google 的增长有多重要？", "language": "zh", "category": "business-drivers-revenue-quality", "company": "Google", "ticker": "GOOGL", "source_tier": "maybe"},
    {"case_id": "D074", "query": "AVGO 的 AI 芯片业务是否是主要驱动？", "language": "zh", "category": "business-drivers-revenue-quality", "company": "Broadcom", "ticker": "AVGO", "source_tier": "maybe"},
    {"case_id": "D091", "query": "MSFT 的盈利质量怎么样？", "language": "zh", "category": "margins-business-model", "company": "Microsoft", "ticker": "MSFT", "source_tier": "maybe"},
    {"case_id": "D094", "query": "NVDA 的高利润率可持续吗？", "language": "zh", "category": "margins-business-model", "company": "NVIDIA", "ticker": "NVDA", "source_tier": "maybe"},
    {"case_id": "D097", "query": "TSLA 的盈利能力最大问题是什么？", "language": "zh", "category": "margins-business-model", "company": "Tesla", "ticker": "TSLA", "source_tier": "maybe"},
    {"case_id": "D099", "query": "COST 的低利润率是不是正常？", "language": "zh", "category": "margins-business-model", "company": "Costco", "ticker": "COST", "source_tier": "maybe"},
    {"case_id": "D103", "query": "ORCL 的利润率质量怎么样？", "language": "zh", "category": "margins-business-model", "company": "Oracle", "ticker": "ORCL", "source_tier": "maybe"},
    {"case_id": "D121", "query": "NVDA 最近现金流质量怎么样？", "language": "zh", "category": "cash-flow-capex", "company": "NVIDIA", "ticker": "NVDA", "source_tier": "maybe"},
    {"case_id": "D149", "query": "Is Salesforce's cash-flow quality healthy?", "language": "en", "category": "cash-flow-capex", "company": "Salesforce", "ticker": "CRM", "source_tier": "maybe"},
    {"case_id": "D166", "query": "How stable is Microsoft's balance sheet?", "language": "en", "category": "balance-sheet-valuation-risk", "company": "Microsoft", "ticker": "MSFT", "source_tier": "maybe"},
    {"case_id": "D176", "query": "How should we assess JPMorgan's balance-sheet risk?", "language": "en", "category": "balance-sheet-valuation-risk", "company": "JPMorgan", "ticker": "JPM", "source_tier": "maybe"},
    {"case_id": "D181", "query": "NVDA 的估值贵不贵？", "language": "zh", "category": "balance-sheet-valuation-risk", "company": "NVIDIA", "ticker": "NVDA", "source_tier": "maybe"},
    {"case_id": "D208", "query": "Is Salesforce's valuation reasonable?", "language": "en", "category": "balance-sheet-valuation-risk", "company": "Salesforce", "ticker": "CRM", "source_tier": "maybe"},
    {"case_id": "D220", "query": "WMT 的主要风险按重要性排序。", "language": "zh", "category": "risk-ranking-disclosure", "company": "Walmart", "ticker": "WMT", "source_tier": "maybe"},
    {"case_id": "D225", "query": "INTC 的转型风险主要在哪里？", "language": "zh", "category": "risk-ranking-disclosure", "company": "Intel", "ticker": "INTC", "source_tier": "maybe"},
    {"case_id": "D230", "query": "What regulatory risks does Google face?", "language": "en", "category": "risk-ranking-disclosure", "company": "Google", "ticker": "GOOGL", "source_tier": "maybe"},
    {"case_id": "D245", "query": "如果消费者支出放缓，AAPL 最大风险是什么？", "language": "zh", "category": "scenario-comparison", "company": "Apple", "ticker": "AAPL", "source_tier": "maybe"},
    {"case_id": "D260", "query": "If consumer spending slows, what is Apple's biggest risk?", "language": "en", "category": "scenario-comparison", "company": "Apple", "ticker": "AAPL", "source_tier": "maybe"},
]

DEMO_SELECTED_MODULES: list[dict[str, Any]] = [
    {
        "id": "business-drivers-revenue-quality",
        "title": "业务驱动与收入质量",
        "description": "展示 FilingLens 能从 filing 文本里解释增长驱动，而不是只报收入数字。",
        "capability": "Explain growth drivers and revenue quality from filing evidence.",
        "caseIds": ["D077", "D056", "D061", "D066", "D067", "D074"],
    },
    {
        "id": "margins-business-model",
        "title": "利润率与商业模式",
        "description": "展示“低利润率是否正常”“高毛利是否可持续”这类分析能力。",
        "capability": "Connect margin profile to business model and sustainability.",
        "caseIds": ["D114", "D099", "D094", "D097", "D103", "D091"],
    },
    {
        "id": "cash-flow-capex",
        "title": "现金流与资本开支",
        "description": "展示 OCF、FCF、capex、cash conversion 的因果解释，尤其是 AMZN FCF 压力。",
        "capability": "Explain operating cash flow, free cash flow, capex, and cash conversion.",
        "caseIds": ["D137", "D139", "D149", "D121"],
    },
    {
        "id": "balance-sheet-valuation-risk",
        "title": "资产负债表与估值风险",
        "description": "展示债务、现金、D/E、FCF yield、负盈利估值风险。",
        "capability": "Assess balance-sheet resilience and valuation-risk boundaries.",
        "caseIds": ["D162", "D170", "D171", "D207", "D210", "D166", "D176", "D181", "D208"],
    },
    {
        "id": "risk-ranking-disclosure",
        "title": "风险排序与披露解释",
        "description": "展示把 10-K/10-Q 风险文本转成优先级和财务传导路径。",
        "capability": "Turn filing risk factors into ranked risks and financial transmission paths.",
        "caseIds": ["D211", "D218", "D236", "D239", "D240", "D220", "D225", "D230"],
    },
    {
        "id": "scenario-comparison",
        "title": "情景分析与公司对比",
        "description": "展示“如果需求放缓 / AI demand slows / 谁现金流更好”这种投资研究式问答。",
        "capability": "Answer scenario analysis and company-comparison questions with evidence boundaries.",
        "caseIds": ["D251", "D275", "D245", "D260"],
    },
]

SUITES: dict[str, dict[str, Any]] = {
    "manual_api_probe_20": {
        "cases": CASES_20,
        "out_dir": ROOT / "eval" / "reports" / "manual_api_probe_20",
        "trace_prefix": "manual-probe-20",
        "title": "Manual API Probe 20",
    },
    "manual_api_probe_60": {
        "cases": CASES_60,
        "out_dir": ROOT / "eval" / "reports" / "manual_api_probe_60",
        "trace_prefix": "manual-probe-60",
        "title": "Manual API Probe 60",
    },
    "manual_api_probe_5": {
        "cases": CASES_5,
        "out_dir": ROOT / "eval" / "reports" / "manual_api_probe_5",
        "trace_prefix": "manual-probe-5",
        "title": "Manual API Probe 5",
    },
    "manual_api_probe_300_demo": {
        "cases": CASES_300_DEMO,
        "out_dir": ROOT / "eval" / "reports" / "manual_api_probe_300_demo",
        "trace_prefix": "manual-probe-300-demo",
        "title": "Manual API Probe 300 Demo",
    },
    "manual_api_probe_demo_selected": {
        "cases": DEMO_SELECTED_CASES,
        "out_dir": ROOT / "eval" / "reports" / "manual_api_probe_demo_selected",
        "trace_prefix": "manual-probe-demo-selected",
        "title": "FilingLens Selected Demo Probe",
    },
}
DEFAULT_SUITE = "manual_api_probe_20"
CASES = CASES_20


def _select_cases(case_ids: str, all_cases: list[dict[str, str]]) -> list[dict[str, str]]:
    text = str(case_ids or "").strip()
    if not text:
        return list(all_cases)
    case_by_id = {case["case_id"]: case for case in all_cases}
    selected: list[dict[str, str]] = []
    for raw in text.replace(",", " ").split():
        case_id = raw.strip().upper()
        if not case_id:
            continue
        if case_id.startswith("Q") and len(case_id) == 2:
            case_id = f"Q0{case_id[1:]}"
        if case_id not in case_by_id:
            valid = ", ".join(case["case_id"] for case in all_cases)
            raise ValueError(f"Unknown case_id {raw!r}. Valid case ids: {valid}")
        selected.append(case_by_id[case_id])
    return selected


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _trace_id_for_case(run_id: str, case_id: str, trace_prefix: str) -> str:
    return f"{trace_prefix}-{run_id}-{case_id.lower()}"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: Any, *, limit: int | None = None) -> str:
    text = str(value or "").strip()
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _json_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _markdown_text(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "_empty_"


def _markdown_inline(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    return text.replace("|", "\\|").replace("\n", " ")


def _request_json(client: httpx.Client, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    response = client.request(method, url, **kwargs)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(data).__name__}")
    return data


def _extract_evidence_counts(trace_ui: Mapping[str, Any]) -> dict[str, Any]:
    packet = _as_dict(trace_ui.get("evidence_packet"))
    summary = _as_dict(trace_ui.get("evidence_packet_summary")) or _as_dict(packet.get("summary"))
    counts = {
        "numeric_count": int(summary.get("numeric_count") or 0),
        "text_count": int(summary.get("text_count") or 0),
        "computed_count": int(summary.get("computed_count") or 0),
        "event_count": int(summary.get("event_count") or 0),
        "limitations_count": int(summary.get("limitations_count") or 0),
        "numeric_returned_count": 0,
        "numeric_validated_count": 0,
        "text_returned_count": 0,
        "text_validated_count": 0,
        "computed_returned_count": 0,
        "computed_validated_count": 0,
        "text_candidate_count": int(summary.get("text_candidate_count") or 0),
        "text_pre_citation_validated_count": int(summary.get("text_pre_citation_validated_count") or 0),
        "text_citable_count": int(summary.get("text_citable_count") or 0),
        "text_final_packet_count": int(summary.get("text_final_packet_count") or summary.get("text_snippet_count") or 0),
        "text_drop_stage_counts": _as_dict(summary.get("text_drop_stage_counts")),
    }
    for item in _as_list(trace_ui.get("evidence_validation_records")):
        if not isinstance(item, Mapping):
            continue
        evidence_type = str(item.get("evidence_type") or "")
        returned = int(item.get("tool_returned_count") or 0)
        validated = int(item.get("validated_evidence_count") or 0)
        if evidence_type == "numeric":
            counts["numeric_returned_count"] += returned
            counts["numeric_validated_count"] += validated
        elif evidence_type == "text":
            counts["text_returned_count"] += returned
            counts["text_validated_count"] += validated
        elif evidence_type == "calculation":
            counts["computed_returned_count"] += returned
            counts["computed_validated_count"] += validated
    return counts


def _answer_history_summary(trace_ui: Mapping[str, Any]) -> list[str]:
    summary: list[str] = []
    for item in _as_list(trace_ui.get("answer_history")):
        if not isinstance(item, Mapping):
            continue
        transform = str(item.get("transform") or "")
        owner = str(item.get("new_owner") or "")
        changed = "changed" if bool(item.get("body_changed")) else "owner"
        summary.append(f"{item.get('index', len(summary) + 1)}:{transform}->{owner}:{changed}")
    return summary


def _extract_missing_requirements(trace_ui: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = _as_dict(trace_ui.get("evidence_plan"))
    requirements = _as_list(plan.get("requirements"))
    out: list[dict[str, Any]] = []
    for item in requirements:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("raw_status") or item.get("status") or "")
        if status not in {"missing", "partial", "failed", "blocked", "optional_missing"}:
            continue
        out.append(
            {
                "requirement_id": item.get("requirement_id", ""),
                "status": status,
                "scope": item.get("scope", ""),
                "required": bool(item.get("required", False)),
                "blocking": bool(item.get("blocking", False)),
                "evidence_type": item.get("evidence_type", ""),
                "tool": item.get("tool", ""),
                "dimension": item.get("dimension", ""),
                "company": item.get("company", ""),
                "missing_reason": item.get("missing_reason", ""),
            }
        )
    return out


def _extract_contract_items(trace_ui: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    contract = _as_dict(trace_ui.get("contract"))
    warnings = _as_list(contract.get("warnings"))
    violations = _as_list(contract.get("violations"))
    decision = _as_dict(trace_ui.get("contract_decision"))
    warnings.extend(_as_list(decision.get("warnings")))
    violations.extend(_as_list(decision.get("violations")))
    seen: set[str] = set()

    def dedupe(items: list[Any]) -> list[Any]:
        out: list[Any] = []
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    return dedupe(warnings), dedupe(violations)


def _progress_summary(trace_ui: Mapping[str, Any], *, limit: int = 30) -> list[dict[str, Any]]:
    events = _as_list(trace_ui.get("progress_events"))
    summary: list[dict[str, Any]] = []
    for item in events[:limit]:
        if not isinstance(item, Mapping):
            continue
        summary.append(
            {
                "time": item.get("created_at") or item.get("timestamp") or item.get("time") or "",
                "event": item.get("event") or item.get("event_type") or item.get("type") or "",
                "node": item.get("node") or item.get("node_id") or "",
                "status": item.get("status") or "",
                "message": _text(item.get("message") or item.get("summary"), limit=180),
            }
        )
    if len(events) > limit:
        summary.append({"time": "", "event": "truncated", "node": "", "status": "", "message": f"{len(events) - limit} more events omitted"})
    return summary


def _answer_part_status(trace_ui: Mapping[str, Any]) -> dict[str, Any]:
    direct = _as_dict(trace_ui.get("answer_part_status_by_id"))
    if direct:
        return direct
    research = _as_dict(trace_ui.get("research_plan"))
    return _as_dict(research.get("answer_part_status_by_id"))


def _case_from_responses(
    case: Mapping[str, str],
    requested_trace_id: str,
    chat_response: Mapping[str, Any],
    trace_ui: Mapping[str, Any] | None,
    bundle_path: Path | None,
    elapsed_s: float,
    error: str = "",
) -> dict[str, Any]:
    trace = _as_dict(trace_ui)
    output = _as_dict(chat_response.get("output"))
    warnings, violations = _extract_contract_items(trace)
    contract_status = (
        str(trace.get("contract_status") or "")
        or str(chat_response.get("contract_status") or "")
        or "error"
    )
    final_route = str(trace.get("final_route") or output.get("final_route") or "")
    contract_decision = _as_dict(trace.get("contract_decision") or chat_response.get("contract_decision"))
    relevance_decision = _as_dict(trace.get("relevance_decision"))
    evidence_counts = _extract_evidence_counts(trace)
    answer = str(chat_response.get("answer") or trace.get("final_answer") or "")
    return {
        "case_id": case["case_id"],
        "category": str(case.get("category") or ""),
        "language": str(case.get("language") or ""),
        "company": str(case.get("company") or ""),
        "ticker": str(case.get("ticker") or ""),
        "source_tier": str(case.get("source_tier") or ""),
        "query": case["query"],
        "trace_id": str(trace.get("trace_id") or chat_response.get("trace_id") or requested_trace_id),
        "answer": answer,
        "answer_chars": len(answer),
        "contract_status": contract_status,
        "contract_decision": contract_decision,
        "contract_route": str(contract_decision.get("route") or ""),
        "relevance_decision": relevance_decision,
        "relevance_route": str(relevance_decision.get("route") or ""),
        "final_route": final_route,
        "final_answer_source": str(trace.get("final_answer_source") or output.get("final_answer_source") or chat_response.get("final_answer_source") or ""),
        "answer_history_summary": _answer_history_summary(trace),
        "answer_quality_tier": str(trace.get("answer_quality_tier") or output.get("answer_quality_tier") or chat_response.get("answer_quality_tier") or ""),
        "quality_tier_reason": str(trace.get("quality_tier_reason") or output.get("quality_tier_reason") or chat_response.get("quality_tier_reason") or ""),
        "main_question_covered": bool(trace.get("main_question_covered", output.get("main_question_covered", chat_response.get("main_question_covered", True)))),
        "fallback_intent_match": bool(trace.get("fallback_intent_match", output.get("fallback_intent_match", chat_response.get("fallback_intent_match", True)))),
        "answered_dimensions": _as_list(trace.get("answered_dimensions") or output.get("answered_dimensions") or chat_response.get("answered_dimensions")),
        "unresolved_relevance_failures": _as_list(trace.get("unresolved_relevance_failures") or output.get("unresolved_relevance_failures") or chat_response.get("unresolved_relevance_failures")),
        "format_constraints_satisfied": bool(trace.get("format_constraints_satisfied", output.get("format_constraints_satisfied", chat_response.get("format_constraints_satisfied", True)))),
        "source_before_repair": str(trace.get("source_before_repair") or output.get("source_before_repair") or chat_response.get("source_before_repair") or ""),
        "repair_applied": bool(trace.get("repair_applied", output.get("repair_applied", chat_response.get("repair_applied", False)))),
        "repair_types": _as_list(trace.get("repair_types") or output.get("repair_types") or chat_response.get("repair_types")),
        "material_claim_uncited_count": int(trace.get("material_claim_uncited_count", output.get("material_claim_uncited_count", chat_response.get("material_claim_uncited_count", 0))) or 0),
        "core_missing_parts": _as_list(trace.get("core_missing_parts") or output.get("core_missing_parts") or chat_response.get("core_missing_parts")),
        "optional_missing_parts": _as_list(trace.get("optional_missing_parts") or output.get("optional_missing_parts") or chat_response.get("optional_missing_parts")),
        "risk_items_directly_supported_count": int(trace.get("risk_items_directly_supported_count", output.get("risk_items_directly_supported_count", chat_response.get("risk_items_directly_supported_count", 0))) or 0),
        "risk_items_template_only_count": int(trace.get("risk_items_template_only_count", output.get("risk_items_template_only_count", chat_response.get("risk_items_template_only_count", 0))) or 0),
        "company_specific_token_leakage": int(trace.get("company_specific_token_leakage", output.get("company_specific_token_leakage", chat_response.get("company_specific_token_leakage", 0))) or 0),
        "output_language": str(trace.get("output_language") or output.get("output_language") or chat_response.get("output_language") or ""),
        "language_leakage": int(trace.get("language_leakage", output.get("language_leakage", chat_response.get("language_leakage", 0))) or 0),
        "language_leakage_unresolved": bool(trace.get("language_leakage_unresolved", output.get("language_leakage_unresolved", chat_response.get("language_leakage_unresolved", False)))),
        "segment_or_product_scope": str(trace.get("segment_or_product_scope") or output.get("segment_or_product_scope") or chat_response.get("segment_or_product_scope") or ""),
        "synthesis_mode": str(trace.get("synthesis_mode") or output.get("synthesis_mode") or ""),
        "blocked": str(contract_status) == "blocked" or final_route == "blocked",
        "task_type": str(trace.get("task_type") or chat_response.get("task_type") or ""),
        "answer_mode": str(trace.get("answer_mode") or output.get("answer_mode") or ""),
        "canonical_intent": _as_dict(trace.get("canonical_intent") or chat_response.get("canonical_intent")),
        "evidence_policy_id": str(trace.get("evidence_policy_id") or ""),
        "evidence_policy": _as_dict(trace.get("evidence_policy")),
        "research_plan_source": str(trace.get("research_plan_source") or _as_dict(trace.get("research_plan")).get("source") or ""),
        "research_plan_fallback_reason": str(
            trace.get("research_plan_fallback_reason")
            or _as_dict(trace.get("research_plan")).get("fallback_reason")
            or ""
        ),
        "research_plan_duration_ms": int(trace.get("research_plan_duration_ms") or _as_dict(trace.get("research_plan")).get("duration_ms") or 0),
        "evidence_health": str(trace.get("evidence_health") or _as_dict(trace.get("analytical_reasoning")).get("evidence_health") or ""),
        "evidence_counts": evidence_counts,
        "missing_requirements": _extract_missing_requirements(trace),
        "missing_required_answer_parts": _as_list(trace.get("missing_required_answer_parts")),
        "partial_required_answer_parts": _as_list(trace.get("partial_required_answer_parts")),
        "missing_but_analyzable_answer_parts": _as_list(trace.get("missing_but_analyzable_answer_parts")),
        "missing_and_unanswerable_answer_parts": _as_list(trace.get("missing_and_unanswerable_answer_parts")),
        "answer_part_status_by_id": _answer_part_status(trace),
        "contract_warnings": warnings,
        "contract_violations": violations,
        "progress_events_summary": _progress_summary(trace),
        "bundle_path": str(bundle_path) if bundle_path else "",
        "elapsed_s": round(elapsed_s, 2),
        "error": error,
    }


def _write_bundle(path: Path, trace_ui: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_pretty(trace_ui) + "\n", encoding="utf-8")


def _run_case(
    client: httpx.Client,
    api_base: str,
    case: Mapping[str, str],
    *,
    run_id: str,
    bundle_dir: Path,
    trace_prefix: str,
) -> dict[str, Any]:
    case_id = case["case_id"]
    trace_id = _trace_id_for_case(run_id, case_id, trace_prefix)
    t0 = time.time()
    chat_response: dict[str, Any] = {}
    trace_ui: dict[str, Any] | None = None
    bundle_path: Path | None = None
    error = ""

    try:
        chat_response = _request_json(
            client,
            "POST",
            f"{api_base.rstrip('/')}/chat",
            json={"query": case["query"], "client_trace_id": trace_id},
        )
    except Exception as exc:
        error = f"POST /chat failed: {exc}"

    used_trace_id = str(chat_response.get("trace_id") or trace_id)
    try:
        trace_ui = _request_json(client, "GET", f"{api_base.rstrip('/')}/trace/{used_trace_id}/ui")
        bundle_path = bundle_dir / f"{case_id}_{used_trace_id}.json"
        _write_bundle(bundle_path, trace_ui)
    except Exception as exc:
        trace_error = f"GET /trace/{used_trace_id}/ui failed: {exc}"
        error = f"{error}; {trace_error}" if error else trace_error

    return _case_from_responses(
        case,
        trace_id,
        chat_response,
        trace_ui,
        bundle_path,
        time.time() - t0,
        error=error,
    )


def _is_blocked(case: Mapping[str, Any]) -> bool:
    return str(case.get("contract_status") or "") == "blocked" or str(case.get("final_route") or "") == "blocked"


def _is_warning(case: Mapping[str, Any]) -> bool:
    return (
        str(case.get("contract_status") or "") == "passed_with_warnings"
        or bool(_as_list(case.get("contract_warnings")))
        or bool(_as_list(case.get("contract_violations")))
    )


def _is_scope_limited(case: Mapping[str, Any]) -> bool:
    return (
        str(case.get("answer_quality_tier") or "") == "scope_limit"
        or str(case.get("contract_status") or "") == "scope_limited"
        or str(case.get("final_route") or "") == "scope_limited"
    )


def _is_invalid_fallback(case: Mapping[str, Any]) -> bool:
    return str(case.get("answer_quality_tier") or "") == "invalid_fallback" or case.get("fallback_intent_match") is False


def _is_safe_refusal(case: Mapping[str, Any]) -> bool:
    return str(case.get("answer_quality_tier") or "") == "safe_refusal" or str(case.get("final_route") or "") == "safe_refusal"


def _write_user_report(
    cases: list[dict[str, Any]],
    path: Path,
    generated_at: str,
    api_base: str,
    run_id: str,
    *,
    is_full_probe: bool,
    full_case_count: int,
    title: str,
) -> None:
    lines = [
        f"# {title} - User Review",
        "",
        f"- run_id: `{run_id}`",
        f"- generated_at: {generated_at}",
        f"- api_base: `{api_base}`",
        f"- total: {len(cases)}",
        f"- selected_case_ids: `{', '.join(str(item.get('case_id', '')) for item in cases)}`",
        f"- selected_case_count: {len(cases)}",
        f"- full_case_count: {full_case_count}",
        f"- is_full_probe: `{is_full_probe}`",
        "",
    ]
    if not is_full_probe:
        lines.extend(["**Note**: This is a targeted run, not the full manual_api_probe_20.", ""])
    for item in cases:
        lines.extend(
            [
                f"## {item['case_id']}",
                "",
                f"**category**: `{_markdown_inline(item.get('category'))}`",
                "",
                f"**language**: `{_markdown_inline(item.get('language'))}`",
                "",
                f"**问题**: {_markdown_text(item.get('query'))}",
                "",
                f"**trace_id**: `{_markdown_inline(item.get('trace_id'))}`",
                "",
                f"**contract_status**: `{_markdown_inline(item.get('contract_status'))}`",
                "",
                f"**quality_tier**: `{_markdown_inline(item.get('answer_quality_tier'))}`",
                "",
                f"**main_question_covered**: `{_markdown_inline(item.get('main_question_covered'))}`",
                "",
                "**用户可见最终答案**",
                "",
                _markdown_text(item.get("answer")),
                "",
            ]
        )
        if item.get("error"):
            lines.extend(["**Error**", "", _markdown_text(item.get("error")), ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _format_list_block(value: Any) -> str:
    items = _as_list(value)
    if not items:
        return "- none"
    return "\n".join(f"- `{_markdown_inline(item)}`" if not isinstance(item, Mapping) else f"- `{_markdown_inline(item.get('requirement_id') or item.get('id') or item)}`: {_markdown_inline(item.get('status') or item.get('missing_reason') or item)}" for item in items)


def _write_debug_report(
    cases: list[dict[str, Any]],
    path: Path,
    generated_at: str,
    api_base: str,
    run_id: str,
    *,
    is_full_probe: bool,
    full_case_count: int,
    title: str,
) -> None:
    lines = [
        f"# {title} - Debug Review",
        "",
        f"- run_id: `{run_id}`",
        f"- generated_at: {generated_at}",
        f"- api_base: `{api_base}`",
        f"- total: {len(cases)}",
        f"- selected_case_ids: `{', '.join(str(item.get('case_id', '')) for item in cases)}`",
        f"- selected_case_count: {len(cases)}",
        f"- full_case_count: {full_case_count}",
        f"- is_full_probe: `{is_full_probe}`",
        "",
    ]
    if not is_full_probe:
        lines.extend(["**Note**: This is a targeted run, not the full manual_api_probe_20.", ""])
    lines.extend(
        [
            "| Case | Category | Lang | Contract | Route | Tier | Source | Synth | Blocked | Answer chars | Bundle | Error |",
            "|---|---|---|---|---|---|---|---|---|---:|---|---|",
        ]
    )
    for item in cases:
        lines.append(
            "| {case} | `{category}` | `{language}` | `{contract}` | `{route}` | `{tier}` | `{source}` | `{synth}` | `{blocked}` | {chars} | `{bundle}` | {error} |".format(
                case=item["case_id"],
                category=_markdown_inline(item.get("category")),
                language=_markdown_inline(item.get("language")),
                contract=_markdown_inline(item.get("contract_status")),
                route=_markdown_inline(item.get("final_route")),
                tier=_markdown_inline(item.get("answer_quality_tier")),
                source=_markdown_inline(item.get("final_answer_source")),
                synth=_markdown_inline(item.get("synthesis_mode")),
                blocked=_markdown_inline(item.get("blocked")),
                chars=item.get("answer_chars", 0),
                bundle=_markdown_inline(item.get("bundle_path")),
                error=_markdown_inline(_text(item.get("error"), limit=120)),
            )
        )
    lines.append("")

    for item in cases:
        counts = _as_dict(item.get("evidence_counts"))
        lines.extend(
            [
                f"## {item['case_id']}",
                "",
                f"- category / language: `{_markdown_inline(item.get('category'))}` / `{_markdown_inline(item.get('language'))}`",
                f"- query: {_markdown_text(item.get('query'))}",
                f"- trace_id: `{_markdown_inline(item.get('trace_id'))}`",
                f"- task_type / answer_mode: `{_markdown_inline(item.get('task_type'))}` / `{_markdown_inline(item.get('answer_mode'))}`",
                f"- evidence policy: `{_markdown_inline(item.get('evidence_policy_id'))}`",
                f"- research plan: source=`{_markdown_inline(item.get('research_plan_source'))}`, fallback=`{_markdown_inline(item.get('research_plan_fallback_reason'))}`, duration_ms={item.get('research_plan_duration_ms', 0)}",
                f"- answer source / history: source=`{_markdown_inline(item.get('final_answer_source'))}`, history=`{_markdown_inline('; '.join(_as_list(item.get('answer_history_summary'))))}`",
                f"- contract: status=`{_markdown_inline(item.get('contract_status'))}`, decision=`{_markdown_inline(_as_dict(item.get('contract_decision')).get('decision'))}`, route=`{_markdown_inline(item.get('contract_route'))}`, relevance_route=`{_markdown_inline(item.get('relevance_route'))}`, final_route=`{_markdown_inline(item.get('final_route'))}`, blocked=`{_markdown_inline(item.get('blocked'))}`",
                f"- quality: tier=`{_markdown_inline(item.get('answer_quality_tier'))}`, reason=`{_markdown_inline(item.get('quality_tier_reason'))}`, covered=`{_markdown_inline(item.get('main_question_covered'))}`, intent_match=`{_markdown_inline(item.get('fallback_intent_match'))}`, format_ok=`{_markdown_inline(item.get('format_constraints_satisfied'))}`, answered_dims=`{_markdown_inline(', '.join(_as_list(item.get('answered_dimensions'))))}`, unresolved_relevance=`{_markdown_inline(', '.join(_as_list(item.get('unresolved_relevance_failures'))))}`",
                f"- repair/risk/scope: source_before_repair=`{_markdown_inline(item.get('source_before_repair'))}`, repair_applied=`{_markdown_inline(item.get('repair_applied'))}`, repair_types=`{_markdown_inline(', '.join(_as_list(item.get('repair_types'))))}`, uncited_material={item.get('material_claim_uncited_count', 0)}, core_missing=`{_markdown_inline(', '.join(_as_list(item.get('core_missing_parts'))))}`, optional_missing=`{_markdown_inline(', '.join(_as_list(item.get('optional_missing_parts'))))}`, direct_risks={item.get('risk_items_directly_supported_count', 0)}, template_only_risks={item.get('risk_items_template_only_count', 0)}, token_leakage={item.get('company_specific_token_leakage', 0)}, output_language=`{_markdown_inline(item.get('output_language'))}`, language_leakage={item.get('language_leakage', 0)}, segment_scope=`{_markdown_inline(item.get('segment_or_product_scope'))}`",
                f"- synthesis_mode: `{_markdown_inline(item.get('synthesis_mode'))}`",
                f"- evidence health: `{_markdown_inline(item.get('evidence_health'))}`",
                f"- evidence counts: numeric={counts.get('numeric_count', 0)} returned/validated={counts.get('numeric_returned_count', 0)}/{counts.get('numeric_validated_count', 0)}, text={counts.get('text_count', 0)} returned/validated={counts.get('text_returned_count', 0)}/{counts.get('text_validated_count', 0)}, computed={counts.get('computed_count', 0)} returned/validated={counts.get('computed_returned_count', 0)}/{counts.get('computed_validated_count', 0)}, event={counts.get('event_count', 0)}",
                f"- raw trace UI JSON: `{_markdown_inline(item.get('bundle_path'))}`",
                "",
                "### Canonical Intent",
                "",
                "```json",
                _json_pretty(item.get("canonical_intent")),
                "```",
                "",
                "### Evidence Policy",
                "",
                "```json",
                _json_pretty(item.get("evidence_policy")),
                "```",
                "",
                "### Missing Requirements",
                "",
                _format_list_block(item.get("missing_requirements")),
                "",
                "### Missing / Partial Answer Parts",
                "",
                f"- missing_required: {_markdown_inline(item.get('missing_required_answer_parts'))}",
                f"- partial_required: {_markdown_inline(item.get('partial_required_answer_parts'))}",
                f"- missing_but_analyzable: {_markdown_inline(item.get('missing_but_analyzable_answer_parts'))}",
                f"- missing_and_unanswerable: {_markdown_inline(item.get('missing_and_unanswerable_answer_parts'))}",
                "",
                "### Answer Part Status",
                "",
                "```json",
                _json_pretty(item.get("answer_part_status_by_id")),
                "```",
                "",
                "### Warnings / Violations",
                "",
                "**warnings**",
                "",
                "```json",
                _json_pretty(item.get("contract_warnings")),
                "```",
                "",
                "**violations**",
                "",
                "```json",
                _json_pretty(item.get("contract_violations")),
                "```",
                "",
                "### Progress Events",
                "",
                "| Time | Event | Node | Status | Message |",
                "|---|---|---|---|---|",
            ]
        )
        progress = _as_list(item.get("progress_events_summary"))
        if progress:
            for event in progress:
                if not isinstance(event, Mapping):
                    continue
                lines.append(
                    "| {time} | {event} | {node} | {status} | {message} |".format(
                        time=_markdown_inline(event.get("time")),
                        event=_markdown_inline(event.get("event")),
                        node=_markdown_inline(event.get("node")),
                        status=_markdown_inline(event.get("status")),
                        message=_markdown_inline(event.get("message")),
                    )
                )
        else:
            lines.append("| - | - | - | - | none |")
        if item.get("error"):
            lines.extend(["", "### Error", "", _markdown_text(item.get("error"))])
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_summary(
    cases: list[dict[str, Any]],
    path: Path,
    generated_at: str,
    api_base: str,
    elapsed_s: float,
    run_id: str,
    *,
    is_full_probe: bool,
    full_case_count: int,
    suite_name: str,
    trace_prefix: str,
) -> dict[str, Any]:
    summary_cases = []
    for item in cases:
        summary_cases.append(
            {
                "case_id": item.get("case_id", ""),
                "category": item.get("category", ""),
                "language": item.get("language", ""),
                "company": item.get("company", ""),
                "ticker": item.get("ticker", ""),
                "source_tier": item.get("source_tier", ""),
                "query": item.get("query", ""),
                "trace_id": item.get("trace_id", ""),
                "contract_status": item.get("contract_status", ""),
                "contract_route": item.get("contract_route", ""),
                "relevance_route": item.get("relevance_route", ""),
                "final_route": item.get("final_route", ""),
                "final_answer_source": item.get("final_answer_source", ""),
                "answer_history_summary": item.get("answer_history_summary", []),
                "answer_quality_tier": item.get("answer_quality_tier", ""),
                "quality_tier_reason": item.get("quality_tier_reason", ""),
                "main_question_covered": item.get("main_question_covered", True),
                "fallback_intent_match": item.get("fallback_intent_match", True),
                "answered_dimensions": item.get("answered_dimensions", []),
                "unresolved_relevance_failures": item.get("unresolved_relevance_failures", []),
                "format_constraints_satisfied": item.get("format_constraints_satisfied", True),
                "source_before_repair": item.get("source_before_repair", ""),
                "repair_applied": item.get("repair_applied", False),
                "repair_types": item.get("repair_types", []),
                "material_claim_uncited_count": item.get("material_claim_uncited_count", 0),
                "core_missing_parts": item.get("core_missing_parts", []),
                "optional_missing_parts": item.get("optional_missing_parts", []),
                "risk_items_directly_supported_count": item.get("risk_items_directly_supported_count", 0),
                "risk_items_template_only_count": item.get("risk_items_template_only_count", 0),
                "company_specific_token_leakage": item.get("company_specific_token_leakage", 0),
                "output_language": item.get("output_language", ""),
                "language_leakage": item.get("language_leakage", 0),
                "language_leakage_unresolved": item.get("language_leakage_unresolved", False),
                "segment_or_product_scope": item.get("segment_or_product_scope", ""),
                "synthesis_mode": item.get("synthesis_mode", ""),
                "blocked": item.get("blocked", False),
                "evidence_counts": item.get("evidence_counts", {}),
                "answer_chars": item.get("answer_chars", 0),
                "answer_length": item.get("answer_chars", 0),
                "bundle_path": item.get("bundle_path", ""),
                "debug_bundle_path": item.get("bundle_path", ""),
                "error": item.get("error", ""),
                "task_type": item.get("task_type", ""),
                "answer_mode": item.get("answer_mode", ""),
            }
        )
    blocked_case_ids = [str(item.get("case_id", "")) for item in cases if _is_blocked(item)]
    scope_limit_case_ids = [str(item.get("case_id", "")) for item in cases if _is_scope_limited(item)]
    invalid_fallback_case_ids = [str(item.get("case_id", "")) for item in cases if _is_invalid_fallback(item)]
    safe_refusal_case_ids = [str(item.get("case_id", "")) for item in cases if _is_safe_refusal(item)]
    error_case_ids = [str(item.get("case_id", "")) for item in cases if item.get("error")]
    summary = {
        "run_id": run_id,
        "suite": suite_name,
        "trace_prefix": trace_prefix,
        "total": len(cases),
        "selected_case_ids": [str(item.get("case_id", "")) for item in cases],
        "selected_case_count": len(cases),
        "full_case_count": full_case_count,
        "is_full_probe": bool(is_full_probe),
        "completed": len([item for item in cases if not item.get("error")]),
        "api_errors": len([item for item in cases if item.get("error")]),
        "blocked_count": len(blocked_case_ids),
        "language_leakage": sum(int(item.get("language_leakage", 0) or 0) for item in cases),
        "language_leakage_unresolved_count": len([item for item in cases if item.get("language_leakage_unresolved")]),
        "passed_count": len([item for item in cases if str(item.get("contract_status") or "") in {"passed", "repaired"}]),
        "warning_count": len([item for item in cases if _is_warning(item)]),
        "blocked_case_ids": blocked_case_ids,
        "scope_limit_case_ids": scope_limit_case_ids,
        "invalid_fallback_case_ids": invalid_fallback_case_ids,
        "safe_refusal_case_ids": safe_refusal_case_ids,
        "error_case_ids": error_case_ids,
        "generated_at": generated_at,
        "api_base": api_base,
        "elapsed_s": round(elapsed_s, 2),
        "cases": summary_cases,
    }
    path.write_text(_json_pretty(summary) + "\n", encoding="utf-8")
    return summary


def _module_by_case_id() -> dict[str, dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for index, module in enumerate(DEMO_SELECTED_MODULES, start=1):
        for case_id in _as_list(module.get("caseIds")):
            data = dict(module)
            data["displayOrder"] = index
            by_case[str(case_id)] = data
    return by_case


def _sentence_like_parts(text: str, *, language: str) -> list[str]:
    clean = " ".join(str(text or "").strip().split())
    if not clean:
        return []
    if language == "zh":
        parts: list[str] = []
        current = ""
        for char in clean:
            current += char
            if char in "。！？":
                parts.append(current.strip())
                current = ""
        if current.strip():
            parts.append(current.strip())
        return parts
    pieces: list[str] = []
    current = ""
    for char in clean:
        current += char
        if char in ".!?":
            pieces.append(current.strip())
            current = ""
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _trim_display_answer(answer: str, *, language: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return ""
    paragraphs = [item.strip() for item in text.splitlines() if item.strip()]
    text = "\n\n".join(paragraphs)
    if language == "zh":
        if len(text) <= 350:
            return text
        out = ""
        for part in _sentence_like_parts(text, language=language):
            if len(out) + len(part) > 350:
                break
            out += part
            if len(out) >= 180:
                break
        return out.strip() or text[:350].rstrip()
    words = text.split()
    if len(words) <= 220:
        return text
    return " ".join(words[:220]).rstrip()


def _short_title(query: str, company: str, ticker: str, *, language: str) -> str:
    text = str(query or "").strip().rstrip("?.。？")
    if language == "zh":
        return text[:28]
    for prefix in ("Why is ", "How should I understand ", "How should we evaluate ", "How should we assess ", "What are ", "What kind of "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if len(text) > 54:
        text = text[:51].rstrip() + "..."
    return text or f"{company or ticker} analysis"


def _subtitle_for_case(item: Mapping[str, Any], module: Mapping[str, Any]) -> str:
    language = str(item.get("language") or "")
    ticker = str(item.get("ticker") or "")
    module_id = str(module.get("id") or "")
    if language == "zh":
        labels = {
            "business-drivers-revenue-quality": f"展示 {ticker} 增长驱动和收入质量解释。",
            "margins-business-model": f"展示 {ticker} 利润率与商业模式分析。",
            "cash-flow-capex": f"展示 {ticker} 现金流、FCF 与资本开支解释。",
            "balance-sheet-valuation-risk": f"展示 {ticker} 资产负债表或估值风险边界。",
            "risk-ranking-disclosure": f"展示 {ticker} 风险披露排序和传导路径。",
            "scenario-comparison": "展示情景分析或公司对比能力。",
        }
    else:
        labels = {
            "business-drivers-revenue-quality": f"Shows how FilingLens explains {ticker} growth drivers from filings.",
            "margins-business-model": f"Shows margin quality and business-model interpretation for {ticker}.",
            "cash-flow-capex": f"Shows cash-flow, FCF, and capex reasoning for {ticker}.",
            "balance-sheet-valuation-risk": f"Shows balance-sheet or valuation-risk boundaries for {ticker}.",
            "risk-ranking-disclosure": f"Shows disclosure-based risk ranking for {ticker}.",
            "scenario-comparison": "Shows scenario analysis or company comparison with evidence boundaries.",
        }
    return labels.get(module_id, str(module.get("description") or ""))


def _collect_citation_ids(value: Any, *, limit: int = 20) -> list[str]:
    out: list[str] = []

    def add(item: Any) -> None:
        text = str(item or "").strip()
        if text and text not in out and len(out) < limit:
            out.append(text)

    def walk(node: Any) -> None:
        if len(out) >= limit:
            return
        if isinstance(node, Mapping):
            for key in ("citation_id", "citationId", "source_id", "sourceId"):
                if key in node:
                    add(node.get(key))
            for key in ("citation_ids", "citationIds", "source_ids", "sourceIds", "citations"):
                if key in node:
                    raw = node.get(key)
                    if isinstance(raw, list):
                        for item in raw:
                            if isinstance(item, Mapping):
                                walk(item)
                            else:
                                add(item)
                    else:
                        add(raw)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return out


def _walk_mappings(value: Any) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            out.append(node)
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return out


def _extract_numeric_evidence(trace_ui: Mapping[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _walk_mappings(trace_ui.get("evidence_packet")):
        value = item.get("value", item.get("metric_value", item.get("numeric_value", item.get("amount"))))
        label = item.get("label", item.get("metric", item.get("metric_name", item.get("name"))))
        if value is None or label is None:
            continue
        key = f"{label}:{value}"
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "label": str(label),
                "value": str(value),
                "unit": str(item.get("unit") or item.get("currency") or item.get("scale") or ""),
                "citationIds": _collect_citation_ids(item, limit=6),
            }
        )
        if len(evidence) >= limit:
            break
    return evidence


def _extract_text_evidence(trace_ui: Mapping[str, Any], *, limit: int = 6) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in _walk_mappings(trace_ui.get("evidence_packet")):
        text = item.get("summary", item.get("snippet", item.get("text", item.get("quote"))))
        if not text:
            continue
        summary = _text(text, limit=220)
        if not summary or summary in seen:
            continue
        seen.add(summary)
        evidence.append({"summary": summary, "citationIds": _collect_citation_ids(item, limit=6)})
        if len(evidence) >= limit:
            break
    return evidence


def _evidence_boundary(item: Mapping[str, Any], trace_ui: Mapping[str, Any]) -> str:
    answer = str(item.get("answer") or "")
    language = str(item.get("language") or "")
    parts = [
        part
        for part in _sentence_like_parts(answer, language=language)
        if any(token in part.lower() for token in ("bounded", "limited", "evidence", "scope", "仍需", "边界", "证据", "有限"))
    ]
    if parts:
        return " ".join(parts[:3])
    counts = _as_dict(item.get("evidence_counts"))
    return (
        f"Evidence boundary: numeric={counts.get('numeric_count', 0)}, "
        f"text={counts.get('text_count', 0)}, computed={counts.get('computed_count', 0)}; "
        "use the raw trace bundle for exact citations."
    )


def _demo_exclusion_reasons(item: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    status = str(item.get("contract_status") or "")
    tier = str(item.get("answer_quality_tier") or "")
    final_route = str(item.get("final_route") or "")
    if item.get("error"):
        reasons.append("api_error")
    if _is_blocked(item):
        reasons.append("blocked")
    if _is_scope_limited(item):
        reasons.append("scope_limit")
    if _is_safe_refusal(item):
        reasons.append("safe_refusal")
    if _is_invalid_fallback(item):
        reasons.append("invalid_fallback")
    if int(item.get("company_specific_token_leakage", 0) or 0) > 0:
        reasons.append("company_or_ticker_leakage")
    if int(item.get("language_leakage", 0) or 0) > 0 or item.get("language_leakage_unresolved"):
        reasons.append("language_leakage")
    if not str(item.get("answer") or "").strip():
        reasons.append("missing_answer")
    if item.get("main_question_covered") is False:
        reasons.append("main_question_not_covered")
    if status == "not_run" or final_route == "not_run" or tier == "not_run":
        reasons.append("not_run")
    return reasons


def _score_for_demo_case(item: Mapping[str, Any], reasons: list[str]) -> int:
    if reasons:
        return 0
    score = 90 if str(item.get("source_tier") or "") == "strong" else 75
    if _is_warning(item):
        score -= 10
    if str(item.get("contract_status") or "") not in {"passed", "repaired"}:
        score -= 10
    return max(score, 1)


def _write_frontend_demo_files(
    cases: list[dict[str, Any]],
    *,
    run_id: str,
    generated_at: str,
    run_dir: Path,
) -> dict[str, Any]:
    module_by_case = _module_by_case_id()
    case_by_id = {case["case_id"]: case for case in cases}
    ordered_case_ids = [case["case_id"] for case in DEMO_SELECTED_CASES]
    frontend_cases: list[dict[str, Any]] = []
    for case_id in ordered_case_ids:
        item = case_by_id[case_id]
        module = module_by_case.get(case_id, {})
        bundle_path = Path(str(item.get("bundle_path") or ""))
        trace_ui: dict[str, Any] = {}
        if bundle_path.exists():
            trace_ui = json.loads(bundle_path.read_text(encoding="utf-8"))
        reasons = _demo_exclusion_reasons(item)
        demo_ready = not reasons
        source_tier = str(item.get("source_tier") or ("strong" if case_id in DEMO_SELECTED_STRONG_IDS else "maybe"))
        raw_answer = str(item.get("answer") or "")
        language = str(item.get("language") or "")
        numeric_evidence = _extract_numeric_evidence(trace_ui)
        text_evidence = _extract_text_evidence(trace_ui)
        citation_ids = _collect_citation_ids(trace_ui)
        frontend_cases.append(
            {
                "caseId": case_id,
                "sourceTier": source_tier,
                "demoReady": demo_ready,
                "language": language,
                "category": str(item.get("category") or ""),
                "company": str(item.get("company") or ""),
                "ticker": str(item.get("ticker") or ""),
                "query": str(item.get("query") or ""),
                "traceId": str(item.get("trace_id") or ""),
                "contractStatus": str(item.get("contract_status") or ""),
                "qualityTier": str(item.get("answer_quality_tier") or ""),
                "score": _score_for_demo_case(item, reasons),
                "moduleId": str(module.get("id") or ""),
                "moduleTitle": str(module.get("title") or ""),
                "moduleDescription": str(module.get("description") or ""),
                "displayOrder": int(module.get("displayOrder") or 0),
                "isHeroCase": case_id in {"D077", "D137", "D251"},
                "title": _short_title(str(item.get("query") or ""), str(item.get("company") or ""), str(item.get("ticker") or ""), language=language),
                "subtitle": _subtitle_for_case(item, module),
                "keyMetrics": [f"{entry['label']}: {entry['value']}{(' ' + entry['unit']) if entry.get('unit') else ''}" for entry in numeric_evidence[:4]],
                "highlights": [entry["summary"] for entry in text_evidence[:3]],
                "tags": [str(item.get("ticker") or ""), str(module.get("title") or ""), source_tier],
                "displayAnswer": _trim_display_answer(raw_answer, language=language),
                "rawAnswer": raw_answer,
                "evidenceBoundary": _evidence_boundary(item, trace_ui),
                "trimNote": "Deterministically trimmed for frontend display; rawAnswer preserves the exact rerun answer.",
                "numericEvidence": numeric_evidence,
                "textEvidence": text_evidence,
                "citationIds": citation_ids,
                "sourceReportPath": str(run_dir / "user_report.md"),
                "generatedAt": generated_at,
                "exclusionReason": ", ".join(reasons),
                "reviewNotes": "Needs human review before default display." if reasons or _is_warning(item) else "",
            }
        )

    data = {
        "version": generated_at[:10],
        "project": "FilingLens",
        "purpose": "Static preset demo questions and answers for frontend display without live model/API calls.",
        "sourceRun": f"manual_api_probe_demo_selected/{run_id}",
        "heroCaseIds": ["D077", "D137", "D251"],
        "modules": DEMO_SELECTED_MODULES,
        "cases": frontend_cases,
    }
    frontend_dir = ROOT / "frontend" / "data"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    json_path = frontend_dir / "filinglens-demo-cases.json"
    js_path = frontend_dir / "filinglens-demo-cases.js"
    readme_path = frontend_dir / "filinglens-demo-cases.README.md"
    json_text = _json_pretty(data) + "\n"
    json_path.write_text(json_text, encoding="utf-8")
    js_path.write_text("window.FILINGLENS_DEMO_CASES = " + json_text, encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# FilingLens Demo Cases",
                "",
                "Static preset Q&A data for the FilingLens frontend demo. The page can render these cases without calling the live model/API.",
                "",
                "## Files",
                "",
                "- `filinglens-demo-cases.json`: canonical JSON data.",
                "- `filinglens-demo-cases.js`: browser global wrapper, `window.FILINGLENS_DEMO_CASES`.",
                "",
                "## Case Fields",
                "",
                "- `rawAnswer`: exact final user-visible answer from the rerun.",
                "- `displayAnswer`: trimmed frontend display copy; it must not change conclusions, numbers, or evidence boundaries.",
                "- `demoReady`: false means keep the case in data but exclude it from default demo display.",
                "- `numericEvidence`, `textEvidence`, and `citationIds`: evidence extracted from the saved trace UI bundle.",
                "",
                "## Frontend Loading",
                "",
                "Load the JSON directly in build tooling, or include `filinglens-demo-cases.js` and read `window.FILINGLENS_DEMO_CASES`.",
                "",
                "## Excluded Cases",
                "",
                "Cases with `demoReady=false` retain `rawAnswer` for audit/review, but should not appear in the default preset carousel/list.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "json": str(json_path),
        "js": str(js_path),
        "readme": str(readme_path),
        "demoReadyCount": len([case for case in frontend_cases if case["demoReady"]]),
        "excludedCount": len([case for case in frontend_cases if not case["demoReady"]]),
        "strongReadyCount": len([case for case in frontend_cases if case["sourceTier"] == "strong" and case["demoReady"]]),
        "maybeReadyCount": len([case for case in frontend_cases if case["sourceTier"] == "maybe" and case["demoReady"]]),
        "casesNeedingReview": [case["caseId"] for case in frontend_cases if case["reviewNotes"]],
        "casesExcluded": [case["caseId"] for case in frontend_cases if not case["demoReady"]],
    }


def run(
    *,
    cases: list[dict[str, str]] | None = None,
    is_full_probe: bool | None = None,
    suite_name: str = DEFAULT_SUITE,
) -> dict[str, Any]:
    if suite_name not in SUITES:
        valid = ", ".join(sorted(SUITES))
        raise ValueError(f"Unknown suite {suite_name!r}. Valid suites: {valid}")
    suite = SUITES[suite_name]
    all_cases = list(suite["cases"])
    out_dir = Path(suite["out_dir"])
    trace_prefix = str(suite["trace_prefix"])
    title = str(suite["title"])
    api_base = os.environ.get("API_BASE") or DEFAULT_API_BASE
    selected_cases = list(cases or all_cases)
    if is_full_probe is None:
        is_full_probe = len(selected_cases) == len(all_cases) and [case["case_id"] for case in selected_cases] == [case["case_id"] for case in all_cases]
    generated_at = _utc_now()
    run_id = _run_id()
    started = time.time()
    run_dir = out_dir / run_id
    bundle_dir = run_dir / "bundles"
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    timeout = httpx.Timeout(TIMEOUT_S)
    with httpx.Client(timeout=timeout) as client:
        for index, case in enumerate(selected_cases, start=1):
            result = _run_case(client, api_base, case, run_id=run_id, bundle_dir=bundle_dir, trace_prefix=trace_prefix)
            results.append(result)
            error = _text(result.get("error"), limit=160)
            status = result.get("contract_status", "")
            print(
                f"[{index:02d}/{len(selected_cases):02d}] {result['case_id']} "
                f"contract={status or '-'} trace={result.get('trace_id') or '-'} "
                f"answer_chars={result.get('answer_chars', 0)} "
                f"error={error or '-'}",
                flush=True,
            )

    _write_user_report(
        results,
        run_dir / "user_report.md",
        generated_at,
        api_base,
        run_id,
        is_full_probe=bool(is_full_probe),
        full_case_count=len(all_cases),
        title=title,
    )
    _write_debug_report(
        results,
        run_dir / "debug_report.md",
        generated_at,
        api_base,
        run_id,
        is_full_probe=bool(is_full_probe),
        full_case_count=len(all_cases),
        title=title,
    )
    summary = _write_summary(
        results,
        run_dir / "summary.json",
        generated_at,
        api_base,
        time.time() - started,
        run_id,
        is_full_probe=bool(is_full_probe),
        full_case_count=len(all_cases),
        suite_name=suite_name,
        trace_prefix=trace_prefix,
    )

    _write_user_report(
        results,
        out_dir / "user_report.md",
        generated_at,
        api_base,
        run_id,
        is_full_probe=bool(is_full_probe),
        full_case_count=len(all_cases),
        title=title,
    )
    _write_debug_report(
        results,
        out_dir / "debug_report.md",
        generated_at,
        api_base,
        run_id,
        is_full_probe=bool(is_full_probe),
        full_case_count=len(all_cases),
        title=title,
    )
    _write_summary(
        results,
        out_dir / "summary.json",
        generated_at,
        api_base,
        time.time() - started,
        run_id,
        is_full_probe=bool(is_full_probe),
        full_case_count=len(all_cases),
        suite_name=suite_name,
        trace_prefix=trace_prefix,
    )

    if suite_name == "manual_api_probe_demo_selected":
        frontend_info = _write_frontend_demo_files(results, run_id=run_id, generated_at=generated_at, run_dir=run_dir)
        summary["frontend_demo"] = frontend_info
        (run_dir / "summary.json").write_text(_json_pretty(summary) + "\n", encoding="utf-8")
        (out_dir / "summary.json").write_text(_json_pretty(summary) + "\n", encoding="utf-8")

    print(_json_pretty(summary), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the manual API probe through POST /chat.")
    parser.add_argument(
        "--suite",
        default=os.environ.get("MANUAL_PROBE_SUITE", DEFAULT_SUITE),
        choices=sorted(SUITES),
        help="Probe suite to run. Defaults to the original manual_api_probe_20 suite.",
    )
    parser.add_argument(
        "--cases",
        default=os.environ.get("MANUAL_PROBE_CASES", ""),
        help="Optional case ids to run, e.g. 'Q09,Q11,Q16' or 'Q09 Q11 Q16'.",
    )
    args = parser.parse_args()
    cases_arg = str(args.cases or "").strip()
    all_cases = list(SUITES[str(args.suite)]["cases"])
    run(cases=_select_cases(cases_arg, all_cases), is_full_probe=not bool(cases_arg), suite_name=str(args.suite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
