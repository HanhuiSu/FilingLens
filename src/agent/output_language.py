"""Output-language contract and public language leakage helpers."""

from __future__ import annotations

import re
from typing import Any, Mapping


OutputLanguage = str


_ZH_INSTRUCTION_PATTERNS = (
    r"用中文",
    r"中文回答",
    r"中文讲",
    r"请.*中文",
    r"in\s+chinese",
    r"answer\s+in\s+chinese",
)

_EN_INSTRUCTION_PATTERNS = (
    r"in\s+english",
    r"answer\s+in\s+english",
    r"english\s+only",
    r"use\s+english",
    r"用英文",
    r"英文回答",
)


def detect_output_language(user_query: str) -> OutputLanguage:
    """Return the intended public answer language for a user query."""
    text = str(user_query or "").strip()
    lowered = text.lower()
    matches: list[tuple[int, str]] = []
    for pattern in _ZH_INSTRUCTION_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            matches.append((match.start(), "zh"))
    for pattern in _EN_INSTRUCTION_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            matches.append((match.start(), "en"))
    if matches:
        return sorted(matches, key=lambda item: item[0])[-1][1]
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.fullmatch(
        r"(?i)\s*(?:[A-Z]{1,5}|Amazon|Apple|Microsoft|Nvidia|NVIDIA|Google|Meta|Tesla)\s+overview\s*",
        text,
    ):
        return "zh"
    return "en"


RISK_THEME_DISPLAY: dict[str, dict[str, str]] = {
    "product_demand_uncertainty": {
        "zh": "新产品和需求不确定性",
        "en": "new-product and demand uncertainty",
    },
    "fulfillment_inventory_capex_pressure": {
        "zh": "履约/库存/资本开支压力",
        "en": "fulfillment, inventory, and capex pressure",
    },
    "regulatory_compliance_pressure": {
        "zh": "监管/合规风险",
        "en": "regulatory and compliance risk",
    },
    "cloud_competition_pressure": {
        "zh": "AWS/云竞争",
        "en": "AWS/cloud competition",
    },
    "competition_pressure": {
        "zh": "竞争加剧",
        "en": "intensifying competition",
    },
    "supply_inventory_pressure": {
        "zh": "供应链或库存压力",
        "en": "supply-chain or inventory pressure",
    },
    "customer_market_concentration": {
        "zh": "客户/市场集中风险",
        "en": "customer or market concentration risk",
    },
    "margin_pressure": {
        "zh": "利润率压力",
        "en": "margin pressure",
    },
    "macro_regulatory_legal": {
        "zh": "宏观、监管或法律风险",
        "en": "macro, regulatory, or legal risk",
    },
    "customer_it_cloud_spend_slowdown": {
        "zh": "客户 IT/云支出放缓",
        "en": "customer IT/cloud-spend slowdown",
    },
    "validated_risk_disclosure": {
        "zh": "已验证风险披露",
        "en": "validated risk disclosure",
    },
    "validated_risk_context": {
        "zh": "已验证风险披露",
        "en": "validated risk disclosure",
    },
    "inventory": {"zh": "库存风险", "en": "inventory risk"},
    "regulatory_cross_border": {
        "zh": "监管 / 跨境合规风险",
        "en": "regulatory and cross-border compliance risk",
    },
    "competition": {"zh": "竞争风险", "en": "competition risk"},
    "supply_chain": {"zh": "供应链风险", "en": "supply-chain risk"},
    "fulfillment_staffing": {
        "zh": "履约与人员配置风险",
        "en": "fulfillment and staffing risk",
    },
}


_LEGACY_THEME_TO_KEY = {
    values["zh"]: key for key, values in RISK_THEME_DISPLAY.items()
}
_LEGACY_THEME_TO_KEY.update(
    {
        values["en"].lower(): key
        for key, values in RISK_THEME_DISPLAY.items()
    }
)


def risk_theme_key(theme: Any) -> str:
    if isinstance(theme, Mapping):
        for key_name in ("theme_key", "theme_id", "risk_theme"):
            value = str(theme.get(key_name) or "").strip()
            if value:
                return _LEGACY_THEME_TO_KEY.get(value, _LEGACY_THEME_TO_KEY.get(value.lower(), value))
        value = str(theme.get("theme_name") or theme.get("theme") or "").strip()
    else:
        value = str(theme or "").strip()
    return _LEGACY_THEME_TO_KEY.get(value, _LEGACY_THEME_TO_KEY.get(value.lower(), value))


def display_theme(theme_key: str, output_language: str = "zh") -> str:
    key = risk_theme_key(theme_key)
    labels = RISK_THEME_DISPLAY.get(key)
    if labels:
        return labels.get("zh" if output_language == "zh" else "en", labels.get("en", key))
    if output_language == "en" and re.search(r"[\u4e00-\u9fff]", str(theme_key or "")):
        return "validated risk disclosure"
    return str(theme_key or ("风险主题" if output_language == "zh" else "risk theme"))


EN_FORBIDDEN_CHINESE_TERMS: tuple[str, ...] = (
    "结论",
    "已验证",
    "合理推断",
    "待验证",
    "证据边界",
    "风险判断",
    "财务传导路径",
    "基于业务模型",
    "履约/库存",
    "监管/合规",
    "供应链或库存",
    "新产品和需求不确定性",
    "客户/市场集中",
    "宏观、监管或法律",
    "不能给买卖建议",
    "不构成投资建议",
)


_EN_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("风险判断", "Risk Judgment"),
    ("已验证风险文本", "Verified Risk Text"),
    ("基于业务模型的风险排序", "Business-Model Risk Ranking"),
    ("基于业务模型", "business-model-based"),
    ("财务传导路径", "Financial Transmission Path"),
    ("已验证事实", "Verified Facts"),
    ("已验证线索", "Verified Signals"),
    ("合理推断", "Reasonable Inference"),
    ("待验证数据", "Data to Verify"),
    ("待验证假设", "Hypotheses to Verify"),
    ("证据边界", "Evidence Boundary"),
    ("结论", "Conclusion"),
    ("履约/库存/资本开支压力", "fulfillment, inventory, and capex pressure"),
    ("履约/库存", "fulfillment and inventory"),
    ("监管/合规风险", "regulatory and compliance risk"),
    ("监管/合规", "regulatory and compliance"),
    ("供应链或库存压力", "supply-chain or inventory pressure"),
    ("供应链或库存", "supply-chain or inventory"),
    ("新产品和需求不确定性", "new-product and demand uncertainty"),
    ("客户/市场集中风险", "customer or market concentration risk"),
    ("客户/市场集中", "customer or market concentration"),
    ("宏观、监管或法律风险", "macro, regulatory, or legal risk"),
    ("宏观、监管或法律", "macro, regulatory, or legal"),
    ("不能给买卖建议", "I cannot provide buy/sell advice"),
    ("不构成投资建议", "not investment advice"),
    (" 和 ", " and "),
)


def language_leakage_terms(text: str, output_language: str) -> list[str]:
    if output_language != "en":
        return []
    raw = str(text or "")
    return [term for term in EN_FORBIDDEN_CHINESE_TERMS if term in raw]


def language_leakage_count(text: str, output_language: str) -> int:
    return len(language_leakage_terms(text, output_language))


def repair_language_leakage(text: str, output_language: str) -> str:
    if output_language != "en":
        return str(text or "")
    repaired = str(text or "")
    for old, new in _EN_REPLACEMENTS:
        repaired = repaired.replace(old, new)
    repaired = re.sub(r"\b([A-Z]{1,5})\s+和\s+([A-Z]{1,5})'s\b", r"\1 and \2's", repaired)
    return repaired
