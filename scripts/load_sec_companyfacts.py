#!/usr/bin/env python3
"""Load SEC XBRL companyfacts into financial_facts with explicit provenance."""

from __future__ import annotations

import argparse
from datetime import date
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import sec_edgar_identity, settings
from src.db.queries import (
    FinancialFactRow,
    clear_financial_facts_for_ticker_provider,
    get_connection,
    insert_financial_facts_batch,
)

SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_ALLOWED_FORMS = {
    "annual": {"10-K", "10-K/A"},
    "quarterly": {"10-Q", "10-Q/A"},
}


def normalize_cik(raw: str | int) -> str:
    return str(raw).strip().lstrip("0").zfill(10)


def companyfacts_url(cik: str) -> str:
    return SEC_COMPANYFACTS_URL.format(cik=normalize_cik(cik))


def sec_viewer_url(cik: str, accession: str) -> str:
    acc_nodash = str(accession or "").replace("-", "")
    cik_num = str(int(normalize_cik(cik)))
    return (
        f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik_num}"
        f"&accession_number={acc_nodash}&xbrl_type=v"
    )


def load_company_ciks(path: Path) -> dict[str, str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, str] = {}
    for item in raw.get("companies", []):
        ticker = str(item.get("ticker", "")).upper().strip()
        cik = str(item.get("cik", "")).strip()
        if ticker and cik:
            out[ticker] = normalize_cik(cik)
    return out


def load_metric_mappings(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(raw.get("metrics", {}))


def fetch_companyfacts(cik: str, user_agent: str, timeout: int = 45) -> dict[str, Any]:
    req = Request(companyfacts_url(cik), headers={"User-Agent": user_agent, "Accept-Encoding": "identity"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _canonical_unit(metric: str, cfg: dict[str, Any], sec_unit: str) -> str:
    if metric == "eps":
        return "USD_per_share"
    if str(cfg.get("unit") or "") == "shares":
        return "shares"
    return str(cfg.get("unit") or sec_unit or "")


def _candidate_units(metric: str, cfg: dict[str, Any]) -> list[str]:
    if isinstance(cfg.get("sec_units"), list):
        return [str(x) for x in cfg["sec_units"]]
    unit = str(cfg.get("unit") or "")
    if unit == "shares":
        return ["shares"]
    if unit == "USD_per_share":
        return ["USD/shares"]
    if unit:
        return [unit]
    return []


def _period_type(form: str) -> str | None:
    form_upper = str(form or "").upper()
    if form_upper in SEC_ALLOWED_FORMS["annual"]:
        return "annual"
    if form_upper in SEC_ALLOWED_FORMS["quarterly"]:
        return "quarterly"
    return None


def _fact_value(fact: dict[str, Any]) -> float | None:
    try:
        return float(fact.get("val"))
    except (TypeError, ValueError):
        return None


def _apply_value_transform(value: float, cfg: dict[str, Any]) -> float:
    transform = str(cfg.get("value_transform") or "").strip().lower()
    if transform == "abs":
        return abs(value)
    if transform == "negative_abs":
        return -abs(value)
    return value


def _extract_direct_metric_rows(
    *,
    ticker: str,
    cik: str,
    payload: dict[str, Any],
    metric: str,
    cfg: dict[str, Any],
) -> list[FinancialFactRow]:
    facts_root = payload.get("facts", {}).get("us-gaap", {})
    rows_by_key: dict[tuple[str, date], FinancialFactRow] = {}
    for tag in cfg.get("tags", []) or []:
        tag_payload = facts_root.get(str(tag), {})
        units = tag_payload.get("units", {}) if isinstance(tag_payload, dict) else {}
        for sec_unit in _candidate_units(metric, cfg):
            for fact in units.get(sec_unit, []) or []:
                if not isinstance(fact, dict):
                    continue
                period_type = _period_type(str(fact.get("form", "")))
                period_end = _parse_date(fact.get("end"))
                raw_value = _fact_value(fact)
                value = _apply_value_transform(raw_value, cfg) if raw_value is not None else None
                if period_type is None or period_end is None or value is None:
                    continue
                key = (period_type, period_end)
                if key in rows_by_key:
                    continue
                accession = str(fact.get("accn", "") or "")
                rows_by_key[key] = FinancialFactRow(
                    ticker=ticker,
                    period_end=period_end,
                    period_type=period_type,
                    metric=metric,
                    value=value,
                    unit=_canonical_unit(metric, cfg, sec_unit),
                    filing_date=_parse_date(fact.get("filed")),
                    source_provider="sec_companyfacts",
                    source_url=sec_viewer_url(cik, accession) if accession else companyfacts_url(cik),
                    source_filing_id=accession,
                    confidence="high",
                    extraction_method="xbrl_companyfacts",
                    source_tag=str(tag),
                )
    return list(rows_by_key.values())


def _derive_ratio_rows(
    *,
    metric: str,
    numerator_metric: str,
    denominator_metric: str,
    direct_rows_by_metric: dict[str, list[FinancialFactRow]],
) -> list[FinancialFactRow]:
    numerator_rows = {
        (row.period_type, row.period_end): row
        for row in direct_rows_by_metric.get(numerator_metric, [])
        if row.value is not None
    }
    denominator_rows = {
        (row.period_type, row.period_end): row
        for row in direct_rows_by_metric.get(denominator_metric, [])
        if row.value not in (None, 0)
    }
    out: list[FinancialFactRow] = []
    for key, numerator in numerator_rows.items():
        denominator = denominator_rows.get(key)
        if denominator is None or denominator.value in (None, 0):
            continue
        out.append(
            FinancialFactRow(
                ticker=numerator.ticker,
                period_end=numerator.period_end,
                period_type=numerator.period_type,
                metric=metric,
                value=float(numerator.value) / float(denominator.value),
                unit="ratio",
                filing_date=numerator.filing_date or denominator.filing_date,
                source_provider="sec_companyfacts",
                source_url=numerator.source_url or denominator.source_url,
                source_filing_id=numerator.source_filing_id or denominator.source_filing_id,
                confidence="high",
                extraction_method="xbrl_companyfacts_derived_ratio",
                source_tag=f"{numerator.source_tag}/{denominator.source_tag}",
            )
        )
    return out


def _derive_binary_rows(
    *,
    metric: str,
    left_metric: str,
    right_metric: str,
    operation: str,
    unit: str,
    direct_rows_by_metric: dict[str, list[FinancialFactRow]],
) -> list[FinancialFactRow]:
    left_rows = {
        (row.period_type, row.period_end): row
        for row in direct_rows_by_metric.get(left_metric, [])
        if row.value is not None
    }
    right_rows = {
        (row.period_type, row.period_end): row
        for row in direct_rows_by_metric.get(right_metric, [])
        if row.value is not None
    }
    out: list[FinancialFactRow] = []
    for key, left in left_rows.items():
        right = right_rows.get(key)
        if right is None or left.value is None or right.value is None:
            continue
        if operation == "sum":
            value = float(left.value) + float(right.value)
        elif operation == "difference":
            value = float(left.value) - float(right.value)
        else:
            continue
        out.append(
            FinancialFactRow(
                ticker=left.ticker,
                period_end=left.period_end,
                period_type=left.period_type,
                metric=metric,
                value=value,
                unit=unit,
                filing_date=left.filing_date or right.filing_date,
                source_provider="sec_companyfacts",
                source_url=left.source_url or right.source_url,
                source_filing_id=left.source_filing_id or right.source_filing_id,
                confidence="high",
                extraction_method=f"xbrl_companyfacts_derived_{operation}",
                source_tag=f"{left.source_tag}/{right.source_tag}",
            )
        )
    return out


def extract_companyfacts_rows(
    ticker: str,
    cik: str,
    payload: dict[str, Any],
    mappings: dict[str, Any],
) -> list[FinancialFactRow]:
    direct_rows_by_metric: dict[str, list[FinancialFactRow]] = {}
    for metric, cfg in mappings.items():
        cfg_dict = dict(cfg or {})
        if cfg_dict.get("derived"):
            continue
        direct_rows_by_metric[metric] = _extract_direct_metric_rows(
            ticker=ticker,
            cik=cik,
            payload=payload,
            metric=metric,
            cfg=cfg_dict,
        )

    rows: list[FinancialFactRow] = []
    for metric, cfg in mappings.items():
        cfg_dict = dict(cfg or {})
        if cfg_dict.get("internal_only") or cfg_dict.get("derived"):
            continue
        rows.extend(direct_rows_by_metric.get(metric, []))

    for metric, cfg in mappings.items():
        derived = dict((cfg or {}).get("derived") or {})
        if not derived:
            continue
        operation = str(derived.get("operation") or "ratio").strip().lower()
        if operation == "ratio":
            derived_rows = _derive_ratio_rows(
                metric=metric,
                numerator_metric=str(derived.get("numerator", "")),
                denominator_metric=str(derived.get("denominator", "")),
                direct_rows_by_metric=direct_rows_by_metric,
            )
        else:
            derived_rows = _derive_binary_rows(
                metric=metric,
                left_metric=str(derived.get("left", "")),
                right_metric=str(derived.get("right", "")),
                operation=operation,
                unit=str((cfg or {}).get("unit") or "USD"),
                direct_rows_by_metric=direct_rows_by_metric,
            )
        if derived_rows:
            direct_rows_by_metric[metric] = derived_rows
            rows.extend(derived_rows)
    return rows


def load_ticker(conn, ticker: str, cik: str, mappings: dict[str, Any], user_agent: str, dry_run: bool = False) -> int:
    payload = fetch_companyfacts(cik, user_agent=user_agent)
    rows = extract_companyfacts_rows(ticker=ticker, cik=cik, payload=payload, mappings=mappings)
    if dry_run:
        print(f"  {ticker}: {len(rows)} SEC companyfacts rows (dry-run)")
        return len(rows)
    clear_financial_facts_for_ticker_provider(conn, ticker, "sec_companyfacts")
    insert_financial_facts_batch(conn, rows)
    print(f"  {ticker}: {len(rows)} SEC companyfacts rows")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load SEC XBRL companyfacts into DuckDB financial_facts.")
    parser.add_argument("--ticker", action="append", help="Ticker to load. Can be repeated. Defaults to target_tickers.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse, but do not write DuckDB.")
    parser.add_argument("--sleep", type=float, default=0.12, help="Delay between SEC requests.")
    args = parser.parse_args()

    name, email = sec_edgar_identity()
    user_agent = f"{name} {email}"
    ciks = load_company_ciks(settings.companies_yaml)
    mappings = load_metric_mappings(settings.data_dir / "metric_mappings.yaml")
    tickers = [t.upper() for t in (args.ticker or settings.target_tickers)]

    conn = get_connection()
    try:
        for ticker in tickers:
            cik = ciks.get(ticker)
            if not cik:
                print(f"  {ticker}: missing CIK in {settings.companies_yaml}")
                continue
            load_ticker(conn, ticker=ticker, cik=cik, mappings=mappings, user_agent=user_agent, dry_run=args.dry_run)
            time.sleep(max(args.sleep, 0.0))
    finally:
        conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
