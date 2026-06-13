"""Shared helpers for loading static data files."""

from __future__ import annotations

from pathlib import Path
import yaml

from src.db.queries import CompanyRow


def load_companies_yaml(path: Path) -> list[CompanyRow]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows: list[CompanyRow] = []
    for item in raw.get("companies", []):
        rows.append(
            CompanyRow(
                ticker=str(item["ticker"]).upper(),
                company_name=str(item["company_name"]),
                sector=str(item["sector"]),
            )
        )
    return rows
