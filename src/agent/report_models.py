"""Structured company-analysis report models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ReportSection(BaseModel):
    section_id: str
    title: str
    markdown: str
    section_status: Literal["satisfied", "partial", "missing"] = "missing"
    key_evidence_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    contract_status: Literal["not_checked", "passed", "repaired", "failed", "blocked"] = "not_checked"


class CompanyAnalysisReport(BaseModel):
    title: str
    company: str
    ticker: str
    period: str | None = None
    report_type: Literal["company_analysis", "comparison_analysis"] = "company_analysis"
    sections: list[ReportSection] = Field(default_factory=list)
    overall_limitations: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    contract_status: Literal["passed", "repaired", "failed", "blocked"] = "failed"
    markdown: str
    generated_at: str
