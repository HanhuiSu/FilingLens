"""Unit tests for the three financial-agent tools.

Covers:
  - compute_metrics: pure-computation, no external deps
  - query_financial_data: reads live DuckDB (integration, skipped if DB absent)
  - search_filings: reads live ChromaDB + embedding (integration, skipped if absent)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.tools.compute_metrics import compute_metrics

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# compute_metrics — pure unit tests (no DB / model)
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for compute_metrics (pure computation, no external deps)."""

    @staticmethod
    def _invoke(data, computation, denominator_data=None):
        return compute_metrics.invoke(
            {"data": data, "computation": computation, "denominator_data": denominator_data}
        )

    def test_growth_basic(self):
        data = [
            {"period": "2023Q1", "value": 100},
            {"period": "2023Q2", "value": 120},
            {"period": "2023Q3", "value": 90},
        ]
        result = self._invoke(data, "growth")
        assert result["computation"] == "growth"
        assert len(result["results"]) == 2
        assert result["results"][0]["growth"] == pytest.approx(0.2, abs=1e-4)
        assert result["results"][1]["growth"] == pytest.approx(-0.25, abs=1e-4)

    def test_qoq_basic(self):
        data = [
            {"period": "2023Q1", "value": 200},
            {"period": "2023Q2", "value": 250},
        ]
        result = self._invoke(data, "qoq")
        assert len(result["results"]) == 1
        assert result["results"][0]["qoq_change"] == pytest.approx(0.25, abs=1e-4)

    def test_yoy_with_4_quarter_gap(self):
        data = [
            {"period": "2023Q1", "value": 100},
            {"period": "2023Q2", "value": 110},
            {"period": "2023Q3", "value": 120},
            {"period": "2023Q4", "value": 130},
            {"period": "2024Q1", "value": 150},
        ]
        result = self._invoke(data, "yoy")
        assert len(result["results"]) == 1
        assert result["results"][0]["yoy_change"] == pytest.approx(0.5, abs=1e-4)
        assert result["results"][0]["compare_period"] == "2023Q1"

    def test_yoy_fallback_with_2_points(self):
        data = [
            {"period": "2023", "value": 100},
            {"period": "2024", "value": 120},
        ]
        result = self._invoke(data, "yoy")
        assert len(result["results"]) == 1
        assert "fallback" in result["results"][0].get("note", "")

    def test_margin_with_denominator(self):
        num = [
            {"period": "2023Q1", "value": 40},
            {"period": "2023Q2", "value": 50},
        ]
        denom = [
            {"period": "2023Q1", "value": 100},
            {"period": "2023Q2", "value": 200},
        ]
        result = self._invoke(num, "margin", denominator_data=denom)
        assert result["results"][0]["margin"] == pytest.approx(0.4, abs=1e-4)
        assert result["results"][1]["margin"] == pytest.approx(0.25, abs=1e-4)

    def test_margin_without_denominator(self):
        data = [
            {"period": "2023Q1", "value": 100},
            {"period": "2023Q2", "value": 150},
        ]
        result = self._invoke(data, "margin")
        assert result["results"][0]["ratio"] == pytest.approx(1.0)
        assert result["results"][1]["ratio"] == pytest.approx(1.5)

    def test_ratio_difference_and_valuation_multiple(self):
        left = [{"period": "2024", "value": 100}]
        right = [{"period": "2024", "value": 25}]

        ratio = self._invoke(left, "ratio", denominator_data=right)
        difference = self._invoke(left, "difference", denominator_data=right)
        multiple = self._invoke(left, "valuation_multiple", denominator_data=right)

        assert ratio["results"][0]["ratio"] == pytest.approx(4.0)
        assert difference["results"][0]["difference"] == pytest.approx(75.0)
        assert multiple["results"][0]["multiple"] == pytest.approx(4.0)
        assert multiple["results"][0]["multiple_label"] == "4.00x"

    def test_cagr_basic(self):
        data = [
            {"period": "2021-01-01", "value": 100},
            {"period": "2024-01-01", "value": 200},
        ]
        result = self._invoke(data, "cagr")
        r = result["results"][0]
        assert r["years"] == 3
        assert r["cagr"] == pytest.approx(0.259921, abs=1e-3)

    def test_cagr_negative_values(self):
        data = [
            {"period": "2021-01-01", "value": -50},
            {"period": "2023-01-01", "value": 100},
        ]
        result = self._invoke(data, "cagr")
        assert "error" in result["results"][0]

    def test_unknown_computation_rejected_by_schema(self):
        """Pydantic Literal validates computation before the function body runs."""
        from pydantic import ValidationError

        data = [{"period": "2023Q1", "value": 100}]
        with pytest.raises((ValidationError, Exception)):
            self._invoke(data, "invalid_op")

    def test_division_by_zero(self):
        data = [
            {"period": "2023Q1", "value": 0},
            {"period": "2023Q2", "value": 100},
        ]
        result = self._invoke(data, "growth")
        assert result["results"][0]["growth"] is None

    def test_single_point_growth(self):
        data = [{"period": "2023Q1", "value": 100}]
        result = self._invoke(data, "growth")
        assert result["results"] == []


# ---------------------------------------------------------------------------
# query_financial_data — integration (needs live DuckDB with data)
# ---------------------------------------------------------------------------
_DB_EXISTS = Path(ROOT / "data" / "db" / "financial.duckdb").exists()


@pytest.mark.skipif(not _DB_EXISTS, reason="DuckDB not populated")
@pytest.mark.integration
class TestQueryFinancialData:
    """Integration tests for query_financial_data against live DuckDB."""

    @staticmethod
    def _invoke(**kwargs):
        from src.tools.query_financial_data import query_financial_data
        return query_financial_data.invoke(kwargs)

    def test_revenue_query(self):
        result = self._invoke(ticker="AAPL", metrics=["revenue"], period_type="quarterly", limit=4)
        assert result["ticker"] == "AAPL"
        assert len(result["financial_facts"]) > 0
        assert result["financial_facts"][0]["metric"] == "revenue"

    def test_multiple_metrics(self):
        result = self._invoke(
            ticker="MSFT",
            metrics=["revenue", "net_income", "eps"],
            period_type="quarterly",
            limit=8,
        )
        metrics_returned = {r["metric"] for r in result["financial_facts"]}
        assert "revenue" in metrics_returned

    def test_price_query(self):
        result = self._invoke(ticker="AAPL", metrics=["close", "volume"], limit=5)
        assert len(result["price_data"]) > 0
        assert "close" in result["price_data"][0]
        assert "volume" in result["price_data"][0]

    def test_mixed_metrics(self):
        result = self._invoke(
            ticker="AAPL",
            metrics=["revenue", "close"],
            period_type="quarterly",
            limit=5,
        )
        assert len(result["financial_facts"]) > 0
        assert len(result["price_data"]) > 0

    def test_date_range_filter(self):
        result = self._invoke(
            ticker="AAPL",
            metrics=["revenue"],
            date_start="2024-01-01",
            date_end="2024-12-31",
            limit=20,
        )
        for row in result["financial_facts"]:
            assert row["period_end"] >= "2024-01-01"
            assert row["period_end"] <= "2024-12-31"

    def test_invalid_metric_returns_empty(self):
        result = self._invoke(ticker="AAPL", metrics=["nonexistent_metric"], limit=5)
        assert result["financial_facts"] == []
        assert result["price_data"] == []

    def test_limit_respected(self):
        result = self._invoke(ticker="AAPL", metrics=["close"], limit=3)
        assert len(result["price_data"]) <= 3

    def test_year_quarter_filter(self):
        result = self._invoke(
            ticker="AAPL",
            metrics=["revenue"],
            period_type="quarterly",
            year=2025,
            quarter=1,
            year_basis="fiscal",
            strict_period_match=True,
            limit=8,
        )
        assert len(result["financial_facts"]) >= 1
        assert all(r.get("fiscal_year") == 2025 for r in result["financial_facts"])
        assert all(r.get("fiscal_quarter") == 1 for r in result["financial_facts"])

    def test_trailing_returns_n(self):
        result = self._invoke(
            ticker="MSFT",
            metrics=["revenue"],
            period_type="trailing",
            target_period_type="quarterly",
            trailing_n=4,
            limit=20,
        )
        assert 0 < len(result["financial_facts"]) <= 4
        assert result.get("period_context", {}).get("applied_filters", {}).get("mode") == "trailing"

    def test_period_context_in_result(self):
        result = self._invoke(
            ticker="GOOGL",
            metrics=["revenue"],
            period_type="annual",
            year=2025,
            year_basis="fiscal",
            comparison_basis="same_period",
            strict_period_match=True,
            limit=5,
        )
        assert "period_context" in result
        ctx = result["period_context"]
        assert ctx.get("year_basis") == "fiscal"
        assert ctx.get("comparison_basis") == "same_period"

    def test_latest_annual_calendar_strict_not_mappable(self):
        result = self._invoke(
            ticker="AAPL",
            metrics=["revenue"],
            period_type="latest",
            target_period_type="annual",
            year_basis="calendar",
            strict_period_match=True,
            limit=5,
        )
        assert result["financial_facts"] == []
        notes = result.get("period_context", {}).get("notes", [])
        assert "calendar_year_not_strictly_mappable_for_non_dec_fiscal_year" in notes


# ---------------------------------------------------------------------------
# search_filings — integration (needs ChromaDB + embedding model)
# ---------------------------------------------------------------------------
_CHROMA_EXISTS = Path(ROOT / "data" / "vectorstore").exists() and any(
    Path(ROOT / "data" / "vectorstore").iterdir()
)


def _embedding_model_available() -> bool:
    """Check if the embedding model can be loaded (offline-safe)."""
    try:
        from src.tools.search_filings import _get_model
        _get_model()
        return True
    except Exception:
        return False


_MODEL_OK = False
if _CHROMA_EXISTS and os.getenv("ENABLE_SEARCH_FILINGS_TESTS", "0") == "1":
    _MODEL_OK = _embedding_model_available()


@pytest.mark.skipif(not _CHROMA_EXISTS, reason="ChromaDB vectorstore not built")
@pytest.mark.skipif(not _DB_EXISTS, reason="DuckDB not populated")
@pytest.mark.skipif(not _MODEL_OK, reason="Embedding model not available")
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_embedding
@pytest.mark.requires_vectorstore
@pytest.mark.timeout(120)
class TestSearchFilings:
    """Integration tests for search_filings (requires embedding model + ChromaDB)."""

    @staticmethod
    def _invoke(**kwargs):
        from src.tools.search_filings import search_filings
        return search_filings.invoke(kwargs)

    def test_basic_search(self):
        result = self._invoke(ticker="AAPL", query="revenue growth", top_k=3)
        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= 3
        chunk = result[0]
        assert "text" in chunk
        assert "filing_id" in chunk
        assert "section" in chunk
        assert "score" in chunk
        assert chunk["ticker"] == "AAPL"

    def test_form_type_filter(self):
        result = self._invoke(ticker="AAPL", query="risk factors", top_k=5, form_type="10-K")
        for chunk in result:
            assert chunk["form_type"] == "10-K"

    def test_top_k_capped(self):
        result = self._invoke(ticker="AAPL", query="management discussion", top_k=2)
        assert len(result) <= 2

    def test_nonexistent_ticker(self):
        result = self._invoke(ticker="ZZZZZ", query="anything", top_k=3)
        assert result == []

    def test_date_range_filter(self):
        result = self._invoke(
            ticker="AAPL",
            query="net income",
            top_k=5,
            date_start="2024-01-01",
            date_end="2025-12-31",
        )
        assert isinstance(result, list)
