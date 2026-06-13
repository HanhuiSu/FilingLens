from __future__ import annotations

from src.agent.driver_evidence import (
    annotate_driver_evidence,
    apply_scope_aware_summary,
    classify_evidence_scope,
    evidence_summary_scope_warning,
)


def test_segment_driver_scope_is_bounded_inference():
    scoped = classify_evidence_scope(
        "Compute & Networking revenue increased 57% and the year over year increase was driven by accelerated computing and AI platform transitions."
    )

    assert scoped["driver_level"] == "segment_level_driver"
    assert scoped["claim_scope"] == "segment"
    assert scoped["allowed_claim_strength"] == "bounded_inference"


def test_product_driver_scope_is_bounded_inference():
    scoped = classify_evidence_scope(
        "Networking revenue was up 13% sequentially, driven by XDR InfiniBand products, NVLink, and Ethernet for AI solutions."
    )

    assert scoped["driver_level"] == "product_level_driver"
    assert scoped["claim_scope"] == "product"
    assert scoped["allowed_claim_strength"] == "bounded_inference"


def test_total_revenue_driver_scope_can_be_definitive():
    scoped = classify_evidence_scope(
        "Total revenue increased 58%, driven by demand for accelerated computing and AI platforms."
    )

    assert scoped["driver_level"] == "company_level_driver"
    assert scoped["claim_scope"] == "company"
    assert scoped["allowed_claim_strength"] == "definitive"


def test_segment_paragraph_with_total_table_stays_segment_scope():
    row = annotate_driver_evidence(
        {
            "evidence_id": "T1",
            "claim": "NVIDIA的营收增长主要由加速计算和人工智能的平台转型驱动",
            "supporting_snippet": (
                "Compute & Networking $130,141 $82,875 57%. Graphics $9,156 $5,085 80%. "
                "Total $139,297 $87,960 58%. Compute & Networking revenue - The year over year "
                "increase was driven by platform shifts to accelerated computing and AI."
            ),
        }
    )

    assert row["driver_level"] == "segment_level_driver"
    assert row["claim_scope"] == "segment"
    assert row["allowed_claim_strength"] == "bounded_inference"


def test_segment_evidence_summary_is_scope_aware():
    row = apply_scope_aware_summary(
        {
            "evidence_id": "T1",
            "claim": "NVIDIA的营收增长主要由加速计算和人工智能的平台转型驱动",
            "supporting_snippet": (
                "Compute & Networking revenue - The year over year increase was driven by "
                "platform shifts to accelerated computing and AI."
            ),
        }
    )

    assert row["summary_scope_warning"] == "evidence_summary_scope_overclaim"
    assert row["claim_scope"] == "segment"
    assert "Compute & Networking" in row["claim"]
    assert "分部" in row["claim"]
    assert "NVIDIA的营收增长主要由" not in row["claim"]


def test_summary_scope_warning_reports_bad_generated_summary():
    warning = evidence_summary_scope_warning(
        {
            "evidence_id": "T1",
            "claim": "NVIDIA 营收增长主要由 AI 平台转型驱动",
            "supporting_snippet": (
                "Compute & Networking revenue - The year over year increase was driven by "
                "platform shifts to accelerated computing and AI."
            ),
        }
    )

    assert warning is not None
    assert warning["code"] == "evidence_summary_scope_overclaim"
    assert "Compute & Networking" in warning["suggested_summary"]
