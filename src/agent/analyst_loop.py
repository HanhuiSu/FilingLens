"""Analyst draft generation / validation / revision loop."""

from __future__ import annotations

from typing import Any

from src.agent.analyst_draft import generate_analyst_draft, summarize_analyst_draft
from src.agent.draft_validation import summarize_draft_validation, validate_analyst_draft


def run_analyst_draft_loop(
    *,
    evidence_packet: dict[str, Any],
    answer_language: str,
    synthesis_mode: str,
    safety_policy: dict[str, Any],
    comparison_judgment_frame: dict[str, Any] | None = None,
    methodology_context: dict[str, Any] | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    all_generation_issues: list[dict[str, Any]] = []
    prior_draft: dict[str, Any] = {}
    repair_instructions: list[str] = []
    final_validation: dict[str, Any] = {}
    final_draft: dict[str, Any] = {}

    for attempt_number in range(1, max_attempts + 1):
        draft, generation_issues = generate_analyst_draft(
            evidence_packet=evidence_packet,
            answer_language=answer_language,
            synthesis_mode=synthesis_mode,
            comparison_judgment_frame=comparison_judgment_frame,
            methodology_context=methodology_context,
            prior_draft=prior_draft or None,
            repair_instructions=repair_instructions or None,
        )
        all_generation_issues.extend(generation_issues)
        validation = validate_analyst_draft(
            draft=draft,
            evidence_packet=evidence_packet,
            safety_policy=safety_policy,
            synthesis_mode=synthesis_mode,
        ).model_dump(exclude_none=True)
        attempts.append(
            {
                "attempt_index": attempt_number,
                "attempt_number": attempt_number,
                "stage": "initial" if attempt_number == 1 else "revision",
                "draft": draft,
                "draft_summary": summarize_analyst_draft(draft, validation),
                "generation_issues": generation_issues,
                "validation": validation,
                "validation_summary": summarize_draft_validation(validation),
            }
        )
        final_validation = validation
        final_draft = draft
        if bool(validation.get("passed", False)):
            return {
                "draft": final_draft,
                "validation": final_validation,
                "accepted_draft": dict(final_validation.get("accepted_draft", {}) or {}),
                "attempts": attempts,
                "generation_issues": all_generation_issues,
                "violations": list(final_validation.get("violations", []) or []),
                "draft_final_status": "passed",
            }
        prior_draft = draft
        repair_instructions = list(final_validation.get("repair_instructions", []) or [])
        if not repair_instructions:
            break

    return {
        "draft": final_draft,
        "validation": final_validation,
        "accepted_draft": {},
        "attempts": attempts,
        "generation_issues": all_generation_issues,
        "violations": list(final_validation.get("violations", []) or []),
        "draft_final_status": "fallback",
    }
