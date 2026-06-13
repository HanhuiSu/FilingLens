"""Lightweight answer ownership and provenance helpers.

This module is intentionally small for the stabilization pass: it records every
body-changing transition and provides the minimal AnswerAssembler surface needed
while legacy generation still builds most answer text.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from src.agent.types import AnswerCandidate

_REF_RE = re.compile(r"\[([NT]\d+)\]")


def evidence_refs_from_body(body: str) -> list[str]:
    return list(dict.fromkeys(_REF_RE.findall(str(body or ""))))


def _body_hash(body: str) -> str:
    return hashlib.sha1(str(body or "").encode("utf-8")).hexdigest()[:12]


def _owner(state: Mapping[str, Any], default: str = "") -> str:
    return str(
        state.get("final_answer_source")
        or (state.get("output", {}) if isinstance(state.get("output"), Mapping) else {}).get("final_answer_source")
        or (state.get("synthesis", {}) if isinstance(state.get("synthesis"), Mapping) else {}).get("final_answer_source")
        or default
        or ""
    )


def answer_history_event(
    *,
    state: Mapping[str, Any],
    previous_body: str,
    new_body: str,
    new_owner: str,
    transform: str,
    reason: str,
    previous_owner: str | None = None,
    evidence_refs: list[str] | None = None,
    claim_change_allowed: bool = False,
    validator_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    refs = list(dict.fromkeys(evidence_refs if evidence_refs is not None else evidence_refs_from_body(new_body)))
    history = [dict(item) for item in state.get("answer_history", []) or [] if isinstance(item, Mapping)]
    event: dict[str, Any] = {
        "index": len(history) + 1,
        "previous_owner": str(previous_owner if previous_owner is not None else _owner(state)),
        "new_owner": str(new_owner or ""),
        "transform": str(transform or ""),
        "reason": str(reason or ""),
        "evidence_refs": refs,
        "claim_change_allowed": bool(claim_change_allowed),
        "body_changed": str(previous_body or "") != str(new_body or ""),
        "previous_body_hash": _body_hash(previous_body),
        "new_body_hash": _body_hash(new_body),
        "previous_body_chars": len(str(previous_body or "")),
        "new_body_chars": len(str(new_body or "")),
    }
    if validator_result:
        event["validator_result"] = dict(validator_result)
    return event


def answer_transition_payload(
    state: Mapping[str, Any],
    *,
    previous_body: str,
    new_body: str,
    new_owner: str,
    transform: str,
    reason: str,
    previous_owner: str | None = None,
    evidence_refs: list[str] | None = None,
    claim_change_allowed: bool = False,
    validator_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    history = [dict(item) for item in state.get("answer_history", []) or [] if isinstance(item, Mapping)]
    event = answer_history_event(
        state={**dict(state), "answer_history": history},
        previous_body=previous_body,
        new_body=new_body,
        new_owner=new_owner,
        transform=transform,
        reason=reason,
        previous_owner=previous_owner,
        evidence_refs=evidence_refs,
        claim_change_allowed=claim_change_allowed,
        validator_result=validator_result,
    )
    history.append(event)
    return {
        "answer_history": history,
        "final_answer_source": str(new_owner or _owner(state)),
    }


class AnswerAssembler:
    """Minimal final-answer owner for candidate selection and trace ownership."""

    @staticmethod
    def candidate(
        *,
        body: str,
        owner: str,
        requested_dimensions: list[str] | None = None,
        answered_dimensions: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        limitations: list[str] | None = None,
        allowed_repairs: list[str] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> AnswerCandidate:
        return AnswerCandidate(
            body=body,
            owner=owner,
            requested_dimensions=list(requested_dimensions or []),
            answered_dimensions=list(answered_dimensions or []),
            evidence_refs=list(evidence_refs or evidence_refs_from_body(body)),
            limitations=list(limitations or []),
            allowed_repairs=list(allowed_repairs or []),
            provenance=dict(provenance or {}),
        )

    @staticmethod
    def select(
        candidate: AnswerCandidate,
        state: Mapping[str, Any],
        *,
        transform: str,
        reason: str,
        previous_body: str | None = None,
        claim_change_allowed: bool = True,
        validator_result: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous_body = str(previous_body if previous_body is not None else (state.get("final_answer") or state.get("draft_answer") or ""))
        payload = answer_transition_payload(
            state,
            previous_body=previous_body,
            new_body=candidate.body,
            new_owner=candidate.owner,
            transform=transform,
            reason=reason,
            evidence_refs=list(candidate.evidence_refs),
            claim_change_allowed=claim_change_allowed,
            validator_result=validator_result,
        )
        candidate_payload = candidate.model_dump(exclude_none=True)
        candidates = [dict(item) for item in state.get("answer_candidates", []) or [] if isinstance(item, Mapping)]
        candidates.append(candidate_payload)
        return {
            **payload,
            "draft_answer": candidate.body,
            "final_answer": candidate.body,
            "answer_candidate": candidate_payload,
            "answer_candidates": candidates[-8:],
        }
