"""Progress-event helpers for trace-backed live run monitoring."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from config import settings

logger = logging.getLogger(__name__)

_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,80}$")


def validate_trace_id(trace_id: str) -> str:
    """Return a safe trace id or raise ValueError."""
    value = str(trace_id or "").strip()
    if not _TRACE_ID_RE.fullmatch(value):
        raise ValueError("trace id must be 8-80 chars and contain only letters, numbers, and hyphen")
    return value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def trace_path_for_id(trace_id: str) -> Path:
    return settings.traces_dir / f"{validate_trace_id(trace_id)}.json"


def _safe_json_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(payload).items():
        try:
            json.dumps(value, ensure_ascii=False, default=str)
            out[str(key)] = value
        except (TypeError, ValueError):
            out[str(key)] = str(value)
    return out


def read_trace_payload(trace_id: str) -> dict[str, Any]:
    path = trace_path_for_id(trace_id)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read trace payload for %s", trace_id, exc_info=True)
        return {}


def write_trace_payload(trace_id: str, payload: Mapping[str, Any]) -> Path:
    path = trace_path_for_id(trace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _safe_json_payload(payload)
    clean["trace_id"] = validate_trace_id(str(clean.get("trace_id") or trace_id))
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(clean, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _elapsed_ms(payload: Mapping[str, Any], now_text: str) -> int | None:
    started = _parse_iso(payload.get("run_started_at") or payload.get("started_at"))
    now = _parse_iso(now_text)
    if not started or not now:
        return None
    return max(0, int((now - started).total_seconds() * 1000))


def append_progress_event(
    trace_id: str,
    event: str,
    status: str,
    message: str,
    *,
    node: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    run_started_at: str | None = None,
) -> None:
    """Append a progress event to a trace file without interrupting the run."""
    try:
        try:
            safe_trace_id = validate_trace_id(trace_id)
        except ValueError:
            logger.debug("Skipping progress event for unsafe trace id %r", trace_id)
            return
        payload = read_trace_payload(safe_trace_id)
        if not payload:
            payload = {"trace_id": safe_trace_id, "progress_events": []}
        payload["trace_id"] = safe_trace_id
        if run_started_at and not payload.get("run_started_at"):
            payload["run_started_at"] = str(run_started_at)
        events = payload.get("progress_events")
        if not isinstance(events, list):
            events = []
        now_text = utc_now_iso()
        item: dict[str, Any] = {
            "event": str(event),
            "status": str(status),
            "message": str(message),
            "timestamp": now_text,
        }
        if node:
            item["node"] = str(node)
        elapsed = _elapsed_ms(payload, now_text)
        if elapsed is not None:
            item["elapsed_ms"] = elapsed
        if metadata:
            item["metadata"] = _safe_json_payload(dict(metadata))

        last = events[-1] if events and isinstance(events[-1], dict) else {}
        duplicate_keys = ("event", "status") if str(event) == "run_failed" else ("event", "status", "message", "node")
        duplicate = all(str(last.get(key) or "") == str(item.get(key) or "") for key in duplicate_keys)
        if not duplicate:
            events.append(item)
        payload["progress_events"] = events[-80:]
        write_trace_payload(safe_trace_id, payload)
    except Exception:
        logger.warning("Could not append progress event %s for trace %s", event, trace_id, exc_info=True)
