from __future__ import annotations

import json

import pytest

from config import settings
from src.agent.progress import append_progress_event, trace_path_for_id, validate_trace_id


def test_append_progress_event_creates_partial_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    trace_id = "progress-test-1"

    append_progress_event(
        trace_id,
        "run_started",
        "started",
        "已接收研究请求，正在初始化分析任务。",
        node="api_chat",
        metadata={"query": "分析 AMZN"},
        run_started_at="2026-05-21T12:00:00.000Z",
    )

    path = trace_path_for_id(trace_id)
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["trace_id"] == trace_id
    assert payload["run_started_at"] == "2026-05-21T12:00:00.000Z"
    assert payload["progress_events"][0]["event"] == "run_started"
    assert payload["progress_events"][0]["status"] == "started"
    assert payload["progress_events"][0]["message"]
    assert payload["progress_events"][0]["timestamp"]


def test_append_progress_event_preserves_existing_trace_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    trace_id = "progress-test-2"
    path = trace_path_for_id(trace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trace_id": trace_id, "final_answer": "old", "progress_events": []}))

    append_progress_event(trace_id, "intent_resolved", "completed", "已识别为单公司分析。")

    payload = json.loads(path.read_text())
    assert payload["final_answer"] == "old"
    assert payload["progress_events"][0]["event"] == "intent_resolved"


def test_progress_event_schema_is_stable(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    trace_id = "progress-test-3"

    append_progress_event(trace_id, "draft_started", "started", "正在生成分析草稿。")

    payload = json.loads(trace_path_for_id(trace_id).read_text())
    event = payload["progress_events"][0]
    assert {"event", "status", "message", "timestamp"} <= set(event)
    assert json.loads(trace_path_for_id(trace_id).read_text())["trace_id"] == trace_id


def test_research_planner_progress_events_are_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    trace_id = "progress-test-planner"

    append_progress_event(
        trace_id,
        "research_plan_started",
        "started",
        "正在生成研究计划。",
        metadata={"question_type": "causal_explanation"},
    )
    append_progress_event(
        trace_id,
        "research_plan_built",
        "completed",
        "研究计划已生成。",
        metadata={"question_type": "causal_explanation"},
    )
    append_progress_event(
        trace_id,
        "relevance_checked",
        "completed",
        "答案相关性已检查。",
        metadata={"relevance_status": "released_with_warnings"},
    )

    events = json.loads(trace_path_for_id(trace_id).read_text())["progress_events"]
    assert [event["event"] for event in events] == [
        "research_plan_started",
        "research_plan_built",
        "relevance_checked",
    ]
    assert events[-1]["metadata"]["relevance_status"] == "released_with_warnings"


@pytest.mark.parametrize("trace_id", ["bad.trace", "../bad", "abc/defgh", "short"])
def test_validate_trace_id_rejects_pathlike_values(trace_id):
    with pytest.raises(ValueError):
        validate_trace_id(trace_id)
