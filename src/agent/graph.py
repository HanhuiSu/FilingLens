"""LangGraph workflow — compiles the financial-analysis agent graph."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    check_evidence,
    classify_and_extract,
    contract_check_node,
    evaluate_evidence,
    execute_tools,
    finalize_node,
    generate_answer,
    prepare_contract_evidence_retry_node,
    relevance_check_node,
    relevance_repair_node,
    repair_generate_node,
    route_after_contract,
    route_after_relevance,
    safe_blocked_answer_node,
    research_plan_node,
)
from src.agent.progress import append_progress_event
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


def _timed_node(name: str, fn: Callable[[AgentState], dict[str, Any]]) -> Callable[[AgentState], dict[str, Any]]:
    def wrapped(state: AgentState) -> dict[str, Any]:
        started = time.monotonic()
        trace_id = str(state.get("trace_id") or "")
        logger.warning("agent stage start: %s trace=%s", name, trace_id or "pending")
        try:
            out = fn(state)
        except Exception as exc:
            logger.exception("agent stage failed: %s trace=%s elapsed=%.1fs", name, trace_id or "pending", time.monotonic() - started)
            if trace_id:
                error_summary = str(exc).strip() or type(exc).__name__
                append_progress_event(
                    trace_id,
                    "run_failed",
                    "failed",
                    f"{name} 阶段失败：{error_summary[:180]}",
                    node=name,
                    metadata={"node": name, "error": str(exc)[:500]},
                )
            raise
        out_trace_id = trace_id or str(out.get("trace_id") or "")
        logger.warning("agent stage done: %s trace=%s elapsed=%.1fs", name, out_trace_id or "pending", time.monotonic() - started)
        return out

    return wrapped


def build_graph() -> StateGraph:
    """Construct and return the (uncompiled) StateGraph."""
    graph = StateGraph(AgentState)

    graph.add_node("classify", _timed_node("classify", classify_and_extract))
    graph.add_node("research_plan", _timed_node("research_plan", research_plan_node))
    graph.add_node("execute_tools", _timed_node("execute_tools", execute_tools))
    graph.add_node("evaluate", _timed_node("evaluate", evaluate_evidence))
    graph.add_node("generate", _timed_node("generate", generate_answer))
    graph.add_node("contract_check", _timed_node("contract_check", contract_check_node))
    graph.add_node("relevance_check", _timed_node("relevance_check", relevance_check_node))
    graph.add_node("relevance_repair", _timed_node("relevance_repair", relevance_repair_node))
    graph.add_node("repair_generate", _timed_node("repair_generate", repair_generate_node))
    graph.add_node("prepare_contract_evidence_retry", _timed_node("prepare_contract_evidence_retry", prepare_contract_evidence_retry_node))
    graph.add_node("safe_blocked_answer", _timed_node("safe_blocked_answer", safe_blocked_answer_node))
    graph.add_node("finalize", _timed_node("finalize", finalize_node))

    graph.set_entry_point("classify")
    graph.add_edge("classify", "research_plan")
    graph.add_edge("research_plan", "execute_tools")
    graph.add_edge("execute_tools", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        check_evidence,
        {
            "sufficient": "generate",
            "insufficient": "execute_tools",
        },
    )
    graph.add_edge("generate", "contract_check")
    graph.add_conditional_edges(
        "contract_check",
        route_after_contract,
        {
            "relevance_check": "relevance_check",
            "repair_generate": "repair_generate",
            "prepare_contract_evidence_retry": "prepare_contract_evidence_retry",
            "blocked": "safe_blocked_answer",
        },
    )
    graph.add_conditional_edges(
        "relevance_check",
        route_after_relevance,
        {
            "finalize": "finalize",
            "relevance_repair": "relevance_repair",
            "blocked": "safe_blocked_answer",
        },
    )
    graph.add_edge("relevance_repair", "contract_check")
    graph.add_edge("repair_generate", "contract_check")
    graph.add_edge("prepare_contract_evidence_retry", "execute_tools")
    graph.add_edge("safe_blocked_answer", "finalize")
    graph.add_edge("finalize", END)

    return graph


def compile_agent():
    """Build and compile the agent, ready to .invoke() / .stream()."""
    return build_graph().compile()


# Convenience singleton
agent = compile_agent()
