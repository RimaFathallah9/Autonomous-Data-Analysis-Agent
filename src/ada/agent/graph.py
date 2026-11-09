"""The agent as a LangGraph state machine.

    START -> plan --(model asked for tools & budget left)--> act -> plan ...
                  \\--(no tool calls OR budget exhausted)--> finalize -> END

WHY LangGraph instead of a `while` loop? Explicit nodes/edges make the control flow
inspectable and testable, give you a place to add checkpoints, human-in-the-loop
or branching later, and every transition is a clear unit to log. The step budget
(`max_steps`) is a hard guard so a confused model cannot loop forever or burn money.
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from ada.agent.llm import LLMClient
from ada.agent.prompts import SYSTEM_PROMPT
from ada.agent.tools import Session, call_tool, openai_tool_schemas


class AgentState(TypedDict):
    # `operator.add` reducer: nodes return ONLY new messages, LangGraph appends them.
    messages: Annotated[list[dict], operator.add]
    steps: int
    stop_reason: str


def build_graph(client: LLMClient, session: Session, max_steps: int = 12):
    schemas = openai_tool_schemas()

    def plan(state: AgentState) -> dict:
        resp = client.chat(state["messages"], schemas)
        msg: dict = {"role": "assistant", "content": resp.content}
        if resp.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in resp.tool_calls
            ]
        return {"messages": [msg]}

    def act(state: AgentState) -> dict:
        last = state["messages"][-1]
        out = []
        for tc in last["tool_calls"]:
            result = call_tool(session, tc["function"]["name"], tc["function"]["arguments"])
            out.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(result, default=str)[:4000]})
        return {"messages": out, "steps": state["steps"] + 1}

    def route(state: AgentState) -> str:
        wants_tools = bool(state["messages"][-1].get("tool_calls"))
        if wants_tools and state["steps"] < max_steps:
            return "act"
        return "finalize"

    def finalize(state: AgentState) -> dict:
        wants_tools = bool(state["messages"][-1].get("tool_calls"))
        return {"stop_reason": "max_steps" if wants_tools else "model_finished"}

    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("finalize", finalize)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route, {"act": "act", "finalize": "finalize"})
    g.add_edge("act", "plan")
    g.add_edge("finalize", END)
    return g.compile()


@dataclass
class AgentResult:
    final_message: str | None
    trace: list[dict]  # full message log: prompts, tool calls, tool results
    session: Session
    steps: int
    stop_reason: str


def run_agent(task: str, session: Session, client: LLMClient, max_steps: int = 12) -> AgentResult:
    graph = build_graph(client, session, max_steps)
    init: AgentState = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        "steps": 0,
        "stop_reason": "",
    }
    final = graph.invoke(init, {"recursion_limit": 4 * max_steps + 10})
    return AgentResult(
        final_message=final["messages"][-1].get("content"),
        trace=final["messages"],
        session=session,
        steps=final["steps"],
        stop_reason=final["stop_reason"],
    )
