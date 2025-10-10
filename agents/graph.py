"""
The agent graph: a declarative LangGraph wiring of five thin nodes.

Every node here is an adapter. Routing lives in ``agents/actions.py``,
retrieval in ``agents/retrieval.py``, composition in ``agents/answering.py``,
the grounding check in ``agents/grounding.py`` and the approval gate in
``agents/approval.py``. A node reads the state, calls one layer, writes the
state back. That is the whole job.

The split matters for two reasons. It makes each stage testable without
constructing a graph — most of this repository's tests never touch LangGraph
at all. And it makes the graph itself *readable*: the topology is the
``NODES``/``EDGES`` tables below, so "what does this agent do" is answered by
twenty lines of data rather than by tracing calls through a builder function.

    START ─▶ router ─┬─(knowledge)─▶ retrieve ─▶ compose ─▶ critic ─▶ END
                     └─(action)────▶ propose ─▶ END
                                        │
                                        └─▶ waits at the approval gate
                                            (agents/approval.py — not a node:
                                             a human decision is not a step in
                                             a function call)

The action path deliberately ends at a *proposal*. Execution is not a node,
because a node runs to completion inside ``graph.invoke`` and a human decision
does not. The proposal is registered with the gate, its id is returned on the
state, and the side effect happens later — if it happens at all.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents import actions, approval, grounding
from agents.answering import compose
from agents.retrieval import clamp_top_k, get_retriever

logger = logging.getLogger(__name__)


# ---- State ----
class AgentState(TypedDict, total=False):
    """What flows between nodes.

    Note what is *not* here: the pending approval. Only its id is, because the
    request outlives the graph invocation that created it.
    """

    query: str
    intent: str
    operator: str
    top_k: int
    style: str
    temperature: float
    contexts: list[dict]
    answer: str
    grounded: bool
    grounding: dict
    proposed_action: dict
    approval_id: str | None
    retrieval: dict
    answer_meta: dict
    log: list[str]


def _log(state: AgentState, message: str) -> list[str]:
    entries = list(state.get("log", []))
    entries.append(message)
    return entries


# ---- Nodes ----
def router_node(state: AgentState) -> AgentState:
    """Question or command?

    The trigger words come from the action registry, so registering an action
    teaches the router about it. They used to be a literal list here, which
    meant a new action was routable only if someone remembered to edit two
    files.
    """
    query = state.get("query") or ""
    intent = "action" if actions.looks_like_action(query) else "knowledge"
    state["intent"] = intent
    state["log"] = _log(state, f"Router → {intent.title()}")
    return state


def retriever_node(state: AgentState) -> AgentState:
    result = get_retriever().retrieve(state.get("query", ""), clamp_top_k(state.get("top_k", 4)))
    state["contexts"] = result.as_contexts()
    state["retrieval"] = {
        "count": len(result.passages),
        "embed_ms": round(result.embed_ms, 2),
        "query_ms": round(result.query_ms, 2),
        "error": result.error,
    }
    if result.error:
        state["log"] = _log(state, f"Retriever → Compose (retrieval failed: {result.error})")
    else:
        state["log"] = _log(
            state,
            f"Retriever → Compose ({len(result.passages)} passages, {result.total_ms:.0f}ms)",
        )
    return state


def compose_node(state: AgentState) -> AgentState:
    outcome = compose(
        state.get("query", ""),
        state.get("contexts", []),
        temperature=state.get("temperature", 0.2),
        style=state.get("style", "detailed"),
    )
    state["answer"] = outcome.text
    state["answer_meta"] = outcome.to_dict()
    state["log"] = _log(state, f"Compose → Critic (provider={outcome.provider}, calls={outcome.cost.calls})")
    return state


def critic_node(state: AgentState) -> AgentState:
    report = grounding.assess(state.get("answer", ""), state.get("contexts", []))
    state["grounded"] = report.grounded
    state["grounding"] = report.to_state()
    state["log"] = _log(
        state,
        f"Critic → END (grounded={report.grounded}, score={report.score:.2f})",
    )
    return state


def action_node(state: AgentState) -> AgentState:
    """Parse a command into a proposal and register it with the approval gate.

    Nothing is executed here, and nothing can be: the node has no dispatcher.
    An unrecognised command proposes ``unknown``, which the gate refuses to
    accept — guessing at an operational change is worse than declining one.
    """
    query = state.get("query", "")
    proposed = actions.propose(query)
    state["proposed_action"] = proposed.to_state()
    state["approval_id"] = None

    if not proposed.is_known:
        state["answer"] = (
            "That looks like a command, but it does not match any action this agent can "
            "propose. Registered actions: "
            + ", ".join(sorted(actions.ACTIONS))
            + "."
        )
        state["log"] = _log(state, "Action → END (no registered action matched)")
        return state

    try:
        request = approval.get_gate().submit(proposed, requested_by=state.get("operator"))
    except approval.ApprovalError as exc:
        state["answer"] = f"The action could not be proposed: {exc}"
        state["log"] = _log(state, f"Action → END (gate refused: {type(exc).__name__})")
        return state

    state["approval_id"] = request.id
    state["answer"] = (
        f"**Proposed action — awaiting your approval.**\n\n"
        f"{request.summary}\n\n"
        f"Risk: **{request.risk.value}**. Nothing has been sent to the ops API. "
        f"This proposal expires in {request.seconds_remaining():.0f} seconds if no decision is recorded."
    )
    state["log"] = _log(state, f"Action → Approval gate ({request.action_type}, id={request.id})")
    return state


# ---- Topology ----
@dataclass(frozen=True)
class NodeSpec:
    name: str
    run: Callable[[AgentState], AgentState]


#: The graph, as data. Adding a stage is a row here and a row in ``EDGES``.
NODES: tuple[NodeSpec, ...] = (
    NodeSpec("router", router_node),
    NodeSpec("retriever", retriever_node),
    NodeSpec("compose", compose_node),
    NodeSpec("critic", critic_node),
    NodeSpec("action", action_node),
)

#: Unconditional edges.
EDGES: tuple[tuple[str, str], ...] = (
    (START, "router"),
    ("retriever", "compose"),
    ("compose", "critic"),
    ("critic", END),
    ("action", END),
)

#: Where the router sends each intent.
INTENT_ROUTES: dict[str, str] = {
    "knowledge": "retriever",
    "action": "action",
}


def route_selector(state: AgentState) -> str:
    return state.get("intent", "knowledge")


def build_graph(nodes: Sequence[NodeSpec] = NODES):
    """Compile the graph from the tables above."""
    builder = StateGraph(AgentState)
    for node in nodes:
        builder.add_node(node.name, node.run)
    for source, target in EDGES:
        builder.add_edge(source, target)
    builder.add_conditional_edges("router", route_selector, INTENT_ROUTES)
    return builder.compile()


_graph: Any | None = None


def get_graph():
    """The compiled graph, built once."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph() -> None:
    """Drop the compiled graph so the next call rebuilds it. Used by tests."""
    global _graph
    _graph = None


def invoke_graph(
    query: str,
    top_k: int = 4,
    style: str | None = None,
    temperature: float = 0.2,
    operator: str | None = None,
) -> AgentState:
    """Run one turn."""
    initial: AgentState = {"query": query, "top_k": clamp_top_k(top_k), "log": [], "temperature": temperature}
    if style:
        initial["style"] = style
    if operator:
        initial["operator"] = operator
    return get_graph().invoke(initial)  # type: ignore[return-value]
