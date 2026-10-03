r"""LangGraph wiring.

recall -> planner -> approve_plan -(HITL)-> supervisor <-> {researcher, analyst}
                                              |
                                           writer -> critic -> (pass | max loops) -> finalize
                                                        \-> (fail) -> supervisor  (reflection loop)
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from sentinel.agents.critic import critic, route_after_critic
from sentinel.agents.memory_nodes import approve_plan, finalize, recall, rejected, route_after_approval
from sentinel.agents.planner import planner
from sentinel.agents.supervisor import route_after_supervisor, supervisor
from sentinel.agents.workers import analyst, researcher
from sentinel.agents.writer import writer
from sentinel.memory.checkpointer import get_checkpointer
from sentinel.state import AgentState


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    for name, fn in [
        ("recall", recall), ("planner", planner), ("approve_plan", approve_plan),
        ("supervisor", supervisor), ("researcher", researcher), ("analyst", analyst),
        ("writer", writer), ("critic", critic), ("finalize", finalize), ("rejected", rejected),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("recall")
    g.add_edge("recall", "planner")
    g.add_edge("planner", "approve_plan")
    g.add_conditional_edges("approve_plan", route_after_approval, {"supervisor": "supervisor", "rejected": "rejected"})
    g.add_conditional_edges(
        "supervisor", route_after_supervisor,
        {"researcher": "researcher", "analyst": "analyst", "writer": "writer"},
    )
    g.add_edge("researcher", "supervisor")
    g.add_edge("analyst", "supervisor")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"finalize": "finalize", "supervisor": "supervisor"})
    g.add_edge("finalize", END)
    g.add_edge("rejected", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else get_checkpointer())
