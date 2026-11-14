"""Module 3 · a LangGraph ReAct agent with search_docs (Module 3,).

    START → agent ─ tool call ─→ tools ─┐
              ↑ │
              └─────────────────────────┘
              └─ no tool call → END

v2_langgraph_agent.ipynb builds this graph cell by cell; keep the two in step.
run_all.ipynb and eval.py import it from here.
"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from tools.search_docs import format_hits
from tools.search_docs import search_docs as search_index
from tools.settings import CHAT_KWARGS, CHAT_MODEL


@tool
def search_docs(query: str) -> str:
    """Search Acme filings for facts and numbers."""
    return format_hits(search_index(query, k=5))


SYSTEM = SystemMessage(content=(
    "You answer questions about Acme Corp's Q3 2026 quarterly filing (10-Q); Q2 2026 results are in "
    "an appendix. Use search_docs for Acme facts and numbers, one quarter or topic per search: to "
    "compare Q3 with Q2, call it once for Q3 and once for Q2. Don't search for greetings or thanks. "
    "Answer only from what the searches return, and be brief."))


def build_v2(model=None, checkpointer=None):
    """The ReAct graph, compiled. A checkpointer gives it memory per thread_id."""
    model = model or init_chat_model(CHAT_MODEL, **CHAT_KWARGS)
    tools = [search_docs]
    llm = model.bind_tools(tools)

    def call_llm(state: MessagesState):
        return {"messages": [llm.invoke([SYSTEM] + state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", call_llm)
    g.add_node("tools", ToolNode(tools))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    app = build_v2(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "acme-1"}, "recursion_limit": 10}
    for q in ["Thanks!", "How did Acme's Q3 margin compare to Q2, and why?"]:
        out = app.invoke({"messages": [("user", q)]}, cfg)
        print(f"› {q}\n{out['messages'][-1].content}\n")
