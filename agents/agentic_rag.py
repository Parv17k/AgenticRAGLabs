"""Module 4 · agentic RAG (Module 4), and the same graph served on AgentShip (Module 5).

    question → agent ─ no tool call ──────────────────────────→ verify → END
                 └─ search_docs → retrieve → grade ─ relevant ─────────→ agent (answers)
                                               ├─ weak, retries < 2 ──→ rewrite → agent (searches again)
                                               └─ weak, out of retries → give_up → END

build_v3_graph(model, tools) is the Module 4 graph. module_4_search_with_judgment.ipynb builds
the same graph cell by cell; keep the two in step. AgenticRAG wraps it, unchanged, for AgentShip
:

    agentship run agents/agentic_rag.yaml --input "How did Acme's Q3 margin compare to Q2, and why?"
    agentship serve --agents-dir agents
"""

from __future__ import annotations

import json
import re

import httpx

from agentship.spec import AgentSpec, ModelParams, ObservabilitySpec
from agentship_langgraph import LangGraphAgent
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from tools.settings import API_BASE, DETERMINISTIC, HEADERS, MODEL, REASONING_EFFORT, RERANK_MODEL, TEMPERATURE

PROMPT = """You answer questions about Acme Corp's quarterly filing (Form 10-Q for Q3 2026).
Use search_docs for any Acme-specific fact or number, once per quarter or topic you need: to
compare two quarters, search for each. Answer greetings, thanks and general definitions directly,
without searching. Answer only from the search results, cite chunk ids like [chunk 36], and say
so when the filings don't contain the answer. Be concise."""

MAX_RETRIES = 2 # rewrites per question
MAX_SEARCHES = 4 # search rounds per question, a second cap so the loop always ends
SEPARATOR = "\n\n---\n\n" # between chunks in a search_docs result


class RAGState(MessagesState):
    retries: int # rewrites so far this question: the counter that ends the loop
    relevant: int # chunks the grader kept from the last search
    note: str | None # a one-off instruction for the agent's next call, from verify
    query: str | None # the rewritten query the agent must search with next
    verified: bool


def _text(message) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def _this_turn(messages: list) -> list:
    """Everything from the latest user message on."""
    start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return messages[start:]


def relevant(query: str, passages: list[str], ask) -> list[bool]:
    """The grader: which passages state what the search looked for?

    A cross-encoder (DEMO_RERANK_MODEL) reads each query–passage pair and scores it: above 0 means
    relevant. It is small, fast and steadier than asking the chat model. Without one, the chat model
    picks the passages instead.
    """
    if RERANK_MODEL:
        resp = httpx.post(f"{API_BASE}/reranking", headers=HEADERS, timeout=60,
                          json={"model": RERANK_MODEL, "query": query, "documents": passages})
        resp.raise_for_status()
        scores = {r["index"]: r["relevance_score"] for r in resp.json()["results"]}
        return [scores[i] > 0 for i in range(len(passages))]
    listing = "\n\n".join(f"<passage {n}>\n{p}\n</passage {n}>" for n, p in enumerate(passages, 1))
    verdict = ask(f"Search: {query}\n\n{listing}\n\nWhich passages state what the search looks for? "
                  f'Reply with JSON only, like {{"relevant": [1, 3]}}, or {{"relevant": []}}.')
    try:
        keep = {int(n) for n in json.loads(re.search(r"\{.*\}", verdict, re.S).group())["relevant"]}
    except Exception: # an unreadable verdict: keep everything rather than lose evidence
        keep = set(range(1, len(passages) + 1))
    return [n in keep for n in range(1, len(passages) + 1)]


def build_v3_graph(model, tools, verify: bool = True) -> StateGraph:
    """The Module 4 graph over any chat model and the search_docs tool."""
    llm = model.bind_tools(tools)
    judge = model.bind(**DETERMINISTIC) # the small decisions should not be a coin toss

    def ask(prompt: str) -> str: # the small decisions: grade, rewrite, verify
        return _text(judge.invoke([HumanMessage(content=prompt)])).strip()

    def agent(state: RAGState) -> dict:
        """LLM + search_docs: decides whether to search, what for, and when to answer."""
        messages = state["messages"]
        new_turn = isinstance(messages[-1], HumanMessage)
        note = None if new_turn else state.get("note")
        searches = sum(1 for m in _this_turn(messages) if isinstance(m, AIMessage) and m.tool_calls)
        system = PROMPT + (f"\n\n{note}" if note else "")
        history = [m for m in messages if not isinstance(m, SystemMessage)]
        if state.get("query") and not new_turn and searches < MAX_SEARCHES: # after a rewrite
            call = {"name": "search_docs", "args": {"query": state["query"]}, "id": f"rewrite-{state['retries']}"}
            reply = AIMessage(content="", tool_calls=[call])
        elif searches < MAX_SEARCHES:
            reply = llm.invoke([SystemMessage(content=system)] + history)
        else: # out of searches: hand over the results as plain text, with no tools and no tool calls in sight
            turn = _this_turn(history)
            earlier = [m for m in history[:len(history) - len(turn)] # past questions and final answers only
                       if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)]
            results = SEPARATOR.join(_text(m) for m in turn if isinstance(m, ToolMessage))
            system += "\n\nYou have used every search for this question. Answer from the search results given."
            ask_once = HumanMessage(content=f"{_text(turn[0])}\n\nSearch results:\n\n{results}")
            reply = model.invoke([SystemMessage(content=system)] + earlier + [ask_once])
            if reply.tool_calls or "<tool_call>" in _text(reply): # a search anyway, as a call or as text
                answer = _text(reply).split("<tool_call>")[0].strip()
                reply = AIMessage(content=answer or "The filings' search results don't answer this.")
        update = {"messages": [reply], "note": None, "query": None}
        if new_turn:
            update |= {"retries": 0, "relevant": 0, "verified": False}
        return update

    def grade(state: RAGState) -> dict:
        """Small model: which chunks state what each search looked for? Drop the rest."""
        messages = state["messages"]
        results = []
        for m in reversed(messages): # the ToolMessages this search just added
            if not isinstance(m, ToolMessage):
                break
            results.insert(0, m)
        call = next(m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls)
        queries = {c["id"]: c["args"].get("query", "") for c in call.tool_calls}
        graded, n_relevant = [], 0
        for m in results: # each search is graded against its own query
            passages = [p for p in _text(m).split(SEPARATOR) if p.strip()]
            kept = [p for p, ok in zip(passages, relevant(queries.get(m.tool_call_id, ""), passages, ask)) if ok]
            n_relevant += len(kept)
            graded.append(ToolMessage(content=SEPARATOR.join(kept) or "No relevant passages.",
                                      tool_call_id=m.tool_call_id, name=m.name, id=m.id))
        return {"messages": graded, "relevant": n_relevant}

    def rewrite(state: RAGState) -> dict:
        """Nothing relevant: write a better query, count one retry."""
        messages = state["messages"]
        question = _text(_this_turn(messages)[0])
        last = next(m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls)
        queries = [c["args"].get("query", "") for c in last.tool_calls]
        better = ask(f"Question: {question}\nThese searches of a company's 10-Q filing found nothing "
                     f"relevant: {queries}\nWrite one better search query, in the words a 10-Q would "
                     f"use. Reply with the query only.").splitlines()[0][:200]
        return {"retries": state.get("retries", 0) + 1, "query": better}

    def give_up(state: RAGState) -> dict:
        """Out of retries: say so instead of guessing."""
        return {"messages": [AIMessage(content="I couldn't find that in Acme's filings, even after "
                                               "rewriting the search.")]}

    def check(state: RAGState) -> dict:
        """Verify: is every claim in the answer in the evidence? If not, answer once more."""
        messages = state["messages"]
        evidence = "\n\n".join(_text(m) for m in _this_turn(messages) if isinstance(m, ToolMessage))
        if state.get("verified") or not evidence: # already checked, or nothing was searched
            return {}
        answer = messages[-1]
        verdict = ask(f"EVIDENCE:\n{evidence}\n\nANSWER:\n{_text(answer)}\n\nIs every number and factual "
                      f"claim in ANSWER supported by EVIDENCE? Saying the evidence doesn't contain something counts as "
                      f"supported. Reply with yes or no only.")
        if verdict.strip("*_ .").lower().startswith("yes"):
            return {"verified": True}
        return {"verified": True, "messages": [RemoveMessage(id=answer.id)],
                "note": "Your last answer had claims that are not in the search results. Answer again "
                        "using only the search results, or say the filings don't contain it."}

    def after_grade(state: RAGState) -> str:
        if state["relevant"]:
            return "agent" # good: answer
        if state.get("retries", 0) < MAX_RETRIES:
            return "rewrite" # weak: try again
        return "give_up" # out of retries

    g = StateGraph(RAGState)
    g.add_node("agent", agent) # LLM + search_docs
    g.add_node("retrieve", ToolNode(tools))
    g.add_node("grade", grade) # small model: relevant?
    g.add_node("rewrite", rewrite) # better query, retries+1
    g.add_node("give_up", give_up)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition, {"tools": "retrieve", END: "verify" if verify else END})
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, ["agent", "rewrite", "give_up"])
    g.add_edge("rewrite", "agent")
    g.add_edge("give_up", END)
    if verify: # the last step, before the final answer
        g.add_node("verify", check)
        g.add_conditional_edges("verify", lambda s: "agent" if s.get("note") else END, ["agent", END])
    return g


# ---------------------------------------------------------------- Module 5: ship it

class AgenticRAG(LangGraphAgent):
    """The Module 4 graph as an AgentShip agent: an API, sessions, traces."""

    #: Nodes whose model output is bookkeeping, kept out of the streamed answer.
    internal_nodes = frozenset({"grade", "rewrite", "verify"})

    def build_graph(self, model, tools):
        # the Module 4 graph, unchanged
        return build_v3_graph(model, tools)


def build_agent() -> AgenticRAG:
    return AgenticRAG(AgentSpec(
        name="agentic-rag", engine="langgraph", model=MODEL, prompt=PROMPT,
        tools=["search_docs"], # AgentShip reads the spec from here when the YAML sets code:
        params=ModelParams(temperature=TEMPERATURE, reasoning_effort=REASONING_EFFORT, max_tokens=4096),
        durability="checkpoint", observability=ObservabilitySpec(),
    ))
