"""Agentic RAG as an AgentShip custom agent (slides 48–51), used by v3, v4 and v5.

    question → agent ─ no tool call ──────────────────────────→ verify → END
                 │                                                 │
                 └─ search_docs → retrieve → grade ─ relevant ─→ agent (answers from what's left)
                                               └─ nothing relevant → rewrite → agent (searches again)

agent    the LLM with search_docs bound. It decides whether to search, and how many times.
grade    one call marks which retrieved chunks are relevant; the rest are dropped from context.
rewrite  when nothing was relevant: write a better query, count a retry (max_retries), search again.
verify   is every claim in the answer in the evidence? If not, answer once more or say so.

Every loop is bounded: retries and search rounds are counted per turn, in state.
The versions differ only in configuration (bottom of this file):

    v3  vector search over fixed chunks (tools/search_docs.py)
    v4  OpenSearch hybrid + RRF + rerank (tools/opensearch_search.py)
    v5  v4, tuned: rerank to 3, one retry, no thinking for small decisions, trimmed history
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated

from agentship.spec import AgentSpec, ModelParams, ObservabilitySpec
from agentship_langgraph import LangGraphAgent
from agentship_langgraph import models
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tools.settings import MODEL
from typing_extensions import TypedDict

PROMPT = """You are an analyst answering questions about Acme Corp's quarterly filing (Form 10-Q, Q3 2026).
Use search_docs for any Acme-specific fact or number. Search once per quarter or topic you need, and
when the searches are independent, request them together in one turn. Do not search for greetings,
thanks, or general definitions; answer those directly and briefly.
Answer only from the search results, cite the chunk ids you used like [chunk 36], and say you don't
know if the results do not contain the answer. Be concise."""

OUT_OF_ROUNDS = ("You have used every search allowed for this question. Do not search again: answer now "
                 "from the search results above, or say the filing does not answer it.")
CHUNK_SEPARATOR = "\n\n---\n\n"
MAX_TOKENS = 4096  # per call: a short answer needs far less, and a looping model stops here


@dataclass(frozen=True)
class Config:
    max_retries: int = 2  # rewrite + search again when the grader finds nothing relevant
    max_search_rounds: int = 4  # hard cap on tool rounds per turn
    verify: bool = True  # check the answer against the evidence before returning it
    history_turns: int | None = None  # keep only the last N turns of the thread in the prompt
    small_effort: str | None = None  # reasoning effort for grade / rewrite / verify


class RAGState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    retries: int
    relevant: int
    hint: str | None
    verified: bool


def _turn_start(messages: list) -> int:
    """Index of the latest user message: everything after it belongs to this turn."""
    return max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))


def _text(message) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def _ask(model, prompt: str) -> str:
    return _text(model.invoke([HumanMessage(content=prompt)])).strip()


class AgenticRAG(LangGraphAgent):
    """The graph above, over AgentShip's wired model and tools."""

    #: Nodes whose model output is bookkeeping, kept out of the streamed answer.
    internal_nodes = frozenset({"grade", "rewrite", "verify"})

    def __init__(self, spec: AgentSpec, config: Config) -> None:
        super().__init__(spec)
        self.config = config

    def build_graph(self, model, tools) -> StateGraph:
        cfg = self.config
        small = models.resolve_model(MODEL, reasoning_effort=cfg.small_effort, max_tokens=MAX_TOKENS,
                                     streaming=False)
        with_tools = model.bind_tools(tools)

        def agent(state: RAGState) -> dict:
            messages = state["messages"]
            start = _turn_start(messages)
            new_turn = start == len(messages) - 1
            rounds = sum(1 for m in messages[start:] if isinstance(m, AIMessage) and m.tool_calls)

            system = [m for m in messages[:1] if isinstance(m, SystemMessage)]
            history = [m for m in messages[len(system):] if not isinstance(m, RemoveMessage)]
            if cfg.history_turns is not None:  # keep the last N turns: from the Nth-last user message on
                humans = [i for i, m in enumerate(history) if isinstance(m, HumanMessage)]
                history = history[humans[-cfg.history_turns]:] if len(humans) > cfg.history_turns else history
            hint = None if new_turn else state.get("hint")
            out_of_rounds = rounds >= cfg.max_search_rounds
            if out_of_rounds:  # no tools bound now, so say why, or the model writes a search as text
                hint = (hint + "\n\n" if hint else "") + OUT_OF_ROUNDS
            if hint:  # one-off guidance for this call; never stored in the thread
                system = [SystemMessage(content=(system[0].content if system else PROMPT) + "\n\n" + hint)]

            reply = (model if out_of_rounds else with_tools).invoke(system + history)
            if not reply.tool_calls and "<tool_call>" in _text(reply):  # a search written out as text
                answer = _text(reply).split("<tool_call>")[0].strip()
                reply = AIMessage(content=answer or "The filing's search results don't answer this.", id=reply.id)
            update = {"messages": [reply], "hint": None}
            if new_turn:
                update |= {"retries": 0, "relevant": 0, "verified": False}
            return update

        def grade(state: RAGState) -> dict:
            messages = state["messages"]
            question = _text(messages[_turn_start(messages)])
            results = []
            for m in reversed(messages):
                if not isinstance(m, ToolMessage):
                    break
                results.insert(0, m)
            passages = [(m, p) for m in results for p in _text(m).split(CHUNK_SEPARATOR) if p.strip()]
            if not passages:
                return {"relevant": 0}
            listing = "\n\n".join(f"<passage {n}>\n{p}\n</passage {n}>" for n, (_, p) in enumerate(passages, 1))
            verdict = _ask(small, f"Question: {question}\n\n{listing}\n\nWhich passages contain information "
                                  f"that helps answer the question? Reply with JSON only, like "
                                  f'{{"relevant": [1, 3]}}, or {{"relevant": []}} if none do.')
            try:
                keep = {int(n) for n in json.loads(re.search(r"\{.*\}", verdict, re.S).group())["relevant"]}
            except Exception:  # unparseable verdict: keep everything rather than lose evidence
                keep = set(range(1, len(passages) + 1))
            kept = {}
            for n, (m, p) in enumerate(passages, 1):
                if n in keep:
                    kept.setdefault(m.id, []).append(p)
            graded = [ToolMessage(content=CHUNK_SEPARATOR.join(kept[m.id]) if m.id in kept
                                  else "No relevant passages for this search.",
                                  tool_call_id=m.tool_call_id, name=m.name, id=m.id) for m in results]
            return {"messages": graded, "relevant": len([n for n in keep if 1 <= n <= len(passages)])}

        def rewrite(state: RAGState) -> dict:
            messages = state["messages"]
            start = _turn_start(messages)
            question = _text(messages[start])
            last_call = next(m for m in reversed(messages) if isinstance(m, AIMessage) and m.tool_calls)
            queries = [c["args"].get("query", "") for c in last_call.tool_calls]
            better = _ask(small, f"Question: {question}\nThese searches of a company's 10-Q filing found nothing "
                                 f"relevant: {queries}\nWrite one better search query, using the words a 10-Q "
                                 f"would use. Reply with the query only.")
            hint = (f"Your last search ({', '.join(queries)}) found nothing relevant. "
                    f"Search again, for example with: {better.splitlines()[0][:200]}")
            return {"hint": hint, "retries": state.get("retries", 0) + 1}

        def verify(state: RAGState) -> dict:
            messages = state["messages"]
            start = _turn_start(messages)
            answer = messages[-1]
            evidence = "\n\n".join(_text(m) for m in messages[start:] if isinstance(m, ToolMessage))
            verdict = _ask(small, f"EVIDENCE:\n{evidence}\n\nANSWER:\n{_text(answer)}\n\nIs every number and "
                                  f"factual claim in ANSWER supported by EVIDENCE? Reply with yes or no only.")
            if verdict.lower().startswith("yes"):
                return {"verified": True}
            return {"verified": True, "messages": [RemoveMessage(id=answer.id)],
                    "hint": "Your previous answer had claims that are not in the search results. Answer "
                            "again using only the search results, or say you don't know."}

        def after_agent(state: RAGState) -> str:
            messages = state["messages"]
            if messages[-1].tool_calls:
                return "retrieve"
            searched = any(isinstance(m, ToolMessage) for m in messages[_turn_start(messages):])
            return "verify" if cfg.verify and searched and not state.get("verified") else END

        def after_grade(state: RAGState) -> str:
            return "rewrite" if not state.get("relevant") and state.get("retries", 0) < cfg.max_retries else "agent"

        g = StateGraph(RAGState)
        g.add_node("agent", agent)
        g.add_node("retrieve", ToolNode(tools))
        g.add_node("grade", grade)
        g.add_node("rewrite", rewrite)
        g.add_node("verify", verify)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", after_agent, ["retrieve", "verify", END])
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", after_grade, ["rewrite", "agent"])
        g.add_edge("rewrite", "agent")
        g.add_conditional_edges("verify", lambda s: "agent" if s.get("hint") else END, ["agent", END])
        return g


def _agent(name: str, tool: str, config: Config, effort: str | None) -> AgenticRAG:
    return AgenticRAG(AgentSpec(
        name=name, engine="langgraph", model=MODEL, prompt=PROMPT, tools=[tool],
        params=ModelParams(reasoning_effort=effort, max_tokens=MAX_TOKENS),
        durability="checkpoint", observability=ObservabilitySpec(),
    ), config)


def build_v3() -> AgenticRAG:
    """v3: agentic RAG over v1's index. The model thinks a little on every call."""
    return _agent("agentic-rag", "search_docs", Config(small_effort="low"), "low")


def build_v4() -> AgenticRAG:
    """v4: the same graph, with OpenSearch hybrid search and a reranker behind search_docs."""
    return _agent("agentic-rag-v4", "search_docs_hybrid", Config(small_effort="low"), "low")


def build_v5() -> AgenticRAG:
    """v5: v4 tuned. Rerank to 3, one retry, no thinking on small decisions, the last 3 turns only."""
    return _agent("researcher-v5", "search_docs_hybrid_top3",
                  Config(max_retries=1, history_turns=3, small_effort="none"), "low")
