"""search_docs as AgentShip Tools, one per retrieval backend.

pyproject.toml registers them under the agentship.tools entry point, so an agent YAML lists one
by name, like the built-in calculator. Every variant shows the model the same name and
description: the agent cannot tell, and does not need to know, what runs behind it.

    tools: [search_docs] v1 index: fixed chunks, vectors only, top 5
    tools: [search_docs_hybrid] v4: OpenSearch hybrid + RRF + rerank, top 5
    tools: [search_docs_hybrid_top3] v5: the same, reranked down to 3
"""

from __future__ import annotations

from collections.abc import Callable

from agentship.tools.tool import Tool
from pydantic import BaseModel, Field

from . import opensearch_search
from . import search_docs as basic

#: The description IS the routing policy: when to search, and when not to.
DESCRIPTION = (
    "Search Acme's internal filings (Form 10-Q for Q3 2026, with Q2 2026 results in an appendix). "
    "Use for Acme-specific numbers and facts. One topic or quarter per search: to compare two "
    "quarters, call it once for each. Do NOT use for definitions, small talk, or live prices."
)


class SearchArgs(BaseModel):
    query: str = Field(description="What to look for, e.g. 'Q2 2026 operating margin'.")


def _tool(search: Callable[[str, int], list[dict]], k: int) -> Tool:
    def run(query: str) -> str:
        return basic.format_hits(search(query, k))

    return Tool(name="search_docs", description=DESCRIPTION, func=run, args_schema=SearchArgs)


search_docs = _tool(basic.search_docs, 5)
search_docs_hybrid = _tool(opensearch_search.search_docs, 5)
search_docs_hybrid_top3 = _tool(opensearch_search.search_docs, 3)
