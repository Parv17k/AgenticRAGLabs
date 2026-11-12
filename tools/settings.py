"""Settings every demo shares, read from the environment and from demos/.env.

Point DEMO_MODEL and OPENAI_API_BASE at any OpenAI-compatible server. Local models have no list
price, so PRICE_AS names a model whose price stands in for the $ columns.
"""

from __future__ import annotations

import os
from pathlib import Path

import litellm

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path = ROOT / ".env") -> None:
    """Load KEY=value lines into os.environ, never overwriting a variable already set."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.split(" #")[0].strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip())


load_env()

MODEL = os.environ.get("DEMO_MODEL", "openai/gpt-4.1-mini")
REASONING_EFFORT = os.environ.get("DEMO_REASONING_EFFORT") or None
#: 0 makes a live demo repeat itself. Self-hosted servers often default to 1.0. "default" sends no
#: temperature at all, as Gemini 3 wants (it warns on every call that sets one).
_temperature = os.environ.get("DEMO_TEMPERATURE", "0")
TEMPERATURE = None if _temperature == "default" else float(_temperature)
#: temperature=0 for the small decisions (grading, judging), unless the model wants its default.
DETERMINISTIC = {} if TEMPERATURE is None else {"temperature": 0}
EMBED_MODEL = os.environ.get("DEMO_EMBED_MODEL", "openai/text-embedding-3-small")
RERANK_MODEL = os.environ.get("DEMO_RERANK_MODEL") or None
PRICE_AS = os.environ.get("PRICE_AS") or None
API_BASE = (os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1").rstrip("/")
#: Extra arguments for every direct chat call, e.g. turning thinking off.
LLM_KWARGS = {"max_tokens": 4096} | ({"temperature": TEMPERATURE} if TEMPERATURE is not None else {}) | (
    {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {})
#: The same model as LangChain's init_chat_model names it, for the LangGraph notebooks:
#: "openai:gpt-4.1-mini" (ChatOpenAI reads OPENAI_API_BASE and OPENAI_API_KEY itself), and any other
#: provider through LiteLLM, e.g. "litellm:gemini/gemini-3.5-flash", which keeps Gemini's thought
#: signatures on tool calls (without them Gemini 3 rejects the turn after a tool call).
CHAT_MODEL = MODEL.replace("/", ":", 1) if MODEL.startswith("openai/") else f"litellm:{MODEL}"
#: Some gateways (Cloudflare) refuse Python's default user agents.
HEADERS = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}", "User-Agent": "agentic-rag-demos/0.1"}

litellm.suppress_debug_info = True


def price(prompt_tokens: int, completion_tokens: int, model: str = MODEL) -> float | None:
    """$ for these tokens at the model's list price, else at PRICE_AS's; None if neither is known."""
    for name in (model, PRICE_AS):
        if not name:
            continue
        try:
            inp, out = litellm.cost_per_token(model=name, prompt_tokens=prompt_tokens,
                                              completion_tokens=completion_tokens)
            if inp or out:
                return inp + out
        except Exception:  # not in LiteLLM's price list
            continue
    return None
