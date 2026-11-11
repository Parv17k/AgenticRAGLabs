"""eval.py: run versions of the Acme assistant on the golden set and score them side by side.

Retrieval and generation are scored separately, as on the evaluation slides:

    routing    retrieved when it should, skipped when it shouldn't
    recall     share of gold facts found in the retrieved chunks (top 5 per search)
    MRR        1 / rank of the first retrieved chunk that holds a gold fact
               (an agent that searches twice is scored on all the chunks it got back)
    faithful   share of the answer's claims supported by what the model was given (LLM judge)
    correct    answer matches the reference: 1, 0.5 or 0 (LLM judge)

plus input tokens, $ per query and p95 latency. The judge's own cost is reported separately.

    python eval.py --agents v0,v1                              # the whole golden set
    python eval.py --agents v1 --ids q3_vs_q2_revenue,thanks   # a few questions
    python eval.py --agents v0,v1 "How did Acme's Q3 margin compare to Q2, and why?"

v2 onwards run the AgentShip agents in agents/ (install with `pip install -e .`, Python 3.13+).
golden.json has 14 questions; a real golden set
has 200 to 500, with gold chunks labelled by people.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import litellm
import numpy as np
import pandas as pd
from litellm.integrations.custom_logger import CustomLogger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tools import settings  # noqa: E402  (loads .env)
from tools.search_docs import CALLS, DOC, format_hits, load_index, search_docs  # noqa: E402

MODEL = settings.MODEL
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", MODEL)
K = 5
#: The judge reads at most this many tokens of context. v0's context is the whole filing; for it the
#: judge reads the chunks nearest to the answer instead, so judging does not cost another v0 call.
JUDGE_CONTEXT_TOKENS = 12_000


@dataclass
class Run:
    answer: str
    context: str  # everything the model was given to answer from
    chunks: list[str] | None  # retrieved chunks in rank order; None when there is no retrieval step
    searches: list[str] = field(default_factory=list)
    retrieved: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    latency: float = 0.0


def complete(messages: list[dict], model: str = MODEL, **kwargs):
    """One chat call, streamed (gateways such as Cloudflare cut a silent request after 100s)."""
    start = time.perf_counter()
    chunks = list(litellm.completion(model=model, messages=messages, stream=True,
                                     stream_options={"include_usage": True}, **settings.LLM_KWARGS | kwargs))
    resp = litellm.stream_chunk_builder(chunks, messages=messages)
    cost = settings.price(resp.usage.prompt_tokens, resp.usage.completion_tokens, model)
    return resp, cost, time.perf_counter() - start


# ---------------------------------------------------------------- versions

def v0(question: str) -> Run:
    """The whole filing in every prompt."""
    doc = DOC.read_text()
    resp, cost, latency = complete([
        {"role": "system", "content": "You answer questions about the attached quarterly filing. Be brief."},
        {"role": "user", "content": f"<document name=\"Acme 10-Q\">\n{doc}\n</document>\n\n{question}"},
    ])
    return Run(resp.choices[0].message.content, doc, None, retrieved=True,
               prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens,
               cost=cost, latency=latency)


V1_PROMPT = """Answer using only this context. If the context does not contain the answer, say so.

{context}

Q: {question}"""


def v1(question: str) -> Run:
    """Basic RAG: top-5 chunks for the question as typed, on every message."""
    start = time.perf_counter()
    hits = search_docs(question, k=K)
    context = format_hits(hits)
    resp, cost, _ = complete([{"role": "user", "content": V1_PROMPT.format(context=context, question=question)}])
    return Run(resp.choices[0].message.content, context, [h["text"] for h in hits], [question], retrieved=True,
               prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens,
               cost=cost, latency=time.perf_counter() - start)


class Meter(CustomLogger):
    """Adds up tokens and $ over every model call LiteLLM makes while it is switched on."""

    def __init__(self) -> None:
        super().__init__()
        self.on = False
        self.reset()

    def reset(self) -> None:
        self.prompt_tokens = self.completion_tokens = 0
        self.cost = 0.0
        self._seen: set[str] = set()

    def _record(self, kwargs: dict, response) -> None:
        call_id = kwargs.get("litellm_call_id") or str(id(response))
        if not self.on or call_id in self._seen:  # LiteLLM may report one call to both hooks
            return
        self._seen.add(call_id)
        usage = getattr(response, "usage", None)
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        self.cost += kwargs.get("response_cost") or settings.price(
            getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0) or 0.0

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self._record(kwargs, response_obj)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self._record(kwargs, response_obj)


METER = Meter()
litellm.callbacks.append(METER)


@lru_cache(maxsize=None)
def agent(spec: str):
    from agentship import build_agent  # only v2 onwards need AgentShip installed

    os.chdir(HERE)  # code: references in the specs are relative to demos/
    return build_agent(str(HERE / spec))


def agentship_run(spec: str, question: str) -> Run:
    """One turn of an AgentShip agent, with its searches, tokens and cost collected."""
    CALLS.clear()
    METER.reset()
    METER.on = True
    start = time.perf_counter()
    try:
        result = asyncio.run(agent(spec).run(question))
    finally:
        METER.on = False
    hits = list({h["id"]: h for call in CALLS for h in call["hits"]}.values())  # first-seen order, no repeats
    return Run(str(result.output), format_hits(hits), [h["text"] for h in hits], [c["query"] for c in CALLS],
               retrieved=bool(CALLS), prompt_tokens=METER.prompt_tokens,
               completion_tokens=METER.completion_tokens, cost=METER.cost, latency=time.perf_counter() - start)


def v2(question: str) -> Run:
    """An agent team: a coordinator routes to a researcher that calls search_docs when it decides to."""
    return agentship_run("agents/team.yaml", question)


def v3(question: str) -> Run:
    """Agentic RAG: route, retrieve, grade, rewrite, verify, over v1's index."""
    return agentship_run("agents/agentic_rag.yaml", question)


def v4(question: str) -> Run:
    """v3's graph with OpenSearch hybrid search + RRF + rerank behind search_docs."""
    return agentship_run("agents/agentic_rag_v4.yaml", question)


def v5(question: str) -> Run:
    """v4 tuned by configuration: a team router, rerank to 3, fewer retries, less thinking."""
    return agentship_run("agents_v5/team.yaml", question)


VERSIONS = {"v0": v0, "v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5}


# ---------------------------------------------------------------- scoring

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def holds(text: str, fact: str | list[str]) -> bool:
    return any(norm(alt) in norm(text) for alt in ([fact] if isinstance(fact, str) else fact))


def retrieval_scores(item: dict, run: Run) -> tuple[float | None, float | None]:
    """recall@k and MRR against the gold facts; None when not applicable."""
    if run.chunks is None or not item["gold"]:
        return None, None
    recall = sum(any(holds(c, f) for c in run.chunks) for f in item["gold"]) / len(item["gold"])
    ranks = [r for r, c in enumerate(run.chunks, 1) if any(holds(c, f) for f in item["gold"])]
    return recall, (1 / ranks[0] if ranks else 0.0)


JUDGE_PROMPT = """You grade an assistant's answer about a company's quarterly filing.

CONTEXT (everything the assistant was given):
{context}

QUESTION: {question}
REFERENCE ANSWER: {reference}
ANSWER: {answer}

1. List each factual claim in ANSWER and say whether CONTEXT supports it.
2. Score ANSWER against REFERENCE: 1 if it gives the same key facts, 0.5 if partly, 0 if wrong or missing.
   Saying the information is unavailable when REFERENCE has it scores 0.

Return JSON only: {{"claims": [{{"claim": "...", "supported": true}}], "correct": 1, "why": "one short sentence"}}"""


def judge(item: dict, run: Run) -> tuple[float | None, float | None, str, float]:
    """faithfulness, correctness (None without a reference), the judge's reason, and the judge's cost."""
    if not run.answer.strip():  # nothing to judge, and a judge may still call it correct
        return None, (0.0 if item.get("reference") else None), "empty answer", 0.0
    context = run.context or "(nothing)"
    if len(context) > JUDGE_CONTEXT_TOKENS * 4:  # v0: judge against the chunks nearest the answer
        context = format_hits(search_docs(run.answer[:2000], k=JUDGE_CONTEXT_TOKENS // 500))
        CALLS.pop()
    resp, cost, _ = complete([{"role": "user", "content": JUDGE_PROMPT.format(
        context=context, question=item["question"],
        reference=item.get("reference") or "(none: skip step 2 and return null)", answer=run.answer[:8000])}],
        model=JUDGE_MODEL, temperature=0)
    text = resp.choices[0].message.content or ""
    verdict = json.loads(re.search(r"\{.*\}", text, re.S).group())
    claims = verdict.get("claims") or []
    if not item.get("should_retrieve", True) and not run.chunks:
        faithful = None  # small talk answered without evidence: nothing to be faithful to
    else:
        faithful = sum(bool(c.get("supported")) for c in claims) / len(claims) if claims else 1.0
    correct = verdict.get("correct")
    return faithful, (float(correct) if correct is not None else None), verdict.get("why", ""), cost or 0.0


def evaluate(version: str, item: dict) -> dict:
    run = VERSIONS[version](item["question"])
    recall, mrr = retrieval_scores(item, run)
    faithful, correct, why, judge_cost = judge(item, run)
    return {"version": version, "id": item["id"], "question": item["question"],
            "routing_ok": run.retrieved == item["should_retrieve"] if "should_retrieve" in item else None,
            "searches": len(run.searches), "recall": recall, "mrr": mrr, "faithful": faithful,
            "correct": correct, "input_tokens": run.prompt_tokens, "cost": run.cost,
            "latency": run.latency, "judge_cost": judge_cost, "why": why, "run": asdict(run) | {"context": None}}


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("routing_ok", "recall", "mrr", "faithful", "correct", "cost"):  # None means n/a, not 0
        df[col] = pd.to_numeric(df[col].astype(object).where(df[col].notna()), errors="coerce")
    return df.groupby("version", sort=False).agg(
        questions=("id", "count"),
        input_tok=("input_tokens", "mean"),
        routing=("routing_ok", "mean"),
        recall=("recall", "mean"),
        mrr=("mrr", "mean"),
        faithful=("faithful", "mean"),
        correct=("correct", "mean"),
        usd_per_query=("cost", "mean"),
        p95_s=("latency", lambda s: float(np.percentile(s, 95))),
    ).round({"input_tok": 0, "routing": 2, "recall": 2, "mrr": 2, "faithful": 2, "correct": 2,
             "usd_per_query": 4, "p95_s": 2})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?", help="ask one question instead of running the golden set")
    parser.add_argument("--agents", default="v0,v1", help=f"comma-separated versions: {', '.join(VERSIONS)}")
    parser.add_argument("--golden", default=str(HERE / "golden.json"))
    parser.add_argument("--ids", help="comma-separated golden ids to run")
    parser.add_argument("--out", help="write every row, answers included, to this JSON file")
    args = parser.parse_args()

    versions = args.agents.split(",")
    unknown = [v for v in versions if v not in VERSIONS]
    if unknown:
        parser.error(f"unknown version(s) {unknown}; available: {list(VERSIONS)}")
    golden = json.loads(Path(args.golden).read_text())
    if args.question:
        items = [next((g for g in golden if g["question"].lower() == args.question.lower()),
                      {"id": "adhoc", "question": args.question, "gold": []})]
    else:
        items = [g for g in golden if not args.ids or g["id"] in args.ids.split(",")]

    load_index()  # build or load the chunk index up front so it doesn't count as latency
    rows = []
    for version in versions:
        for item in items:
            try:
                row = evaluate(version, item)
            except Exception as e:  # one bad turn (a timeout, a runaway answer) shouldn't end the run
                print(f"{version}  {item['id']:<22} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
                continue
            rows.append(row)
            score = "n/a" if row["correct"] is None else f"{row['correct']:g}"
            faithful = "n/a" if row["faithful"] is None else f"{row['faithful']:.2f}"
            print(f"{version}  {item['id']:<22} in={row['input_tokens']:>7,}  searches={row['searches']}  "
                  f"faithful={faithful}  correct={score}  {row['latency']:.1f}s", flush=True)
            if args.question:
                print(f"\n{row['run']['answer']}\n")

    print()
    print(summarize(rows).to_string())
    print(f"\nJudge ({JUDGE_MODEL}) cost: ${sum(r['judge_cost'] for r in rows):.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
