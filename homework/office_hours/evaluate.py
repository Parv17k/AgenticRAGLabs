"""Score the Office Hours Bot on golden.json: routing accuracy and answer accuracy, separately.

    python evaluate.py                 # every question
    python evaluate.py --only s3 e1    # a few, while debugging

Routing is checked against each question's expected `route`. Answers are judged by the model
against `expected`; escalations pass if the run PAUSED (nothing sent) with the right recipient.
Results go to results.json; the summary table is printed.

This compiles the graph directly (not through `agentship serve`) so it can read the route from the
graph's state. The send tool is wrapped exactly as AgentShip wraps it, so escalations pause here
too and nothing is ever written to outbox.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agentship_langgraph.tools import to_langchain_tool  # noqa: E402
from langchain.chat_models import init_chat_model  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import kb  # noqa: E402
from graph import build_office_hours_graph, text_of  # noqa: E402
from tools.settings import CHAT_KWARGS, CHAT_MODEL  # noqa: E402

JUDGE = """You are grading a course chatbot.
QUESTION: {question}
EXPECTED: {expected}
ANSWER: {answer}
Does ANSWER state the facts in EXPECTED, without contradicting them? Extra correct detail is fine.
If EXPECTED says the bot should not find the information, ANSWER passes only if it says so instead
of inventing an answer. Reply with PASS or FAIL, then one short sentence of reasoning."""


async def run_one(app, model, item: dict) -> dict:
    cfg = {"configurable": {"thread_id": f"eval-{item['id']}-{time.time()}"}, "recursion_limit": 25}
    start = time.perf_counter()
    state = await app.ainvoke({"messages": [HumanMessage(content=item["question"])]}, cfg)
    seconds = round(time.perf_counter() - start, 1)
    route = state.get("route")
    row = {"id": item["id"], "question": item["question"], "expected_route": item["route"],
           "route": route, "route_ok": route == item["route"], "seconds": seconds}

    if item["route"] == "escalate":
        pause = state.get("__interrupt__")
        args = pause[0].value.get("args", {}) if pause else {}
        row |= {"answer": f"PAUSED for approval: email to {args.get('to')}" if pause else "did not pause",
                "answer_ok": bool(pause) and args.get("to") == item["expected_to"]}
    else:
        answer = text_of(state["messages"][-1])
        verdict = text_of(model.invoke([HumanMessage(content=JUDGE.format(
            question=item["question"], expected=item["expected"], answer=answer))])).strip()
        row |= {"answer": answer, "answer_ok": verdict.upper().startswith("PASS"), "judge": verdict}
    return row


async def main(only: list[str]) -> None:
    golden = json.loads((HERE / "golden.json").read_text())
    if only:
        golden = [g for g in golden if g["id"] in only]
    model = init_chat_model(CHAT_MODEL, **CHAT_KWARGS)
    tools = [to_langchain_tool(kb.send_ta_email, confirm_writes=True)]  # as AgentShip wraps it
    app = build_office_hours_graph(model, tools).compile(checkpointer=InMemorySaver())

    rows = []
    for item in golden:
        row = await run_one(app, model, item)
        rows.append(row)
        mark = lambda ok: "✅" if ok else "❌"  # noqa: E731
        print(f"{row['id']:>3}  route {mark(row['route_ok'])} {row['route'] or '-':<9} "
              f"answer {mark(row['answer_ok'])}  {row['seconds']:>5}s  {row['question'][:60]}")

    by_route: dict[str, list[dict]] = {}
    for r in rows:
        by_route.setdefault(r["expected_route"], []).append(r)
    print("\n| route     | n  | routing | answers |\n|-----------|----|---------|---------|")
    for name, group in [*by_route.items(), ("ALL", rows)]:
        n = len(group)
        print(f"| {name:<9} | {n:<2} | {sum(r['route_ok'] for r in group) / n:>6.0%}  "
              f"| {sum(r['answer_ok'] for r in group) / n:>6.0%}  |")
    (HERE / "results.json").write_text(json.dumps(rows, indent=2))
    print("\nper-question details in results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=[], help="question ids to run, e.g. s3 e1")
    asyncio.run(main(parser.parse_args().only))
