# Homework: The Office Hours Bot

*A multi-agent course assistant with LangGraph and AgentShip.* The same handout as a PDF: [HOMEWORK.pdf](HOMEWORK.pdf).

**Builds on:** workshop Modules 3–5 · **Work:** individually · **Due:** ____________________

## The idea

Build the **Office Hours Bot** for a fictional course, *AGNT 6100: Building Agentic Systems*. A
**supervisor** routes each student message to one of three specialists:

```
                      ┌─ syllabus ─ agentic RAG you build in LangGraph:
                      │     retrieve → grade → (rewrite → retrieve)* → answer with [citations]
Student ─→ route ─────┼─ concepts ─ an AgentShip agent in pure YAML (search_concepts tool)
      (supervisor)    │
                      └─ escalate ─ draft an email to the right TA → send_ta_email
                                    ⏸ PAUSES until a human approves
```

> **Disclaimer: AI-generated synthetic data.** The data in `data/`, including the course, people, email addresses, places, dates and
> policies in this homework were generated with AI for teaching. They are fictional, and any
> resemblance to real persons, courses or institutions is coincidental. Not affiliated with or
> endorsed by any university. Nothing is really emailed: "sending" appends a line to `outbox.jsonl`.

## Setup

Use the workshop repo, [AgenticRAGLabs](https://github.com/Parv17k/AgenticRAGLabs), with its virtual environment and `.env`. The starter
code is in `homework/office_hours/`. Every place you write code is marked `TODO (Step N)`, and
until you fill it in, it raises an error naming its step.

```bash
source .venv/bin/activate && docker compose up -d   # Postgres (checkpoints) + Phoenix (traces)
cd homework/office_hours && export PYTHONPATH=.
python kb.py "slip days"                            # BM25 search, works out of the box
```

> Add `--env-file ../../.env` to every `agentship` command below so it finds your API key.

## Steps

**1 · Explore.** Read `data/syllabus.md` and try `python kb.py "<query>"`. Find one query where
BM25 ranks the wrong section first. Save it for your report.

**2 · AgentShip warm-up (no Python).** In `concepts.yaml`, add the `kb:search_concepts` tool and
finish the prompt. In `team.yaml`, add the `concepts` member. Run
`agentship run team.yaml -v --input "Explain embeddings"` and watch `classify → route → dispatch`.
*Why go further?* This team can't grade or retry, and a pause inside a `members:` specialist never
reaches the supervisor, so the real bot is a graph you build.

**3 · Build the syllabus branch in LangGraph** (`graph.py`; the diagram at the top is your spec):

| TODO | Node | What it does |
|----|-----------|---------------------------------------------------|
| 3a | `route` | One model call over the `ROUTES` descriptions; fall back to `DEFAULT_ROUTE` |
| 3b | `grade` | Keep only the passages that help answer the question |
| 3c | `after_grade` | Conditional edge: `answer`, `rewrite` (if retries remain) or `give_up` |
| 3d | `rewrite` | Write a better query and **increment `retries`**, so the loop ends |
| 3e | edges | Wire `retrieve → grade → …` as in the diagram |

Test: `agentship run office_hours.yaml --input "Is there a weekly lab section?"` must not invent one.

**4 · Plug in the YAML specialist.** In `ask_concepts`, build the agent from `concepts.yaml` with
AgentShip's `build_agent` and `await` its `run()`.

**5 · Escalation with human approval.** (5a) In `draft_email`, return the drafted email as a
`send_ta_email` tool call on an `AIMessage`. (5b) Mark the tool `side_effecting=True` in `kb.py`,
and add `durability="checkpoint"` and `confirm_writes=True` to the spec in `build_agent()`. Then:

```bash
agentship serve --agents-dir .        # Studio: http://127.0.0.1:8000/studio
curl -s -X POST localhost:8000/v1/agents/office-hours:invoke \
  -H "Authorization: Bearer dev-secret" -H "Content-Type: application/json" \
  -d '{"input": "Please email the TA: I have the flu and need an extension on Homework 4."}'
# → "paused": true, with the draft in "interrupt". outbox.jsonl is still empty.
curl -s -X POST localhost:8000/v1/agents/office-hours:resume \
  -H "Authorization: Bearer dev-secret" -H "Content-Type: application/json" \
  -d '{"session_id": "<from above>", "resume_value": {"approved": true}}'
```

Rejecting (`"approved": false`) must write nothing. Pause a run, **restart the server**, then approve:
it continues, because the pause is stored in Postgres.

**6 · Evaluate and improve.** Grow `golden.json` to **15+ questions**: at least 4 per route, 2
*ambiguous* ones (e.g. "Can I get an extension for a job interview?") and 1 the syllabus can't
answer. Run `python evaluate.py` (it scores routing and answers separately) and, using `results.json` and
Phoenix traces (http://localhost:6006), **fix at least two failures**, e.g. multi-part questions or
policy questions misrouted as requests. Run each version at least twice: LLM outputs vary.

**7 · Report.** `REPORT.md`, at most 2 pages: before/after tables with one paragraph per fix,
a Phoenix trace screenshot, your Step 1 BM25 miss and how you'd fix it, **what doesn't work
yet**, and which AI tools you used.

## Submit

- The completed `homework/office_hours/` folder, with `golden.json` (15+) and `results.json`
- `REPORT.md`
- A ≤ 60-second recording: escalate → **paused** → approve → `outbox.jsonl` updated
- One `curl` command in your README that runs against your server

## Grading (100 points + 10 bonus)

| Area | Points | Full marks when… |
|--------------------------|------:|-------------------------------------------------|
| Syllabus branch (3) | 25 | Grades, bounded rewrites, citations, admits what it can't find |
| AgentShip (2, 4) | 15 | Team routes correctly; YAML specialist called from the graph |
| Human-in-the-loop (5) | 20 | Pauses; approve writes once; reject writes nothing; recording |
| Evaluation (6) | 25 | 15+ questions; two fixes with before/after numbers; repeated runs |
| Report (7) | 15 | Clear, honest, trace screenshot, real limitations |

Staff will also run a **hidden question set**. Invented facts cost more than "I couldn't find
that". **Bonus:** hybrid search with RRF that fixes your Step 1 miss (+4); a prompt-injection
attack planted in the syllabus, plus a defence (+3); tokens and cost per question for each
route (+3).

## Troubleshooting

| Problem | Fix |
|---|---|
| `tool 'kb:…' could not be imported` | Run `export PYTHONPATH=.` inside `homework/office_hours`. |
| `agentship serve` refuses a durable agent | Start Postgres with `docker compose up -d`, or for a quick test add `--allow-in-memory-durability` (runs won't survive a restart). |
| Port 8000 already in use | Add `--port 8077` and use that port in your `curl` commands. |
| `429` / rate-limit errors | The Gemini free tier allows 5 requests/minute. Enable billing, or evaluate with `--only`. |
| The bot answers "the passages don't mention…" when they do | Your grader dropped the passage. Print `state["passages"]` after `grade`. |
| An escalation sent an email without pausing | Check both halves of 5b: `side_effecting=True` *and* `confirm_writes=True`. |

## Questions or career guidance?

Have a question about this homework, or want career guidance? Schedule a call with us on Topmate:

- **Parv Khatri**: [topmate.io/parv39](https://topmate.io/parv39)
- **Harshul Jain**: [topmate.io/harshuljain13](https://topmate.io/harshuljain13)
