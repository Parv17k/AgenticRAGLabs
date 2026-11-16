# Homework: The Office Hours Bot

*A multi-agent course assistant with LangGraph and AgentShip.* The same handout as a PDF: [HOMEWORK.pdf](HOMEWORK.pdf).

<div class="meta">
<span><b>Builds on</b> workshop Modules 3–5</span>
<span><b>Work</b> individually</span>
<span><b>Due</b> ____________________</span>
</div>

## The idea

Build the **Office Hours Bot** for a fictional course, *AGNT 6100: Building Agentic Systems*. A
**supervisor** routes each student message to one of three specialists:

<div class="diagram">
<svg viewBox="0 0 760 270" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial, sans-serif" font-size="13">
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>
  <rect x="10" y="112" width="110" height="46" rx="8" fill="#f1f5f9" stroke="#64748b"/>
  <text x="65" y="133" text-anchor="middle" font-weight="bold">Student</text>
  <text x="65" y="149" text-anchor="middle" fill="#475569" font-size="11">question</text>
  <rect x="160" y="112" width="120" height="46" rx="8" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="220" y="133" text-anchor="middle" font-weight="bold">route</text>
  <text x="220" y="149" text-anchor="middle" fill="#5b21b6" font-size="11">supervisor (1 LLM call)</text>
  <line x1="120" y1="135" x2="158" y2="135" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M280 125 C 310 60, 320 40, 345 40" fill="none" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
  <line x1="280" y1="135" x2="345" y2="135" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
  <path d="M280 145 C 310 210, 320 230, 345 230" fill="none" stroke="#555" stroke-width="1.5" marker-end="url(#a)"/>
  <rect x="348" y="12" width="400" height="58" rx="8" fill="#ecfdf5" stroke="#059669"/>
  <text x="362" y="33" font-weight="bold" fill="#065f46">syllabus — agentic RAG you build in LangGraph</text>
  <text x="362" y="54" fill="#065f46" font-size="12">retrieve → grade → (rewrite → retrieve)* → answer with [citations]</text>
  <rect x="348" y="106" width="400" height="58" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="362" y="127" font-weight="bold" fill="#1e40af">concepts — an AgentShip agent in pure YAML</text>
  <text x="362" y="148" fill="#1e40af" font-size="12">search_concepts tool over the course notes, zero Python</text>
  <rect x="348" y="200" width="400" height="58" rx="8" fill="#fff7ed" stroke="#ea580c"/>
  <text x="362" y="221" font-weight="bold" fill="#9a3412">escalate — draft an email to the right TA</text>
  <text x="362" y="242" fill="#9a3412" font-size="12">send_ta_email ⏸ PAUSES until a human approves</text>
</svg>
</div>

All data in `data/` is **synthetic**: a 20-section syllabus and 17 concept notes. Nothing is really
emailed. "Sending" appends a line to `outbox.jsonl`.

## Setup

Use the workshop repo, [AgenticRAGLabs](https://github.com/Parv17k/AgenticRAGLabs), with its virtual environment and `.env`. The starter
code is in `homework/office_hours/`. Every place you write code is marked `TODO (Step N)`, and
until you fill it in, it raises an error naming its step.

```bash
source .venv/bin/activate && docker compose up -d   # Postgres (checkpoints) + Phoenix (traces)
cd homework/office_hours && export PYTHONPATH=.
python kb.py "slip days"                            # BM25 search, works out of the box
```

::: note
Add `--env-file ../../.env` to every `agentship` command below so it finds your API key.
:::

## Steps

**1 · Explore.** Read `data/syllabus.md` and try `python kb.py "<query>"`. Find one query where
BM25 ranks the wrong section first. Save it for your report.

**2 · AgentShip warm-up (no Python).** In `concepts.yaml`, add the `kb:search_concepts` tool and
finish the prompt. In `team.yaml`, add the `concepts` member. Run
`agentship run team.yaml -v --input "Explain embeddings"` and watch `classify → route → dispatch`.
*Why go further?* This team's syllabus agent can't grade or retry, and an approval pause inside a
`members:` specialist never reaches the supervisor. So the real bot is a graph you build.

**3 · Build the syllabus branch in LangGraph** (`graph.py`; the diagram at the top is your spec):

| TODO | Node | What it does |
|----|-----------|---------------------------------------------------|
| 3a | `route` | One model call over the `ROUTES` descriptions; fall back to `DEFAULT_ROUTE` |
| 3b | `grade` | Keep only the passages that help answer the question |
| 3c | `after_grade` | Conditional edge: `answer`, `rewrite` (if retries remain) or `give_up` |
| 3d | `rewrite` | Write a better query and **increment `retries`**, so the loop ends |
| 3e | edges | Wire `retrieve → grade → …` as in the diagram |

Test with `agentship run office_hours.yaml --input "Is there a weekly lab section?"`. It should
say it can't find this, not invent an answer.

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

Check that rejecting (`"approved": false`) writes nothing. Then pause a run, **restart the
server**, and approve it. It continues, because the pause is stored in Postgres.

**6 · Evaluate and improve.** Grow `golden.json` to **15+ questions**: at least 4 per route, 2
*ambiguous* ones (e.g. "Can I get an extension for a job interview?") and 1 the syllabus can't
answer. Run `python evaluate.py`, which scores routing and answers separately. Using
`results.json` and the Phoenix traces (http://localhost:6006), **fix at least two failures**.
Typical ones: multi-part questions (try one search per part), policy questions misrouted as
requests (sharpen `ROUTES` or add router examples), and a grader that drops good evidence.
Run each version at least twice, because LLM outputs vary.

**7 · Report.** `REPORT.md`, at most 2 pages: before/after tables with one paragraph per fix,
a Phoenix trace screenshot, your Step 1 BM25 miss and how you'd fix it, **what doesn't work
yet**, and which AI tools you used.

## Submit

::: checklist
- The completed `homework/office_hours/` folder, with `golden.json` (15+) and `results.json`
- `REPORT.md`
- A ≤ 60-second recording: escalate → **paused** → approve → `outbox.jsonl` updated
- One `curl` command in your README that runs against your server
:::

## Grading (100 points + 10 bonus)

| Area | Points | Full marks when… |
|--------------------------|------:|-------------------------------------------------|
| Syllabus branch (3) | 25 | Grades, rewrites in a bounded loop, cites ids, admits when it can't find something |
| AgentShip integration (2, 4) | 15 | The team routes correctly; the YAML specialist is called from the graph |
| Human-in-the-loop (5) | 20 | Pauses before sending; approve writes once; reject writes nothing; recording |
| Evaluation (6) | 25 | 15+ good questions; two real fixes with before/after numbers; repeated runs |
| Report (7) | 15 | Clear and honest, with a trace screenshot and real limitations |

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
