# Agentic RAG Labs

**NYU · Agentic RAG, Lecture 1: Foundations**

One company filing. One question. Four ways to answer it, each smarter than the last.

> **The question:** *"How did Acme's Q3 margin compare to Q2, and why?"*
>
> **The document:** Acme Corp's quarterly report (a synthetic Form 10-Q, about 100,000 tokens).
> The Q3 numbers are in the main report. The Q2 numbers sit in an appendix near the end.

That second line is the trap. A system that only finds Q3 will answer half the question and then
say it can't find Q2. Watching each version fall into the trap, or avoid it, is the lecture.

---

## The four versions at a glance

| Version | Idea | What happens on our question |
|---|---|---|
| **v0** · Whole document | Paste the entire filing into every prompt. | ✅ Right answer, but every question sends 107K tokens. |
| **v1** · Basic RAG | Split the filing into chunks; send only the 5 closest to the question. | ❌ Finds Q3, misses the Q2 appendix, says it can't compare. |
| **v2** · Agent | Give the model a search tool and let it decide what to look up. | ✅ Searches Q3, then Q2. But it can loop when the answer isn't there. |
| **v3** · Agentic RAG | An agent that also *grades* what it found, *rewrites* bad searches, and *checks* its answer. | ✅ Right answer, and it knows when to stop and say "not in the filing". |

Then **Chapter 5** puts v3 into production with [AgentShip](https://github.com/Agent-Ship/agent-ship):
an API, saved conversations that survive a server crash, and traces you can inspect.

### Measured results

All four versions on the 14-question test set (`golden.json`), running on `gemini-3.5-flash`:

| Version | Correct | Input tokens / question | Cost / question | Typical time |
|---|:---:|---:|---:|---:|
| v0 · Whole document | **100%** | 106,800 | $0.161 | 2–4 s |
| v1 · Basic RAG | 46% | 2,700 | $0.006 | 1–2 s |
| v2 · Agent | 79% | 6,700 | $0.014 | 6–9 s |
| v3 · Agentic RAG | **93%** | 11,800 | $0.020 | ~6 s |

**How to read this:** v0 is accurate but expensive, at about 8× the cost of v3, and it gets worse as documents grow.
v1 is cheap but misses whatever the top 5 chunks don't cover. v2 recovers most of the accuracy;
the three questions it missed were loops it couldn't stop. v3 gets close to v0's accuracy at an
eighth of the cost. Its one miss was an honest "I couldn't find it", not a made-up number.

Full answers and the judge's reasoning for every question are in `results_golden.json`.

---

## Slides → labs

| Chapter | Slides | What you build | Open this |
|---|---|---|---|
| 1 · Just upload the PDF | 12 | v0 | `v0_whole_doc.ipynb` |
| 2 · Send only what matters | 23–24 | v1 | `v1_basic_rag.ipynb` |
| 3 · Give the model judgment | 35–45 | v2, built step by step in LangGraph | `v2_langgraph_agent.ipynb` |
| 4 · Search with judgment | 48–54 | v3: grade, rewrite, verify | `v3_agentic_rag.ipynb` |
| 5 · Ship it | 59–65 | v3 as a service, a single agent, a team | `agents/*.yaml` + the commands below |
| Finale | 68–69 | v0 → v3 side by side | `run_all.ipynb` |

Run the notebooks in order: each one starts where the previous one stopped.

---

## Setup

You need **Python 3.13+**, a **Gemini API key**, and (for Chapter 5 only) **Docker**.

### 1. Get a Gemini API key

1. Go to [Google AI Studio → API keys](https://aistudio.google.com/apikey) and create a key.
2. **Turn on billing** for the key's project in AI Studio.
   The free tier allows only 5 requests a minute, and a single v3 question makes 5 to 10.
   Running every notebook once costs about **$1.20**, most of it v0.

### 2. Install

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .

# The latest AgentShip, straight from GitHub
pip install "agentship-core @ git+https://github.com/Agent-Ship/agent-ship.git@main#subdirectory=packages/agentship-core" \
            "agentship-langgraph @ git+https://github.com/Agent-Ship/agent-ship.git@main#subdirectory=packages/agentship-langgraph"

python -m ipykernel install --user --name agentic-rag-demos   # the kernel the notebooks use
```

### 3. Add your key

```bash
cp .env.example .env
```

Open `.env` and paste your key after `GEMINI_API_KEY=`. Everything else is already set.
`.env` is in `.gitignore`, so your key never gets committed.

### 4. Start the services (Chapter 5 only)

```bash
docker compose up -d   # Postgres (saved sessions) · Phoenix (traces) · OpenSearch (lecture 2)
agentship db upgrade --allow-migrations --database-url postgresql://agentship:agentship@localhost:5434/agentship
```

The notebooks for Chapters 1–4 don't need Docker.

---

## Running the labs

### Chapters 1–4: notebooks

Open each notebook, choose the **agentic-rag-demos** kernel, and run all cells from the top.

### Chapter 5: ship it

```bash
# Slide 59 · one agent, one tool
agentship run agents/calc.yaml --input "What is 17 times 23?"

# Slide 60 · the v3 graph from Chapter 4, served unchanged
agentship run agents/agentic_rag.yaml --input "How did Acme's Q3 margin compare to Q2, and why?"

# Slide 62 · a team: a coordinator routes each question to the right specialist
agentship run agents/team.yaml --input "Compare Q3 and Q2 revenue."

# Slide 65 · serve every agent as an API
agentship serve --agents-dir agents
```

With the server running:

| What | Where |
|---|---|
| Studio (chat with any agent) | http://127.0.0.1:8000/studio · key `dev-secret` |
| Traces (every model call and search) | http://localhost:6006 |
| REST API | `POST /v1/agents/agentic-rag:invoke` with `{"input": "...", "session_id": "..."}` |

Reuse the same `session_id` and the agent remembers the conversation, even after you stop and restart the server.

### Finale: measure it

```bash
# One question, every version, scored
python eval.py --agents v0,v1,v2,v3 "How did Acme's Q3 margin compare to Q2, and why?"

# The full 14-question test set (about 5 minutes, about $3)
python eval.py --agents v0,v1,v2,v3 --out results.json
```

`eval.py` scores **retrieval** (did it find the right chunks?) separately from **generation** (is the
answer faithful to what it found, and correct?). An LLM judge grades the answers.

---

## How v3 works

```
question ─→ agent ─┬─ no search needed (e.g. "Hi!") ────────────────────────→ verify ─→ answer
                   │
                   └─ search_docs ─→ grade each chunk
                                       ├─ something relevant ──→ agent writes the answer
                                       ├─ nothing relevant ────→ rewrite the query, search again
                                       └─ still nothing, 2 tries later ─→ "The filings don't contain this."
```

- **grade** removes chunks that don't answer the search, so the agent can't build on bad evidence.
- **rewrite** turns a vague question into the filing's own words. *"Work contracted but not yet
  delivered"* becomes *"remaining performance obligations"*.
- **verify** checks that every number in the answer appears in the evidence before it goes out.
- Hard limits (2 rewrites, 4 searches) guarantee the loop always ends. That is the guarantee v2 lacks.

The graph lives in `agents/agentic_rag.py`. The Chapter 4 notebook builds it cell by cell, and
Chapter 5 serves the same function on AgentShip without changing a line.

---

## What's in this folder

```
v0_whole_doc.ipynb … v3_agentic_rag.ipynb   the four chapter notebooks
run_all.ipynb                               the finale: v0 → v3 side by side
agents/
  langgraph_agent.py                        v2, the agent from Chapter 3
  agentic_rag.py                            v3, the graph from Chapter 4, plus its AgentShip wrapper
  agentic_rag.yaml, calc.yaml, team.yaml    the Chapter 5 agents (researcher.yaml is a team member)
tools/
  search_docs.py                            chunking, embeddings and search, shared by every version
  settings.py                               reads .env: model, keys, prices
data/acme_10q.txt                           the filing (make_acme_10q.py generates it)
golden.json                                 14 test questions with reference answers
eval.py                                     scores any version on golden.json
results_golden.json                         the results in the table above
lecture2/                                   v4 and v5, for Lecture 2 (hybrid search, reranking, tuning)
```

---

## Troubleshooting

| You see | Why | Fix |
|---|---|---|
| `429` or *"exceeded your current quota"* | The key's project is on the free tier. | Turn on billing in AI Studio. |
| `402` *"prepayment credits are depleted"* | The project's prepaid balance is empty. | Add credits in AI Studio. |
| `403` *"Gemini API has not been used in project…"* | The Gemini API is off for that project. | Open the link in the error and click **Enable**. Wait 2–3 minutes. |
| `400` right after the agent's first search | The model is set to `openai/gemini-…`, which drops Gemini's tool-call signature. | Use `gemini/gemini-3.5-flash`, as in `.env.example`. |
| v2 stops with `GraphRecursionError` | v2 has no search limit and can loop. | That's expected. It's the problem v3 solves. |
| The first v3 run takes a minute; later runs take seconds | On a Mac, iCloud is syncing the folder, and Python waits on its files. | See the note below. |
| v3 prints *"verify → unsupported claims"* and then repeats the same answer | The verify check is occasionally too strict. | Harmless: it costs one extra call. |

**On a Mac with iCloud Drive:** if this folder is under Desktop or Documents, iCloud may hold or
evict files while it syncs them. Rename the environment to `.venv.nosync` and link `.venv` to it
(`mv .venv .venv.nosync && ln -s .venv.nosync .venv`), and set the workshop folder to
**Keep Downloaded**.

---

## Using a different model

Every lab reads its model from `.env`, so you can point it anywhere:

- **Any OpenAI-compatible server** (OpenAI, a local Ollama or llama.cpp server, a gateway): set
  `OPENAI_API_KEY`, `OPENAI_API_BASE`, `DEMO_MODEL=openai/<model>`,
  `DEMO_EMBED_MODEL=openai/<embedding model>` and `DEMO_TEMPERATURE=0`.
- **A local model** has no list price, so set `PRICE_AS` to a model whose prices should fill the $ columns.
- **The agent YAMLs** in `agents/` name their model directly. Change `model:` there too.

The slides show `openai/gpt-4o-mini` (and `anthropic/claude-sonnet-4-6` for the team's writer).
The labs use one model everywhere so that the results are easy to compare.

On a self-hosted model, v0's first question has to read the whole filing before it writes anything,
which can take minutes. Run `run_all.ipynb` once before a live demo to warm the server's cache.

---

## Lecture 2

`lecture2/` and `tools/opensearch_search.py` continue the story: **v4** runs v3 on OpenSearch hybrid
search (keywords and vectors) with reranking, and **v5** tunes it into a team. They need the OpenSearch
container from `docker compose up -d`, and an index built with the same embedding model as in your `.env`.
