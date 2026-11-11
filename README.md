# Agentic RAG labs · NYU lecture 1, Foundations

One synthetic Acme 10-Q (`data/acme_10q.txt`, ~100K tokens) and one question, answered five times:
"How did Acme's Q3 margin compare to Q2, and why?" Each lab is the live demo of one chapter.

| Chapter | Slides | Version | Lab |
|---|---|---|---|
| 1 · Just upload the PDF | 12 | v0 · the whole filing in every prompt | `v0_whole_doc.ipynb` |
| 2 · Send only what matters | 23–24 | v1 · basic RAG: chunk, embed, top 5 | `v1_basic_rag.ipynb` |
| 3 · Give the model judgment | 35–45 | v2 · a LangGraph agent with `search_docs` | `v2_langgraph_agent.ipynb`, `agents/langgraph_agent.py` |
| 4 · Search with judgment | 48–54 | v3 · agentic RAG: route, retrieve, grade, rewrite, verify | `v3_agentic_rag.ipynb`, `agents/agentic_rag.py` |
| 5 · Ship it | 59–65 | v3 served on AgentShip, an agent, a team | `agents/agentic_rag.yaml`, `agents/calc.yaml`, `agents/team.yaml` |
| Finale | 68–69 | v0 → v3 on one question, side by side | `run_all.ipynb` |

`v3_agentic_rag.ipynb` builds the graph that `agents/agentic_rag.py` defines as `build_v3_graph`, and
chapter 5 serves that same function on AgentShip unchanged. `eval.py` scores any version on the 14-question
golden set (`golden.json`).

`lecture2/` and `tools/opensearch_search.py` are for lecture 2 (hybrid search, reranking, evaluation,
tuning): v4 runs the agentic RAG graph on OpenSearch, and v5 tunes it as a team.

## Setup

Python 3.13+ (AgentShip needs it) and Docker.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e .                       # AgentShip, LiteLLM; registers the search_docs tools
cp .env.example .env                   # model, server and keys
docker compose up -d                   # Postgres (checkpoints), Phoenix (traces, :6006), OpenSearch (:9200)
agentship db upgrade --allow-migrations --database-url postgresql://agentship:agentship@localhost:5434/agentship
python -m ipykernel install --user --name agentic-rag-demos   # the notebooks' kernel
```

Until [Agent-Ship/agent-ship#85](https://github.com/Agent-Ship/agent-ship/pull/85) is released, the
AgentShip labs need its fixes (a team re-answering turn one, tools ignored without a template, team params):

```bash
git clone -b fix/supervisor-turn-state-and-tools-default https://github.com/Agent-Ship/agent-ship.git ../agent-ship
pip install -e ../agent-ship/packages/agentship-core -e ../agent-ship/packages/agentship-langgraph
```

Any OpenAI-compatible server works: set `OPENAI_API_BASE`, `DEMO_MODEL`, `DEMO_EMBED_MODEL` and,
for lecture 2, `DEMO_RERANK_MODEL` (a `/reranking` model on the same server). A local model has no list
price, so `PRICE_AS` names the model whose price fills the $ columns. The agent YAMLs name their model
directly; change `model:` there too if you switch.

If this folder is synced by iCloud Drive (Desktop & Documents on a Mac), make the environment
`.venv.nosync` with `.venv` as a symlink to it, and set the workshop folder to **Keep Downloaded**.
iCloud otherwise evicts files, and reads from them stall.

## Running each demo

Open the notebooks with the *Agentic RAG demos* kernel and run them top to bottom. The rest is shell:

```bash
# Chapter 5 · slides 59–65
agentship run agents/calc.yaml --input "What is 17 times 23?"                      # slide 59
agentship run agents/agentic_rag.yaml --input "How did Acme's Q3 margin compare to Q2, and why?"
agentship run agents/team.yaml --input "Compare Q3 and Q2 revenue."                # slide 62
agentship serve --agents-dir agents    # slide 65 · Studio at http://127.0.0.1:8000/studio, key dev-secret
                                       # traces in Phoenix at http://localhost:6006

# Finale · slides 68–69: run_all.ipynb, or the whole golden set, scored
python eval.py --agents v0,v1,v2,v3 "How did Acme's Q3 margin compare to Q2, and why?"
python eval.py --agents v1,v2,v3 --out results.json

# Lecture 2
python tools/opensearch_search.py "How many shares did Acme buy back last quarter?"  # BM25, k-NN, hybrid, rerank
agentship run lecture2/v4.yaml --input "How many shares did Acme buy back last quarter?"
python eval.py --agents v3,v4,v5
```

The slides name `openai/gpt-4o-mini` (and the team's writer `anthropic/claude-sonnet-4-6`). The labs
run every model call on `DEMO_MODEL`'s server instead, at `DEMO_TEMPERATURE` (0 by default, so a live
demo repeats itself).

On a self-hosted model, v0's first question processes the whole ~117K-token filing before it writes a
word (about 4 minutes on the lecture server). Later questions reuse the server's prompt cache, so run
`run_all.ipynb` once before the talk.
