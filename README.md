# Demos · Agentic RAG guest lecture

One synthetic Acme 10-Q (`data/acme_10q.txt`, ~100K tokens), one golden set, and a version per module.
Every version answers the same questions, so `eval.py` can put them side by side (module 06).

| Module | Version | What changes | Run |
|---|---|---|---|
| 00 | v0 | the whole filing in every prompt | `v0_whole_doc.ipynb` |
| 01 | v1 | basic RAG: 500-token chunks, vectors, top 5, every message | `v1_basic_rag.ipynb` |
| 02 | v2 | AgentShip agent and team; `search_docs` is a tool the agent may call | `agents/calc.yaml`, `agents/team.yaml` |
| 03 | v3 | agentic RAG graph: route, retrieve, grade, rewrite, verify, loops capped | `agents/agentic_rag.yaml` |
| 04 | v4 | v3 with OpenSearch: structural chunks, BM25 + k-NN, RRF, rerank | `agents/agentic_rag_v4.yaml`, `tools/opensearch_search.py` |
| 05 | v5 | v4 tuned by configuration: router, rerank to 3, fewer retries, less thinking | `agents_v5/team.yaml` |
| 06 | all | one question through every version, then the golden set | `eval.py` |

`tools/search_docs.py` (v1–v3) and `tools/opensearch_search.py` (v4–v5) return the same thing, so
the agents cannot tell which index they search. `agents/agentic_rag.py` holds the v3–v5 graph.

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

Until [Agent-Ship/agent-ship#85](https://github.com/Agent-Ship/agent-ship/pull/85) is released, v2 and
v5 need its fixes (a team re-answering turn one, tools ignored without a template, team params):

```bash
git clone -b fix/supervisor-turn-state-and-tools-default https://github.com/Agent-Ship/agent-ship.git ../agent-ship
pip install -e ../agent-ship/packages/agentship-core -e ../agent-ship/packages/agentship-langgraph
```

Any OpenAI-compatible server works: set `OPENAI_API_BASE`, `DEMO_MODEL`, `DEMO_EMBED_MODEL` and,
for v4–v5, `DEMO_RERANK_MODEL` (a `/reranking` model on the same server). A local model has no list
price, so `PRICE_AS` names the model whose price fills the $ columns. The agent YAMLs name their model
directly; change `model:` there too if you switch.

If this folder is synced by iCloud Drive (Desktop & Documents on a Mac), make the environment
`.venv.nosync` with `.venv` as a symlink to it, and set the workshop folder to **Keep Downloaded**.
iCloud otherwise evicts files, and reads from them stall.

## Running each demo

```bash
# v2 · slides 35–43
agentship run agents/calc.yaml --input "What is 17 times 23?"
agentship run agents/team.yaml --input "Compare Q3 and Q2 revenue."
agentship serve --agents-dir agents    # Studio at http://127.0.0.1:8000/studio, key dev-secret

# v3 · slide 52
agentship run agents/agentic_rag.yaml --input "Hi!"                                   # no search
agentship run agents/agentic_rag.yaml --input "How did Acme's Q3 margin compare to Q2, and why?"

# v4 · slide 68: the same question through BM25, k-NN, hybrid and hybrid + rerank
python tools/opensearch_search.py "How many shares did Acme buy back last quarter?"
agentship run agents/agentic_rag_v4.yaml --input "How many shares did Acme buy back last quarter?"

# v5 and module 06 · slides 82, 86–87
python eval.py --agents v4,v5
python eval.py --agents v0,v1,v2,v3,v4,v5 "How did Acme's Q3 margin compare to Q2, and why?"
python eval.py --agents v0,v1,v2,v3,v4,v5 --out results.json      # the whole golden set
```

On a self-hosted model, v0's first question processes the whole ~117K-token filing before it writes a
word (about 10 minutes on the lecture server). Later questions reuse the server's prompt cache, so run
v0 once before the talk.
