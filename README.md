<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6366f1,50:8b5cf6,100:ec4899&height=200&section=header&text=Agentic%20RAG%20Labs&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=From%20stuffing%20the%20whole%20document%20to%20an%20agent%20that%20knows%20when%20to%20search&descAlignY=58&descSize=14" alt="Agentic RAG Labs" width="100%" />

### Four ways to answer one question about one document. Measured side by side.

<p>
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13+" />
  <img src="https://img.shields.io/badge/LangGraph-agent%20graphs-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Gemini-3.5%20Flash-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini 3.5 Flash" />
  <img src="https://img.shields.io/badge/AgentShip-deploy-FF6B35?style=for-the-badge&logo=rocket&logoColor=white" alt="AgentShip" />
</p>

<p>
  <a href="https://drive.google.com/drive/folders/1Odf5nXXHFy1TUnrkO89I67ct3VH-Gelb?usp=sharing">
    <img src="https://img.shields.io/badge/📑%20Course%20Slides-open%20in%20Drive-34A853?style=for-the-badge&logo=googledrive&logoColor=white" alt="Course slides" />
  </a>
</p>

<p>
  <img src="https://img.shields.io/badge/best%20accuracy-93%25-brightgreen?style=flat-square" alt="Best accuracy 93%" />
  <img src="https://img.shields.io/badge/at%20cost-%240.020%2Fquestion-blue?style=flat-square" alt="Cost per question" />
  <img src="https://img.shields.io/badge/8×%20cheaper-than%20module%201-orange?style=flat-square" alt="8x cheaper than module 1" />
  <img src="https://img.shields.io/badge/golden%20set-14%20questions-8b5cf6?style=flat-square" alt="Golden set" />
</p>

<p>
  <a href="https://drive.google.com/drive/folders/1Odf5nXXHFy1TUnrkO89I67ct3VH-Gelb?usp=sharing">Slides</a> ·
  <a href="#-the-modules">The Modules</a> ·
  <a href="#-setup">Setup</a> ·
  <a href="#-running-the-demos">Running</a> ·
  <a href="#-how-module-4-works">How Module 4 Works</a> ·
  <a href="#-troubleshooting">Troubleshooting</a> ·
  <a href="#-contributors">Contributors</a>
</p>

</div>

---

Course labs on **agentic retrieval-augmented generation (RAG)**. Six modules answer the *same*
question about the *same* document, each one fixing what the previous one got wrong.

- **The document.** `data/acme_10q.txt` — a quarterly report (Form 10-Q) for Acme Corp, a
  fictitious company. About 100,000 tokens.
- **The question.** *"How did Acme's Q3 margin compare to Q2, and why?"*
- **Why it is hard.** Q3's figures are in the main body. Q2's appear only in an appendix near the
  end. Answering needs two distant parts of one document — exactly where naive approaches break.

## 📚 The Modules

Each module is one file in [`demos/`](demos/). Run them in order; each builds on the one before.
The lecture slides that go with them are
[here](https://drive.google.com/drive/folders/1Odf5nXXHFy1TUnrkO89I67ct3VH-Gelb?usp=sharing).

| # | Module | The idea | What happens | Accuracy | Cost/q |
|---|---|---|---|---:|---:|
| **1** | [Just upload the PDF](demos/module_1_just_upload_the_pdf.ipynb) | Put the whole report in every prompt. | Right answer — at ~107,000 tokens per question, every question. Charts the cost. | 100% | $0.161 |
| **2** | [Send only what matters](demos/module_2_send_only_what_matters.ipynb) | Chunk the report, store the vectors in Chroma, send the 5 chunks closest to the question, with the chat history. | Finds Q3, misses the Q2 appendix, admits it cannot compare. Charted against Module 1. | 46% | $0.006 |
| **3** | [Give the model judgment](demos/module_3_give_the_model_judgment.ipynb) | Hand the model a search tool and let it decide what to look for. | Searches Q3, then Q2, gets it right — but can loop forever when the answer isn't there. | 79% | $0.014 |
| **4** | [Search with judgment](demos/module_4_search_with_judgment.ipynb) | The agent also **grades** what it retrieved, **rewrites** failed searches, and **verifies** its answer. | Right answer — and says so plainly when the report doesn't contain one. Holds a conversation, charted against Modules 1 and 2. | **93%** | **$0.020** |
| **5** | [Ship it](demos/module_5_ship_it.md) | Deploy Module 4 unchanged on [AgentShip](https://github.com/Agent-Ship/agent-ship). | HTTP API, sessions that survive restarts, execution traces. | — | — |
| **6** | [Compare them all](demos/module_6_compare_them_all.ipynb) | Run 1–4 side by side on the same questions. | The table above, reproduced live. | — | — |

Accuracy and cost come from the 14 questions in `golden.json`, scored with `gemini-3.5-flash`.
Every answer and the judge's reasoning is in `results_golden.json`.

**The punchline.** Module 1 is accurate but costs ~8× Module 4 per question, and gets worse as
documents grow. Module 2 is cheap but blind outside its 5 chunks. Module 3 recovers most of the
accuracy — all three of its failures were running out of steps mid-task. Module 4 nearly matches
Module 1's accuracy at an eighth of the cost, and its one miss was a question where it didn't find
the passage and *said so* instead of inventing a number.

> **How retrieval works here.** Chunks are embedded once and the vectors cached in `data/.index/`.
> Module 2 stores them in Chroma, an in-memory vector database, and checks that Chroma returns
> the same chunks as a plain NumPy dot product. From Module 3 on, `search_docs` queries the same
> Chroma collection. Chroma runs inside Python, so there is no extra service to start.

> **The same conversation, three ways.** Modules 1, 2 and 4 each answer the same three questions
> as one conversation: *"What was Q3 revenue?"*, *"How about Q2?"*, *"And the margin?"*. Each
> saves its token counts and cost in `data/.runs/`, and the next one charts itself against them,
> so run the modules in order.

## 🚀 Setup

Python 3.13+ and a Gemini API key. Docker is needed for Module 5 only.

**1. Get a Gemini API key.** Create one at [Google AI Studio](https://aistudio.google.com/apikey),
then enable billing for its project. The free tier allows five requests per minute and a single
Module 4 question makes five to ten. Running every notebook once costs about USD 1.20, mostly
Module 1.

**2. Install.**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .

pip install "agentship-core @ git+https://github.com/Agent-Ship/agent-ship.git@main#subdirectory=packages/agentship-core" \
            "agentship-langgraph @ git+https://github.com/Agent-Ship/agent-ship.git@main#subdirectory=packages/agentship-langgraph"

python -m ipykernel install --user --name agentic-rag-demos
```

The second command installs the current AgentShip from GitHub; the third registers the Jupyter
kernel the notebooks use.

**3. Add your key.**

```bash
cp .env.example .env
```

Put the key after `GEMINI_API_KEY=` in `.env`. Nothing else needs changing. `.env` is gitignored
and never committed. Keep comments on their own lines — VS Code loads `.env` into notebooks itself
and treats everything after the `=` as the value.

**4. Start the services (Module 5 only).**

```bash
docker compose up -d
agentship db upgrade --allow-migrations --database-url postgresql://agentship:agentship@localhost:5434/agentship
```

## ▶️ Running the Demos

**Modules 1–4 and 6.** Open the notebook in [`demos/`](demos/), select the `agentic-rag-demos`
kernel, and run all cells in order.

**Module 5.** Terminal commands only — see [`demos/module_5_ship_it.md`](demos/module_5_ship_it.md).

**Scoring it yourself.** `eval.py` addresses the systems by build order, where `v0` is Module 1 and
`v3` is Module 4:

```bash
# one question, all four systems
python eval.py --agents v0,v1,v2,v3 "How did Acme's Q3 margin compare to Q2, and why?"

# the full golden set — about 5 minutes and USD 3
python eval.py --agents v0,v1,v2,v3 --out results.json
```

It scores retrieval and generation separately: retrieval by whether the right passages were found,
generation by whether a model judges the answer supported by the retrieved text and in agreement
with the reference answer.

## 🧠 How Module 4 Works

```
question --> agent --+-- no search needed --------------------------------> verify --> answer
                     |
                     +-- search --> grade the retrieved chunks
                                      +-- relevant chunks found ----------> write the answer
                                      +-- none relevant -----------------> rewrite the query, search again
                                      +-- none relevant after 2 rewrites -> report the filing lacks it
```

- **Grade.** Chunks that don't address the search are discarded, so the answer rests only on
  relevant evidence.
- **Rewrite.** A query that finds nothing is rephrased in the report's own words — "work contracted
  but not yet delivered" becomes "remaining performance obligations".
- **Verify.** Every figure in the answer is checked against the evidence before it is returned.
- **Limits.** At most two rewrites and four searches per question, so it always terminates.

The graph lives in `agents/agentic_rag.py`. Module 4 builds it step by step; Module 5 deploys that
exact definition.

## 🗂️ Repository Structure

| Path | Contents |
|---|---|
| `demos/` | The six modules — five notebooks and Module 5's commands |
| `agents/langgraph_agent.py` | The Module 3 agent |
| `agents/agentic_rag.py` | The Module 4 graph, plus its AgentShip wrapper |
| `agents/*.yaml` | The Module 5 agents; `researcher.yaml` is a team member |
| `tools/search_docs.py` | Chunking, embedding and search, shared by every module |
| `tools/settings.py` | Configuration read from `.env` |
| `data/acme_10q.txt` | The report, generated by `data/make_acme_10q.py` |
| `golden.json` | The 14 evaluation questions and reference answers |
| `eval.py` | The scoring script |
| `results_golden.json` | The results in the table above |

## 🩺 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 429, "exceeded your current quota" | The project is on the free tier. | Enable billing in AI Studio. |
| 402, "prepayment credits are depleted" | The prepaid balance is gone. | Add credit in AI Studio. |
| 403, "Gemini API has not been used in project" | The API is disabled for that project. | Follow the link in the error, enable it, wait a few minutes. |
| 400 right after the agent's first search | The model is set as `openai/gemini-...`, which drops information Gemini needs for tool calls. | Use `gemini/gemini-3.5-flash`, as in `.env.example`. |
| Module 3 stops with `GraphRecursionError` | Module 3 has no search limit and can loop. | Expected — it's the exact limitation Module 4 fixes. |
| First Module 4 run takes a minute, later runs seconds | On macOS, iCloud Drive is syncing the folder and delaying file access. | See the note below. |
| Module 4 reports "unsupported claims", then repeats the same answer | Verification is occasionally too strict. | Harmless; costs one extra model call. |

**macOS note.** If this folder lives under Desktop or Documents with iCloud Drive on, iCloud may
delay or evict local files. Move the virtual environment out of sync with
`mv .venv .venv.nosync && ln -s .venv.nosync .venv`, and set the folder to **Keep Downloaded** in
Finder.

## 🔧 Using a Different Model

Every model setting comes from `.env`.

- **OpenAI-compatible servers** (OpenAI itself, or local Ollama / llama.cpp): set
  `OPENAI_API_KEY`, `OPENAI_API_BASE`, `DEMO_MODEL=openai/<model>`,
  `DEMO_EMBED_MODEL=openai/<embedding model>` and `DEMO_TEMPERATURE=0`.
- **Local models** have no published price — set `PRICE_AS` to the model whose prices the cost
  columns should use.
- **Agent definitions.** The YAML files in `agents/` name their model directly, so change `model:`
  there too.

The labs use one model throughout so the numbers stay comparable. With a self-hosted model, the
first Module 1 question must process the whole report before emitting anything, which can take
minutes — run Module 6 once before a live demo to warm the server's cache.

## 🔭 Going Further: The OpenSearch Modules

These six modules deliberately keep retrieval as simple as possible — vector search in an
in-memory Chroma collection, no services to run. The next step up is **real hybrid retrieval**:
BM25 keyword search and k-NN vector search in a single query, fused with reciprocal rank fusion, then reranked.

That material lives in its own repository:

> ### 👉 [**harshuljain13/opensearch-in-action**](https://github.com/harshuljain13/opensearch-in-action)
>
> Hybrid BM25 + k-NN retrieval on OpenSearch, RRF fusion, reranking, and the tuning that follows —
> the natural continuation of Module 4.

Start there once Modules 1–6 make sense: the agent design carries over unchanged, and only the
retrieval layer underneath it is swapped.

---

## 👥 Contributors

Thanks to everyone who has shaped these labs.

<div align="center">

| Contributor | Role |
|---|---|
| [**@Parv17k**](https://github.com/Parv17k) | Author and maintainer — labs, agents, evaluation harness |
| [**@harshuljain13**](https://github.com/harshuljain13) | Author and maintainer — labs, agents, evaluation harness |

</div>

Contributions are welcome. Open an issue to discuss a change, or send a pull request with a short
description of what it improves.

<div align="center">

<sub>Built for the agentic RAG course · <a href="https://github.com/Agent-Ship/agent-ship">AgentShip</a> · <a href="https://github.com/Parv17k/AgenticRAGLabs">Repository</a></sub>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ec4899,50:8b5cf6,100:6366f1&height=120&section=footer" alt="" width="100%" />

</div>
