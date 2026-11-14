<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6366f1,50:8b5cf6,100:ec4899&height=200&section=header&text=Agentic%20RAG%20Labs&fontSize=50&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=From%20stuffing%20the%20whole%20document%20to%20an%20agent%20that%20knows%20when%20to%20search&descAlignY=58&descSize=14" alt="Agentic RAG Labs" width="100%" />

### Four RAG systems. One question. One document. Measured side by side.

<p>
  <img src="https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.13+" />
  <img src="https://img.shields.io/badge/LangGraph-agent%20graphs-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Gemini-3.5%20Flash-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini 3.5 Flash" />
  <img src="https://img.shields.io/badge/AgentShip-deploy-FF6B35?style=for-the-badge&logo=rocket&logoColor=white" alt="AgentShip" />
</p>

<p>
  <img src="https://img.shields.io/badge/OpenSearch-hybrid%20search-005EB8?style=for-the-badge&logo=opensearch&logoColor=white" alt="OpenSearch" />
  <img src="https://img.shields.io/badge/Docker-compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Jupyter-notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/Phoenix-tracing-9A4DFF?style=for-the-badge&logo=arrow&logoColor=white" alt="Phoenix" />
</p>

<p>
  <img src="https://img.shields.io/badge/accuracy-93%25-brightgreen?style=flat-square" alt="Accuracy 93%" />
  <img src="https://img.shields.io/badge/cost%2Fquestion-%240.020-blue?style=flat-square" alt="Cost per question" />
  <img src="https://img.shields.io/badge/golden%20set-14%20questions-orange?style=flat-square" alt="Golden set" />
  <img src="https://img.shields.io/badge/lectures-1%20%26%202-8b5cf6?style=flat-square" alt="Lectures 1 and 2" />
</p>

<p>
  <a href="#-the-four-versions">Four Versions</a> ·
  <a href="#-results">Results</a> ·
  <a href="#-setup">Setup</a> ·
  <a href="#-design-of-v3">Design of v3</a> ·
  <a href="#-troubleshooting">Troubleshooting</a> ·
  <a href="#-contributors">Contributors</a>
</p>

</div>

---

> **TL;DR** — Laboratory materials for Lecture 1, *Foundations*, of a university course on
> agentic retrieval-augmented generation (RAG). Build v0 → v3, then ship v3 as an API.

## 📖 Overview

The labs answer a single question about a single document with four successively more capable
systems, so that each design can be compared directly against the previous one.

- **Document.** The quarterly report (Form 10-Q) of Acme Corp, a fictitious company, stored in
  `data/acme_10q.txt`. It is approximately 100,000 tokens long.
- **Question.** "How did Acme's Q3 margin compare to Q2, and why?"

The Q3 figures appear in the main body of the report, while the Q2 figures appear only in an
appendix near the end. Answering the question therefore requires information from two distant
parts of the document, and this is where the simpler systems fail.

## ✨ The Four Versions

| Version | Approach | Behaviour on the question |
|---|---|---|
| v0, whole document | The entire report is included in every prompt. | Answers correctly, but sends about 107,000 tokens per question. |
| v1, basic RAG | The report is split into chunks, and only the five chunks most similar to the question are sent. | Retrieves the Q3 figures but not the Q2 appendix, and reports that it cannot make the comparison. |
| v2, agent | The model receives a search tool and decides for itself what to search for. | Searches for Q3 and then for Q2, and answers correctly. It has no limit on searching, however, and can loop when the information does not exist. |
| v3, agentic RAG | An agent that additionally grades the retrieved text, rewrites unsuccessful searches, and verifies its answer against the evidence. | Answers correctly, and states that the information is absent when the report does not contain it. |

In Chapter 5, v3 is deployed with [AgentShip](https://github.com/Agent-Ship/agent-ship), which
provides an HTTP API, conversation sessions that persist across server restarts, and execution
traces.

## 📊 Results

The four versions were evaluated on the 14 questions in `golden.json` using `gemini-3.5-flash`.

| Version | Correct | Input tokens per question | Cost per question (USD) | Typical latency |
|---|---:|---:|---:|---:|
| v0, whole document | 100% | 106,800 | 0.161 | 2 to 4 s |
| v1, basic RAG | 46% | 2,700 | 0.006 | 1 to 2 s |
| v2, agent | 79% | 6,700 | 0.014 | 6 to 9 s |
| v3, agentic RAG | 93% | 11,800 | 0.020 | about 6 s |

v0 is accurate but costs roughly eight times as much as v3 per question, and its cost grows with the
length of the document. v1 is inexpensive but fails whenever the answer lies outside the five
retrieved chunks. v2 recovers most of the accuracy; all three of its errors occurred when it reached
its step limit without finishing. v3 approaches the accuracy of v0 at one eighth of the cost. Its
single error was a question for which it did not find the relevant passage and said so, rather than
producing an unsupported figure.

The answers and the evaluator's reasoning for every question are recorded in `results_golden.json`.

## 🎓 Correspondence with the Lecture Slides

| Chapter | Slides | Content | Material |
|---|---|---|---|
| 1. Just upload the PDF | 12 | v0 | `v0_whole_doc.ipynb` |
| 2. Send only what matters | 23–24 | v1 | `v1_basic_rag.ipynb` |
| 3. Give the model judgment | 35–45 | v2, built step by step in LangGraph | `v2_langgraph_agent.ipynb` |
| 4. Search with judgment | 48–54 | v3: grading, rewriting and verification | `v3_agentic_rag.ipynb` |
| 5. Ship it | 59–65 | v3 as a service, a single agent and a team | `agents/*.yaml` and the commands below |
| Finale | 68–69 | Comparison of v0 to v3 | `run_all.ipynb` |

The notebooks are intended to be run in order, since each builds on the previous one.

## 🚀 Setup

The labs require Python 3.13 or later and a Gemini API key. Docker is required for Chapter 5 only.

### Step 1. Obtain a Gemini API key

1. Create a key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Enable billing for the key's project in AI Studio. The free tier permits only five requests per
   minute, whereas a single v3 question makes between five and ten. Running every notebook once
   costs approximately USD 1.20, most of it for v0.

### Step 2. Install the software

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .

pip install "agentship-core @ git+https://github.com/Agent-Ship/agent-ship.git@main#subdirectory=packages/agentship-core" \
            "agentship-langgraph @ git+https://github.com/Agent-Ship/agent-ship.git@main#subdirectory=packages/agentship-langgraph"

python -m ipykernel install --user --name agentic-rag-demos
```

The second command installs the current version of AgentShip from GitHub. The last command
registers the Jupyter kernel that the notebooks use.

### Step 3. Configure the key

```bash
cp .env.example .env
```

Open `.env` and enter the key after `GEMINI_API_KEY=`. No other settings need to be changed.
The file `.env` is listed in `.gitignore` and is never committed.
Keep comments in `.env` on lines of their own: VS Code loads the file into notebooks itself and
treats anything after the `=` as part of the value.

### Step 4. Start the supporting services (Chapter 5 only)

```bash
docker compose up -d
agentship db upgrade --allow-migrations --database-url postgresql://agentship:agentship@localhost:5434/agentship
```

This starts PostgreSQL, which stores agent sessions; Phoenix, which displays traces; and
OpenSearch, which is used in Lecture 2.

## ▶️ Running the Labs

### Chapters 1 to 4

Open each notebook, select the `agentic-rag-demos` kernel, and run all cells in order.

### Chapter 5

```bash
# Slide 59: a single agent with one tool
agentship run agents/calc.yaml --input "What is 17 times 23?"

# Slide 60: the v3 graph from Chapter 4, deployed without modification
agentship run agents/agentic_rag.yaml --input "How did Acme's Q3 margin compare to Q2, and why?"

# Slide 62: a team in which a coordinator assigns each question to a specialist
agentship run agents/team.yaml --input "Compare Q3 and Q2 revenue."

# Slide 65: serve all agents through an HTTP API
agentship serve --agents-dir agents
```

While the server is running, the following are available:

| Service | Location |
|---|---|
| AgentShip Studio, for conversing with any agent | http://127.0.0.1:8000/studio (key: `dev-secret`) |
| Phoenix, for inspecting model calls and searches | http://localhost:6006 |
| REST API | `POST /v1/agents/agentic-rag:invoke` with the body `{"input": "...", "session_id": "..."}` |

Requests that share a `session_id` belong to the same conversation. The conversation is retained
even if the server is stopped and restarted.

### Evaluation

```bash
# Score every version on one question
python eval.py --agents v0,v1,v2,v3 "How did Acme's Q3 margin compare to Q2, and why?"

# Score every version on the full set of 14 questions (about 5 minutes and USD 3)
python eval.py --agents v0,v1,v2,v3 --out results.json
```

`eval.py` measures retrieval and generation separately. Retrieval is scored by whether the
correct passages were found. Generation is scored by whether the answer is supported by the
retrieved text and whether it agrees with the reference answer, as judged by a language model.

## 🧠 Design of v3

```
question --> agent --+-- no search required ------------------------------> verify --> answer
                     |
                     +-- search --> grade the retrieved chunks
                                      +-- relevant chunks found ----------> agent writes the answer
                                      +-- none relevant -----------------> rewrite the query, search again
                                      +-- none relevant after 2 rewrites -> report that the filing lacks it
```

- **Grade.** Retrieved chunks that do not address the search are discarded, so the answer is
  built only on relevant evidence.
- **Rewrite.** A query that finds nothing relevant is rephrased in the terminology of the
  report. For example, "work contracted but not yet delivered" becomes "remaining performance
  obligations".
- **Verify.** Before the answer is returned, each figure in it is checked against the evidence.
- **Limits.** At most two rewrites and four searches are permitted per question, so the process
  always terminates.

The graph is defined in `agents/agentic_rag.py`. The Chapter 4 notebook constructs the same graph
step by step, and Chapter 5 deploys that definition on AgentShip without modification.

## 🗂️ Repository Structure

| Path | Contents |
|---|---|
| `v0_whole_doc.ipynb` to `v3_agentic_rag.ipynb` | The notebooks for Chapters 1 to 4 |
| `run_all.ipynb` | Side-by-side comparison of v0 to v3 |
| `agents/langgraph_agent.py` | v2, the agent from Chapter 3 |
| `agents/agentic_rag.py` | v3, the graph from Chapter 4, and its AgentShip wrapper |
| `agents/*.yaml` | The Chapter 5 agents; `researcher.yaml` is a member of the team |
| `tools/search_docs.py` | Chunking, embedding and search, shared by all versions |
| `tools/settings.py` | Configuration read from `.env` |
| `data/acme_10q.txt` | The report, generated by `data/make_acme_10q.py` |
| `golden.json` | The 14 evaluation questions and their reference answers |
| `eval.py` | The evaluation script |
| `results_golden.json` | The results reported above |
| `lecture2/` | v4 and v5, for Lecture 2 |

## 🩺 Troubleshooting

| Symptom | Cause | Remedy |
|---|---|---|
| Error 429, "exceeded your current quota" | The project is on the free tier. | Enable billing in AI Studio. |
| Error 402, "prepayment credits are depleted" | The project's prepaid balance is exhausted. | Add credit in AI Studio. |
| Error 403, "Gemini API has not been used in project" | The Gemini API is disabled for the project. | Follow the link in the error message, enable the API, and wait a few minutes. |
| Error 400 immediately after the agent's first search | The model is configured as `openai/gemini-...`, which discards information Gemini requires on tool calls. | Set the model to `gemini/gemini-3.5-flash`, as in `.env.example`. |
| v2 stops with `GraphRecursionError` | v2 has no search limit and can loop. | This is expected; it is the limitation that v3 addresses. |
| The first v3 run takes about a minute and later runs take seconds | On macOS, iCloud Drive is synchronising the folder and delays file access. | See the note below. |
| v3 reports "unsupported claims" and then repeats the same answer | The verification step is occasionally too strict. | No action is needed; the cost is one additional model call. |

**Note for macOS users.** If this folder is located under Desktop or Documents and iCloud Drive is
enabled, iCloud may delay or remove local files during synchronisation. Move the virtual
environment out of synchronisation with `mv .venv .venv.nosync && ln -s .venv.nosync .venv`, and
set the folder to **Keep Downloaded** in Finder.

## 🔧 Using a Different Model

All model settings are read from `.env`.

- **OpenAI-compatible servers**, including OpenAI itself and local servers such as Ollama or
  llama.cpp: set `OPENAI_API_KEY`, `OPENAI_API_BASE`, `DEMO_MODEL=openai/<model>`,
  `DEMO_EMBED_MODEL=openai/<embedding model>` and `DEMO_TEMPERATURE=0`.
- **Local models** have no published price. Set `PRICE_AS` to the model whose prices should be used
  in the cost columns.
- **Agent definitions.** The YAML files in `agents/` specify their model directly, so the `model:`
  field must also be changed there.

The slides refer to `openai/gpt-4o-mini`, and to `anthropic/claude-sonnet-4-6` for the team's
writer. The labs use a single model throughout so that the results remain comparable.

With a self-hosted model, the first v0 question must process the entire report before producing
any output, which can take several minutes. Running `run_all.ipynb` once before a live
demonstration warms the server's cache.

## 🔭 Lecture 2

The directory `lecture2/` and the script `tools/opensearch_search.py` support Lecture 2. v4 runs
v3 on OpenSearch hybrid search, which combines keyword and vector retrieval, with reranking; v5
tunes it as a team. Both require the OpenSearch container started by `docker compose up -d` and an
index built with the embedding model configured in `.env`.

---

## 👥 Contributors

Thanks to everyone who has shaped these labs.

<div align="center">

| Contributor | Role |
|---|---|
| [**@Parv17k**](https://github.com/Parv17k) | Author and maintainer — labs, agents, evaluation harness |
| [**@harshuljain13**](https://github.com/harshuljain13) | Author and maintainer — labs, agents, evaluation harness |

</div>

Contributions are welcome. Open an issue to discuss a change, or send a pull request with
a short description of what it improves.

<div align="center">

<sub>Built for the agentic RAG lecture series · <a href="https://github.com/Agent-Ship/agent-ship">AgentShip</a> · <a href="https://github.com/Parv17k/AgenticRAGLabs">Repository</a></sub>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ec4899,50:8b5cf6,100:6366f1&height=120&section=footer" alt="" width="100%" />

</div>
