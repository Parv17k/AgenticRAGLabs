# Module 5 · Ship it

The graph from Module 4, deployed unchanged on
[AgentShip](https://github.com/Agent-Ship/agent-ship) — an HTTP API, sessions that survive a
restart, and execution traces. There is no notebook for this module; it runs from the terminal.

Start the services first (from the repo root):

```bash
docker compose up -d
agentship db upgrade --allow-migrations --database-url postgresql://agentship:agentship@localhost:5434/agentship
```

## One agent, one tool

```bash
agentship run agents/calc.yaml --input "What is 17 times 23?"
```

## The Module 4 graph, deployed without modification

```bash
agentship run agents/agentic_rag.yaml --input "How did Acme's Q3 margin compare to Q2, and why?"
```

## A team, where a coordinator assigns each question to a specialist

```bash
agentship run agents/team.yaml --input "Compare Q3 and Q2 revenue."
```

## Serve every agent over HTTP

```bash
agentship serve --agents-dir agents
```

While the server runs:

| What | Where |
|---|---|
| AgentShip Studio, to converse with any agent | http://127.0.0.1:8000/studio (key: `dev-secret`) |
| Phoenix, to inspect model calls and searches | http://localhost:6006 |
| REST API | `POST /v1/agents/agentic-rag:invoke` with `{"input": "...", "session_id": "..."}` |

Requests that share a `session_id` are one conversation, and it is retained across server
restarts.
