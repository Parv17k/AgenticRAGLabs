"""The Office Hours Bot's knowledge bases and tools.

Two small knowledge bases, each a markdown file split into one chunk per `## ` section:

    data/syllabus.md  → search_syllabus   (deadlines, grading, policies, people)
    data/concepts.md  → search_concepts   (course topics: RAG, embeddings, LangGraph, ...)

Search is BM25 in plain Python: no server, no embeddings bill, and the same answer every run.
Each search tool is an AgentShip `Tool`, so a YAML agent lists it as `kb:search_syllabus`.

`send_ta_email` is the one tool that changes the world. It is marked `side_effecting=True`, so an
agent with `confirm_writes: true` pauses for a human before it runs. "Sending" appends the email
to outbox.jsonl, so you can see exactly what would have gone out.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from agentship.tools import Tool
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTBOX = HERE / "outbox.jsonl"

STOPWORDS = set("""a an and are as at be by can do does for from has have how i in is it its my
of on or so that the their this to was we what when where which who why will with you your""".split())


# ------------------------------------------------------------------ chunking

def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def load_chunks(path: Path) -> list[dict]:
    """One chunk per `## ` section: {id, title, text}. The id is what answers cite."""
    chunks, title, lines = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            if title:
                chunks.append({"id": slug(title), "title": title, "text": " ".join(lines).strip()})
            title, lines = line[3:].strip(), []
        elif title and line.strip():
            lines.append(line.strip())
    if title:
        chunks.append({"id": slug(title), "title": title, "text": " ".join(lines).strip()})
    return chunks


# ------------------------------------------------------------------ BM25

def stem(word: str) -> str:
    """A tiny stemmer, so "grades" matches "grade" and "extensions" matches "extension"."""
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 over a list of chunks. Titles count twice: they are short and on-topic."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(f"{c['title']} {c['title']} {c['text']}")) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.lengths)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 3) -> list[dict]:
        terms = tokenize(query)
        scored = []
        for chunk, doc, length in zip(self.chunks, self.docs, self.lengths):
            score = 0.0
            for t in terms:
                if t in doc:
                    tf = doc[t]
                    score += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: -pair[0])
        return [c | {"score": round(s, 2)} for s, c in scored[:k]]


SYLLABUS = BM25(load_chunks(DATA / "syllabus.md"))
CONCEPTS = BM25(load_chunks(DATA / "concepts.md"))

SEPARATOR = "\n\n---\n\n"  # between passages in a search result


def format_hits(hits: list[dict]) -> str:
    """Passages as the model sees them: an id to cite, a title, the text."""
    if not hits:
        return "No results."
    return SEPARATOR.join(f"[{h['id']}] {h['title']}\n{h['text']}" for h in hits)


# ------------------------------------------------------------------ AgentShip tools

class SearchArgs(BaseModel):
    query: str = Field(description="A short, focused search query, e.g. 'late policy slip days'.")


search_syllabus = Tool(
    name="search_syllabus",
    description=("Search the AGNT 6100 syllabus: dates, deadlines, grading, late policy, extensions, "
                 "exams, office hours, staff and course policies. One topic per search."),
    func=lambda query: format_hits(SYLLABUS.search(query, k=3)),
    args_schema=SearchArgs,
)

search_concepts = Tool(
    name="search_concepts",
    description=("Search the AGNT 6100 concept notes: explanations of course topics such as embeddings, "
                 "BM25, hybrid search, RAG, agents, LangGraph, checkpointing and evaluation."),
    func=lambda query: format_hits(CONCEPTS.search(query, k=3)),
    args_schema=SearchArgs,
)


class EmailArgs(BaseModel):
    to: str = Field(description="The TA's email address.")
    subject: str = Field(description="A short subject line.")
    body: str = Field(description="The email body, written on the student's behalf.")


def _send(to: str, subject: str, body: str) -> str:
    record = {"sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "to": to, "subject": subject, "body": body}
    with OUTBOX.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return f"Email sent to {to} with subject {subject!r}."


send_ta_email = Tool(
    name="send_ta_email",
    description="Send an email to a course TA on the student's behalf. Needs human approval.",
    func=_send,
    args_schema=EmailArgs,
    # TODO (Step 5b): this tool changes the world. Tell AgentShip, so confirm_writes can pause it.
)


if __name__ == "__main__":  # python kb.py "late policy" — try the search by hand
    import sys

    q = " ".join(sys.argv[1:]) or "when is the midterm"
    print(f"syllabus ({len(SYLLABUS.chunks)} chunks):\n{format_hits(SYLLABUS.search(q))}\n")
    print(f"concepts ({len(CONCEPTS.chunks)} chunks):\n{format_hits(CONCEPTS.search(q))}")
