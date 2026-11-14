"""search_docs: the retrieval tool every version from Module 2 on uses.

Chunks the Acme 10-Q exactly as the v1 notebook does (paragraph-aware, size=500, overlap=50),
embeds the chunks once, and caches the vectors in data/.index/ so later runs start instantly.
The vectors are loaded into Chroma, an in-memory vector database (Module 2 builds the same
collection by hand), and each search is a Chroma query.
v1 calls it on every message; the agents in v2 and later call it as a tool when they decide to
(tools/agentship_tool.py registers it with AgentShip under the name search_docs).
v4 keeps this signature and swaps the body for OpenSearch hybrid search.

    python tools/search_docs.py "What was Q3 revenue?" -k 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from functools import lru_cache

import chromadb
import litellm
import numpy as np
import tiktoken

try:
    from .settings import EMBED_MODEL, ROOT
except ImportError: # run as a script: python tools/search_docs.py
    from settings import EMBED_MODEL, ROOT

DATA = ROOT / "data"
DOC = DATA / "acme_10q.txt"
SIZE, OVERLAP = 500, 50
#: Qwen3 embedding models expect an instruction on the query side only.
QUERY_PREFIX = ("Instruct: Given a question about a company's quarterly filing, retrieve passages that "
                "answer it\nQuery: ") if "qwen3-embedding" in EMBED_MODEL.lower() else ""

enc = tiktoken.get_encoding("o200k_base")


def ntok(text: str) -> int:
    return len(enc.encode(text))


def is_heading(line: str) -> bool:
    return line.isupper() or line.startswith("Note ")


def split(text: str, size: int = SIZE, overlap: int = OVERLAP) -> list[dict]:
    pieces, section = [], ""
    for para in text.split("\n\n"):
        first = para.strip().split("\n")[0]
        if is_heading(first):
            section = first
        parts = [para] if ntok(para) <= size else para.split("\n")
        pieces += [(p, section) for p in parts]

    chunks, cur, sections = [], [], []
    for piece, section in pieces:
        if cur and ntok("\n\n".join(cur + [piece])) > size:
            chunks.append({"text": "\n\n".join(cur), "sections": sections})
            tail = enc.decode(enc.encode(chunks[-1]["text"])[-overlap:])
            cur, sections = [tail], []
        cur.append(piece)
        if section not in sections:
            sections.append(section)
    chunks.append({"text": "\n\n".join(cur), "sections": sections})
    return chunks


def label(chunk: dict) -> str:
    return " → ".join(s.split(" (")[0][:40] for s in chunk["sections"])


def embed(texts: list[str]) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), 32):
        resp = litellm.embedding(model=EMBED_MODEL, input=texts[i:i + 32])
        vectors += [d["embedding"] for d in resp.data]
    v = np.array(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@lru_cache(maxsize=1)
def load_index() -> tuple[list[dict], np.ndarray]:
    """Chunks and their vectors, built once per document + embedding model and cached on disk."""
    text = DOC.read_text()
    key = hashlib.sha256(f"{text}|{EMBED_MODEL}|{SIZE}|{OVERLAP}".encode()).hexdigest()[:16]
    cache = DATA / ".index" / key
    if (cache / "vectors.npy").exists():
        return json.loads((cache / "chunks.json").read_text()), np.load(cache / "vectors.npy")
    chunks = split(text)
    vectors = embed([c["text"] for c in chunks])
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "chunks.json").write_text(json.dumps(chunks))
    np.save(cache / "vectors.npy", vectors)
    return chunks, vectors


_building = threading.Lock()


@lru_cache(maxsize=1)
def _collection() -> chromadb.Collection:
    chunks, vectors = load_index()
    db = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))
    collection = db.get_or_create_collection("search_docs", metadata={"hnsw:space": "cosine"})
    collection.upsert(ids=[str(i) for i in range(len(chunks))], embeddings=vectors.tolist(),
                      documents=[c["text"] for c in chunks], metadatas=[{"section": label(c)} for c in chunks])
    return collection


def load_collection() -> chromadb.Collection:
    """The cached chunks and vectors in a Chroma collection: in memory, no server, no telemetry.

    Built once per process. The lock matters when an agent's first turn runs two searches at once:
    two threads creating Chroma clients together break each other.
    """
    with _building:
        return _collection()


#: Every call made in this process, oldest first, so eval.py can see what an agent searched for.
CALLS: list[dict] = []


def search_docs(query: str, k: int = 5) -> list[dict]:
    """Search Acme filings (10-Q). Use for Acme-specific numbers."""
    found = load_collection().query(query_embeddings=[embed([QUERY_PREFIX + query])[0].tolist()], n_results=k)
    hits = [{"id": int(i), "section": meta["section"], "score": round(1 - distance, 3), "text": text}
            for i, distance, meta, text in zip(found["ids"][0], found["distances"][0],
                                               found["metadatas"][0], found["documents"][0])]
    CALLS.append({"query": query, "hits": hits})
    return hits


def format_hits(hits: list[dict]) -> str:
    """Hits as one string, the form a model reads in a prompt or a tool result."""
    return "\n\n---\n\n".join(f"[chunk {h['id']} · {h['section']}]\n{h['text']}" for h in hits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()
    for h in search_docs(args.query, args.k):
        print(f"chunk {h['id']:3} {h['score']:.3f} {h['section']}")
