"""search_docs on OpenSearch (v4): structural chunks, hybrid BM25 + k-NN, RRF, then a reranker.

Same signature and output as tools/search_docs.py, so an agent cannot tell which one it calls.
What changes is everything behind it (module 04):

    chunk     start a chunk at every heading, keep tables whole, prepend a context header
              ("[Acme 10-Q · Q3 2026 · Note 3. Remaining Performance Obligations]")
    index     one OpenSearch index with a BM25 text field and an HNSW k-NN vector field
    retrieve  hybrid query: BM25 and k-NN in one request, fused with RRF (normalization on <2.19)
    rerank    a cross-encoder rescores the top RETRIEVE_K, and only the best k go to the model

    docker compose up -d opensearch
    python tools/opensearch_search.py --reindex "How many shares did Acme buy back?"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import httpx

try:
    from . import search_docs as basic
    from .settings import API_BASE, EMBED_MODEL, HEADERS, RERANK_MODEL
except ImportError:  # run as a script
    import search_docs as basic
    from settings import API_BASE, EMBED_MODEL, HEADERS, RERANK_MODEL

OS_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200").rstrip("/")
INDEX = "acme-filings"
PIPELINE = "acme-hybrid"
SIZE, OVERLAP = 400, 50
RETRIEVE_K = 30  # wide net for the reranker; only k of these reach the model

os_client = httpx.Client(base_url=OS_URL, timeout=60)


# ---------------------------------------------------------------- ingest

def structural_chunks(text: str, size: int = SIZE, overlap: int = OVERLAP) -> list[dict]:
    """A new chunk at every heading, packed by paragraph up to `size` tokens, with a context header."""
    chunks, cur, section = [], [], ""

    def flush() -> None:
        if cur:
            header = f"[Acme 10-Q · Q3 2026 · {section.split(' (')[0][:80]}]"
            chunks.append({"text": header + "\n" + "\n\n".join(cur), "sections": [section]})

    for para in text.split("\n\n"):
        first = para.strip().split("\n")[0]
        if basic.is_heading(first):
            flush()
            cur, section = [], first
        for piece in ([para] if basic.ntok(para) <= size else para.split("\n")):
            if cur and basic.ntok("\n\n".join(cur + [piece])) > size:
                flush()
                cur = [basic.enc.decode(basic.enc.encode("\n\n".join(cur))[-overlap:])]
            cur.append(piece)
    flush()
    return [c for c in chunks if c["text"].strip()]


def _put_pipeline() -> str:
    """Hybrid score fusion: RRF where the cluster has it (2.19+), min-max normalization otherwise."""
    rrf = {"phase_results_processors": [
        {"score-ranker-processor": {"combination": {"technique": "rrf", "rank_constant": 60}}}]}
    if os_client.put(f"/_search/pipeline/{PIPELINE}", json=rrf).is_success:
        return "rrf"
    norm = {"phase_results_processors": [{"normalization-processor": {
        "normalization": {"technique": "min_max"},
        "combination": {"technique": "arithmetic_mean", "parameters": {"weights": [0.5, 0.5]}}}}]}
    os_client.put(f"/_search/pipeline/{PIPELINE}", json=norm).raise_for_status()
    return "min_max"


def ensure_index(reindex: bool = False) -> dict:
    """Create and fill the index unless it already holds this document under this embedding model."""
    text = basic.DOC.read_text()
    fingerprint = hashlib.sha256(f"{text}|{EMBED_MODEL}|{SIZE}|{OVERLAP}".encode()).hexdigest()[:16]
    current = os_client.get(f"/{INDEX}/_mapping")
    if current.is_success and not reindex:
        meta = current.json()[INDEX]["mappings"].get("_meta", {})
        if meta.get("fingerprint") == fingerprint:
            return meta | {"fusion": _put_pipeline()}
    os_client.delete(f"/{INDEX}")

    chunks = structural_chunks(text)
    vectors = basic.embed([c["text"] for c in chunks])
    meta = {"fingerprint": fingerprint, "embed_model": EMBED_MODEL, "chunks": len(chunks)}
    os_client.put(f"/{INDEX}", json={
        "settings": {"index.knn": True, "number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"_meta": meta, "properties": {
            "text": {"type": "text", "analyzer": "english"},
            "embedding": {"type": "knn_vector", "dimension": int(vectors.shape[1]),
                          "method": {"name": "hnsw", "engine": "faiss", "space_type": "innerproduct",
                                     "parameters": {"m": 16, "ef_construction": 128}}},
            "section": {"type": "keyword"},
            "chunk_id": {"type": "integer"},
            "doc_type": {"type": "keyword"},
            "date": {"type": "date"},
        }},
    }).raise_for_status()
    lines = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        lines.append(json.dumps({"index": {"_index": INDEX, "_id": str(i)}}))
        lines.append(json.dumps({"text": chunk["text"], "embedding": vector.tolist(), "chunk_id": i,
                                 "section": chunk["sections"][0], "doc_type": "10-Q", "date": "2026-09-30"}))
    resp = os_client.post("/_bulk?refresh=true", content="\n".join(lines) + "\n",
                          headers={"Content-Type": "application/x-ndjson"})
    resp.raise_for_status()
    assert not resp.json()["errors"], "bulk indexing failed"
    return meta | {"fusion": _put_pipeline()}


# ---------------------------------------------------------------- query

def _hits(resp: httpx.Response) -> list[dict]:
    resp.raise_for_status()
    return [{"id": h["_source"]["chunk_id"], "section": h["_source"]["section"].split(" (")[0][:40],
             "score": round(h["_score"], 4), "text": h["_source"]["text"]} for h in resp.json()["hits"]["hits"]]


def _query_vector(query: str) -> list[float]:
    return basic.embed([basic.QUERY_PREFIX + query])[0].tolist()


def bm25(query: str, k: int = RETRIEVE_K) -> list[dict]:
    return _hits(os_client.post(f"/{INDEX}/_search", json={
        "size": k, "_source": {"excludes": ["embedding"]}, "query": {"match": {"text": query}}}))


def knn(query: str, k: int = RETRIEVE_K) -> list[dict]:
    return _hits(os_client.post(f"/{INDEX}/_search", json={
        "size": k, "_source": {"excludes": ["embedding"]},
        "query": {"knn": {"embedding": {"vector": _query_vector(query), "k": k}}}}))


def hybrid(query: str, k: int = RETRIEVE_K) -> list[dict]:
    return _hits(os_client.post(f"/{INDEX}/_search", params={"search_pipeline": PIPELINE}, json={
        "size": k, "_source": {"excludes": ["embedding"]},
        "query": {"hybrid": {"queries": [
            {"match": {"text": query}},
            {"knn": {"embedding": {"vector": _query_vector(query), "k": k}}}]}}}))


def rerank(query: str, hits: list[dict], k: int) -> list[dict]:
    """Rescore query–chunk pairs with a cross-encoder and keep the best k (skipped without a model)."""
    if not RERANK_MODEL or not hits:
        return hits[:k]
    resp = httpx.post(f"{API_BASE}/reranking", headers=HEADERS, timeout=120, json={
        "model": RERANK_MODEL, "query": query, "documents": [h["text"] for h in hits]})
    resp.raise_for_status()
    order = sorted(resp.json()["results"], key=lambda r: -r["relevance_score"])
    return [hits[r["index"]] | {"score": round(r["relevance_score"], 3)} for r in order[:k]]


_ready: dict = {}


def search_docs(query: str, k: int = 5) -> list[dict]:
    """Search Acme filings (10-Q). Use for Acme-specific numbers."""
    if not _ready:
        _ready.update(ensure_index())
    hits = rerank(query, hybrid(query), k)
    basic.CALLS.append({"query": query, "hits": hits})
    return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()
    print(ensure_index(args.reindex))
    for name, fn in [("bm25", bm25), ("knn", knn), ("hybrid", hybrid)]:
        print(f"\n{name}:")
        for h in fn(args.query, args.k):
            print(f"  chunk {h['id']:3}  {h['score']:.4f}  {h['section']}")
    print("\nhybrid + rerank:")
    for h in search_docs(args.query, args.k):
        print(f"  chunk {h['id']:3}  {h['score']:.3f}  {h['section']}")
