"""
Retrieval: a three-stage pipeline over a local FAISS vector database.

    query ─┬─► BM25 (lexical)          top-30 ─┐
           └─► bge-small + FAISS       top-30 ─┴─► Reciprocal Rank Fusion
                                                        │
                        metadata filter (student batch) ▼
                                          cross-encoder rerank top-20 ─► top-k

Why each stage exists
---------------------
* BM25 catches exact tokens the embedder blurs: "DATA301", "Clause 7.2", "65%".
* Dense retrieval (BAAI/bge-small-en-v1.5, 384-d) catches paraphrase:
  "can I move to third year?" ~ "Progression to Year 3".
* Reciprocal Rank Fusion merges the two by RANK, not by score, so there is no
  need to put a BM25 score and a cosine on the same scale.
* The cross-encoder (ms-marco-MiniLM-L-6-v2) reads query and passage TOGETHER,
  which is far more precise than comparing two independent vectors. Its logit
  is also an ABSOLUTE relevance score -- unlike a min-max-normalised fused score,
  which always rates the best hit 1.0 -- so it is what the abstention decision
  is made on.

Vector database choice
----------------------
FAISS IndexFlatIP (exact inner product on L2-normalised vectors = cosine).
With ~400 chunks, exact search takes well under a millisecond and has perfect
recall; an approximate index (HNSW / IVF) only pays off past ~10^5 vectors.
The index is persisted to disk and keyed by a hash of the corpus + model name,
so it is rebuilt automatically when documents change and never otherwise.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from advisor import config
from advisor.ingest import Chunk, load_chunks

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")
_STOP = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "are", "be",
    "on", "at", "as", "by", "with", "that", "this", "it", "i", "my", "can",
    "do", "does", "if", "shall", "will", "any", "have", "has", "what", "which",
    "me", "am", "how", "there", "their", "they", "from", "per", "must", "need",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


# --------------------------------------------------------------------------
# Models (loaded once per process)
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device="cpu")


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL, device="cpu")


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus)
        self.doc_len = np.array([len(d) for d in corpus], dtype=float)
        self.avgdl = self.doc_len.mean() if self.N else 1.0
        self.tf = [Counter(d) for d in corpus]
        df = Counter()
        for d in corpus:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.N)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for t in tokenize(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            f = np.array([tf.get(t, 0) for tf in self.tf], dtype=float)
            out += idf * f * (self.k1 + 1) / (f + norm)
        return out


# --------------------------------------------------------------------------
# FAISS vector store
# --------------------------------------------------------------------------

class VectorStore:
    def __init__(self, chunks: list[Chunk]):
        import faiss
        self.chunks = chunks
        texts = [c.embed_text() for c in chunks]
        key = hashlib.sha1(("||".join(texts) + config.EMBED_MODEL).encode()).hexdigest()[:16]
        folder = config.VECTORSTORE_DIR
        folder.mkdir(parents=True, exist_ok=True)
        idx_path, meta_path = folder / "index.faiss", folder / "meta.json"

        if idx_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("key") == key:
                self.index = faiss.read_index(str(idx_path))
                self.built = False
                return

        vecs = embedder().encode(texts, batch_size=32, normalize_embeddings=True,
                                 show_progress_bar=False).astype("float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        faiss.write_index(self.index, str(idx_path))
        meta_path.write_text(json.dumps({
            "key": key, "model": config.EMBED_MODEL, "dim": int(vecs.shape[1]),
            "n": len(texts), "index": "IndexFlatIP (exact cosine)",
            "chunk_ids": [c.chunk_id for c in chunks]}, indent=1))
        self.built = True

    def search(self, query: str, n: int) -> tuple[np.ndarray, np.ndarray]:
        q = embedder().encode([config.EMBED_QUERY_PREFIX + query],
                              normalize_embeddings=True).astype("float32")
        sims, ids = self.index.search(q, min(n, self.index.ntotal))
        return ids[0], sims[0]

    def all_sims(self, query: str) -> np.ndarray:
        ids, sims = self.search(query, self.index.ntotal)
        out = np.zeros(self.index.ntotal, dtype="float32")
        out[ids] = sims
        return out


# --------------------------------------------------------------------------
# Hybrid retriever
# --------------------------------------------------------------------------

@dataclass
class Hit:
    chunk: Chunk
    score: float        # cross-encoder logit -- ABSOLUTE relevance, used for abstention
    bm25: float         # raw BM25
    dense: float        # raw cosine similarity
    coverage: float     # share of query content-words present in the chunk
    rrf: float = 0.0    # fused rank score (pre-rerank)

    def as_evidence(self) -> str:
        return f"[{self.chunk.citation()}]\n{self.chunk.text}"


def _rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + r + 1)
    return fused


class HybridRetriever:
    def __init__(self, chunks: list[Chunk] | None = None, api_key: str | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        self.bm25 = BM25([tokenize(c.embed_text()) for c in self.chunks])
        self.store = VectorStore(self.chunks)
        self.mode = f"{config.EMBED_MODEL.split('/')[-1]} + FAISS + BM25 (RRF) + rerank"
        self._by_parent: dict[tuple, list[int]] = {}
        for i, c in enumerate(self.chunks):
            self._by_parent.setdefault((c.doc_key, c.page, c.parent), []).append(i)

    def _allowed(self, c: Chunk, batch: str | None, doc_keys: list[str] | None) -> bool:
        if doc_keys and c.doc_key not in doc_keys:
            return False
        # metadata filter: a 2025 student must not be shown the 2022 prerequisite list
        if batch and c.doc_key == "catalogue" and c.batch and c.batch != str(batch):
            return False
        return True

    def search(self, query: str, k: int | None = None, batch: str | None = None,
               doc_keys: list[str] | None = None, rerank: bool = True) -> list[Hit]:
        k = k or config.TOP_K
        n = config.CANDIDATES

        raw_b = self.bm25.scores(query)
        bm_rank = [int(i) for i in np.argsort(-raw_b)[:n] if raw_b[i] > 0]
        d_ids, _ = self.store.search(query, n)
        dense_rank = [int(i) for i in d_ids if i >= 0]
        raw_d = self.store.all_sims(query)

        fused = _rrf([bm_rank, dense_rank])
        cand = [i for i in sorted(fused, key=fused.get, reverse=True)
                if self._allowed(self.chunks[i], batch, doc_keys)][:config.RERANK_POOL]
        if not cand:
            return []

        if rerank:
            pairs = [(query, self.chunks[i].embed_text()) for i in cand]
            ce = reranker().predict(pairs, show_progress_bar=False)
            order = [cand[j] for j in np.argsort(-ce)]
            ce_by = {cand[j]: float(ce[j]) for j in range(len(cand))}
        else:
            order, ce_by = cand, {i: fused[i] for i in cand}

        qterms = set(tokenize(query))
        hits = []
        for i in order[:k]:
            c = self.chunks[i]
            cov = len(qterms & set(tokenize(c.embed_text()))) / max(1, len(qterms))
            hits.append(Hit(c, ce_by[i], float(raw_b[i]), float(raw_d[i]), cov, fused[i]))
        return hits

    def is_out_of_corpus(self, hits: list[Hit]) -> bool:
        """Abstain when even the best passage is judged irrelevant by the
        cross-encoder. The threshold is calibrated on the test set by
        scripts/calibrate_abstention.py."""
        return (not hits) or max(h.score for h in hits) < config.RERANK_MIN


if __name__ == "__main__":
    import time
    t = time.time()
    r = HybridRetriever()
    print(f"{r.mode} | {len(r.chunks)} chunks | built={r.store.built} | {time.time()-t:.1f}s\n")
    for q in ["What is the minimum attendance requirement?",
              "Can I move to third year with a low CGPA?",
              "prerequisites for DATA301",
              "How much does a canteen meal cost?"]:
        t = time.time()
        hits = r.search(q, k=3, batch="2025")
        print(f"Q: {q}   ({(time.time()-t)*1000:.0f} ms)  abstain={r.is_out_of_corpus(hits)}")
        for h in hits:
            print(f"   ce={h.score:6.2f} cos={h.dense:.2f}  {h.chunk.citation()[:78]}")
        print()
