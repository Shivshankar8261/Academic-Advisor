"""
Hybrid retriever: BM25 (lexical) fused with dense embeddings (semantic).

Why hybrid? Academic queries mix two very different needs:
  * exact-token lookups  -- "DATA301", "Clause 12.1", "65%"   -> BM25 wins
  * paraphrased concepts -- "can I retake a failed subject?"  -> dense wins
Neither alone is sufficient, so scores are min-max normalised and blended.

If no API key is present the retriever silently degrades to BM25 + TF-IDF,
so the whole pipeline (and the evaluation harness) still runs offline.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from advisor import config
from advisor.ingest import Chunk, load_chunks

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "are", "be",
    "on", "at", "as", "by", "with", "that", "this", "it", "i", "my", "can",
    "do", "does", "if", "shall", "will", "any", "have", "has",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.N = len(corpus)
        self.doc_len = np.array([len(d) for d in corpus], dtype=float)
        self.avgdl = self.doc_len.mean() if self.N else 0.0
        self.tf = [Counter(d) for d in corpus]
        df = Counter()
        for d in corpus:
            df.update(set(d))
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(self.N)
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out


# --------------------------------------------------------------------------
# Dense embeddings (Gemini, with on-disk cache and TF-IDF fallback)
# --------------------------------------------------------------------------

class DenseIndex:
    def __init__(self, texts: list[str], api_key: str | None = None):
        self.texts = texts
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.mode = "gemini" if self.api_key else "tfidf"
        self._client = None
        self._vec = None
        self.matrix = self._build()

    # -- cache key ties the index to the exact corpus contents ------------
    def _cache_path(self):
        h = hashlib.sha1("||".join(self.texts).encode()).hexdigest()[:16]
        return config.PROCESSED / f"embeddings_{self.mode}_{h}.pkl"

    def _client_or_none(self):
        if self._client is None and self.api_key:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _embed_batch(self, texts: list[str], task: str,
                     progress: bool = False) -> np.ndarray:
        """
        Embed with free-tier rate limiting.

        The quota meters individual CONTENTS, not API calls, so a single
        request carrying 100 texts consumes 100 units. Batches are therefore
        kept small and paced, with exponential backoff on 429.
        """
        import time as _t
        client = self._client_or_none()
        vecs: list = []
        B = config.EMBED_BATCH
        n_batches = (len(texts) + B - 1) // B
        for bi, i in enumerate(range(0, len(texts), B), 1):
            chunk = texts[i:i + B]
            for attempt in range(6):
                try:
                    resp = client.models.embed_content(
                        model=config.EMBED_MODEL,
                        contents=chunk,
                        config={"task_type": task},
                    )
                    vecs.extend(e.values for e in resp.embeddings)
                    break
                except Exception as exc:                     # noqa: BLE001
                    if "429" not in str(exc) and "RESOURCE_EXHAUSTED" not in str(exc):
                        raise
                    wait = min(60, 5 * 2 ** attempt)
                    if progress:
                        print(f"    rate limited, waiting {wait}s "
                              f"(batch {bi}/{n_batches})", flush=True)
                    _t.sleep(wait)
            else:
                raise RuntimeError("embedding quota exhausted after 6 retries")
            if progress and bi % 5 == 0:
                print(f"    embedded {min(i+B, len(texts))}/{len(texts)}", flush=True)
            if bi < n_batches and config.EMBED_PAUSE:
                _t.sleep(config.EMBED_PAUSE)
        arr = np.array(vecs, dtype=np.float32)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)

    def _build(self) -> np.ndarray:
        cache = self._cache_path()
        if cache.exists():
            with open(cache, "rb") as fh:
                obj = pickle.load(fh)
            self._vec = obj.get("vectorizer")
            return obj["matrix"]

        if self.mode == "gemini":
            try:
                matrix = self._embed_batch(self.texts, "RETRIEVAL_DOCUMENT", progress=True)
                payload = {"matrix": matrix, "vectorizer": None}
            except Exception as exc:                        # noqa: BLE001
                # Quota exhaustion must not take the whole advisor down: fall
                # back to the local TF-IDF index and carry on. The abstention
                # thresholds differ per mode, so `self.mode` is updated too.
                print(f"  [retriever] dense embedding unavailable ({exc}); "
                      f"falling back to TF-IDF", flush=True)
                self.mode = "tfidf"
                return self._build()
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(
                sublinear_tf=True, ngram_range=(1, 2), min_df=1, stop_words="english"
            )
            matrix = self._vec.fit_transform(self.texts).toarray().astype(np.float32)
            matrix /= (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
            payload = {"matrix": matrix, "vectorizer": self._vec}

        with open(cache, "wb") as fh:
            pickle.dump(payload, fh)
        return matrix

    def scores(self, query: str) -> np.ndarray:
        if self.mode == "gemini":
            q = self._embed_batch([query], "RETRIEVAL_QUERY")[0]
        else:
            q = self._vec.transform([query]).toarray().astype(np.float32)[0]
            q /= (np.linalg.norm(q) + 1e-9)
        return self.matrix @ q


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


@dataclass
class Hit:
    chunk: Chunk
    score: float        # fused, min-max normalised -- for RANKING only
    bm25: float         # raw BM25, unbounded
    dense: float        # raw cosine similarity in [-1, 1]
    coverage: float     # fraction of query content-words present in the chunk

    def as_evidence(self) -> str:
        return f"[{self.chunk.citation()}]\n{self.chunk.text}"


class HybridRetriever:
    def __init__(self, chunks: list[Chunk] | None = None, api_key: str | None = None):
        self.chunks = chunks if chunks is not None else load_chunks()
        texts = [f"{c.section} {c.clause} {c.text}" for c in self.chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.dense = DenseIndex(texts, api_key=api_key)
        self.mode = self.dense.mode

    def search(self, query: str, k: int | None = None,
               doc_keys: list[str] | None = None) -> list[Hit]:
        k = k or config.TOP_K
        raw_b = self.bm25.scores(query)
        raw_d = self.dense.scores(query)
        b, d = _minmax(raw_b), _minmax(raw_d)
        w = config.BM25_WEIGHT
        fused = w * b + (1 - w) * d

        mask = np.ones(len(self.chunks), dtype=bool)
        if doc_keys:
            mask = np.array([c.doc_key in doc_keys for c in self.chunks])
        fused = np.where(mask, fused, -1.0)

        order = np.argsort(-fused)[:k]
        qterms = set(tokenize(query))
        hits = []
        for i in order:
            if fused[i] <= 0:
                continue
            ctoks = set(tokenize(self.chunks[i].text))
            cov = len(qterms & ctoks) / max(1, len(qterms))
            hits.append(Hit(self.chunks[i], float(fused[i]),
                            float(raw_b[i]), float(raw_d[i]), cov))
        return hits

    def is_out_of_corpus(self, hits: list[Hit]) -> bool:
        """
        Absolute (not rank-normalised) confidence test.

        min-max normalisation always sets the best hit to 1.0, so the fused
        score says nothing about whether the corpus actually covers the
        question. We therefore judge on two scale-free signals:
          * raw cosine similarity of the best hit
          * how many content words of the query appear in it at all
        A question is treated as unanswerable when BOTH are weak.
        """
        if not hits:
            return True
        best = max(hits, key=lambda h: h.dense)
        weak_semantic = best.dense < config.MIN_DENSE_SIM
        weak_lexical = max(h.coverage for h in hits) < config.MIN_TERM_COVERAGE
        return weak_semantic and weak_lexical


if __name__ == "__main__":
    r = HybridRetriever()
    print(f"index mode = {r.mode}, {len(r.chunks)} chunks\n")
    for q in ["What is the minimum attendance requirement?",
              "prerequisites for Machine Learning DATA301",
              "CGPA needed to be promoted to year 3",
              "How much does the campus canteen charge for lunch?"]:
        hits = r.search(q, k=3)
        print(f"Q: {q}")
        print(f"   out_of_corpus={r.is_out_of_corpus(hits)}")
        for h in hits:
            print(f"   {h.score:.3f}  {h.chunk.citation()}  :: {h.chunk.text[:100]}")
        print()
