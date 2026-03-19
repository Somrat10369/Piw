# file: clustering/cluster.py
"""
Clustering Layer — groups articles by topic.

Phase 1 (MVP): keyword/TF-IDF cosine similarity (no GPU needed).
Phase 2 (upgrade): set USE_SEMANTIC_CLUSTERING=True in config.py after
                   installing sentence-transformers + faiss-cpu.
"""

from __future__ import annotations
import math
import re
from collections import defaultdict, Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collector.collector import Article

try:
    from config import USE_SEMANTIC_CLUSTERING, CLUSTER_SIMILARITY_THRESHOLD
except ImportError:
    USE_SEMANTIC_CLUSTERING = False
    CLUSTER_SIMILARITY_THRESHOLD = 0.30

# ── Keyword-based clustering (MVP) ────────────────────────────────────────────

_STOPWORDS = set("""
a an the and or but in on at to for of with by from as is was are were be been
being have has had do does did will would could should may might shall not no nor
so yet both either neither each few more most other some such than that this
those through under until up very while who which whom whose about above after
before between during into through too very just because since though although
""".split())

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w not in _STOPWORDS and len(w) > 2]

def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    total = sum(tf.values()) or 1
    return {t: (c / total) * idf.get(t, 1.0) for t, c in tf.items()}

def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[k] * b[k] for k in shared)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    return dot / (mag_a * mag_b + 1e-9)

def cluster_articles(articles: list["Article"],
                     threshold: float | None = None) -> list[list["Article"]]:
    """
    Group articles by topic similarity.
    Returns list of clusters (each cluster = list of Article).
    """
    if threshold is None:
        threshold = CLUSTER_SIMILARITY_THRESHOLD

    if USE_SEMANTIC_CLUSTERING:
        return _semantic_cluster(articles, threshold)

    # ── keyword TF-IDF path ──
    corpus_tokens = [_tokenize(a.title + " " + a.summary) for a in articles]

    # build IDF
    doc_freq: dict[str, int] = defaultdict(int)
    N = len(corpus_tokens)
    for tokens in corpus_tokens:
        for t in set(tokens):
            doc_freq[t] += 1
    idf = {t: math.log(N / (freq + 1)) + 1 for t, freq in doc_freq.items()}

    vectors = [_tfidf_vector(tokens, idf) for tokens in corpus_tokens]

    visited = [False] * len(articles)
    clusters: list[list["Article"]] = []

    for i, article in enumerate(articles):
        if visited[i]:
            continue
        cluster = [article]
        visited[i] = True
        for j in range(i + 1, len(articles)):
            if visited[j]:
                continue
            sim = _cosine(vectors[i], vectors[j])
            if sim >= threshold:
                cluster.append(articles[j])
                visited[j] = True
        clusters.append(cluster)

    return clusters


def _semantic_cluster(articles, threshold):
    """Phase 2: sentence-transformers + FAISS. Activated by config flag."""
    try:
        from sentence_transformers import SentenceTransformer
        import faiss, numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [a.title + " " + a.summary for a in articles]
        embeddings = model.encode(texts, normalize_embeddings=True)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype("float32"))

        visited = [False] * len(articles)
        clusters = []
        for i, article in enumerate(articles):
            if visited[i]:
                continue
            sims, idxs = index.search(embeddings[i:i+1].astype("float32"), len(articles))
            cluster = []
            for sim, j in zip(sims[0], idxs[0]):
                if not visited[j] and sim >= threshold:
                    cluster.append(articles[j])
                    visited[j] = True
            clusters.append(cluster)
        return clusters
    except ImportError:
        print("[Clustering] sentence-transformers/faiss not installed. Falling back to TF-IDF.")
        return cluster_articles(articles, threshold)


def pick_top_clusters(clusters: list[list["Article"]],
                      top_n: int = 5) -> list[list["Article"]]:
    """Return the N largest clusters (most-covered topics)."""
    return sorted(clusters, key=len, reverse=True)[:top_n]