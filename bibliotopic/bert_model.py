"""Procedimiento BERTopic (Grootendorst, 2022) implementado sin dependencias nativas para poder
ejecutarse en el navegador (Pyodide):

    embeddings (transformers.js, MiniLM)  ->  UMAP (umap-js)  ->  HDBSCAN (scikit-learn)
    ->  c-TF-IDF con reducción de palabras frecuentes  ->  fusión de temas  ->  reasignación de outliers

Los pasos de embeddings y UMAP se calculan fuera (JavaScript) y llegan aquí como matrices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .lda_model import match_topics
from .text import Corpus, topic_diversity

DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2"


def lsa_embeddings(texts, dims=256, seed=42):
    """Respaldo si no se puede descargar el modelo de lenguaje."""
    tf = TfidfVectorizer(min_df=3, max_df=0.5, stop_words="english", sublinear_tf=True)
    Z = tf.fit_transform(texts)
    emb = TruncatedSVD(n_components=min(dims, Z.shape[1] - 1), random_state=seed).fit_transform(Z)
    return normalize(emb).astype(np.float32)


# ------------------------------------------------------------------ c-TF-IDF
def class_counts(X, labels, classes):
    rows = []
    for c in classes:
        idx = np.where(labels == c)[0]
        rows.append(sp.csr_matrix(X[idx].sum(axis=0)))
    return sp.vstack(rows).tocsr()


def ctfidf(counts, reduce_frequent_words=True):
    """W(t,c) = tf_norm(t,c) · log(1 + A / f_t), igual que ClassTfidfTransformer de BERTopic."""
    tf = normalize(counts, norm="l1", axis=1)
    if reduce_frequent_words:
        tf = tf.copy()
        tf.data = np.sqrt(tf.data)
    f_t = np.asarray(counts.sum(axis=0)).ravel()
    A = counts.sum(axis=1).mean()
    idf = np.log(A / np.maximum(f_t, 1e-12) + 1)
    idf[f_t == 0] = 0
    return np.asarray((tf @ sp.diags(idf)).todense()), idf


def _cluster(reduced, min_cluster_size):
    return HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean",
                   cluster_selection_method="eom").fit_predict(reduced)


def _reduce(X, labels, nr_topics):
    """Fusiona temas por similitud c-TF-IDF (aglomerativo, enlace promedio, coseno)."""
    topics = sorted(t for t in set(labels) if t >= 0)
    if not nr_topics or len(topics) <= nr_topics:
        return labels
    W, _ = ctfidf(class_counts(X, labels, topics))
    ag = AgglomerativeClustering(n_clusters=nr_topics, metric="cosine", linkage="average").fit(W)
    mapping = {t: int(g) for t, g in zip(topics, ag.labels_)}
    return np.array([mapping.get(l, -1) for l in labels])


def _sort_by_size(labels):
    ids = [t for t, _ in pd.Series(labels[labels >= 0]).value_counts().items()]
    m = {t: i for i, t in enumerate(ids)}
    return np.array([m.get(l, -1) for l in labels])


def fit_topics(X, reduced, nr_topics, min_cluster_size):
    raw = _cluster(reduced, min_cluster_size)
    native = len(set(raw) - {-1})
    lab = _sort_by_size(_reduce(X, raw, nr_topics))
    k = int(lab.max()) + 1 if (lab >= 0).any() else 0
    W, idf = ctfidf(class_counts(X, lab, list(range(k))))
    return lab, native, W, idf


def reassign_outliers(X, labels, W, idf):
    out = np.where(labels < 0)[0]
    if len(out) == 0:
        return labels.copy()
    D = normalize(X[out], norm="l1", axis=1)
    D.data = np.sqrt(D.data)
    D = D @ sp.diags(idf)
    sim = cosine_similarity(D, W)
    new = labels.copy()
    new[out] = sim.argmax(axis=1)
    return new


class BERTResult:
    def __init__(self, corpus: Corpus, reduced_list, xy, emb_name, nr_topics=None,
                 min_cluster_size=None):
        """reduced_list: lista de matrices UMAP (n x 5), la primera es la principal y el resto
        (otras semillas) se usan para la estabilidad."""
        X = corpus.X.tocsr()
        n = corpus.n_docs
        self.emb_name = emb_name
        self.min_cluster_size = min_cluster_size or max(10, int(round(n * 0.005)))
        self.topics_raw, self.native_topics, self.ctfidf, idf = fit_topics(
            X, reduced_list[0], nr_topics, self.min_cluster_size)
        self.k = self.ctfidf.shape[0]
        self.outlier_rate = float((self.topics_raw == -1).mean())
        self.topics = reassign_outliers(X, self.topics_raw, self.ctfidf, idf)
        vocab = corpus.vocab
        self.top10 = [list(vocab[np.argsort(-r)[:10]]) for r in self.ctfidf]
        self.top25 = [list(vocab[np.argsort(-r)[:25]]) for r in self.ctfidf]
        self.npmi = corpus.npmi(self.top10)
        self.umass = corpus.umass(self.top10)
        self.diversity = topic_diversity(self.top25)
        self.theta = np.zeros((n, self.k))
        self.theta[np.arange(n), self.topics] = 1.0
        self.prevalence = self.theta.mean(axis=0)
        self.sizes = np.bincount(self.topics_raw[self.topics_raw >= 0], minlength=self.k)
        self.ids = [f"B{i + 1}" for i in range(self.k)]
        self.labels = [f"B{i + 1}: " + ", ".join(w[:3]) for i, w in enumerate(self.top10)]
        self.xy = np.asarray(xy) if xy is not None else None
        self.n_stab = len(reduced_list) - 1
        if self.n_stab:
            sims = []
            for red in reduced_list[1:]:
                _, _, W2, _ = fit_topics(X, red, nr_topics or self.k, self.min_cluster_size)
                r, c, S = match_topics(self.ctfidf, W2)
                v = np.zeros(self.k)
                v[r] = S[r, c]
                sims.append(v)
            self.stab = np.mean(sims, axis=0)
        else:
            self.stab = np.full(self.k, np.nan)

    def topic_table(self):
        return pd.DataFrame({
            "Topic": self.ids,
            "Label": self.labels,
            "Top terms": [", ".join(w) for w in self.top10],
            "Prevalence (%)": (100 * self.prevalence).round(2),
            "Documents (before outlier reassignment)": self.sizes,
            "NPMI": np.round(self.npmi, 3),
            "Stability": np.round(self.stab, 3),
        })
