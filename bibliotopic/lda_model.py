"""LDA: selección del número de temas (4 métricas), ajuste final y estabilidad entre semillas."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

from .text import Corpus, topic_diversity


def fit_lda(X, k, seed=42, max_iter=60):
    m = LatentDirichletAllocation(n_components=k, learning_method="batch", max_iter=max_iter,
                                  random_state=seed, doc_topic_prior=None, topic_word_prior=None,
                                  evaluate_every=-1, n_jobs=-1)
    theta = m.fit_transform(X)
    beta = m.components_ / m.components_.sum(axis=1, keepdims=True)
    return m, theta, beta


def top_words(beta, vocab, n=10):
    return [list(vocab[np.argsort(-row)[:n]]) for row in beta]


def cao_juan(beta):  # Cao et al. (2009): menor es mejor
    s = cosine_similarity(beta)
    k = len(beta)
    return float((s.sum() - k) / (k * (k - 1)))


def arun(beta, theta, doc_len):  # Arun et al. (2010): menor es mejor
    cm1 = np.linalg.svd(beta, compute_uv=False)
    cm1 = cm1 / cm1.sum()
    cm2 = doc_len @ theta
    cm2 = np.sort(cm2 / cm2.sum())[::-1]
    cm1 = np.sort(cm1)[::-1]
    eps = 1e-12
    return float(np.sum(cm1 * np.log((cm1 + eps) / (cm2 + eps))) +
                 np.sum(cm2 * np.log((cm2 + eps) / (cm1 + eps))))


def deveaud(beta):  # Deveaud et al. (2014): mayor es mejor
    k = len(beta)
    js = [jensenshannon(beta[i], beta[j]) ** 2 for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(js)) if js else 0.0


def select_k(corpus: Corpus, ks, seed=42, max_iter=40, progress=None) -> pd.DataFrame:
    doc_len = np.asarray(corpus.X.sum(axis=1)).ravel()
    rows = []
    for i, k in enumerate(ks):
        m, theta, beta = fit_lda(corpus.X, k, seed, max_iter)
        tw = top_words(beta, corpus.vocab, 10)
        rows.append({
            "K": k,
            "NPMI": float(np.nanmean(corpus.npmi(tw))),
            "CaoJuan2009": cao_juan(beta),
            "Arun2010": arun(beta, theta, doc_len),
            "Deveaud2014": deveaud(beta),
            "Perplexity": float(m.perplexity(corpus.X)),
            "Diversity": topic_diversity(top_words(beta, corpus.vocab, 25)),
        })
        if progress:
            progress((i + 1) / len(ks), f"LDA K={k}")
    t = pd.DataFrame(rows)

    def mm(s, invert=False):
        r = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return 1 - r if invert else r
    t["Score"] = (mm(t["NPMI"]) + mm(t["Deveaud2014"]) + mm(t["CaoJuan2009"], True)
                  + mm(t["Arun2010"], True)) / 4
    return t


def match_topics(A, B):
    """Empareja filas de A y B (Hungarian sobre similitud coseno)."""
    s = cosine_similarity(A, B)
    r, c = linear_sum_assignment(-s)
    return r, c, s


def stability(corpus: Corpus, k, beta_ref, seeds=(1, 2, 3, 4), max_iter=60):
    """Similitud media de cada tema de referencia con su mejor par en réplicas con otra semilla."""
    sims = np.zeros((len(seeds), k))
    for i, s in enumerate(seeds):
        _, _, b = fit_lda(corpus.X, k, seed=s, max_iter=max_iter)
        r, c, S = match_topics(beta_ref, b)
        sims[i, r] = S[r, c]
    return sims.mean(axis=0)


class LDAResult:
    def __init__(self, corpus: Corpus, k: int, seed=42, n_stab=4, max_iter=60):
        self.k = k
        self.model, self.theta, self.beta = fit_lda(corpus.X, k, seed, max_iter)
        self.top10 = top_words(self.beta, corpus.vocab, 10)
        self.top25 = top_words(self.beta, corpus.vocab, 25)
        self.npmi = corpus.npmi(self.top10)
        self.umass = corpus.umass(self.top10)
        self.diversity = topic_diversity(self.top25)
        self.dominant = self.theta.argmax(axis=1)
        self.prevalence = self.theta.mean(axis=0)
        self.stab = (stability(corpus, k, self.beta, seeds=tuple(range(seed + 1, seed + 1 + n_stab)),
                               max_iter=max_iter) if n_stab else np.full(k, np.nan))
        self.labels = [f"L{i + 1}: " + ", ".join(w[:3]) for i, w in enumerate(self.top10)]
        self.ids = [f"L{i + 1}" for i in range(k)]

    def topic_table(self):
        return pd.DataFrame({
            "Topic": self.ids,
            "Label": self.labels,
            "Top terms": [", ".join(w) for w in self.top10],
            "Prevalence (%)": (100 * self.prevalence).round(2),
            "Dominant docs": np.bincount(self.dominant, minlength=self.k),
            "NPMI": np.round(self.npmi, 3),
            "Stability": np.round(self.stab, 3),
        })
