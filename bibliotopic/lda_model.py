"""LDA: selección del número de temas (cinco métricas, dos etapas), ajuste final y estabilidad entre semillas.

Parámetros principales (todos se registran en la configuración efectiva):
- method: "batch" (EM variacional por lotes, determinista para una semilla; por defecto) u "online"
  (Hoffman et al., 2010; más rápido, usado en versiones anteriores).
- max_iter y tol: número máximo de iteraciones y criterio de convergencia (cambio de la perplejidad,
  evaluada cada 5 iteraciones en el modo batch).
- doc_prior / word_prior: priors de Dirichlet α y η; None = 1/K (el valor por defecto de scikit-learn).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

from .text import Corpus, topic_diversity

LDA_DEFAULTS = dict(method="batch", max_iter=100, tol=0.1, doc_prior=None, word_prior=None)


def lda_params(k, method="batch", max_iter=100, tol=0.1, doc_prior=None, word_prior=None, **_):
    """Parámetros efectivos (con los priors resueltos a número) para registrarlos en la configuración."""
    return {"learning_method": method, "max_iter": int(max_iter),
            "perp_tol": float(tol) if method == "batch" else None,
            "doc_topic_prior": float(doc_prior) if doc_prior else 1.0 / k,
            "topic_word_prior": float(word_prior) if word_prior else 1.0 / k}


def fit_lda(X, k, seed=42, max_iter=100, method="batch", tol=0.1, doc_prior=None, word_prior=None):
    p = lda_params(k, method, max_iter, tol, doc_prior, word_prior)
    kw = dict(n_components=k, learning_method=p["learning_method"], max_iter=p["max_iter"],
              doc_topic_prior=p["doc_topic_prior"], topic_word_prior=p["topic_word_prior"],
              random_state=seed, n_jobs=None)
    if method == "batch":
        kw.update(evaluate_every=10, perp_tol=p["perp_tol"])
    else:
        kw.update(batch_size=256, learning_offset=10.0, evaluate_every=-1)
    m = LatentDirichletAllocation(**kw)
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


def ranking_agreement(tw_a, tw_b, top=10):
    """Acuerdo entre dos soluciones (Greene et al., 2014): Jaccard medio de las listas de top-términos de los
    temas emparejados con el algoritmo húngaro."""
    sa = [set(t[:top]) for t in tw_a]
    sb = [set(t[:top]) for t in tw_b]
    J = np.array([[len(a & b) / len(a | b) for b in sb] for a in sa])
    r, c = linear_sum_assignment(-J)
    return float(J[r, c].mean())


METRICS = ["CaoJuan2009", "Arun2010", "Deveaud2014", "NPMI", "Stability (Greene 2014)"]
MINIMIZE = {"CaoJuan2009", "Arun2010"}


def _fit_metrics(job):
    """Un ajuste (K, semilla) -> métricas. Función de nivel de módulo para poder paralelizarla."""
    X, corpus, doc_len, k, seed, params = job
    m, theta, beta = fit_lda(X, k, seed, **params)
    tw = top_words(beta, corpus.vocab, 10)
    return {"K": k, "seed": seed, "NPMI": float(np.nanmean(corpus.npmi(tw))),
            "CaoJuan2009": cao_juan(beta), "Arun2010": arun(beta, theta, doc_len), "Deveaud2014": deveaud(beta),
            "Perplexity": float(m.perplexity(X)), "Iterations": int(m.n_iter_),
            "Diversity": topic_diversity(top_words(beta, corpus.vocab, 25)), "_tw": tw}


def _evaluate(corpus, ks, seeds, params, progress, done, total, executor=None):
    doc_len = np.asarray(corpus.X.sum(axis=1)).ravel()
    jobs = [(corpus.X, corpus, doc_len, k, s, params) for k in ks for s in seeds]
    results = []
    it = executor.map(_fit_metrics, jobs) if executor is not None else map(_fit_metrics, jobs)
    for r in it:
        results.append(r)
        done[0] += 1
        if progress:
            progress(min(done[0] / total, 1.0), f"LDA K={r['K']} (semilla {r['seed']}) · {done[0]}/{total}")
    rows = []
    for k in ks:
        rs = [r for r in results if r["K"] == k]
        row = {"K": k}
        for m in ("NPMI", "CaoJuan2009", "Arun2010", "Deveaud2014", "Perplexity", "Diversity", "Iterations"):
            v = np.array([r[m] for r in rs], float)
            row[m] = float(v.mean())
            row[m + " SD"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        pairs = [(a, b) for i, a in enumerate(rs) for b in rs[i + 1:]]
        row["Stability (Greene 2014)"] = (float(np.mean([ranking_agreement(a["_tw"], b["_tw"]) for a, b in pairs]))
                                          if pairs else np.nan)
        row["Seeds"] = len(rs)
        rows.append(row)
    return rows


def _score(t):
    def mm(s, invert=False):
        if s.isna().all():
            return pd.Series(np.nan, index=s.index)
        r = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return 1 - r if invert else r
    t = t.copy()
    norm = pd.DataFrame({m: mm(t[m], m in MINIMIZE) for m in METRICS if m in t})
    for m in norm:
        t[m + " (normalized)"] = norm[m]
    t["Score"] = norm.mean(axis=1, skipna=True)
    return t


def select_k(corpus: Corpus, ks, seed=42, n_seeds=3, fine_window=5, params=None, progress=None,
             executor=None, min_k=2, max_iter=None) -> pd.DataFrame:
    """Selección de K en dos etapas: búsqueda gruesa en ks y fina (paso 1) en ±fine_window alrededor del mejor
    K. Cada métrica se promedia sobre n_seeds semillas; el acuerdo entre semillas (Greene et al., 2014) es la
    quinta métrica. Las métricas se normalizan entre 0 y 1 sobre todos los K evaluados; Score es su media."""
    params = {**LDA_DEFAULTS, **(params or {})}
    if max_iter is not None:
        params["max_iter"] = max_iter
    seeds = [seed + 1000 + j for j in range(max(1, int(n_seeds)))]
    ks = sorted(set(int(k) for k in ks if k >= min_k))
    total = (len(ks) + (2 * fine_window if fine_window else 0)) * len(seeds)
    done = [0]
    rows = _evaluate(corpus, ks, seeds, params, progress, done, total, executor)
    stage = {k: "coarse" for k in ks}
    t = _score(pd.DataFrame(rows))
    if fine_window:
        best = int(t.loc[t["Score"].idxmax(), "K"])
        fine = [k for k in range(max(min_k, best - fine_window), best + fine_window + 1) if k not in stage]
        if fine:
            total = (len(ks) + len(fine)) * len(seeds)
            rows += _evaluate(corpus, fine, seeds, params, progress, done, total, executor)
            stage.update({k: "fine" for k in fine})
            t = _score(pd.DataFrame(rows).sort_values("K").reset_index(drop=True))
    t["Stage"] = t["K"].map(stage)
    t.attrs.update(seeds=seeds, params=lda_params(int(t["K"].iloc[0]), **params), fine_window=fine_window)
    return t


def match_topics(A, B):
    """Empareja filas de A y B (Hungarian sobre similitud coseno)."""
    s = cosine_similarity(A, B)
    r, c = linear_sum_assignment(-s)
    return r, c, s


def _beta_job(job):
    X, k, s, params = job
    return fit_lda(X, k, seed=s, **params)[2]


def stability(corpus: Corpus, k, beta_ref, seeds=(1, 2, 3, 4), params=None, executor=None, progress=None,
              max_iter=None):
    """Para cada réplica (otra semilla), coseno de cada tema de referencia con su par húngaro.
    Devuelve la matriz réplicas × temas (estabilidad por tema = media; se informa también la DE)."""
    params = {**LDA_DEFAULTS, **(params or {})}
    if max_iter is not None:
        params["max_iter"] = max_iter
    sims = np.zeros((len(seeds), k))
    jobs = [(corpus.X, k, s, params) for s in seeds]
    betas = executor.map(_beta_job, jobs) if executor is not None else map(_beta_job, jobs)
    for i, b in enumerate(betas):
        r, c, S = match_topics(beta_ref, b)
        sims[i, r] = S[r, c]
        if progress:
            progress((i + 1) / len(seeds), f"Estabilidad LDA: réplica {i + 1}/{len(seeds)}")
    return sims


class LDAResult:
    def __init__(self, corpus: Corpus, k: int, seed=42, n_stab=10, params=None, executor=None, progress=None,
                 max_iter=None):
        params = {**LDA_DEFAULTS, **(params or {})}
        if max_iter is not None:  # compatibilidad con llamadas antiguas
            params["max_iter"] = max_iter
        self.k = k
        self.seed = seed
        self.params = lda_params(k, **params)
        self.model, self.theta, self.beta = fit_lda(corpus.X, k, seed, **params)
        self.n_iter = int(self.model.n_iter_)
        self.top10 = top_words(self.beta, corpus.vocab, 10)
        self.top25 = top_words(self.beta, corpus.vocab, 25)
        self.npmi = corpus.npmi(self.top10)
        self.umass = corpus.umass(self.top10)
        self.diversity = topic_diversity(self.top25)
        self.dominant = self.theta.argmax(axis=1)
        self.prevalence = self.theta.mean(axis=0)
        self.stab_seeds = list(range(seed + 1, seed + 1 + n_stab)) if n_stab else []
        if n_stab:
            self.stab_reps = stability(corpus, k, self.beta, seeds=tuple(self.stab_seeds), params=params,
                                       executor=executor, progress=progress)
            self.stab = self.stab_reps.mean(axis=0)
            self.stab_sd = self.stab_reps.std(axis=0, ddof=1) if n_stab > 1 else np.full(k, np.nan)
        else:
            self.stab_reps = None
            self.stab = np.full(k, np.nan)
            self.stab_sd = np.full(k, np.nan)
        self.labels = [f"L{i + 1}: " + ", ".join(w[:3]) for i, w in enumerate(self.top10)]
        self.ids = [f"L{i + 1}" for i in range(k)]

    def topic_table(self):
        return pd.DataFrame({
            "Topic": self.ids,
            "Label": self.labels,
            "Top terms": [", ".join(w) for w in self.top10],
            "Prevalence (%)": (100 * self.prevalence).round(1),
            "Dominant docs": np.bincount(self.dominant, minlength=self.k),
            "NPMI": np.round(self.npmi, 3),
            "Stability (mean)": np.round(self.stab, 3),
            "Stability (SD)": np.round(getattr(self, "stab_sd", np.full(self.k, np.nan)), 3),
        })
