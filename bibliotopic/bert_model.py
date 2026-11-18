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

from .lda_model import match_topics, subsample_index
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


def auto_min_cluster_size(n):
    """Tamaño mínimo de grupo automático: máx(15, 0.5 % de N)."""
    return max(15, int(round(0.005 * n)))


def _cluster(reduced, min_cluster_size, min_samples=None):
    # min_samples=None -> igual a min_cluster_size (valor por defecto de HDBSCAN); se registra el valor efectivo
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
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


def fit_topics(X, reduced, nr_topics, min_cluster_size, min_samples=None):
    raw = _cluster(reduced, min_cluster_size, min_samples)
    native = len(set(raw) - {-1})
    lab = _sort_by_size(_reduce(X, raw, nr_topics))
    k = int(lab.max()) + 1 if (lab >= 0).any() else 0
    W, idf = ctfidf(class_counts(X, lab, list(range(k))))
    return lab, native, W, idf


def assignment_similarity(X, W, idf):
    """Similitud coseno de cada documento con el vector c-TF-IDF de cada tema (misma ponderación que c-TF-IDF)."""
    D = normalize(X, norm="l1", axis=1)
    D.data = np.sqrt(D.data)
    D = D @ sp.diags(idf)
    return cosine_similarity(D, W)


def reassign_outliers(X, labels, W, idf, sim=None, threshold=0.0):
    """Reducción de outliers por c-TF-IDF (como reduce_outliers(strategy="c-tf-idf") de BERTopic): cada outlier
    (-1) se asigna al tema más similar si la similitud es ≥ threshold; si no, sigue sin asignar (-1).
    Devuelve (etiquetas, similitudes)."""
    out = np.where(labels < 0)[0]
    if sim is None:
        sim = assignment_similarity(X, W, idf)
    new = labels.copy()
    if len(out) and sim.shape[1]:
        best = sim[out].argmax(axis=1)
        ok = sim[out, best] >= threshold
        new[out[ok]] = best[ok]
    return new, sim


class BERTResult:
    def __init__(self, corpus: Corpus, reduced_list, xy, emb_name, nr_topics=None,
                 min_cluster_size=None, min_samples=None, outlier_threshold=0.0, umap_seeds=None,
                 stab_subsample=0.0):
        """reduced_list: lista de matrices UMAP (n x 5), la primera es la principal y el resto
        (otras semillas) se usan para la estabilidad. stab_subsample: fracción de documentos de cada réplica de
        estabilidad (0 = todos; solo cambia la semilla de UMAP)."""
        X = corpus.X.tocsr()
        n = corpus.n_docs
        self.emb_name = emb_name
        self.min_cluster_size = int(min_cluster_size or auto_min_cluster_size(n))
        self.min_samples = int(min_samples) if min_samples else self.min_cluster_size
        self.outlier_threshold = float(outlier_threshold or 0.0)
        self.umap_seeds = list(umap_seeds or [])
        self.topics_raw, self.native_topics, self.ctfidf, idf = fit_topics(
            X, reduced_list[0], nr_topics, self.min_cluster_size, self.min_samples)
        self.k = self.ctfidf.shape[0]
        self.outlier_rate = float((self.topics_raw == -1).mean())
        self.topics, sim = reassign_outliers(X, self.topics_raw, self.ctfidf, idf, threshold=self.outlier_threshold)
        self.outlier_rate_after = float((self.topics == -1).mean())
        # trazabilidad de la asignación de cada documento: original (agrupamiento), final, similitud y motivo
        n_ = len(self.topics)
        best = sim.argmax(axis=1) if self.k else np.zeros(n_, int)
        self.assign_sim = sim[np.arange(n_), np.where(self.topics >= 0, self.topics, best)] if self.k else np.zeros(n_)
        srt = np.sort(sim, axis=1) if self.k > 1 else np.zeros((n_, 2))
        self.assign_margin = srt[:, -1] - srt[:, -2] if self.k > 1 else np.zeros(n_)
        self.reassigned = (self.topics_raw < 0) & (self.topics >= 0)
        clustered = self.assign_sim[~self.reassigned]
        # baja confianza: outlier reasignado con similitud inferior al percentil 25 de la de los documentos
        # agrupados con su propio tema
        self.low_conf_threshold = float(np.percentile(clustered, 25)) if len(clustered) else 0.0
        self.low_confidence = self.reassigned & (self.assign_sim < self.low_conf_threshold)
        vocab = corpus.vocab
        self.top10 = [list(vocab[np.argsort(-r)[:10]]) for r in self.ctfidf]
        self.top25 = [list(vocab[np.argsort(-r)[:25]]) for r in self.ctfidf]
        self.npmi = corpus.npmi(self.top10)
        self.umass = corpus.umass(self.top10)
        self.diversity = topic_diversity(self.top25)
        self.theta = np.zeros((n, self.k))
        ok_ = self.topics >= 0  # con umbral > 0 puede quedar algún documento sin asignar
        self.theta[np.arange(n)[ok_], self.topics[ok_]] = 1.0
        self.prevalence = self.theta.mean(axis=0)
        self.sizes = np.bincount(self.topics_raw[self.topics_raw >= 0], minlength=self.k)
        self.ids = [f"B{i + 1}" for i in range(self.k)]
        self.labels = [f"B{i + 1}: " + ", ".join(w[:3]) for i, w in enumerate(self.top10)]
        self.xy = np.asarray(xy) if xy is not None else None
        self.n_stab = len(reduced_list) - 1
        self.stab_subsample = float(stab_subsample or 0)
        if self.n_stab:
            sims = []
            for r_, red in enumerate(reduced_list[1:]):
                Xr = X
                if 0 < self.stab_subsample < 1:
                    seed_r = self.umap_seeds[r_ + 1] if len(self.umap_seeds) > r_ + 1 else r_ + 1
                    idx = subsample_index(n, self.stab_subsample, seed_r)
                    Xr, red = X[idx], np.asarray(red)[idx]
                _, _, W2, _ = fit_topics(Xr, red, nr_topics or self.k, self.min_cluster_size, self.min_samples)
                v = np.zeros(self.k)
                if W2.shape[0]:
                    r, c, S = match_topics(self.ctfidf, W2)
                    v[r] = S[r, c]
                sims.append(v)
            self.stab_reps = np.array(sims)
            self.stab = self.stab_reps.mean(axis=0)
            self.stab_sd = self.stab_reps.std(axis=0, ddof=1) if self.n_stab > 1 else np.full(self.k, np.nan)
        else:
            self.stab_reps = None
            self.stab = np.full(self.k, np.nan)
            self.stab_sd = np.full(self.k, np.nan)

    def topic_table(self):
        return pd.DataFrame({
            "Topic": self.ids,
            "Label": self.labels,
            "Top terms": [", ".join(w) for w in self.top10],
            "Prevalence (%)": (100 * self.prevalence).round(1),
            "Prevalence, clustered documents only (%)": (100 * self.sizes / max(1, self.sizes.sum())).round(1),
            "Documents (before outlier reassignment)": self.sizes,
            "Outliers reassigned": np.bincount(self.topics[(self.topics_raw < 0) & (self.topics >= 0)],
                                               minlength=self.k),
            "Low-confidence reassignments": np.bincount(self.topics[self.low_confidence & (self.topics >= 0)],
                                                        minlength=self.k) if hasattr(self, "low_confidence")
            else np.zeros(self.k, int),
            "NPMI": np.round(self.npmi, 3),
            "Stability (mean)": np.round(self.stab, 3),
            "Stability (SD)": np.round(getattr(self, "stab_sd", np.full(self.k, np.nan)), 3),
        })


def min_cluster_size_sensitivity(corpus: Corpus, reduced_list, nr_topics, base_mcs, factors=(1.0, 0.5, 0.25),
                                 min_samples=None, n_rep=5, floor=5):
    """Sensibilidad a min_cluster_size (HDBSCAN): se repite el agrupamiento sobre la misma proyección UMAP con el
    valor elegido y fracciones de él (½, ¼). Para cada valor: temas nativos y finales, % de outliers, NPMI media y
    estabilidad media frente a las réplicas UMAP disponibles (hasta n_rep). Devuelve (tabla, etiquetas por valor)
    para calcular después la concordancia con LDA."""
    X = corpus.X.tocsr()
    rows, labels = [], {}
    seen = set()
    for f in factors:
        mcs = max(int(floor), int(round(base_mcs * f)))
        if mcs in seen:
            continue
        seen.add(mcs)
        ms = int(min_samples) if min_samples else mcs  # por defecto, igual a min_cluster_size
        lab, native, W, _ = fit_topics(X, reduced_list[0], nr_topics, mcs, ms)
        k = W.shape[0]
        top10 = [list(corpus.vocab[np.argsort(-r)[:10]]) for r in W]
        npmi = float(np.nanmean(corpus.npmi(top10))) if k else np.nan
        stabs = []
        for red in reduced_list[1:1 + int(n_rep)]:
            _, _, W2, _ = fit_topics(X, red, nr_topics or k, mcs, ms)
            v = np.zeros(k)
            if W2.shape[0] and k:
                r, c, S = match_topics(W, W2)
                v[r] = S[r, c]
            stabs.append(v)
        rows.append({"min_cluster_size": mcs, "Relative to selected": f"×{f:g}",
                     "Native clusters (HDBSCAN)": int(native), "Topics": int(k),
                     "Outlier documents (%)": round(100 * float((lab < 0).mean()), 1),
                     "Mean NPMI": round(npmi, 3),
                     "Mean stability": round(float(np.mean(stabs)), 3) if stabs else np.nan,
                     "Stability replicates": len(stabs)})
        labels[mcs] = lab
    return pd.DataFrame(rows), labels
