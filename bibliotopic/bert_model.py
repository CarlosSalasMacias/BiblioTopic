"""BERTopic con el mismo vocabulario que LDA (para que la comparación sea justa)."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .lda_model import match_topics
from .text import Corpus, topic_diversity

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed(texts, model_name=DEFAULT_MODEL, progress=None):
    """Devuelve (embeddings, nombre_del_método). Si no se puede descargar el modelo usa LSA."""
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_name)
        emb = st.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(emb), model_name
    except Exception as e:  # sin internet / sin torch
        warnings.warn(f"No se pudo cargar {model_name} ({e}). Se usa LSA (TF-IDF + SVD) como respaldo.")
        tf = TfidfVectorizer(min_df=3, max_df=0.5, stop_words="english", sublinear_tf=True)
        Z = tf.fit_transform(texts)
        emb = TruncatedSVD(n_components=min(256, Z.shape[1] - 1), random_state=42).fit_transform(Z)
        return normalize(emb), "LSA fallback (TF-IDF + SVD, 256 dimensions)"


_SEP = "zqz"


def _enc(w):
    return w.replace("_", _SEP)


def _docs(corpus):
    return [d.replace("_", _SEP) for d in corpus.clean]


def _fit(corpus: Corpus, emb, nr_topics, seed, min_cluster_size):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                      random_state=seed, low_memory=True)
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean",
                  cluster_selection_method="eom", prediction_data=True)
    # BERTopic elimina "_" al preprocesar: se codifican las frases y se conserva el orden del vocabulario
    vec = CountVectorizer(vocabulary=[_enc(w) for w in corpus.vocab], tokenizer=str.split,
                          lowercase=False, token_pattern=None)
    tm = BERTopic(umap_model=umap_model, hdbscan_model=hdb, vectorizer_model=vec,
                  ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
                  top_n_words=25, calculate_probabilities=False, verbose=False)
    topics, _ = tm.fit_transform(_docs(corpus), embeddings=emb)
    native = len(set(topics)) - (1 if -1 in topics else 0)
    if nr_topics and native > nr_topics:
        tm.reduce_topics(_docs(corpus), nr_topics=nr_topics + (1 if -1 in topics else 0))
        topics = list(tm.topics_)
    return tm, np.array(topics), native


def _ctfidf_rows(tm, k):
    """Matriz c-TF-IDF (k x V) de los temas 0..k-1 (sin outliers)."""
    M = tm.c_tf_idf_
    M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    off = tm._outliers if hasattr(tm, "_outliers") else 0
    return M[off:off + k]


class BERTResult:
    def __init__(self, corpus: Corpus, emb, emb_name, nr_topics=None, seed=42, n_stab=3,
                 min_cluster_size=None):
        self.emb_name = emb_name
        n = corpus.n_docs
        self.min_cluster_size = min_cluster_size or max(10, int(round(n * 0.005)))
        self.seed = seed
        self.model, topics, self.native_topics = _fit(corpus, emb, nr_topics, seed,
                                                      self.min_cluster_size)
        self.topics_raw = topics
        self.k = int(topics.max()) + 1
        self.outlier_rate = float((topics == -1).mean())
        if (topics == -1).any():
            new = self.model.reduce_outliers(_docs(corpus), list(topics), strategy="c-tf-idf",
                                             threshold=0.0)
            self.topics = np.array(new)
        else:
            self.topics = topics
        self.ctfidf = _ctfidf_rows(self.model, self.k)
        vocab = corpus.vocab
        self.top10 = [list(vocab[np.argsort(-r)[:10]]) for r in self.ctfidf]
        self.top25 = [list(vocab[np.argsort(-r)[:25]]) for r in self.ctfidf]
        self.npmi = corpus.npmi(self.top10)
        self.umass = corpus.umass(self.top10)
        self.diversity = topic_diversity(self.top25)
        valid = self.topics >= 0
        self.theta = np.zeros((n, self.k))
        self.theta[np.where(valid)[0], self.topics[valid]] = 1.0
        self.prevalence = self.theta.mean(axis=0)
        self.sizes = np.bincount(self.topics_raw[self.topics_raw >= 0], minlength=self.k)
        self.ids = [f"B{i + 1}" for i in range(self.k)]
        self.labels = [f"B{i + 1}: " + ", ".join(w[:3]) for i, w in enumerate(self.top10)]
        self.stab = self._stability(corpus, emb, nr_topics, n_stab)
        # mapa 2D para la figura
        try:
            from umap import UMAP
            self.xy = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine",
                           random_state=seed).fit_transform(emb)
        except Exception:
            self.xy = None

    def _stability(self, corpus, emb, nr_topics, n_stab):
        if not n_stab:
            return np.full(self.k, np.nan)
        sims = []
        for s in range(self.seed + 1, self.seed + 1 + n_stab):
            tm, t, _ = _fit(corpus, emb, nr_topics or self.k, s, self.min_cluster_size)
            kk = int(t.max()) + 1
            r, c, S = match_topics(self.ctfidf, _ctfidf_rows(tm, kk))
            v = np.zeros(self.k)
            v[r] = S[r, c]
            sims.append(v)
        return np.mean(sims, axis=0)

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
