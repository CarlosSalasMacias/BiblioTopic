"""Preprocesamiento del texto (título + resumen) común a LDA y BERTopic."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

ACADEMIC_STOPWORDS = {
    "study", "studies", "studied", "result", "results", "using", "use", "used", "uses", "based",
    "paper", "article", "research", "also", "however", "may", "might", "can", "could", "would",
    "show", "shows", "showed", "shown", "found", "find", "findings", "present", "presented",
    "aim", "aimed", "objective", "objectives", "conclusion", "conclusions", "conclude", "work",
    "method", "methods", "approach", "data", "analysis", "analyzed", "analysed", "evaluate",
    "evaluated", "evaluation", "effect", "effects", "different", "difference", "differences",
    "significant", "significantly", "high", "higher", "low", "lower", "increase", "increased",
    "decrease", "decreased", "new", "well", "among", "within", "two", "three", "one", "first",
    "second", "including", "include", "includes", "indicate", "indicated", "indicates", "observed",
    "obtained", "provide", "provides", "provided", "suggest", "suggests", "suggested", "total",
    "respectively", "level", "levels", "compared", "comparison", "therefore", "thus", "due",
    "several", "various", "important", "potential", "main", "per", "et", "al", "year", "years",
    "elsevier", "rights", "reserved", "published", "ltd", "copyright", "author", "authors",
    "springer", "wiley", "taylor", "francis", "mdpi", "licensee", "basel", "switzerland",
    "ii", "iii", "iv", "vs", "via", "e.g", "i.e", "using", "related", "type", "types", "high",
    "make", "made", "way", "case", "order", "according", "investigate", "investigated",
}


IRREGULAR = {"species": "species", "series": "series", "leaves": "leaf", "indices": "index",
             "analyses": "analysis", "data": "data", "larvae": "larva", "fungi": "fungus",
             "diabetes": "diabetes", "mice": "mouse", "children": "child", "women": "woman"}


def lemmatize_simple(tok: str) -> str:
    """Lematizador ligero (plural -> singular) sin dependencias externas."""
    if tok in IRREGULAR:
        return IRREGULAR[tok]
    if len(tok) <= 4:
        return tok
    if tok.endswith("ies") and len(tok) > 5:
        return tok[:-3] + "y"
    if tok.endswith(("sses", "xes", "ches", "shes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is", "as")):
        return tok[:-1]
    return tok


def tokenize(text: str, stop: set) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z\s\-]", " ", text)
    toks = re.split(r"[\s\-]+", text)
    toks = [lemmatize_simple(t) for t in toks if len(t) > 2 and t not in stop]
    return [t for t in toks if t not in stop]


def detect_phrases(token_docs: list[list[str]], min_count=10, threshold=0.35) -> set:
    """Colocaciones (bigramas) por NPMI, p. ej. 'coffee_berry'."""
    uni = Counter()
    bi = Counter()
    for d in token_docs:
        uni.update(d)
        bi.update(zip(d, d[1:]))
    n = sum(uni.values())
    keep = set()
    for (a, b), c in bi.items():
        if c < min_count or a == b:
            continue
        pab = c / n
        npmi = math.log(pab / ((uni[a] / n) * (uni[b] / n))) / -math.log(pab)
        if npmi >= threshold:
            keep.add((a, b))
    return keep


def merge_phrases(doc: list[str], phrases: set) -> list[str]:
    out, i = [], 0
    while i < len(doc):
        if i + 1 < len(doc) and (doc[i], doc[i + 1]) in phrases:
            out.append(doc[i] + "_" + doc[i + 1])
            i += 2
        else:
            out.append(doc[i])
            i += 1
    return out


class Corpus:
    """Contiene textos crudos (para embeddings), tokens limpios y matriz documento-término."""

    def __init__(self, df, extra_stopwords=(), min_df=5, max_df=0.5, query_terms=()):
        self.raw = (df["TI"].fillna("") + ". " + df["AB"].fillna("")).tolist()
        stop = set(ENGLISH_STOP_WORDS) | ACADEMIC_STOPWORDS
        stop |= {w.strip().lower() for w in extra_stopwords if w.strip()}
        # los términos de la ecuación de búsqueda aparecen en casi todos los documentos
        stop |= {lemmatize_simple(w.strip().lower()) for w in query_terms if w.strip()}
        self.stopwords = stop
        toks = [tokenize(t, stop) for t in self.raw]
        self.phrases = detect_phrases(toks)
        self.tokens = [merge_phrases(t, self.phrases) for t in toks]
        self.clean = [" ".join(t) for t in self.tokens]
        self.vectorizer = CountVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None,
                                          min_df=min_df, max_df=max_df)
        self.X = self.vectorizer.fit_transform(self.clean)
        self.vocab = np.array(self.vectorizer.get_feature_names_out())
        self.Xbin = (self.X > 0).astype(np.float32).tocsc()
        self.doc_freq = np.asarray(self.Xbin.sum(axis=0)).ravel()
        self.word_index = {w: i for i, w in enumerate(self.vocab)}

    @property
    def n_docs(self):
        return self.X.shape[0]

    def npmi(self, topics_words: list[list[str]]) -> list[float]:
        """Coherencia NPMI (Bouma, 2009) por co-ocurrencia documental en el propio corpus."""
        n = self.n_docs
        out = []
        for words in topics_words:
            idx = [self.word_index[w] for w in words if w in self.word_index]
            vals = []
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    a, b = idx[i], idx[j]
                    co = self.Xbin[:, a].multiply(self.Xbin[:, b]).sum()
                    if co == 0:
                        vals.append(-1.0)
                        continue
                    pij = co / n
                    pi, pj = self.doc_freq[a] / n, self.doc_freq[b] / n
                    if pij >= 1:
                        vals.append(1.0)
                        continue
                    vals.append(math.log(pij / (pi * pj)) / -math.log(pij))
            out.append(float(np.mean(vals)) if vals else np.nan)
        return out

    def umass(self, topics_words: list[list[str]]) -> list[float]:
        """Coherencia UMass (Mimno et al., 2011)."""
        out = []
        for words in topics_words:
            idx = [self.word_index[w] for w in words if w in self.word_index]
            vals = []
            for i in range(1, len(idx)):
                for j in range(i):
                    co = self.Xbin[:, idx[i]].multiply(self.Xbin[:, idx[j]]).sum()
                    vals.append(math.log((co + 1) / max(self.doc_freq[idx[j]], 1)))
            out.append(float(np.mean(vals)) if vals else np.nan)
        return out


def topic_diversity(topics_words: list[list[str]], topn=25) -> float:
    """Proporción de palabras únicas en el top-n de todos los temas (Dieng et al., 2020)."""
    words = [w for t in topics_words for w in t[:topn]]
    return len(set(words)) / max(len(words), 1)
