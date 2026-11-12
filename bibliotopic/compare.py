"""Comparación LDA vs BERTopic, tendencias temporales e índice de robustez temática."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity


def similarity_matrix(lda, bert, topn=None):
    """Coseno entre la distribución palabra-tema de LDA y el vector c-TF-IDF de BERTopic
    (mismo vocabulario). Si topn, se restringe a los top-n términos de cada tema."""
    A = lda.beta.copy()
    B = bert.ctfidf.copy()
    if topn:
        for M in (A, B):
            thr = -np.sort(-M, axis=1)[:, topn - 1:topn]
            M[M < thr] = 0
    return pd.DataFrame(cosine_similarity(A, B), index=lda.ids, columns=bert.ids)


def jaccard_matrix(lda, bert, topn=10):
    J = np.zeros((lda.k, bert.k))
    for i, a in enumerate(lda.top10[:]):
        for j, b in enumerate(bert.top10):
            sa, sb = set(a[:topn]), set(b[:topn])
            J[i, j] = len(sa & sb) / len(sa | sb)
    return pd.DataFrame(J, index=lda.ids, columns=bert.ids)


def topic_trends(theta, years, min_docs_year=10):
    """Pendiente OLS de la prevalencia anual de cada tema (Griffiths & Steyvers, 2004)."""
    df = pd.DataFrame(theta)
    df["Y"] = years.values.astype(int)
    counts = df["Y"].value_counts()
    # En corpus pequeños pocos años alcanzan el mínimo: se rebaja (a la mitad, hasta 1) hasta tener 3 años
    thr = max(1, int(min_docs_year))
    while thr > 1 and (counts >= thr).sum() < 3:
        thr = max(1, thr // 2)
    keep = counts[counts >= thr].index
    m = df[df["Y"].isin(keep)].groupby("Y").mean().sort_index()
    rows = []
    for t in range(theta.shape[1]):
        if len(m) >= 3:
            r = stats.linregress(m.index.values, m[t].values)
            rows.append((r.slope, r.pvalue))
        else:  # menos de 3 años con datos: no se puede estimar una tendencia
            rows.append((0.0, 1.0))
    out = pd.DataFrame(rows, columns=["Slope", "p-value"])
    out["p-value"] = out["p-value"].fillna(1.0)
    out["Slope"] = out["Slope"].fillna(0.0)
    out["Trend"] = np.where(out["p-value"] < 0.05, np.where(out["Slope"] > 0, "Hot", "Cold"),
                            "Stable")
    out.attrs["min_docs_year"] = thr
    out.attrs["n_years"] = len(m)
    return out, m


def compare(lda, bert, years, min_docs_year=10, tau_strong=0.5, tau_partial=0.3):
    S = similarity_matrix(lda, bert)
    J = jaccard_matrix(lda, bert)
    r, c = linear_sum_assignment(-S.values)
    tl, _ = topic_trends(lda.theta, years, min_docs_year)
    tb, _ = topic_trends(bert.theta, years, min_docs_year)
    rows = []
    matched_b = set()
    for i, j in zip(r, c):
        s = S.values[i, j]
        stab = np.sqrt(np.nan_to_num(lda.stab[i], nan=1.0) * np.nan_to_num(bert.stab[j], nan=1.0))
        tri = s * stab
        cat = ("Robust" if s >= tau_strong else "Partial" if s >= tau_partial else "Divergent")
        matched_b.add(j)
        rows.append({
            "LDA topic": lda.labels[i], "BERTopic topic": bert.labels[j],
            "Cosine": round(s, 3), "Jaccard top-10": round(J.values[i, j], 3),
            "Shared terms": ", ".join(w for w in lda.top10[i] if w in set(bert.top10[j])),
            "Stability LDA": round(lda.stab[i], 3), "Stability BERTopic": round(bert.stab[j], 3),
            "TRI": round(tri, 3), "Agreement": cat,
            "Trend LDA": tl["Trend"][i], "Trend BERTopic": tb["Trend"][j],
            "_i": i, "_j": j,
        })
    for j in range(bert.k):
        if j not in matched_b:
            rows.append({"LDA topic": "-", "BERTopic topic": bert.labels[j], "Cosine": np.nan,
                         "Agreement": "BERTopic-only", "Trend BERTopic": tb["Trend"][j],
                         "_i": -1, "_j": j})
    for i in range(lda.k):
        if i not in set(r):
            rows.append({"LDA topic": lda.labels[i], "BERTopic topic": "-", "Cosine": np.nan,
                         "Agreement": "LDA-only", "Trend LDA": tl["Trend"][i], "_i": i, "_j": -1})
    matches = pd.DataFrame(rows).sort_values("Cosine", ascending=False, na_position="last")

    ok = bert.topics_raw >= 0
    metrics = pd.DataFrame({
        "Metric": ["Number of topics", "Mean NPMI coherence", "Mean UMass coherence",
                   "Topic diversity (top-25)", "Mean topic stability (cosine across seeds)",
                   "Outlier documents (%)", "Largest topic share (%)",
                   "Smallest topic share (%)", "Hot topics", "Cold topics"],
        "LDA": [lda.k, np.nanmean(lda.npmi), np.nanmean(lda.umass), lda.diversity,
                np.nanmean(lda.stab), 0.0, 100 * lda.prevalence.max(), 100 * lda.prevalence.min(),
                int((tl["Trend"] == "Hot").sum()), int((tl["Trend"] == "Cold").sum())],
        "BERTopic": [bert.k, np.nanmean(bert.npmi), np.nanmean(bert.umass), bert.diversity,
                     np.nanmean(bert.stab), 100 * bert.outlier_rate, 100 * bert.prevalence.max(),
                     100 * bert.prevalence.min(), int((tb["Trend"] == "Hot").sum()),
                     int((tb["Trend"] == "Cold").sum())],
    })
    metrics[["LDA", "BERTopic"]] = metrics[["LDA", "BERTopic"]].astype(float).round(3)

    # coincidencia a nivel documento
    agreement = {
        "ARI (all documents)": adjusted_rand_score(lda.dominant, bert.topics),
        "NMI (all documents)": normalized_mutual_info_score(lda.dominant, bert.topics),
        "ARI (non-outliers)": adjusted_rand_score(lda.dominant[ok], bert.topics_raw[ok]),
        "NMI (non-outliers)": normalized_mutual_info_score(lda.dominant[ok], bert.topics_raw[ok]),
    }
    # concordancia de tendencias en temas emparejados (Robust + Partial)
    paired = matches[matches["Agreement"].isin(["Robust", "Partial"])]
    trend_agree = float((paired["Trend LDA"] == paired["Trend BERTopic"]).mean()) if len(paired) else np.nan
    agreement["Trend agreement in matched topics"] = trend_agree
    agreement = {k: round(float(v), 3) for k, v in agreement.items()}

    cont = pd.crosstab(pd.Series([lda.ids[x] for x in lda.dominant], name="LDA"),
                       pd.Series([bert.ids[x] for x in bert.topics], name="BERTopic"))
    cont = cont.reindex(index=lda.ids, columns=bert.ids, fill_value=0)
    return {"S": S, "J": J, "matches": matches, "metrics": metrics, "agreement": agreement,
            "trends_lda": tl, "trends_bert": tb, "contingency": cont}


def strategic_map(theta, df, trends, labels):
    """Crecimiento (pendiente) vs impacto (citas normalizadas por edad) por tema."""
    if not df["TC"].notna().any():
        return None
    age = (pd.Timestamp.now().year - df["PY"].astype(int) + 1).clip(lower=1).values
    tcy = (df["TC"].fillna(0).values / age)
    w = theta / (theta.sum(axis=0, keepdims=True) + 1e-12)
    impact = (w * tcy[:, None]).sum(axis=0)
    return pd.DataFrame({"Topic": labels, "Growth (slope)": trends["Slope"].values,
                         "Impact (citations/year)": impact, "Prevalence": theta.mean(axis=0)})
