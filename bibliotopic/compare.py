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


HOT, COLD = "Hot", "Cold"
NO_TREND = "No significant trend"
NOT_ESTIMABLE = "Not estimable"


def bh_adjust(p):
    """Valores p ajustados por Benjamini-Hochberg (FDR); NaN se conservan."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.any():
        pv = p[ok]
        order = np.argsort(pv)
        ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty_like(pv)
        out[order] = np.minimum(ranked, 1.0)
        q[ok] = out
    return q


def topic_trends(theta, years, min_docs_year=10, exclude_years=(), alpha=0.05):
    """Pendiente OLS de la prevalencia media anual de cada tema (Griffiths & Steyvers, 2004), con IC 95 %.

    - Solo años con al menos min_docs_year documentos; en corpus pequeños el mínimo se rebaja (a la mitad,
      hasta 1) hasta tener 3 años. Con menos de 3 años la tendencia es «Not estimable».
    - exclude_years: años excluidos (p. ej., el año en curso, incompleto).
    - Clasificación con p ajustado por Benjamini-Hochberg entre temas (pruebas múltiples): Hot / Cold si
      q < alpha; si no, «No significant trend» (que no equivale a demostrar estabilidad).
    Las tendencias describen el corpus recuperado y son exploratorias."""
    df = pd.DataFrame(theta)
    df["Y"] = np.asarray(years).astype(int)
    if len(exclude_years):
        df = df[~df["Y"].isin(list(exclude_years))]
    counts = df["Y"].value_counts()
    thr = max(1, int(min_docs_year))
    while thr > 1 and (counts >= thr).sum() < 3:
        thr = max(1, thr // 2)
    keep = counts[counts >= thr].index
    m = df[df["Y"].isin(keep)].groupby("Y").mean().sort_index()
    n_years = len(m)
    rows = []
    for t in range(theta.shape[1]):
        if n_years >= 3:
            r = stats.linregress(m.index.values, m[t].values)
            h = r.stderr * stats.t.ppf(0.975, n_years - 2) if np.isfinite(r.stderr) else np.nan
            p_ = r.pvalue if np.isfinite(r.pvalue) else 1.0
            rows.append((r.slope, r.stderr, r.slope - h, r.slope + h, p_))
        else:
            rows.append((np.nan, np.nan, np.nan, np.nan, np.nan))
    out = pd.DataFrame(rows, columns=["Slope", "SE", "CI low", "CI high", "p-value"])
    out["q-value (BH)"] = bh_adjust(out["p-value"].values)
    out["Years used"] = n_years
    out["Trend"] = np.where(out["p-value"].isna(), NOT_ESTIMABLE,
                            np.where(out["q-value (BH)"] < alpha, np.where(out["Slope"] > 0, HOT, COLD),
                                     NO_TREND))
    out.attrs.update(min_docs_year=thr, n_years=n_years, alpha=alpha,
                     years=(int(m.index.min()), int(m.index.max())) if n_years else None,
                     excluded_years=[int(y) for y in exclude_years])
    return out, m


def _representative(lda, bert, titles):
    """Título del documento más representativo de cada tema: mayor proporción del tema (LDA) y mayor similitud
    c-TF-IDF entre los documentos agrupados en el tema (BERTopic)."""
    if titles is None:
        return None, None
    titles = list(titles)
    rl = [titles[int(np.argmax(lda.theta[:, k]))] for k in range(lda.k)]
    rb = []
    sim = getattr(bert, "assign_sim", None)
    for k in range(bert.k):
        idx = np.where(bert.topics_raw == k)[0]
        if not len(idx):
            idx = np.where(bert.topics == k)[0]
        best = idx[int(np.argmax(sim[idx]))] if sim is not None and len(idx) else (idx[0] if len(idx) else None)
        rb.append(titles[best] if best is not None else "")
    return rl, rb


def compare(lda, bert, years, min_docs_year=10, tau_strong=0.5, tau_partial=0.3, exclude_years=(), titles=None):
    S = similarity_matrix(lda, bert)
    J = jaccard_matrix(lda, bert)
    r, c = linear_sum_assignment(-S.values)
    tl, _ = topic_trends(lda.theta, years, min_docs_year, exclude_years)
    tb, _ = topic_trends(bert.theta, years, min_docs_year, exclude_years)
    rep_l, rep_b = _representative(lda, bert, titles)
    rows = []
    matched_b = set()
    for i, j in zip(r, c):
        s = S.values[i, j]
        # TRI no disponible si falta la estabilidad de alguno de los temas (no se sustituye por 1)
        sl_, sb_ = lda.stab[i], bert.stab[j]
        tri = s * np.sqrt(sl_ * sb_) if np.isfinite(sl_) and np.isfinite(sb_) else np.nan
        cat = ("Robust" if s >= tau_strong else "Partial" if s >= tau_partial else "Divergent")
        matched_b.add(j)
        rows.append({
            "LDA topic": lda.labels[i], "BERTopic topic": bert.labels[j],
            "Cosine": round(s, 3), "Jaccard top-10": round(J.values[i, j], 3),
            "Shared terms": ", ".join(w for w in lda.top10[i] if w in set(bert.top10[j])),
            "Stability LDA": round(lda.stab[i], 3), "Stability BERTopic": round(bert.stab[j], 3),
            "TRI": round(tri, 3) if np.isfinite(tri) else np.nan,
            "TRI note": "" if np.isfinite(tri) else "Not available: topic stability was not computed",
            "Agreement": cat,
            "Trend LDA": tl["Trend"][i], "Trend BERTopic": tb["Trend"][j],
            # el emparejamiento uno a uno siempre propone una pareja; por debajo de τ parcial no es convincente
            "Convincing match": "Yes" if s >= tau_partial else f"No (cosine < {tau_partial})",
            "Representative LDA document": rep_l[i] if rep_l else "",
            "Representative BERTopic document": rep_b[j] if rep_b else "",
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
    paired = paired[(paired["Trend LDA"] != NOT_ESTIMABLE) & (paired["Trend BERTopic"] != NOT_ESTIMABLE)]
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
    if not df["TC"].notna().any() or trends["Slope"].isna().all():
        return None
    age = (pd.Timestamp.now().year - df["PY"].astype(int) + 1).clip(lower=1).values
    tcy = (df["TC"].fillna(0).values / age)
    w = theta / (theta.sum(axis=0, keepdims=True) + 1e-12)
    impact = (w * tcy[:, None]).sum(axis=0)
    return pd.DataFrame({"Topic": labels, "Growth (slope)": trends["Slope"].values,
                         "Impact (citations/year)": impact, "Prevalence": theta.mean(axis=0)})
