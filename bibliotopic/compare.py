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


def mk_hamed_rao(x, y, sen_slope, alpha=0.05):
    """Mann-Kendall modificado por autocorrelación (Hamed y Rao, 1998): la varianza de S se multiplica por
    n/n* = 1 + 2/(n(n-1)(n-2)) Σ (n-k)(n-k-1)(n-k-2) r_k, con r_k las autocorrelaciones significativas (5 %) de
    los rangos de la serie sin tendencia (restando la pendiente de Sen). Devuelve (p, factor n/n*)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = len(y)
    if n < 4 or np.ptp(y) == 0:
        return 1.0, 1.0
    S = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, ties = np.unique(y, return_counts=True)
    var0 = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0
    r = stats.rankdata(y - sen_slope * x)
    r = r - r.mean()
    den = np.sum(r ** 2)
    lim = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    acc = 0.0
    for k in range(1, n - 1):
        rk = np.sum(r[:-k] * r[k:]) / den if den > 0 else 0.0
        if abs(rk) > lim:
            acc += (n - k) * (n - k - 1) * (n - k - 2) * rk
    factor = 1 + 2 * acc / (n * (n - 1) * (n - 2))
    if factor <= 0:
        factor = 1.0
    var = var0 * factor
    if var <= 0:
        return 1.0, float(factor)
    z = (S - np.sign(S)) / np.sqrt(var)
    return float(2 * (1 - stats.norm.cdf(abs(z)))), float(factor)


def topic_trends(theta, years, min_docs_year=10, exclude_years=(), alpha=0.05, test="mann_kendall_hr"):
    """Tendencia de la prevalencia media anual de cada tema.

    - Regresión lineal (OLS; Griffiths & Steyvers, 2004): pendiente con IC 95 %.
    - Mann-Kendall (tau de Kendall frente al año) con pendiente de Sen (Theil-Sen) e IC 95 %: no paramétrico.
    - Mann-Kendall modificado por autocorrelación (Hamed y Rao, 1998): las prevalencias anuales suelen estar
      autocorrelacionadas, lo que infla los falsos positivos del Mann-Kendall clásico.
    - Benjamini-Hochberg sobre todos los temas para cada prueba. La clasificación (Hot / Cold / «No significant
      trend») usa la q de la prueba elegida en test: "mann_kendall_hr" (por defecto), "mann_kendall" u "ols".
      En las pruebas de Mann-Kendall, la dirección es la de la pendiente de Sen.
    - Solo años con al menos min_docs_year documentos (en corpus pequeños el mínimo se rebaja a la mitad, hasta 1,
      hasta tener 3 años); con menos de 3 años la tendencia es «Not estimable». exclude_years: p. ej., el año en
      curso, incompleto. Las tendencias describen el corpus recuperado y son exploratorias."""
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
            x, y = m.index.values.astype(float), m[t].values
            r = stats.linregress(x, y)
            h = r.stderr * stats.t.ppf(0.975, n_years - 2) if np.isfinite(r.stderr) else np.nan
            p_ols = r.pvalue if np.isfinite(r.pvalue) else 1.0
            if np.ptp(y) > 0:
                tau, p_mk = stats.kendalltau(x, y)
                sen = stats.theilslopes(y, x, 0.95)
                sen_s, sen_lo, sen_hi = sen[0], sen[2], sen[3]
                p_hr, fac = mk_hamed_rao(x, y, sen_s)
            else:
                tau, p_mk, sen_s, sen_lo, sen_hi, p_hr, fac = 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0
            rows.append((r.slope, r.stderr, r.slope - h, r.slope + h, p_ols, tau,
                         p_mk if np.isfinite(p_mk) else 1.0, sen_s, sen_lo, sen_hi, p_hr, fac))
        else:
            rows.append((np.nan,) * 12)
    out = pd.DataFrame(rows, columns=["Slope", "SE", "CI low", "CI high", "p-value", "Kendall tau",
                                      "MK p-value", "Sen slope", "Sen CI low", "Sen CI high",
                                      "MK p-value (Hamed-Rao)", "Autocorrelation factor n/n*"])
    out["q-value (BH)"] = bh_adjust(out["p-value"].values)
    out["MK q-value (BH)"] = bh_adjust(out["MK p-value"].values)
    out["MK q-value (Hamed-Rao, BH)"] = bh_adjust(out["MK p-value (Hamed-Rao)"].values)
    out["Years used"] = n_years
    q = {"mann_kendall": out["MK q-value (BH)"], "mann_kendall_hr": out["MK q-value (Hamed-Rao, BH)"]}.get(
        test, out["q-value (BH)"])
    direction = out["Sen slope"] if test.startswith("mann_kendall") else out["Slope"]
    out["Trend"] = np.where(out["p-value"].isna(), NOT_ESTIMABLE,
                            np.where(q < alpha, np.where(direction > 0, HOT, COLD), NO_TREND))
    out["Test"] = {"mann_kendall": "Mann-Kendall", "mann_kendall_hr": "Mann-Kendall (Hamed-Rao)"}.get(test, "OLS")
    out.attrs.update(min_docs_year=thr, n_years=n_years, alpha=alpha, test=test,
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


def compare(lda, bert, years, min_docs_year=10, tau_strong=0.5, tau_partial=0.3, exclude_years=(), titles=None,
            n_perm=0, seed=42, trend_test="mann_kendall_hr", alpha=0.05, n_boot=1000):
    S = similarity_matrix(lda, bert)
    J = jaccard_matrix(lda, bert)
    r, c = linear_sum_assignment(-S.values)
    from .stats import permutation_null_cosine, tri_pair_ci
    tl, _ = topic_trends(lda.theta, years, min_docs_year, exclude_years, alpha=alpha, test=trend_test)
    tb, _ = topic_trends(bert.theta, years, min_docs_year, exclude_years, alpha=alpha, test=trend_test)
    null = permutation_null_cosine(lda.beta, bert.ctfidf, n_perm, seed) if n_perm else None
    reps_l, reps_b = getattr(lda, "stab_reps", None), getattr(bert, "stab_reps", None)
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
            "Cosine null p (permutation)": round(float(null["p"][i, j]), 4) if null else np.nan,
            "Cosine null 95th percentile": round(float(null["p95_pair"][i, j]), 3) if null else np.nan,
            "Above chance": ("Yes" if null["p"][i, j] < 0.05 else "No") if null else "",
            "Re-matched null p": round(float(null["p_rematched"][i, j]), 4) if null else np.nan,
            "Above re-matched null": ("Yes" if null["p_rematched"][i, j] < 0.05 else "No") if null else "",
            "TRI": round(tri, 3) if np.isfinite(tri) else np.nan,
            "TRI 95% CI": (lambda ci: f"[{ci[0]:.3f}, {ci[1]:.3f}]" if np.isfinite(ci[0]) else "")(
                tri_pair_ci(s, reps_l[:, i], reps_b[:, j], n_boot, seed))
            if np.isfinite(tri) and reps_l is not None and reps_b is not None else "",
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

    # mejor correspondencia de cada tema en el otro modelo (muchos a uno), para ver cuándo el emparejamiento
    # húngaro fuerza pares sin sentido
    hung = dict(zip(r, c))
    hung_b = {j: i for i, j in hung.items()}
    best = []
    for i in range(lda.k):
        j = int(np.argmax(S.values[i]))
        best.append({"Model": "LDA", "Topic": lda.labels[i], "Best match": bert.labels[j],
                     "Best cosine": round(float(S.values[i, j]), 3),
                     "Hungarian partner": bert.labels[hung[i]] if i in hung else "-",
                     "Hungarian cosine": round(float(S.values[i, hung[i]]), 3) if i in hung else np.nan,
                     "Hungarian = best": "Yes" if hung.get(i) == j else "No"})
    for j in range(bert.k):
        i = int(np.argmax(S.values[:, j]))
        best.append({"Model": "BERTopic", "Topic": bert.labels[j], "Best match": lda.labels[i],
                     "Best cosine": round(float(S.values[i, j]), 3),
                     "Hungarian partner": lda.labels[hung_b[j]] if j in hung_b else "-",
                     "Hungarian cosine": round(float(S.values[hung_b[j], j]), 3) if j in hung_b else np.nan,
                     "Hungarian = best": "Yes" if hung_b.get(j) == i else "No"})
    best = pd.DataFrame(best)

    # sensibilidad de la clasificación a los umbrales τ (decisiones del análisis)
    pc = np.array([S.values[i, j] for i, j in zip(r, c)])
    sens = []
    for ts in np.round(np.arange(0.40, 0.601, 0.05), 2):
        for tp in np.round(np.arange(0.20, 0.401, 0.05), 2):
            if tp >= ts:
                continue
            sens.append({"tau_strong": ts, "tau_partial": tp, "Convergent": int((pc >= ts).sum()),
                         "Partial": int(((pc >= tp) & (pc < ts)).sum()), "Divergent": int((pc < tp).sum()),
                         "Selected": "Yes" if (abs(ts - tau_strong) < 1e-9 and abs(tp - tau_partial) < 1e-9) else ""})
    sens = pd.DataFrame(sens)

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
    metrics[["LDA", "BERTopic"]] = metrics[["LDA", "BERTopic"]].astype(float)
    pct = metrics["Metric"].str.contains("%")
    metrics.loc[pct, ["LDA", "BERTopic"]] = metrics.loc[pct, ["LDA", "BERTopic"]].round(1)
    metrics.loc[~pct, ["LDA", "BERTopic"]] = metrics.loc[~pct, ["LDA", "BERTopic"]].round(3)

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
                       pd.Series([bert.ids[x] if x >= 0 else "Unassigned" for x in bert.topics], name="BERTopic"))
    cont = cont.reindex(index=lda.ids, columns=bert.ids, fill_value=0)
    return {"S": S, "J": J, "matches": matches, "metrics": metrics, "agreement": agreement,
            "trends_lda": tl, "trends_bert": tb, "contingency": cont, "best_matches": best,
            "threshold_sensitivity": sens, "null": null}


def strategic_map(theta, df, trends, labels):
    """Crecimiento (pendiente) vs impacto (citas normalizadas por edad) por tema."""
    if not df["TC"].notna().any() or trends["Slope"].isna().all():
        return None
    age = (pd.Timestamp.now().year - df["PY"].astype(int) + 1).clip(lower=1).values
    tcy = (df["TC"].fillna(0).values / age)
    w = theta / (theta.sum(axis=0, keepdims=True) + 1e-12)
    impact = (w * tcy[:, None]).sum(axis=0)
    sen = str(trends["Test"].iloc[0]).startswith("Mann-Kendall") if "Test" in trends and len(trends) else False
    gcol, gval = (("Growth (Sen slope)", trends["Sen slope"].values) if sen and "Sen slope" in trends
                  else ("Growth (slope)", trends["Slope"].values))
    return pd.DataFrame({"Topic": labels, gcol: gval, "Trend": trends["Trend"].values,
                         "Impact (citations/year)": impact, "Prevalence": theta.mean(axis=0)})
