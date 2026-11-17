"""Inferencia para la comparación de modelos: bootstrap de documentos y pruebas no paramétricas."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _npmi_matrix(Xb, idx_pairs):
    """NPMI medio por tema a partir de una matriz binaria documento × palabra (densa, pequeña)."""
    n = Xb.shape[0]
    df = Xb.sum(axis=0)
    co = Xb.T @ Xb
    out = []
    for pairs in idx_pairs:
        vals = []
        for a, b in pairs:
            c = co[a, b]
            if c == 0:
                vals.append(-1.0)
                continue
            pij = c / n
            if pij >= 1:
                vals.append(1.0)
                continue
            vals.append(np.log(pij / ((df[a] / n) * (df[b] / n))) / -np.log(pij))
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out)


def fmt_p(p, digits=3) -> str:
    """Formato de valores p: «< 0.001» en lugar de «0.000»; tres decimales."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n.a."
    return "< 0.001" if p < 0.001 else f"{p:.{digits}f}"


def ci_from_boot(boots, point, method="percentile", jack=None, level=0.95):
    """IC a partir de las réplicas bootstrap. method="percentile" o "bca" (Efron, 1987): la aceleración se
    estima con un jackknife (agrupado si n es grande) pasado en jack; sin jack, BCa usa solo la corrección de
    sesgo. Devuelve (lo, hi)."""
    b = np.asarray(boots, float)
    b = b[np.isfinite(b)]
    if len(b) < 10:
        return (np.nan, np.nan)
    a_lo, a_hi = (1 - level) / 2, 1 - (1 - level) / 2
    if method != "bca":
        return float(np.percentile(b, 100 * a_lo)), float(np.percentile(b, 100 * a_hi))
    prop = np.clip((b < point).mean() + 0.5 * (b == point).mean(), 1 / len(b), 1 - 1 / len(b))
    z0 = stats.norm.ppf(prop)
    acc = 0.0
    if jack is not None and len(jack) > 2:
        j = np.asarray(jack, float)
        d = j.mean() - j
        den = 6 * (np.sum(d ** 2) ** 1.5)
        acc = float(np.sum(d ** 3) / den) if den > 0 else 0.0
    out = []
    for a in (a_lo, a_hi):
        z = stats.norm.ppf(a)
        adj = stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
        out.append(float(np.percentile(b, 100 * np.clip(adj, 0, 1))))
    return out[0], out[1]


def _groups(n, g=100, seed=0):
    """Grupos para el jackknife agrupado (delete-d) usado en la aceleración de BCa."""
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, min(n, g))


def bootstrap_coherence(corpus, models: dict, B=1000, seed=42, topn=10, method="percentile"):
    """NPMI medio de cada modelo y su diferencia. La estimación puntual se calcula con todos los documentos (es
    la misma que en el resto de tablas); el bootstrap (remuestreo de documentos con reemplazo) aporta solo el IC
    95 %. Se informa también la media bootstrap, rotulada como tal."""
    words = sorted({w for tw in models.values() for t in tw for w in t[:topn] if w in corpus.word_index})
    col = {w: i for i, w in enumerate(words)}
    Xb = corpus.Xbin[:, [corpus.word_index[w] for w in words]].toarray().astype(np.float32)
    pairs = {}
    for name, tw in models.items():
        pairs[name] = []
        for t in tw:
            ids = [col[w] for w in t[:topn] if w in col]
            pairs[name].append([(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))])
    point = {name: float(np.nanmean(_npmi_matrix(Xb, pairs[name]))) for name in models}
    rng = np.random.default_rng(seed)
    n = Xb.shape[0]
    boots = {name: [] for name in models}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Xs = Xb[idx]
        for name in models:
            boots[name].append(np.nanmean(_npmi_matrix(Xs, pairs[name])))
    jack = None
    if method == "bca":
        jack = {name: [] for name in models}
        for g in _groups(n, seed=seed):
            keep = np.ones(n, bool)
            keep[g] = False
            for name in models:
                jack[name].append(np.nanmean(_npmi_matrix(Xb[keep], pairs[name])))
    res = {}
    for name, v in boots.items():
        lo, hi = ci_from_boot(v, point[name], method, jack[name] if jack else None)
        res[name] = {"point": point[name], "mean": point[name], "boot_mean": float(np.nanmean(v)), "lo": lo, "hi": hi}
    names = list(models)
    if len(names) == 2:
        d = np.array(boots[names[0]]) - np.array(boots[names[1]])
        dp = point[names[0]] - point[names[1]]
        dj = (np.array(jack[names[0]]) - np.array(jack[names[1]])) if jack else None
        lo, hi = ci_from_boot(d, dp, method, dj)
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        res["difference"] = {"label": f"{names[0]} - {names[1]}", "point": float(dp), "mean": float(dp),
                             "boot_mean": float(d.mean()), "lo": lo, "hi": hi, "p": float(max(p, 1 / B))}
    res["method"] = method
    res["B"] = B
    return res


def bootstrap_agreement(a, b, B=1000, seed=42, method="percentile"):
    """IC 95 % de ARI y NMI remuestreando documentos. Devuelve {"ARI": (lo, hi), "NMI": (lo, hi)}."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ari, nmi = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        ari.append(adjusted_rand_score(a[i], b[i]))
        nmi.append(normalized_mutual_info_score(a[i], b[i]))
    jack_a = jack_n = None
    if method == "bca":
        jack_a, jack_n = [], []
        for g in _groups(n, seed=seed):
            keep = np.ones(n, bool)
            keep[g] = False
            jack_a.append(adjusted_rand_score(a[keep], b[keep]))
            jack_n.append(normalized_mutual_info_score(a[keep], b[keep]))
    return {"ARI": ci_from_boot(ari, adjusted_rand_score(a, b), method, jack_a),
            "NMI": ci_from_boot(nmi, normalized_mutual_info_score(a, b), method, jack_n)}


def mean_ci(v, level=0.95, bounds=None):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(np.mean(v)) if len(v) else np.nan, np.nan, np.nan)
    m, se = v.mean(), stats.sem(v)
    h = se * stats.t.ppf((1 + level) / 2, len(v) - 1)
    lo, hi = m - h, m + h
    if bounds is not None:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    return float(m), float(lo), float(hi)


def compare_topics(x, y):
    """Mann-Whitney U (bilateral) entre los valores por tema de dos modelos + tamaño del efecto r."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 3 or len(y) < 3:
        return {"U": np.nan, "p": np.nan, "r": np.nan}
    u = stats.mannwhitneyu(x, y, alternative="two-sided")
    # r = Z / sqrt(N) a partir de la aproximación normal
    n1, n2 = len(x), len(y)
    mu, sd = n1 * n2 / 2, np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u.statistic - mu) / sd
    return {"U": float(u.statistic), "p": float(u.pvalue), "r": float(abs(z) / np.sqrt(n1 + n2))}


def inference_table(corpus, lda, bert, B=1000, seed=42, method="percentile"):
    """Tabla de inferencia (IC 95 % y pruebas). Las estimaciones puntuales son las mismas del resto de tablas y
    del texto (calculadas con todos los datos); el bootstrap solo aporta el intervalo. Los valores p se guardan
    sin redondear en facts y se formatean con fmt_p en la tabla y en el texto."""
    import pandas as pd
    coh = bootstrap_coherence(corpus, {"LDA": lda.top10, "BERTopic": bert.top10}, B=B, seed=seed, method=method)
    st_l, st_b = mean_ci(lda.stab, bounds=(0, 1)), mean_ci(bert.stab, bounds=(0, 1))
    t_coh = compare_topics(lda.npmi, bert.npmi)
    t_st = compare_topics(lda.stab, bert.stab)
    ok = bert.topics_raw >= 0
    # dos poblaciones con su propio denominador e intervalo: (a) todos los documentos con los outliers de
    # BERTopic reasignados; (b) solo los documentos asignados por el agrupamiento (asignación original)
    agr = bootstrap_agreement(lda.dominant, bert.topics, B=B, seed=seed, method=method)
    agr_ok = (bootstrap_agreement(lda.dominant[ok], bert.topics_raw[ok], B=B, seed=seed, method=method)
              if ok.sum() > 2 and (~ok).any() else None)
    d = coh["difference"]
    lab = "BCa" if method == "bca" else "percentile"
    rows = [
        ("Mean NPMI coherence", coh["LDA"], coh["BERTopic"], t_coh,
         f"Point estimates on all documents; {lab} bootstrap CI ({B} resamples). Difference "
         f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}], bootstrap p = {fmt_p(d['p'])}. Bootstrap means: "
         f"LDA {coh['LDA']['boot_mean']:.3f}, BERTopic {coh['BERTopic']['boot_mean']:.3f}"),
        ("Mean topic stability", {"point": st_l[0], "lo": st_l[1], "hi": st_l[2]},
         {"point": st_b[0], "lo": st_b[1], "hi": st_b[2]}, t_st, "t-based CI over topics"),
    ]
    out = []
    for name, a, b, t, note in rows:
        out.append({"Metric": name,
                    "LDA (95% CI)": f"{a['point']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]",
                    "BERTopic (95% CI)": f"{b['point']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]",
                    "Mann-Whitney U": f"{t['U']:.1f}" if not np.isnan(t["U"]) else "",
                    "p-value": fmt_p(t["p"]) if not np.isnan(t["p"]) else "",
                    "Effect size r": f"{t['r']:.3f}" if not np.isnan(t["r"]) else "",
                    "Note": note})
    n_all, n_ok = len(ok), int(ok.sum())
    for metric, fn in (("ARI", adjusted_rand_score), ("NMI", normalized_mutual_info_score)):
        pops = [("all documents, outliers reassigned", lda.dominant, bert.topics, agr, n_all)]
        if agr_ok is not None:
            pops.append(("clustered documents only, original assignment", lda.dominant[ok], bert.topics_raw[ok],
                         agr_ok, n_ok))
        for pop, a_, b_, ci, n_ in pops:
            out.append({"Metric": f"Document agreement {metric} ({pop})", "LDA (95% CI)": "",
                        "BERTopic (95% CI)": f"{fn(a_, b_):.3f} [{ci[metric][0]:.3f}, {ci[metric][1]:.3f}]",
                        "Mann-Whitney U": "", "p-value": "", "Effect size r": "",
                        "Note": f"n = {n_:,} documents; {lab} bootstrap over documents"})
    facts = {"coh": coh, "stab_lda": st_l, "stab_bert": st_b, "t_coh": t_coh, "t_stab": t_st, "agr": agr,
             "agr_clustered": agr_ok, "n_all": n_all, "n_clustered": n_ok, "method": method, "B": B}
    return pd.DataFrame(out), facts


def tri_ci(matches, B=1000, seed=42):
    """IC 95 % bootstrap del TRI medio de los pares emparejados (remuestreo de pares)."""
    v = matches["TRI"].dropna().values.astype(float) if "TRI" in matches else np.array([])
    if len(v) < 3:
        return None
    rng = np.random.default_rng(seed)
    bs = [rng.choice(v, len(v)).mean() for _ in range(B)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def tri_pair_ci(cos, reps_l, reps_b, B=1000, seed=42):
    """IC 95 % del TRI de un par remuestreando las réplicas de estabilidad de cada modelo (el coseno entre los
    temas finales es fijo). reps_*: vector de similitudes del tema en cada réplica."""
    rl, rb = np.asarray(reps_l, float), np.asarray(reps_b, float)
    if len(rl) < 2 or len(rb) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bs = cos * np.sqrt(rng.choice(rl, (B, len(rl))).mean(axis=1) * rng.choice(rb, (B, len(rb))).mean(axis=1))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def permutation_null_cosine(A, B, n_perm=1000, seed=42):
    """Línea base nula del coseno entre temas de dos modelos: se permutan los pesos de vocabulario de cada vector
    de B (rompe la correspondencia término a término y conserva la distribución de pesos). Devuelve la matriz
    de valores p por par (proporción de cosenos nulos ≥ observado) y el percentil 95 nulo por par y global."""
    from sklearn.metrics.pairwise import cosine_similarity
    A, B = np.asarray(A, float), np.asarray(B, float)
    obs = cosine_similarity(A, B)
    if not n_perm:
        return None
    rng = np.random.default_rng(seed)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    ge = np.zeros_like(obs)
    null_all = np.empty((n_perm, *obs.shape), dtype=np.float32)
    for t in range(n_perm):
        P = np.array([row[rng.permutation(len(row))] for row in B])
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        s = An @ Pn.T
        null_all[t] = s
        ge += s >= obs
    p = (ge + 1) / (n_perm + 1)
    return {"p": p, "p95_pair": np.percentile(null_all, 95, axis=0), "p95_global": float(np.percentile(null_all, 95)),
            "n_perm": n_perm}
