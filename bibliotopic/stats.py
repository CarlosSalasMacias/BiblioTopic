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


def topic_npmi(corpus, topics_words, topn=10):
    """NPMI de cada tema (media de sus pares de palabras) con todos los documentos; igual que Corpus.npmi pero
    con matrices dispersas para todos los temas a la vez."""
    words = sorted({w for t in topics_words for w in t[:topn] if w in corpus.word_index})
    col = {w: i for i, w in enumerate(words)}
    Xs = corpus.Xbin[:, [corpus.word_index[w] for w in words]].tocsc().astype(np.float64)
    n = Xs.shape[0]
    df = np.asarray(Xs.sum(axis=0)).ravel()
    out = []
    for t in topics_words:
        ids = [col[w] for w in t[:topn] if w in col]
        vals = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                c = float(Xs[:, a].multiply(Xs[:, b]).sum())
                if c == 0:
                    vals.append(-1.0)
                    continue
                pij = c / n
                if pij >= 1:
                    vals.append(1.0)
                    continue
                vals.append(float(np.log(pij / ((df[a] / n) * (df[b] / n))) / -np.log(pij)))
        out.append(float(np.mean(vals)) if vals else np.nan)
    return np.array(out)


def bootstrap_coherence(corpus, models: dict, B=1000, seed=42, topn=10, method="percentile"):
    """NPMI medio de cada modelo y su diferencia, con IC 95 % por bootstrap **sobre los temas** (cada tema es una
    observación; se remuestrean los temas de cada modelo de forma independiente).

    La estimación puntual es la media del NPMI de los temas calculado con todos los documentos (la misma de las
    demás tablas). No se remuestrean documentos: duplicar unos y omitir otros cambia las co-ocurrencias y sesga
    a la baja el NPMI de los términos poco frecuentes, hasta dejar la estimación fuera de su propio intervalo.
    Remuestrear temas es además coherente con la prueba U de Mann-Whitney, que también compara temas. BCa usa la
    aceleración del jackknife que omite un tema cada vez."""
    vals = {}
    for name, tw in models.items():
        v = topic_npmi(corpus, tw, topn)
        vals[name] = v[np.isfinite(v)]
    point = {name: float(np.mean(v)) if len(v) else np.nan for name, v in vals.items()}
    rng = np.random.default_rng(seed)
    boots = {name: [] for name in models}
    for _ in range(B):
        for name, v in vals.items():
            boots[name].append(float(np.mean(v[rng.integers(0, len(v), len(v))])) if len(v) else np.nan)
    loo = {name: [float(np.mean(np.delete(v, i))) for i in range(len(v))] if len(v) > 2 else None
           for name, v in vals.items()}
    res = {}
    for name, b in boots.items():
        lo, hi = ci_from_boot(b, point[name], method, loo[name] if method == "bca" else None)
        res[name] = {"point": point[name], "mean": point[name], "boot_mean": float(np.nanmean(b)), "lo": lo,
                     "hi": hi, "n_topics": int(len(vals[name]))}
    names = list(models)
    if len(names) == 2:
        a, c = names
        d = np.array(boots[a]) - np.array(boots[c])
        dp = point[a] - point[c]
        dj = None
        if method == "bca" and loo[a] and loo[c]:  # jackknife de dos muestras: se omite un tema de cada modelo
            dj = [x - point[c] for x in loo[a]] + [point[a] - y for y in loo[c]]
        lo, hi = ci_from_boot(d, dp, method, dj)
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        if method == "bca":  # p coherente con el IC BCa: menor nivel al que el intervalo excluye el 0
            p = 1.0
            for alpha in np.linspace(0.001, 0.999, 999):
                l_, h_ = ci_from_boot(d, dp, "bca", dj, level=1 - alpha)
                if np.isfinite(l_) and (l_ > 0 or h_ < 0):
                    p = float(alpha)
                    break
        res["difference"] = {"label": f"{a} - {c}", "point": float(dp), "mean": float(dp),
                             "boot_mean": float(d.mean()), "lo": lo, "hi": hi, "p": float(max(p, 1 / B))}
    res["method"] = method
    res["B"] = B
    res["unit"] = "topics"
    return res


def _agreement_from_counts(C):
    """ARI y NMI (media aritmética, como scikit-learn) a partir de la tabla de contingencia C."""
    from scipy.special import comb
    n = C.sum()
    if n < 2:
        return np.nan, np.nan
    ri, cj = C.sum(axis=1), C.sum(axis=0)
    sum_comb = comb(C, 2).sum()
    sa, sb = comb(ri, 2).sum(), comb(cj, 2).sum()
    expected = sa * sb / comb(n, 2)
    mx = (sa + sb) / 2
    ari = 1.0 if mx == expected else float((sum_comb - expected) / (mx - expected))
    nz = C > 0
    pij = C[nz] / n
    pi = (ri / n)[np.nonzero(nz)[0]]
    pj = (cj / n)[np.nonzero(nz)[1]]
    mi = float(np.sum(pij * (np.log(pij) - np.log(pi) - np.log(pj))))
    ent = lambda v: -float(np.sum((v[v > 0] / n) * np.log(v[v > 0] / n)))  # noqa: E731
    ha, hb = ent(ri), ent(cj)
    if (ri > 0).sum() == 1 and (cj > 0).sum() == 1:
        return ari, 1.0
    denom = (ha + hb) / 2
    nmi = float(max(mi, 0.0) / denom) if denom > 0 else 1.0
    return ari, nmi


def bootstrap_agreement(a, b, B=1000, seed=42, method="percentile"):
    """IC 95 % de ARI y NMI con jackknife delete-d por submuestreo sin reemplazo (Shao y Wu, 1989).

    No se usa el bootstrap con reemplazo: al duplicar documentos, ambas particiones coinciden en las copias y
    ARI/NMI se inflan, hasta dejar la estimación puntual fuera de su intervalo. En su lugar se toman B
    submuestras de la mitad de los documentos (m = n/2, sin reemplazo), se estima el error típico
    EE = √[(m/(n−m)) · var(θ_m)] y el intervalo es θ̂ ± 1.96·EE (centrado en la estimación con todos los
    documentos y recortado al rango del índice). method se ignora (compatibilidad).
    Devuelve {"ARI": (lo, hi), "NMI": (lo, hi), "method": "delete-d jackknife"}."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n = len(a)
    ari0, nmi0 = adjusted_rand_score(a, b), normalized_mutual_info_score(a, b)
    if n < 10:
        return {"ARI": (np.nan, np.nan), "NMI": (np.nan, np.nan), "method": "delete-d jackknife"}
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    code = (ia * len(ub) + ib).astype(np.intp)
    shape = (len(ua), len(ub))
    m = n // 2

    def stat(w):
        C = np.bincount(code, weights=w, minlength=shape[0] * shape[1]).reshape(shape)
        C = C[C.sum(axis=1) > 0][:, C.sum(axis=0) > 0]
        return _agreement_from_counts(C)

    ari, nmi = [], []
    for _ in range(int(B)):
        w = np.zeros(n)
        w[rng.choice(n, m, replace=False)] = 1.0
        r = stat(w)
        ari.append(r[0])
        nmi.append(r[1])
    fac = m / (n - m)
    z = stats.norm.ppf(0.975)

    def ci(v, point, lo_b, hi_b):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if len(v) < 10:
            return (np.nan, np.nan)
        se = np.sqrt(fac * v.var(ddof=1))
        return (float(max(lo_b, point - z * se)), float(min(hi_b, point + z * se)))

    return {"ARI": ci(ari, ari0, -1.0, 1.0), "NMI": ci(nmi, nmi0, 0.0, 1.0), "method": "delete-d jackknife"}


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
         f"Point estimates on all documents; {lab} bootstrap CI over topics ({B} resamples). Difference "
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
                        "Note": f"n = {n_:,} documents; delete-d jackknife CI (half-samples without replacement)"})
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
    from scipy.optimize import linear_sum_assignment
    ge = np.zeros_like(obs)
    null_all = np.empty((n_perm, *obs.shape), dtype=np.float32)
    matched = []  # nula más estricta: cosenos de los pares que el algoritmo húngaro empareja en cada permutación
    for t in range(n_perm):
        P = np.array([row[rng.permutation(len(row))] for row in B])
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
        s = An @ Pn.T
        null_all[t] = s
        ge += s >= obs
        r_, c_ = linear_sum_assignment(-s)
        matched.append(s[r_, c_])
    p = (ge + 1) / (n_perm + 1)
    matched = np.sort(np.concatenate(matched))
    # valor p frente a la nula re-emparejada: proporción de cosenos emparejados bajo la nula ≥ observado
    p_rm = (len(matched) - np.searchsorted(matched, obs, side="left") + 1) / (len(matched) + 1)
    return {"p": p, "p95_pair": np.percentile(null_all, 95, axis=0), "p95_global": float(np.percentile(null_all, 95)),
            "p_rematched": p_rm, "p95_rematched": float(np.percentile(matched, 95)), "n_perm": n_perm}


def _spearman_fast(x, y):
    """ρ de Spearman (rangos medios para empates) sin pruebas: para las réplicas bootstrap."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def tri_expert_correlation(matches, lda_ids, bert_ids, per_topic, min_pairs=5, B=1000, seed=42):
    """Primera validación empírica del TRI: correlación de Spearman, sobre los pares emparejados, entre el TRI (y,
    para comparar, sus componentes por separado: coseno y estabilidad media) y la valoración de los expertos del
    par (media de los dos temas: interpretabilidad, coherencia y precisión de la palabra intrusa).

    Cada ρ lleva un IC 95 % por bootstrap percentil remuestreando pares, y se estiman las diferencias
    ρ(TRI) − ρ(estabilidad) y ρ(TRI) − ρ(coseno) con bootstrap pareado (los mismos pares remuestreados para
    ambos predictores): las correlaciones comparten datos y el TRI contiene la estabilidad, por lo que no son
    independientes y no deben compararse solo por su valor puntual.
    Devuelve None si no hay al menos min_pairs pares con TRI y valoraciones."""
    import pandas as pd
    if matches is None or per_topic is None or "TRI" not in matches or "_i" not in matches:
        return None
    pt = per_topic.set_index(["model", "topic"])
    rows = []
    for _, m in matches.iterrows():
        if pd.isna(m.get("_i")) or pd.isna(m.get("_j")) or not np.isfinite(m.get("TRI", np.nan)):
            continue
        a, b = ("LDA", lda_ids[int(m["_i"])]), ("BERTopic", bert_ids[int(m["_j"])])
        if a not in pt.index or b not in pt.index:
            continue
        r = {"LDA": a[1], "BERTopic": b[1], "TRI": float(m["TRI"]), "Cosine": float(m["Cosine"]),
             "Mean stability": float(np.sqrt(m["Stability LDA"] * m["Stability BERTopic"]))}
        for c in ("interpretability", "coherence", "model_precision"):
            r[c] = float(np.nanmean([pt.loc[a, c], pt.loc[b, c]]))
        rows.append(r)
    d = pd.DataFrame(rows)
    if len(d) < min_pairs:
        return None
    names = {"interpretability": "Interpretability (1-5)", "coherence": "Coherence (1-5)",
             "model_precision": "Word-intrusion precision"}
    preds = ("TRI", "Cosine", "Mean stability")
    rng = np.random.default_rng(seed)
    out, facts, diffs = {p_: {"Predictor": p_} for p_ in preds}, {}, []
    for c, nm in names.items():
        ok = d[list(preds) + [c]].dropna()
        n = len(ok)
        usable = n >= min_pairs and ok[c].nunique() > 1
        boots = {p_: [] for p_ in preds}
        if usable:
            X = {p_: ok[p_].values for p_ in preds}
            y = ok[c].values
            for _ in range(int(B)):
                idx = rng.integers(0, n, n)
                for p_ in preds:
                    boots[p_].append(_spearman_fast(X[p_][idx], y[idx]))
        for p_ in preds:
            if usable and ok[p_].nunique() > 1:
                rho, p = stats.spearmanr(ok[p_], ok[c])
                bb = np.array(boots[p_], float)
                bb = bb[np.isfinite(bb)]
                lo, hi = (np.percentile(bb, [2.5, 97.5]) if len(bb) >= 10 else (np.nan, np.nan))
                out[p_][nm] = (f"ρ = {rho:.3f} [{lo:.3f}, {hi:.3f}] (p {('= ' + fmt_p(p)) if p >= 0.001 else fmt_p(p)}; "
                               f"n = {n})")
                facts[(p_, c)] = {"rho": float(rho), "p": float(p), "n": int(n), "lo": float(lo), "hi": float(hi)}
            else:
                out[p_][nm] = "n.a."
        for other in ("Mean stability", "Cosine"):
            if (("TRI", c) in facts) and ((other, c) in facts):
                dd = np.array(boots["TRI"], float) - np.array(boots[other], float)
                dd = dd[np.isfinite(dd)]
                if len(dd) < 10:
                    continue
                point = facts[("TRI", c)]["rho"] - facts[(other, c)]["rho"]
                lo, hi = np.percentile(dd, [2.5, 97.5])
                pv = float(max(2 * min((dd <= 0).mean(), (dd >= 0).mean()), 1 / len(dd)))
                facts[("TRI-" + other, c)] = {"diff": float(point), "lo": float(lo), "hi": float(hi), "p": pv}
                diffs.append((other, nm, f"Δρ = {point:.3f} [{lo:.3f}, {hi:.3f}] (bootstrap p {('= ' + fmt_p(pv)) if pv >= 0.001 else fmt_p(pv)})"))
    table = pd.DataFrame([out[p_] for p_ in preds])
    for other, label in (("Mean stability", "TRI − mean stability"), ("Cosine", "TRI − cosine")):
        row = {"Predictor": label}
        for c, nm in names.items():
            row[nm] = next((t for o, n_, t in diffs if o == other and n_ == nm), "n.a.")
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    return {"table": table, "pairs": d, "facts": facts, "n_pairs": int(len(d)), "B": int(B)}
