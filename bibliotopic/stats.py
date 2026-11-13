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


def bootstrap_coherence(corpus, models: dict, B=200, seed=42, topn=10):
    """IC 95 % (percentil) del NPMI medio de cada modelo y de la diferencia entre ellos,
    remuestreando documentos con reemplazo. models: {nombre: lista de top-words por tema}."""
    words = sorted({w for tw in models.values() for t in tw for w in t[:topn] if w in corpus.word_index})
    col = {w: i for i, w in enumerate(words)}
    Xb = corpus.Xbin[:, [corpus.word_index[w] for w in words]].toarray().astype(np.float32)
    pairs = {}
    for name, tw in models.items():
        pairs[name] = []
        for t in tw:
            ids = [col[w] for w in t[:topn] if w in col]
            pairs[name].append([(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))])
    rng = np.random.default_rng(seed)
    n = Xb.shape[0]
    boots = {name: [] for name in models}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        Xs = Xb[idx]
        for name in models:
            boots[name].append(np.nanmean(_npmi_matrix(Xs, pairs[name])))
    res = {}
    for name, v in boots.items():
        v = np.array(v)
        res[name] = {"mean": float(np.nanmean(v)), "lo": float(np.nanpercentile(v, 2.5)),
                     "hi": float(np.nanpercentile(v, 97.5))}
    names = list(models)
    if len(names) == 2:
        d = np.array(boots[names[0]]) - np.array(boots[names[1]])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        res["difference"] = {"label": f"{names[0]} - {names[1]}", "mean": float(d.mean()),
                             "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
                             "p": float(max(p, 1 / B))}
    return res


def bootstrap_agreement(a, b, B=200, seed=42):
    rng = np.random.default_rng(seed)
    n = len(a)
    ari, nmi = [], []
    for _ in range(B):
        i = rng.integers(0, n, n)
        ari.append(adjusted_rand_score(a[i], b[i]))
        nmi.append(normalized_mutual_info_score(a[i], b[i]))
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))  # noqa: E731
    return {"ARI": ci(ari), "NMI": ci(nmi)}


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


def inference_table(corpus, lda, bert, B=200, seed=42):
    """Tabla de inferencia para el informe (IC 95 % y pruebas)."""
    import pandas as pd
    coh = bootstrap_coherence(corpus, {"LDA": lda.top10, "BERTopic": bert.top10}, B=B, seed=seed)
    st_l, st_b = mean_ci(lda.stab, bounds=(0, 1)), mean_ci(bert.stab, bounds=(0, 1))
    t_coh = compare_topics(lda.npmi, bert.npmi)
    t_st = compare_topics(lda.stab, bert.stab)
    ok = bert.topics_raw >= 0
    # dos poblaciones con su propio denominador e intervalo: (a) todos los documentos con los outliers de
    # BERTopic reasignados; (b) solo los documentos asignados por el agrupamiento (asignación original)
    agr = bootstrap_agreement(lda.dominant, bert.topics, B=B, seed=seed)
    agr_ok = (bootstrap_agreement(lda.dominant[ok], bert.topics_raw[ok], B=B, seed=seed)
              if ok.sum() > 2 and (~ok).any() else None)
    rows = [
        ("Mean NPMI coherence", coh["LDA"], coh["BERTopic"], t_coh,
         f"Bootstrap difference {coh['difference']['mean']:.3f} "
         f"[{coh['difference']['lo']:.3f}, {coh['difference']['hi']:.3f}], p = {coh['difference']['p']:.3f}"),
        ("Mean topic stability", {"mean": st_l[0], "lo": st_l[1], "hi": st_l[2]},
         {"mean": st_b[0], "lo": st_b[1], "hi": st_b[2]}, t_st, ""),
    ]
    out = []
    for name, a, b, t, note in rows:
        out.append({"Metric": name,
                    "LDA (95% CI)": f"{a['mean']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]",
                    "BERTopic (95% CI)": f"{b['mean']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]",
                    "Mann-Whitney U": round(t["U"], 1) if not np.isnan(t["U"]) else "",
                    "p-value": round(t["p"], 4) if not np.isnan(t["p"]) else "",
                    "Effect size r": round(t["r"], 3) if not np.isnan(t["r"]) else "",
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
                        "Note": f"n = {n_:,} documents; bootstrap over documents"})
    facts = {"coh": coh, "stab_lda": st_l, "stab_bert": st_b, "t_coh": t_coh, "t_stab": t_st, "agr": agr,
             "agr_clustered": agr_ok, "n_all": n_all, "n_clustered": n_ok}
    return pd.DataFrame(out), facts


def tri_ci(matches, B=1000, seed=42):
    """IC 95 % bootstrap del TRI medio de los pares emparejados."""
    v = matches["TRI"].dropna().values.astype(float) if "TRI" in matches else np.array([])
    if len(v) < 3:
        return None
    rng = np.random.default_rng(seed)
    bs = [rng.choice(v, len(v)).mean() for _ in range(B)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
