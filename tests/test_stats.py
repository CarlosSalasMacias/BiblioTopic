import numpy as np
import pandas as pd

from bibliotopic import compare
from bibliotopic.compare import NO_TREND, NOT_ESTIMABLE, bh_adjust, topic_trends


def test_trend_not_estimable_with_fewer_than_three_years():
    theta = np.random.default_rng(0).dirichlet([1, 1, 1], 40)
    out, _ = topic_trends(theta, pd.Series([2021] * 20 + [2022] * 20), 10)
    assert (out["Trend"] == NOT_ESTIMABLE).all() and out["Slope"].isna().all()


def test_trend_classes_and_ci():
    years = np.repeat(np.arange(2010, 2020), 20)
    t = (years - 2010) / 10
    theta = np.c_[0.2 + 0.6 * t, 0.8 - 0.6 * t, np.full_like(t, 0.5)]
    theta = theta + np.random.default_rng(1).normal(0, 0.01, theta.shape)
    out, _ = topic_trends(theta, pd.Series(years), 10)
    assert list(out["Trend"][:2]) == ["Hot", "Cold"]
    assert out["Trend"][2] == NO_TREND  # sin tendencia detectada, no «Stable»
    assert (out["CI low"] <= out["Slope"]).all() and (out["Slope"] <= out["CI high"]).all()
    assert (out["Years used"] == 10).all()


def test_incomplete_year_excluded():
    years = np.repeat(np.arange(2018, 2024), 15)
    theta = np.random.default_rng(2).dirichlet([1, 1], len(years))
    out, m = topic_trends(theta, pd.Series(years), 10, exclude_years=[2023])
    assert 2023 not in m.index and out.attrs["excluded_years"] == [2023]


def test_bh_adjust():
    q = bh_adjust([0.01, 0.04, 0.03, np.nan])
    assert np.isnan(q[3]) and np.allclose(q[:3], [0.03, 0.04, 0.04])


class _M:  # modelo mínimo para compare()
    def __init__(self, k, n, prefix, rng, stab=None):
        self.k, self.ids = k, [f"{prefix}{i + 1}" for i in range(k)]
        self.labels = [f"{x}: t" for x in self.ids]
        self.beta = rng.random((k, 30))
        self.ctfidf = self.beta + rng.random((k, 30)) * 0.1
        self.top10 = [[f"w{j}" for j in np.argsort(-r)[:10]] for r in self.beta]
        self.top25 = self.top10
        self.stab = np.full(k, np.nan) if stab is None else stab
        self.npmi = self.umass = np.zeros(k)
        self.diversity = 1.0
        self.dominant = rng.integers(0, k, n)
        self.topics_raw = np.where(rng.random(n) < 0.3, -1, rng.integers(0, k, n))
        self.topics = np.where(self.topics_raw < 0, rng.integers(0, k, n), self.topics_raw)
        self.theta = np.eye(k)[self.dominant]
        self.prevalence = self.theta.mean(axis=0)
        self.outlier_rate = float((self.topics_raw < 0).mean())


def test_tri_not_available_without_stability():
    rng = np.random.default_rng(3)
    lda, bert = _M(3, 60, "L", rng), _M(3, 60, "B", rng)
    years = pd.Series(np.repeat(np.arange(2015, 2021), 10))
    C = compare.compare(lda, bert, years, 5, titles=[f"doc {i}" for i in range(60)])
    m = C["matches"]
    paired = m[m["_i"] >= 0][m["_j"] >= 0]
    assert paired["TRI"].isna().all() and paired["TRI note"].str.contains("Not available").all()
    assert set(paired["Convincing match"].str[:2]) <= {"Ye", "No"}
    assert paired["Representative LDA document"].str.startswith("doc").all()
    lda.stab, bert.stab = np.full(3, 0.8), np.full(3, 0.5)
    m2 = compare.compare(lda, bert, years, 5)["matches"]
    assert m2["TRI"].notna().sum() == 3


def test_agreement_denominators():
    from bibliotopic import stats as ST
    rng = np.random.default_rng(4)
    lda, bert = _M(3, 80, "L", rng), _M(3, 80, "B", rng)
    ok = bert.topics_raw >= 0
    a_all = ST.bootstrap_agreement(lda.dominant, bert.topics, B=30)
    a_ok = ST.bootstrap_agreement(lda.dominant[ok], bert.topics_raw[ok], B=30)
    assert a_all != a_ok  # dos poblaciones distintas con sus propios intervalos


def test_mann_kendall_and_sen():
    years = np.repeat(np.arange(2010, 2020), 20)
    t = (years - 2010) / 10
    theta = np.c_[0.2 + 0.6 * t, 0.8 - 0.6 * t]
    theta = theta + np.random.default_rng(5).normal(0, 0.01, theta.shape)
    out, _ = topic_trends(theta, pd.Series(years), 10, test="mann_kendall")
    assert list(out["Trend"]) == ["Hot", "Cold"] and (out["Test"] == "Mann-Kendall").all()
    assert out["Sen slope"][0] > 0 > out["Sen slope"][1]
    assert (out["Sen CI low"] <= out["Sen slope"]).all()


def test_permutation_null_and_fmt_p():
    from bibliotopic.stats import fmt_p, permutation_null_cosine
    rng = np.random.default_rng(0)
    A = rng.random((3, 50))
    B = np.vstack([A[0] + rng.random(50) * 0.05, rng.random((2, 50))])
    null = permutation_null_cosine(A, B, 200, 1)
    assert null["p"][0, 0] < 0.05 and null["p"].shape == (3, 3)
    assert fmt_p(0.0001) == "< 0.001" and fmt_p(0.0274) == "0.027"


def test_fleiss_kappa_known_value():
    from bibliotopic.validation import fleiss_kappa
    perfect = np.array([["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]], dtype=object)
    assert abs(fleiss_kappa(perfect) - 1.0) < 1e-9


def test_bca_interval_contains_point():
    from bibliotopic.stats import ci_from_boot
    rng = np.random.default_rng(1)
    boots = rng.normal(0.5, 0.05, 1000)
    lo, hi = ci_from_boot(boots, 0.52, "bca", jack=rng.normal(0.52, 0.01, 50))
    assert lo < 0.52 < hi


def test_parallel_lda_broker_identical_to_serial():
    """Los ajustes repartidos entre workers (JobBroker) dan exactamente el mismo resultado que en serie."""
    import json
    import pandas as pd
    from bibliotopic import lda_model as L
    from bibliotopic.text import Corpus
    rng = np.random.default_rng(0)
    topics = [["coffee", "rust", "fungus", "leaf", "spore", "hemileia"],
              ["soil", "nitrogen", "fertilizer", "yield", "compost", "organic"],
              ["shade", "tree", "bird", "biodiversity", "agroforestry", "canopy"]]
    docs = [{"TI": " ".join(rng.choice(topics[i % 3], 4)), "AB": " ".join(rng.choice(topics[i % 3], 40)),
             "PY": 2000 + i % 20} for i in range(150)]
    C = Corpus(pd.DataFrame(docs))
    params = dict(L.LDA_DEFAULTS, max_iter=20)
    t = L.select_k(C, [2, 3, 4], seed=42, n_seeds=2, fine_window=1, params=params)
    R = L.LDAResult(C, 3, seed=42, n_stab=2, params=params)
    B, lite = L.JobBroker(), L.corpus_from_payload(L.corpus_payload(C))
    for _ in range(10):
        try:
            t2 = L.select_k(C, [2, 3, 4], seed=42, n_seeds=2, fine_window=1, params=params, executor=B)
            R2 = L.LDAResult(C, 3, seed=42, n_stab=2, params=params, executor=B)
            break
        except L.NeedJobs as e:
            for s in e.specs:
                B.put(json.dumps(s), L.run_job_spec(lite, s))
    pd.testing.assert_frame_equal(t, t2)
    assert np.allclose(R.beta, R2.beta) and np.allclose(R.stab_reps, R2.stab_reps)


def test_relative_convergence_stops_early():
    from bibliotopic import lda_model as L
    from scipy import sparse
    rng = np.random.default_rng(1)
    X = sparse.csr_matrix(rng.poisson(0.3, size=(200, 80)))
    m, _, _ = L.fit_lda(X, 4, seed=1, max_iter=200, tol=0.001)
    assert m.n_iter_ < 200
    assert L.lda_params(4)["relative_tol"] == 0.001


def test_palettes_valid_and_theme_choice():
    from bibliotopic import palettes as P
    for k in P.ORDER:
        for v in (0, 1):
            p = P.Palette(k, v)
            assert len(p.cat) == 8 and len(set(p.cat)) == 8 and sorted(p.cat) == sorted(P.PALETTES[k]["cat"])
            Ls = [P.hex2oklch(c)[0] for c in p.seq]
            assert all(a > b for a, b in zip(Ls, Ls[1:]))  # escala secuencial: de clara a oscura
            assert p.increase != p.decrease
    assert P.choose(['TITLE-ABS-KEY ( coffee OR "coffea arabica" )'], ["coffee", "soil", "shade"])[0] == "earth"
    assert P.choose([], ["machine learning", "energy", "battery"])[0] == "tech"
    assert P.choose([], [])[0] == "classic"


def test_coherence_ci_over_topics_contains_point():
    """El IC de la coherencia se obtiene remuestreando temas: contiene la estimación puntual y, con BCa, el valor
    p es coherente con el intervalo."""
    import pandas as pd
    from bibliotopic.stats import bootstrap_coherence, topic_npmi
    from bibliotopic.text import Corpus
    rng = np.random.default_rng(0)
    T = [["coffee", "rust", "fungus", "leaf", "spore"], ["soil", "nitrogen", "fertilizer", "yield", "compost"],
         ["shade", "tree", "bird", "canopy", "agroforestry"], ["roast", "aroma", "flavor", "acid", "cup"]]
    C = Corpus(pd.DataFrame([{"TI": " ".join(rng.choice(T[i % 4], 3)), "AB": " ".join(rng.choice(T[i % 4] + T[(i + 1) % 4][:1], 30)),
                              "PY": 2000} for i in range(200)]))
    lda_tw = [list(t) for t in T]
    bert_tw = [T[0][:3] + T[1][:2], T[2][:3] + T[3][:2], T[1][2:] + T[0][3:], T[3][:3] + T[2][3:]]
    assert np.allclose(topic_npmi(C, lda_tw), C.npmi(lda_tw))
    for m in ("percentile", "bca"):
        r = bootstrap_coherence(C, {"LDA": lda_tw, "BERTopic": bert_tw}, B=300, method=m)
        for k in ("LDA", "BERTopic", "difference"):
            assert r[k]["lo"] <= r[k]["point"] <= r[k]["hi"]
        d = r["difference"]
        assert (d["p"] < 0.05) == (d["lo"] > 0 or d["hi"] < 0) or m == "percentile"


def test_tri_expert_correlation_ci_and_paired_difference():
    """IC bootstrap de cada ρ y diferencia pareada TRI − estabilidad: el IC contiene la estimación puntual."""
    import pandas as pd
    from bibliotopic.stats import tri_expert_correlation
    rng = np.random.default_rng(1)
    n = 30
    cos, sl, sb = rng.uniform(0.2, 0.9, n), rng.uniform(0.3, 1, n), rng.uniform(0.3, 1, n)
    matches = pd.DataFrame({"_i": range(n), "_j": range(n), "Cosine": cos, "Stability LDA": sl,
                            "Stability BERTopic": sb, "TRI": cos * np.sqrt(sl * sb)})
    ids_l, ids_b = [f"L{i + 1}" for i in range(n)], [f"B{i + 1}" for i in range(n)]
    q = np.sqrt(sl * sb)
    pt = pd.DataFrame([{"model": m, "topic": t, "interpretability": 1 + 4 * q[i] + rng.normal(0, .3),
                        "coherence": 1 + 4 * q[i] + rng.normal(0, .3), "model_precision": q[i]}
                       for i in range(n) for m, t in (("LDA", ids_l[i]), ("BERTopic", ids_b[i]))])
    r = tri_expert_correlation(matches, ids_l, ids_b, pt, B=300)
    f = r["facts"][("TRI", "interpretability")]
    assert f["lo"] <= f["rho"] <= f["hi"] and f["n"] == n
    d = r["facts"][("TRI-Mean stability", "interpretability")]
    assert d["lo"] <= d["diff"] <= d["hi"]
    assert len(r["table"]) == 5


def test_hamed_rao_and_agreement_jackknife_and_rematched_null():
    from bibliotopic.compare import mk_hamed_rao
    from bibliotopic.stats import bootstrap_agreement, permutation_null_cosine
    from scipy import stats as sst
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    rng = np.random.default_rng(0)
    # Hamed-Rao: serie con autocorrelación positiva -> p mayor (más conservador) que el MK clásico
    x = np.arange(30.0)
    e = np.zeros(30)
    for i in range(1, 30):
        e[i] = 0.8 * e[i - 1] + rng.normal()
    y = 0.03 * x + e
    p_hr, fac = mk_hamed_rao(x, y, sst.theilslopes(y, x)[0])
    assert 0 <= p_hr <= 1 and fac > 0
    # ARI/NMI: el IC (jackknife delete-d) contiene siempre la estimación puntual
    a = rng.integers(0, 6, 400)
    b = np.where(rng.random(400) < 0.6, a, rng.integers(0, 6, 400))
    r = bootstrap_agreement(a, b, B=200)
    assert r["NMI"][0] <= normalized_mutual_info_score(a, b) <= r["NMI"][1]
    assert r["ARI"][0] <= adjusted_rand_score(a, b) <= r["ARI"][1]
    # nula re-emparejada: más estricta que la nula por par
    A, B = rng.random((5, 60)), rng.random((5, 60))
    nu = permutation_null_cosine(A, B, n_perm=200)
    assert nu["p95_rematched"] >= nu["p95_global"] and nu["p_rematched"].shape == (5, 5)


def test_ref_key_particles_and_anonymous():
    from bibliotopic.citations import ref_key
    assert ref_key("De Souza J, 2013, PLANT DIS, V9") == "De Souza J, 2013"
    assert ref_key("DE SOUZA J, 2013, PLANT DIS") == "De Souza J, 2013"
    assert ref_key("De Souza, J., Smith, A., Title (2013) Plant Dis") == "De Souza J, 2013"
    assert ref_key("Anonymous, 2014, FAOSTAT DATABASE") == "Anonymous [FAOSTAT DATABASE], 2014"
    assert ref_key("Anonymous, 2014") is None


def test_first_period_min_docs():
    import pandas as pd
    from bibliotopic.hjbiplot import make_periods_mode
    y = pd.Series([1950, 1970, 1985] + [2000] * 12 + list(range(2001, 2021)) * 15)
    per, labels, _ = make_periods_mode(y, "width", 5, first_min_docs=10)
    assert labels[0] == "1950-1999" and labels[1].startswith("2000")


def test_stability_subsample_deterministic():
    from bibliotopic.lda_model import subsample_index
    a, b = subsample_index(100, 0.8, 43), subsample_index(100, 0.8, 43)
    assert len(a) == 80 and (a == b).all() and len(set(a)) == 80


def test_trend_sensitivity_columns():
    import numpy as np
    from bibliotopic import compare
    rng = np.random.default_rng(0)
    years = np.repeat(np.arange(2000, 2020), 40)
    theta = rng.dirichlet(np.ones(3), len(years))
    theta[:, 0] += (years - 2000) * 0.01
    main, _ = compare.topic_trends(theta, years, 10)
    ts = compare.trend_sensitivity(theta, years, main, ["a", "b", "c"], 30)
    assert ts["Trend (weighted by documents per year, WLS)"].iloc[0] == compare.HOT
    assert ts.attrs["min_docs_alt"] == 30


def test_broker_key_survives_js_roundtrip():
    import json
    from bibliotopic.lda_model import JobBroker
    spec = {"kind": "beta", "k": 5, "seed": 43, "params": {"tol": 0.001, "stab_subsample": 0.0, "max_iter": 100.0}}
    js = {"kind": "beta", "k": 5, "seed": 43, "params": {"tol": 0.001, "stab_subsample": 0, "max_iter": 100}}  # JS
    assert JobBroker.key(spec) == JobBroker.key(js)
