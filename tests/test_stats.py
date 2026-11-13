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
