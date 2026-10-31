"""Orquestación: datos -> bibliometría -> LDA -> BERTopic -> comparación -> salidas."""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import biblio, compare, figures as F, io as bio
from .bert_model import BERTResult, embed, DEFAULT_MODEL
from .hjbiplot import HJBiplot, make_periods
from .lda_model import LDAResult, select_k
from .text import Corpus

DEFAULTS = dict(
    year_min=None, year_max=None, doc_types=["Article", "Review"], languages=["English"],
    query_terms=[], extra_stopwords=[], min_df=5, max_df=0.5,
    k_min=4, k_max=20, k_step=2, k=None, seed=42, n_stab_lda=4, n_stab_bert=3,
    embed_model=DEFAULT_MODEL, bert_nr_topics="lda", min_cluster_size=None,
    period_width=5, top_countries=15, tau_strong=0.5, tau_partial=0.3, min_docs_year=10,
)


@dataclass
class Results:
    cfg: dict
    merge_stats: dict = field(default_factory=dict)
    availability: dict = field(default_factory=dict)
    excl_steps: list = field(default_factory=list)
    df: pd.DataFrame | None = None
    corpus: Corpus | None = None
    has_countries: bool = False
    ksel: pd.DataFrame | None = None
    lda: LDAResult | None = None
    bert: BERTResult | None = None
    emb_name: str = ""
    tables: dict = field(default_factory=dict)
    figs: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    versions: dict = field(default_factory=dict)


def _versions():
    import importlib.metadata as md
    v = {"python": platform.python_version()}
    for p in ["pandas", "numpy", "scikit-learn", "bertopic", "umap-learn", "hdbscan",
              "sentence-transformers", "networkx", "scipy"]:
        try:
            v[p] = md.version(p)
        except Exception:
            pass
    return v


def compute(files, cfg=None, progress=None):
    """files: lista de (nombre, bytes). Ejecuta todos los modelos (parte costosa)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    say = progress or (lambda p, m: None)
    R = Results(cfg=cfg, versions=_versions())

    say(0.02, "Reading files")
    frames, fmts = [], {}
    for name, raw in files:
        d, fmt = bio.read_file(name, raw)
        frames.append(d)
        fmts[name] = fmt
    merged, R.merge_stats = bio.merge(frames)
    R.merge_stats["formats"] = fmts
    R.availability = bio.field_availability(merged)
    df, R.excl_steps = bio.apply_filters(merged, cfg["year_min"], cfg["year_max"],
                                         cfg["doc_types"], cfg["languages"])
    R.has_countries = R.availability["C1"] > 0.3
    if R.has_countries:
        df = biblio.add_country_columns(df)
    R.df = df

    say(0.08, "Preprocessing text")
    R.corpus = Corpus(df, cfg["extra_stopwords"], cfg["min_df"], cfg["max_df"], cfg["query_terms"])

    ks = list(range(cfg["k_min"], cfg["k_max"] + 1, cfg["k_step"]))
    R.ksel = select_k(R.corpus, ks, seed=cfg["seed"], max_iter=30,
                      progress=lambda p, m: say(0.08 + 0.32 * p, m))
    k = cfg["k"] or int(R.ksel.loc[R.ksel["Score"].idxmax(), "K"])
    say(0.42, f"Fitting final LDA (K={k}) and stability replicates")
    R.lda = LDAResult(R.corpus, k, seed=cfg["seed"], n_stab=cfg["n_stab_lda"])

    say(0.55, "Computing document embeddings")
    emb, R.emb_name = embed(R.corpus.raw, cfg["embed_model"])
    nr = k if cfg["bert_nr_topics"] == "lda" else (cfg["bert_nr_topics"] or None)
    say(0.65, "Fitting BERTopic and stability replicates")
    R.bert = BERTResult(R.corpus, emb, R.emb_name, nr_topics=nr, seed=cfg["seed"],
                        n_stab=cfg["n_stab_bert"], min_cluster_size=cfg["min_cluster_size"])
    say(0.9, "Models ready")
    return R


def render(R: Results, out_dir, progress=None):
    """Tablas, figuras y hechos numéricos (rápido; se repite tras editar etiquetas)."""
    say = progress or (lambda p, m: None)
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    fp = lambda n: str(out / "figures" / n)  # noqa: E731
    cfg, df, lda, bert = R.cfg, R.df, R.lda, R.bert
    T, FIG, facts = {}, {}, {}

    # ---- PRISMA
    ms = R.merge_stats
    FIG["prisma"] = F.prisma(fp("Fig01_PRISMA_flow.png"), ms["records_by_db"], ms["records_total"],
                             ms["duplicates_removed"], ms["unique"], R.excl_steps, len(df))

    # ---- bibliometría
    say(0.91, "Bibliometric indicators")
    info = biblio.main_information(df, R.has_countries)
    T["T1_main_information"] = pd.DataFrame(list(info.items()), columns=["Description", "Results"])
    ap = biblio.annual_production(df)
    T["annual_production"] = ap
    FIG["annual"] = F.annual_production(fp("Fig02_annual_production.png"), ap)
    src = biblio.top_sources(df, 15)
    T["T2_top_sources"] = src
    FIG["sources"] = F.barh(fp("Fig03_top_sources.png"), src["Source"], src["Articles"],
                            "Most relevant sources", "Documents")
    brad = biblio.bradford(df)
    T["bradford"] = brad
    au = biblio.top_authors(df, 15)
    T["T3_top_authors"] = au
    FIG["authors"] = F.barh(fp("Fig04_top_authors.png"), au["Author"], au["Articles"],
                            "Most productive authors", "Documents", color=F.GREEN)
    T["lotka"] = biblio.lotka(df)
    facts["bradford_zone1"] = int((brad["Zone"] == "Zone 1").sum())
    facts["n_sources"] = int(len(brad))
    if R.has_countries:
        cp = biblio.country_production(df, 20)
        T["T4_countries"] = cp
        FIG["countries"] = F.country_scp_mcp(fp("Fig05_country_production.png"), cp)
        col = biblio.country_collaboration(df)
        T["country_collaboration"] = col
        from collections import Counter
        import itertools
        part = Counter(itertools.chain.from_iterable(df["COUNTRIES"]))
        nodes = part.most_common(40)
        edges = {(a, b): w for a, b, w in col.itertuples(index=False)}
        comms = F.network(fp("Fig06_country_collaboration.png"), nodes, edges,
                          "Country collaboration network")
        if comms:
            FIG["collab"] = fp("Fig06_country_collaboration.png")
            facts["collab_clusters"] = [sorted(c) for c in sorted(comms, key=len, reverse=True)]
    kw_field = "DE" if R.availability["DE"] > 0.3 else ("ID" if R.availability["ID"] > 0.3 else None)
    if kw_field:
        kf = biblio.keyword_freq(df, kw_field, 30)
        T["T5_keywords"] = kf
        nodes, edges = biblio.keyword_cooccurrence(df, kw_field, 60)
        comms = F.network(fp("Fig07_keyword_cooccurrence.png"), nodes, edges,
                          "Keyword co-occurrence network")
        if comms:
            FIG["kw"] = fp("Fig07_keyword_cooccurrence.png")
            facts["kw_clusters"] = [sorted(c, key=lambda k: -dict(nodes).get(k, 0))[:6]
                                    for c in sorted(comms, key=len, reverse=True)]
    facts["kw_field"] = kw_field

    # ---- LDA
    say(0.93, "Topic figures")
    FIG["ksel"] = F.k_selection(fp("Fig08_LDA_K_selection.png"), R.ksel, lda.k)
    T["lda_k_selection"] = R.ksel.round(4)
    T["T6_LDA_topics"] = lda.topic_table()
    w_lda = [np.sort(r)[::-1] for r in lda.beta]
    FIG["lda_words"] = F.topic_words(fp("Fig09_LDA_topics.png"), w_lda, lda.top10, lda.labels,
                                     "LDA topics: top terms (word probability)")
    periods, plabels = make_periods(df["PY"], cfg["period_width"])
    facts["periods"] = plabels
    ml = pd.DataFrame(lda.theta, columns=lda.labels).groupby(periods.values).mean().reindex(plabels)
    mb = pd.DataFrame(bert.theta, columns=bert.labels).groupby(periods.values).mean().reindex(plabels)
    T["lda_by_period"] = ml.round(4).reset_index(names="Period")
    T["bert_by_period"] = mb.round(4).reset_index(names="Period")
    FIG["lda_time"] = F.topic_heatmap_time(fp("Fig10_LDA_topics_by_period.png"), ml, lda.labels,
                                           "LDA topic prevalence by period")
    hj_l = HJBiplot(ml.rename(columns=dict(zip(lda.labels, lda.ids))))
    FIG["hj_lda"] = F.biplot(fp("Fig11_HJBiplot_LDA_periods.png"), hj_l,
                             "HJ-Biplot: LDA topics and periods")
    T["hj_lda_inertia"] = hj_l.summary()

    # ---- BERTopic
    w_b = [np.sort(r)[::-1] for r in bert.ctfidf]
    FIG["bert_words"] = F.topic_words(fp("Fig12_BERTopic_topics.png"), w_b, bert.top10, bert.labels,
                                      "BERTopic topics: top terms (c-TF-IDF)", color=F.ORANGE)
    if bert.xy is not None:
        FIG["bert_map"] = F.doc_map(fp("Fig13_BERTopic_document_map.png"), bert.xy, bert.topics_raw,
                                    bert.labels, "BERTopic document map (UMAP projection)")
    T["T7_BERTopic_topics"] = bert.topic_table()
    FIG["bert_time"] = F.topic_heatmap_time(fp("Fig14_BERTopic_topics_by_period.png"), mb,
                                            bert.labels, "BERTopic topic prevalence by period",
                                            cmap="Oranges")
    hj_b = HJBiplot(mb.rename(columns=dict(zip(bert.labels, bert.ids))))
    FIG["hj_bert"] = F.biplot(fp("Fig15_HJBiplot_BERTopic_periods.png"), hj_b,
                              "HJ-Biplot: BERTopic topics and periods")
    T["hj_bert_inertia"] = hj_b.summary()

    # ---- comparación
    say(0.96, "Comparing models")
    C = compare.compare(lda, bert, df["PY"], cfg["min_docs_year"], cfg["tau_strong"],
                        cfg["tau_partial"])
    T["T8_model_metrics"] = C["metrics"]
    T["T9_topic_alignment"] = C["matches"].drop(columns=["_i", "_j"])
    T["doc_agreement"] = pd.DataFrame(list(C["agreement"].items()), columns=["Measure", "Value"])
    T["lda_trends"] = C["trends_lda"].assign(Topic=lda.labels).round(5)
    T["bert_trends"] = C["trends_bert"].assign(Topic=bert.labels).round(5)
    FIG["sim"] = F.similarity_heatmap(fp("Fig16_LDA_vs_BERTopic_similarity.png"), C["S"], C["matches"],
                                      "Topic similarity between LDA and BERTopic")
    FIG["cont"] = F.contingency(fp("Fig17_document_assignment_agreement.png"), C["contingency"],
                                "Document-level agreement between models")
    FIG["trends"] = F.trends(fp("Fig18_hot_cold_topics.png"), C["trends_lda"], C["trends_bert"],
                             lda.labels, bert.labels)

    # ---- países x temas (si hay afiliaciones)
    if R.has_countries:
        topc = [c for c, _ in T["T4_countries"][["Country", "Articles"]].values[:cfg["top_countries"]]]
        rows = []
        for c in topc:
            m = df["COUNTRIES"].map(lambda cs: c in cs).values
            rows.append(lda.theta[m].mean(axis=0))
        mc = pd.DataFrame(rows, index=topc, columns=lda.ids)
        hj_c = HJBiplot(mc)
        FIG["hj_country"] = F.biplot(fp("Fig19_HJBiplot_countries_LDA_topics.png"), hj_c,
                                     "HJ-Biplot: countries and LDA topics", row_color=F.GREEN)
        T["hj_country_inertia"] = hj_c.summary()
        facts["hj_country"] = hj_c

    # ---- mapa estratégico (si hay citas)
    sm = compare.strategic_map(lda.theta, df, C["trends_lda"], lda.labels)
    if sm is not None:
        T["lda_strategic_map"] = sm.round(4)
        FIG["strategic"] = F.strategic(fp("Fig20_strategic_map_LDA.png"), sm,
                                       "Strategic map of LDA topics (growth vs impact)")

    facts.update({"info": info, "ap": ap, "C": C, "hj_l": hj_l, "hj_b": hj_b, "ml": ml, "mb": mb,
                  "periods": plabels})
    R.tables, R.figs, R.facts = T, FIG, facts

    # ---- exportar
    with pd.ExcelWriter(out / "tables" / "all_tables.xlsx") as xw:
        for name, t in T.items():
            t.to_excel(xw, sheet_name=name[:31], index=False)
    docs = df[["UT", "DI", "TI", "SO", "PY", "DB_SOURCES"]].copy()
    docs["LDA_dominant"] = [lda.ids[i] for i in lda.dominant]
    docs["BERTopic"] = [bert.ids[i] for i in bert.topics]
    docs.to_csv(out / "tables" / "documents_topics.csv", index=False)
    (out / "run_config.json").write_text(json.dumps(
        {"config": cfg, "versions": R.versions, "embedding": R.emb_name,
         "merge": {k: v for k, v in ms.items()}, "field_availability": R.availability},
        indent=2, default=str), encoding="utf-8")
    return R


