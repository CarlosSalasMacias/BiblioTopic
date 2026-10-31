"""Orquestación por pasos (la página web llama a cada paso; JS calcula embeddings y UMAP)."""
from __future__ import annotations

import io as _io
import itertools
import json
import platform
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import biblio, compare, figures as F, io as bio
from .bert_model import BERTResult
from .hjbiplot import HJBiplot, make_periods
from .lda_model import LDAResult, select_k
from .text import Corpus

ANALYSES = {
    "biblio": "Bibliometría descriptiva (producción, fuentes, autores, Bradford, Lotka)",
    "countries": "Países y red de colaboración internacional",
    "keywords": "Red de co-ocurrencia de palabras clave",
    "lda": "Modelado de temas LDA",
    "bertopic": "Modelado de temas BERTopic",
    "compare": "Comparación LDA vs BERTopic",
    "temporal": "Tendencias temporales (temas emergentes / en declive)",
    "hjbiplot": "HJ-Biplot (periodos × temas y países × temas)",
    "strategic": "Mapa estratégico (crecimiento vs impacto)",
    "stability": "Estabilidad de los temas entre semillas",
}

DEFAULTS = dict(
    analyses=list(ANALYSES), year_min=None, year_max=None, doc_types=["Article", "Review"],
    languages=["English"], query_terms=[], extra_stopwords=[], min_df=5, max_df=0.5,
    k_min=5, k_max=30, k_step=5, k=None, seed=42, n_stab_lda=3, n_stab_bert=3,
    bert_nr_topics="lda", min_cluster_size=None, period_width=5, top_countries=15,
    tau_strong=0.5, tau_partial=0.3, min_docs_year=10,
)


def _versions():
    import importlib.metadata as md
    v = {"python": platform.python_version()}
    for p in ["pandas", "numpy", "scikit-learn", "scipy", "networkx", "matplotlib"]:
        try:
            v[p] = md.version(p)
        except Exception:
            pass
    return v


class Session:
    def __init__(self):
        self.cfg = dict(DEFAULTS)
        self.frames = []
        self.merged = None
        self.merge_stats, self.availability, self.excl_steps = {}, {}, []
        self.df = self.corpus = self.ksel = self.lda = self.bert = None
        self.has_countries = False
        self.emb_name = ""
        self.tables, self.figs, self.facts = {}, {}, {}
        self.versions = _versions()

    def on(self, a):
        return a in self.cfg["analyses"]

    # ------------------------------------------------------------- 1. datos
    def load(self, files):
        """files: lista de (nombre, bytes). Devuelve un resumen para la interfaz."""
        self.frames, fmts, errors = [], {}, []
        expanded = []
        for name, raw in files:
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(_io.BytesIO(raw)) as z:
                    expanded += [(n.split("/")[-1], z.read(n)) for n in z.namelist()
                                 if n.lower().endswith((".txt", ".csv", ".tsv"))
                                 and not n.startswith("__MACOSX")]
            else:
                expanded.append((name, raw))
        for name, raw in expanded:
            try:
                d, fmt = bio.read_file(name, raw)
                self.frames.append(d)
                fmts[name] = fmt
            except Exception as e:
                errors.append(f"{name}: {e}")
        if not self.frames:
            return {"ok": False, "errors": errors}
        self.merged, self.merge_stats = bio.merge(self.frames)
        self.merge_stats["formats"] = fmts
        self.availability = bio.field_availability(self.merged)
        m = self.merged
        yrs = m["PY"].dropna().astype(int)
        return {
            "ok": True, "errors": errors, "stats": self.merge_stats,
            "availability": {k: round(100 * v, 1) for k, v in self.availability.items()},
            "with_abstract": int((m["AB"].str.len() > 0).sum()),
            "year_min": int(yrs.min()), "year_max": int(yrs.max()),
            "doc_types": m["DT"][m["DT"] != ""].value_counts().to_dict(),
            "languages": m["LA"][m["LA"] != ""].value_counts().to_dict(),
        }

    # ------------------------------------------------- 2. filtros y texto
    def prepare(self, cfg):
        self.cfg = {**DEFAULTS, **cfg}
        c = self.cfg
        df, self.excl_steps = bio.apply_filters(self.merged, c["year_min"], c["year_max"],
                                                c["doc_types"], c["languages"])
        self.has_countries = self.availability["C1"] > 0.3 and self.on("countries")
        if self.availability["C1"] > 0.3:
            df = biblio.add_country_columns(df)
        self.df = df
        self.corpus = Corpus(df, c["extra_stopwords"], c["min_df"], c["max_df"], c["query_terms"])
        self.lda = self.bert = self.ksel = None
        return {"documents": len(df), "vocabulary": len(self.corpus.vocab),
                "phrases": len(self.corpus.phrases), "excluded": self.excl_steps}

    def texts(self):
        return self.corpus.raw

    # --------------------------------------------------------------- 3. LDA
    def run_lda(self, progress=None):
        c = self.cfg
        say = progress or (lambda p, m: None)
        ks = list(range(c["k_min"], c["k_max"] + 1, c["k_step"]))
        if c["k"]:
            self.ksel = None
            k = int(c["k"])
        else:
            self.ksel = select_k(self.corpus, ks, seed=c["seed"],
                                 progress=lambda p, m: say(0.8 * p, m))
            k = int(self.ksel.loc[self.ksel["Score"].idxmax(), "K"])
        say(0.85, f"LDA final (K={k})")
        n_stab = c["n_stab_lda"] if self.on("stability") else 0
        self.lda = LDAResult(self.corpus, k, seed=c["seed"], n_stab=n_stab)
        return {"k": k}

    def bert_nr_topics(self):
        c = self.cfg
        if c["bert_nr_topics"] == "lda":
            return self.lda.k if self.lda is not None else None
        return c["bert_nr_topics"] or None

    def n_umap_runs(self):
        return 1 + (self.cfg["n_stab_bert"] if self.on("stability") else 0)

    # ---------------------------------------------------------- 4. BERTopic
    def run_bertopic(self, reduced_list, xy, emb_name):
        self.emb_name = emb_name
        self.bert = BERTResult(self.corpus, reduced_list, xy, emb_name,
                               nr_topics=self.bert_nr_topics(),
                               min_cluster_size=self.cfg["min_cluster_size"])
        return {"k": self.bert.k, "native": self.bert.native_topics,
                "outliers": round(100 * self.bert.outlier_rate, 1)}

    def topic_labels(self):
        out = {}
        for name, r in (("lda", self.lda), ("bert", self.bert)):
            if r is not None:
                out[name] = [{"id": i, "terms": ", ".join(w[:8]), "label": l.split(":", 1)[1].strip()}
                             for i, w, l in zip(r.ids, r.top10, r.labels)]
        return out

    def set_labels(self, labels):
        for name, r in (("lda", self.lda), ("bert", self.bert)):
            if r is not None and name in labels:
                r.labels = [f"{i}: {l}" for i, l in zip(r.ids, labels[name])]

    # ----------------------------------------------------------- 5. salidas
    def render(self, out_dir="/out"):
        out = Path(out_dir)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        for p in (out / "figures").glob("*.png"):
            p.unlink()
        fp = lambda n: str(out / "figures" / n)  # noqa: E731
        c, df, lda, bert = self.cfg, self.df, self.lda, self.bert
        T, FIG, X = {}, {}, {}
        ms = self.merge_stats

        FIG["prisma"] = F.prisma(fp("Fig01_PRISMA_flow.png"), ms["records_by_db"], ms["records_total"],
                                 ms["duplicates_removed"], ms["unique"], self.excl_steps, len(df))
        info = biblio.main_information(df, self.has_countries)
        T["T1_main_information"] = pd.DataFrame(list(info.items()), columns=["Description", "Results"])
        ap = biblio.annual_production(df)
        T["annual_production"] = ap
        X.update(info=info, ap=ap)

        if self.on("biblio"):
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
            X["bradford_zone1"] = int((brad["Zone"] == "Zone 1").sum())
            X["n_sources"] = int(len(brad))

        if self.has_countries:
            cp = biblio.country_production(df, 20)
            T["T4_countries"] = cp
            FIG["countries"] = F.country_scp_mcp(fp("Fig05_country_production.png"), cp)
            col = biblio.country_collaboration(df)
            T["country_collaboration"] = col
            part = Counter(itertools.chain.from_iterable(df["COUNTRIES"]))
            edges = {(a, b): w for a, b, w in col.itertuples(index=False)}
            comms = F.network(fp("Fig06_country_collaboration.png"), part.most_common(40), edges,
                              "Country collaboration network")
            if comms:
                FIG["collab"] = fp("Fig06_country_collaboration.png")
                X["collab_clusters"] = [sorted(cm, key=lambda k: -part[k])
                                        for cm in sorted(comms, key=len, reverse=True)]

        kw_field = None
        if self.on("keywords"):
            kw_field = ("DE" if self.availability["DE"] > 0.3 else
                        "ID" if self.availability["ID"] > 0.3 else None)
        if kw_field:
            T["T5_keywords"] = biblio.keyword_freq(df, kw_field, 30)
            nodes, edges = biblio.keyword_cooccurrence(df, kw_field, 60)
            comms = F.network(fp("Fig07_keyword_cooccurrence.png"), nodes, edges,
                              "Keyword co-occurrence network")
            if comms:
                FIG["kw"] = fp("Fig07_keyword_cooccurrence.png")
                X["kw_clusters"] = [sorted(cm, key=lambda k: -dict(nodes).get(k, 0))[:6]
                                    for cm in sorted(comms, key=len, reverse=True)]
        X["kw_field"] = kw_field

        width = int(c["period_width"])
        periods, plabels = make_periods(df["PY"], width)
        span = int(df["PY"].max() - df["PY"].min()) + 1
        while len(plabels) < 4 and width > 2 and span >= 8:  # al menos 4 periodos para el HJ-Biplot
            width -= 1
            periods, plabels = make_periods(df["PY"], width)
        X["period_width"] = width
        X["periods"] = plabels
        C = None
        if lda is not None:
            if self.ksel is not None:
                FIG["ksel"] = F.k_selection(fp("Fig08_LDA_K_selection.png"), self.ksel, lda.k)
                T["lda_k_selection"] = self.ksel.round(4)
            T["T6_LDA_topics"] = lda.topic_table()
            FIG["lda_words"] = F.topic_words(fp("Fig09_LDA_topics.png"),
                                             [np.sort(r)[::-1] for r in lda.beta], lda.top10,
                                             lda.labels, "LDA topics: top terms (word probability)")
            ml = pd.DataFrame(lda.theta, columns=lda.labels).groupby(periods.values).mean().reindex(plabels)
            T["lda_by_period"] = ml.round(4).reset_index(names="Period")
            X["ml"] = ml
            if self.on("temporal"):
                FIG["lda_time"] = F.topic_heatmap_time(fp("Fig10_LDA_topics_by_period.png"), ml,
                                                       lda.labels, "LDA topic prevalence by period")
            if self.on("hjbiplot") and len(plabels) >= 3:
                hj = HJBiplot(ml.rename(columns=dict(zip(lda.labels, lda.ids))))
                FIG["hj_lda"] = F.biplot(fp("Fig11_HJBiplot_LDA_periods.png"), hj,
                                         "HJ-Biplot: LDA topics and periods")
                T["hj_lda_inertia"] = hj.summary()
                X["hj_l"] = hj
        if bert is not None:
            T["T7_BERTopic_topics"] = bert.topic_table()
            FIG["bert_words"] = F.topic_words(fp("Fig12_BERTopic_topics.png"),
                                              [np.sort(r)[::-1] for r in bert.ctfidf], bert.top10,
                                              bert.labels, "BERTopic topics: top terms (c-TF-IDF)",
                                              color=F.ORANGE)
            if bert.xy is not None:
                FIG["bert_map"] = F.doc_map(fp("Fig13_BERTopic_document_map.png"), bert.xy,
                                            bert.topics_raw, bert.labels,
                                            "BERTopic document map (UMAP projection)")
            mb = pd.DataFrame(bert.theta, columns=bert.labels).groupby(periods.values).mean().reindex(plabels)
            T["bert_by_period"] = mb.round(4).reset_index(names="Period")
            if self.on("temporal"):
                FIG["bert_time"] = F.topic_heatmap_time(fp("Fig14_BERTopic_topics_by_period.png"), mb,
                                                        bert.labels, "BERTopic topic prevalence by period",
                                                        cmap="Oranges")
            if self.on("hjbiplot") and len(plabels) >= 3:
                hj = HJBiplot(mb.rename(columns=dict(zip(bert.labels, bert.ids))))
                FIG["hj_bert"] = F.biplot(fp("Fig15_HJBiplot_BERTopic_periods.png"), hj,
                                          "HJ-Biplot: BERTopic topics and periods")
                T["hj_bert_inertia"] = hj.summary()
                X["hj_b"] = hj

        trends_l = trends_b = None
        if lda is not None:
            trends_l, _ = compare.topic_trends(lda.theta, df["PY"], c["min_docs_year"])
            T["lda_trends"] = trends_l.assign(Topic=lda.labels).round(5)
        if bert is not None:
            trends_b, _ = compare.topic_trends(bert.theta, df["PY"], c["min_docs_year"])
            T["bert_trends"] = trends_b.assign(Topic=bert.labels).round(5)
        X["trends_lda"], X["trends_bert"] = trends_l, trends_b

        if lda is not None and bert is not None and self.on("compare"):
            C = compare.compare(lda, bert, df["PY"], c["min_docs_year"], c["tau_strong"],
                                c["tau_partial"])
            T["T8_model_metrics"] = C["metrics"]
            T["T9_topic_alignment"] = C["matches"].drop(columns=["_i", "_j"])
            T["doc_agreement"] = pd.DataFrame(list(C["agreement"].items()), columns=["Measure", "Value"])
            FIG["sim"] = F.similarity_heatmap(fp("Fig16_LDA_vs_BERTopic_similarity.png"), C["S"],
                                              C["matches"], "Topic similarity between LDA and BERTopic")
            FIG["cont"] = F.contingency(fp("Fig17_document_assignment_agreement.png"), C["contingency"],
                                        "Document-level agreement between models")
        X["C"] = C
        if self.on("temporal") and (trends_l is not None or trends_b is not None):
            FIG["trends"] = F.trends(fp("Fig18_hot_cold_topics.png"), trends_l, trends_b,
                                     lda.labels if lda else None, bert.labels if bert else None)

        if self.has_countries and lda is not None and self.on("hjbiplot"):
            topc = list(T["T4_countries"]["Country"].values[:c["top_countries"]])
            rows = [lda.theta[df["COUNTRIES"].map(lambda cs, cc=cc: cc in cs).values].mean(axis=0)
                    for cc in topc]
            if len(topc) >= 3:
                hjc = HJBiplot(pd.DataFrame(rows, index=topc, columns=lda.ids))
                FIG["hj_country"] = F.biplot(fp("Fig19_HJBiplot_countries_LDA_topics.png"), hjc,
                                             "HJ-Biplot: countries and LDA topics", row_color=F.GREEN)
                T["hj_country_inertia"] = hjc.summary()
                X["hj_country"] = hjc

        if self.on("strategic") and lda is not None and trends_l is not None:
            sm = compare.strategic_map(lda.theta, df, trends_l, lda.labels)
            if sm is not None:
                T["lda_strategic_map"] = sm.round(4)
                FIG["strategic"] = F.strategic(fp("Fig20_strategic_map_LDA.png"), sm,
                                               "Strategic map of LDA topics (growth vs impact)")

        self.tables, self.figs, self.facts = T, FIG, X
        with pd.ExcelWriter(out / "tables" / "all_tables.xlsx") as xw:
            for name, t in T.items():
                t.to_excel(xw, sheet_name=name[:31], index=False)
        docs = df[["UT", "DI", "TI", "SO", "PY", "DB_SOURCES"]].copy()
        if lda is not None:
            docs["LDA_dominant"] = [lda.ids[i] for i in lda.dominant]
        if bert is not None:
            docs["BERTopic"] = [bert.ids[i] for i in bert.topics]
        docs.to_csv(out / "tables" / "documents_topics.csv", index=False)
        (out / "run_config.json").write_text(json.dumps(
            {"config": c, "versions": self.versions, "embedding": self.emb_name,
             "merge": ms, "field_availability": self.availability}, indent=2, default=str),
            encoding="utf-8")
        return [{"key": k, "path": v} for k, v in FIG.items()]

    def table_json(self, name, n=50):
        t = self.tables.get(name)
        return None if t is None else json.loads(t.head(n).to_json(orient="split", force_ascii=False))

    def zip_outputs(self, out_dir="/out"):
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in Path(out_dir).rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(out_dir))
        return buf.getvalue()
