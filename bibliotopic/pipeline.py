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

from . import biblio, citations as CIT, compare, evolution as EVO, figures as F, io as bio
from . import stats as ST
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
    "inference": "Inferencia estadística (IC 95 % por bootstrap y pruebas entre modelos)",
    "citations": "Análisis de citas (más citados por tema y co-citación)",
    "thematic": "Mapa temático de Callon y evolución temática (Sankey)",
}

DEFAULTS = dict(
    analyses=list(ANALYSES), year_min=None, year_max=None, doc_types=["Article", "Review"],
    languages=["English"], query_terms=[], extra_stopwords=[], min_df=5, max_df=0.5,
    k_min=5, k_max=30, k_step=5, k=None, seed=42, n_stab_lda=3, n_stab_bert=3,
    bert_nr_topics="lda", min_cluster_size=None, period_width=5, top_countries=15,
    tau_strong=0.5, tau_partial=0.3, min_docs_year=10, synonyms="",
    export_tiff=True, cluster_shape="hull", composite=True, n_boot=200,
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
        self.validation = None
        self.ai = {}

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
        self.input_files = [name for name, _ in files]
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
        F.configure(tiff=bool(c["export_tiff"]), cluster_shape=c["cluster_shape"])
        df, self.excl_steps = bio.apply_filters(self.merged, c["year_min"], c["year_max"],
                                                c["doc_types"], c["languages"])
        self.has_countries = self.availability["C1"] > 0.3 and self.on("countries")
        if self.availability["C1"] > 0.3:
            df = biblio.add_country_columns(df)
        df = biblio.normalize_keywords(df, biblio.parse_synonyms(c.get("synonyms", "")))
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
                r.custom_labels = True

    def propose_titles(self, meta=None):
        from .report import propose_titles
        return propose_titles(self, meta or {})

    def _bip_names(self, r):
        """Nombre de los vectores en los biplots: código + nombre corto si los autores lo asignaron."""
        if getattr(r, "custom_labels", False):
            return [f"{i} {l.split(':', 1)[1].strip()}" for i, l in zip(r.ids, r.labels)]
        return list(r.ids)

    # ------------------------------------------------- proyecto / validación / IA
    def save_project(self) -> bytes:
        import gzip
        import pickle
        keep = (self.tables, self.figs, self.facts, self.frames)
        self.tables, self.figs, self.facts, self.frames = {}, {}, {}, []
        try:
            return gzip.compress(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL), 6)
        finally:
            self.tables, self.figs, self.facts, self.frames = keep

    @staticmethod
    def load_project(raw: bytes) -> "Session":
        import gzip
        import pickle
        s = pickle.loads(gzip.decompress(raw))
        F.configure(tiff=bool(s.cfg.get("export_tiff", True)), cluster_shape=s.cfg.get("cluster_shape", "hull"))
        return s

    def validation_sheet(self) -> bytes:
        from .validation import build_sheet
        return build_sheet(self, seed=self.cfg["seed"])

    def load_validation(self, files):
        from .validation import analyse, answer_key
        try:
            key = answer_key(self, seed=self.cfg["seed"])
        except Exception:
            key = None
        self.validation = analyse(files, key)
        v = self.validation
        # Nombres propuestos por los expertos: se aplican a las figuras si están en inglés
        sug = v.get("suggestions", {})
        per_model, spanish, applied = {}, [], False
        for name, model, r in (("lda", "LDA", self.lda), ("bert", "BERTopic", self.bert)):
            if r is None:
                continue
            m = sug.get(model, {})
            current = [l.split(":", 1)[1].strip() if ":" in l else l for l in r.labels]
            per_model[name] = [{"id": tid, **m[tid]} if tid in m else {"id": tid} for tid in r.ids]
            spanish += [tid for tid in r.ids if tid in m and m[tid]["spanish"]]
        if sug and not spanish:
            new = {}
            for name, r in (("lda", self.lda), ("bert", self.bert)):
                if r is not None:
                    cur = [l.split(":", 1)[1].strip() if ":" in l else l for l in r.labels]
                    new[name] = [s_.get("best", c) for s_, c in zip(per_model[name], cur)]
            self.set_labels(new)
            applied = True
        return {"raters": v["raters"], "alpha": {k: (None if np.isnan(x) else round(x, 3))
                                                   for k, x in v["alpha"].items()},
                "table": json.loads(v["table"].to_json(orient="split", force_ascii=False)),
                "suggestions": per_model, "spanish": spanish, "applied": applied}

    def ai_context(self, n_docs=4):
        """Datos que se envían a la IA: temas, documentos representativos y referencias del corpus."""
        df = self.df
        refs, ref_id = [], {}

        def add_ref(i):
            if i not in ref_id:
                ref_id[i] = f"R{len(refs) + 1}"
                r = df.iloc[i]
                refs.append({"id": ref_id[i], "row": i, "cite": CIT.short_ref(r), "title": str(r["TI"]),
                             "year": int(r["PY"]), "citations": int(r["TC"]) if pd.notna(r["TC"]) else None})
            return ref_id[i]
        tc = df["TC"].fillna(0).values
        topics = {}
        for name, res in (("lda", self.lda), ("bert", self.bert)):
            if res is None:
                continue
            assign = res.dominant if name == "lda" else res.topics
            items = []
            for k in range(res.k):
                idx = np.where(assign == k)[0]
                rep = idx[np.argsort(-res.theta[idx, k])][:n_docs] if name == "lda" else idx[:0]
                cited = idx[np.argsort(-tc[idx])][:n_docs]
                if name == "bert":
                    rep = cited
                docs = [{"ref": add_ref(int(i)), "title": str(df.iloc[int(i)]["TI"])} for i in
                        list(dict.fromkeys(list(rep) + list(cited)))[:n_docs + 2]]
                items.append({"id": res.ids[k], "label": res.labels[k].split(":", 1)[1].strip(),
                              "terms": res.top10[k], "prevalence_pct": round(100 * float(res.prevalence[k]), 1),
                              "docs": docs})
            topics[name] = items
        summary = {k: v for k, v in self.facts.get("info", {}).items()}
        X = self.facts
        if X.get("C") is not None:
            m = X["C"]["matches"]
            summary["topic_alignment"] = m[["LDA topic", "BERTopic topic", "Cosine", "Agreement"]].head(15).to_dict("records")
        for name, key in (("lda", "trends_lda"), ("bert", "trends_bert")):
            t = X.get(key)
            res = self.lda if name == "lda" else self.bert
            if t is not None and res is not None:
                summary[f"{name}_trends"] = {res.labels[i]: t["Trend"][i] for i in range(res.k)}
        if "T11_thematic_map" in self.tables:
            summary["thematic_map"] = self.tables["T11_thematic_map"][["Label", "Quadrant"]].to_dict("records")
        if "T4_countries" in self.tables:
            summary["top_countries"] = self.tables["T4_countries"]["Country"].head(8).tolist()
        self.ai_refs = refs
        return json.loads(json.dumps({"topics": topics, "summary": summary, "references": refs},
                                     default=str, ensure_ascii=False))

    def set_ai_text(self, texts: dict):
        self.ai.update({k: v for k, v in texts.items() if v})

    # ----------------------------------------------------------- 5. salidas
    def render(self, out_dir="/out", final=False):
        """final=False: vista rápida (PNG + PDF). final=True: además TIFF 600 dpi y figuras compuestas."""
        F.configure(tiff=bool(final and self.cfg.get("export_tiff", True)),
                    cluster_shape=self.cfg.get("cluster_shape", "hull"))
        self.final = final
        out = Path(out_dir)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        for p in list((out / "figures").glob("*")):
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
            FIG["collab_map"] = F.collaboration_map(fp("Fig05b_country_collaboration_map.png"), dict(part),
                                                    list(col.itertuples(index=False, name=None)),
                                                    "Country collaboration map")
            comms = F.network(fp("Fig06_country_collaboration.png"), part.most_common(40), edges,
                              "Country collaboration network", top_nodes=40, normalize="association",
                              size_label="Documents")
            if comms:
                FIG["collab"] = fp("Fig06_country_collaboration.png")
                X["collab_clusters"] = comms

        kw_field = None
        if self.on("keywords"):
            kw_field = ("DE" if self.availability["DE"] > 0.3 else
                        "ID" if self.availability["ID"] > 0.3 else None)
        if kw_field:
            T["T5_keywords"] = biblio.keyword_freq(df, kw_field, 30)
            nodes, edges = biblio.keyword_cooccurrence(df, kw_field, 60)
            comms = F.network(fp("Fig07_keyword_cooccurrence.png"), nodes, edges,
                              "Keyword co-occurrence network", top_nodes=50, min_weight=2)
            if comms:
                FIG["kw"] = fp("Fig07_keyword_cooccurrence.png")
                X["kw_clusters"] = [cm[:6] for cm in comms]
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
                hj = HJBiplot(ml.rename(columns=dict(zip(lda.labels, self._bip_names(lda)))))
                hj.col_clusters, hj.k_cols, hj.sil_cols = F.biplot_clusters(hj)[:3]
                FIG["hj_lda"] = F.biplot(fp("Fig11_HJBiplot_LDA_periods.png"), hj,
                                         "HJ-Biplot: LDA topics and periods", col_clusters=hj.col_clusters)
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
                hj = HJBiplot(mb.rename(columns=dict(zip(bert.labels, self._bip_names(bert)))))
                hj.col_clusters, hj.k_cols, hj.sil_cols = F.biplot_clusters(hj)[:3]
                FIG["hj_bert"] = F.biplot(fp("Fig15_HJBiplot_BERTopic_periods.png"), hj,
                                          "HJ-Biplot: BERTopic topics and periods",
                                          col_clusters=hj.col_clusters)
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
                hjc = HJBiplot(pd.DataFrame(rows, index=topc, columns=self._bip_names(lda)))
                (hjc.col_clusters, hjc.k_cols, hjc.sil_cols,
                 hjc.row_clusters, hjc.k_rows, hjc.sil_rows) = F.biplot_clusters(hjc, rows=True)
                FIG["hj_country"] = F.biplot(fp("Fig19_HJBiplot_countries_LDA_topics.png"), hjc,
                                             "HJ-Biplot: countries and LDA topics", row_kind="country",
                                             col_clusters=hjc.col_clusters, row_clusters=hjc.row_clusters)
                T["hj_country_inertia"] = hjc.summary()
                X["hj_country"] = hjc

        if self.on("strategic") and lda is not None and trends_l is not None:
            sm = compare.strategic_map(lda.theta, df, trends_l, lda.labels)
            if sm is not None:
                T["lda_strategic_map"] = sm.round(4)
                FIG["strategic"] = F.strategic(fp("Fig20_strategic_map_LDA.png"), sm,
                                               "Strategic map of LDA topics (growth vs impact)")

        # ---- inferencia estadística
        if C is not None and self.on("inference"):
            inf_t, inf = ST.inference_table(self.corpus, lda, bert, B=int(c["n_boot"]), seed=c["seed"])
            T["T8b_inference"] = inf_t
            X["inference"] = inf
            X["tri_ci"] = ST.tri_ci(C["matches"]) if self.on("stability") else None

        # ---- citas
        if self.on("citations"):
            mc = CIT.most_cited(df, 10)
            if mc is not None:
                T["T10_most_cited"] = mc
                X["most_cited"] = mc
                if lda is not None:
                    cb = CIT.cited_by_topic(df, lda.dominant, lda.ids, lda.labels, 2)
                    T["cited_by_topic_LDA"] = cb.drop(columns=["_row"])
                    X["cited_lda"] = cb
                if bert is not None:
                    cb = CIT.cited_by_topic(df, bert.topics, bert.ids, bert.labels, 2)
                    T["cited_by_topic_BERTopic"] = cb.drop(columns=["_row"])
            if self.availability.get("CR", 0) > 0.3:
                nodes, edges, ctab = CIT.cocitation(df, 40)
                T["T13_cocited_references"] = ctab
                comms = F.network(fp("Fig21_cocitation_network.png"), nodes, edges, "Co-citation network",
                                  top_nodes=40, min_weight=2)
                if comms:
                    FIG["cocit"] = fp("Fig21_cocitation_network.png")
                    X["cocit_clusters"] = comms

        # ---- mapa temático y evolución
        if self.on("thematic") and kw_field:
            kwl = df[kw_field].map(biblio.split_list).tolist()
            tm = EVO.callon_map(kwl)
            if len(tm) >= 3:
                T["T11_thematic_map"] = tm.round(3)
                FIG["callon"] = EVO.plot_callon(fp("Fig22_thematic_map.png"), tm,
                                                "Thematic map (Callon centrality and density)")
            evp, evl = make_periods(df["PY"], max(int(X["period_width"]), (int(df["PY"].max()) - int(df["PY"].min()) + 1) // 4))
            layers, links = EVO.evolution(df, kw_field, evp, evl)
            if sum(len(l) for l in layers) >= 4 and links:
                FIG["sankey"] = EVO.plot_sankey(fp("Fig23_thematic_evolution.png"), layers, links,
                                                "Thematic evolution of keyword clusters")
                T["thematic_evolution_links"] = pd.DataFrame(
                    [{"From period": layers[l["t"]][l["i"]]["period"], "From theme": layers[l["t"]][l["i"]]["top"],
                      "To period": layers[l["t"] + 1][l["j"]]["period"], "To theme": layers[l["t"] + 1][l["j"]]["top"],
                      "Inclusion index": round(l["inclusion"], 3), "Shared keywords": l["shared"]} for l in links])
                X["evolution"] = (layers, links)

        # ---- validación por expertos
        if self.validation is not None:
            T["T12_expert_validation"] = self.validation["table"]

        # ---- figuras compuestas (paneles a, b, c...)
        X["composites"] = {}
        if c["composite"] and final:
            groups = [("comp_biblio", ["annual", "sources", "authors", "countries"], False),
                      ("comp_collab", ["collab_map", "collab"], True),
                      ("comp_keywords", ["kw", "callon"], False),
                      ("comp_time", ["lda_time", "bert_time"], False),
                      ("comp_hj", ["hj_lda", "hj_bert"], False),
                      ("comp_compare", ["sim", "cont"], False)]
            for name, keys, full in groups:
                keys = [k for k in keys if k in FIG]
                if len(keys) >= 2:
                    pth = F.composite(fp(f"Composite_{name[5:]}.png"), [FIG[k] for k in keys], ncols=2,
                                      full_width_first=full)
                    if pth:
                        X["composites"][name] = keys
                        FIG[name] = pth

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

    def script_package(self, meta=None) -> bytes:
        """Código y configuración exacta del análisis (material suplementario)."""
        from .script_export import build_package
        return build_package(self, meta)

    def zip_outputs(self, out_dir="/out", meta=None):
        buf = _io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in Path(out_dir).rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(out_dir))
            # el script del análisis también va dentro del .zip de resultados
            with zipfile.ZipFile(_io.BytesIO(self.script_package(meta))) as sz:
                for n in sz.namelist():
                    z.writestr(n, sz.read(n))
        return buf.getvalue()
