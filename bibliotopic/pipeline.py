"""Orquestación por pasos (la página web llama a cada paso; JS calcula embeddings y UMAP)."""
from __future__ import annotations

import hashlib
import io as _io
import itertools
import math
import json
import re
import platform
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import biblio, citations as CIT, compare, evolution as EVO, figures as F, io as bio
from . import stats as ST
from .bert_model import BERTResult
from .hjbiplot import HJBiplot, make_periods, make_periods_mode
from . import palettes as PAL
from .lda_model import JobBroker, LDAResult, NeedJobs, corpus_payload, run_job_spec, select_k
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

__version__ = "2.0.0"
CONFIG_VERSION = 2

# Valores comunes a ambos perfiles
BASE = dict(
    analyses=list(ANALYSES), year_min=None, year_max=None, doc_types=["Article", "Review"],
    languages=["English"], query_terms=[], extra_stopwords=[], min_df=5, max_df=0.5,
    k=None, seed=42, bert_nr_topics="lda", min_cluster_size=None, min_samples=None, outlier_threshold=0.0,
    period_mode="width", period_width=5, n_periods=5, custom_periods="", top_countries=15,
    tau_strong=0.5, tau_partial=0.3, min_docs_year=10, synonyms="", citation_source="max",
    keep_unknown_language=True, keep_unknown_type=True, keep_unknown_year=True, exclude_incomplete_year=True,
    trend_test="mann_kendall", alpha=0.05, umap_metric="cosine", umap_neighbors=15, umap_components=5,
    umap_min_dist=0.0, embedding_model="Xenova/all-MiniLM-L6-v2",
    search_scopus="", search_wos="", search_date="", wos_indexes=[],
    export_tiff=True, cluster_shape="hull", composite=True, palette="auto", palette_variant=None,
)

# Perfiles. «publication» es el valor por defecto y está pensado para artículos en revistas Q1.
PROFILES = {
    "publication": dict(
        n_boot=1000, boot_method="percentile", n_perm=1000,
        n_stab_lda=10, n_stab_bert=10,
        k_min=5, k_max=40, k_step=5, k_fine_window=5, k_seeds=3,
        lda_method="batch", lda_max_iter=100, lda_tol=0.001, lda_doc_prior=None, lda_word_prior=None,
    ),
    "quick": dict(
        n_boot=200, boot_method="percentile", n_perm=200,
        n_stab_lda=3, n_stab_bert=3,
        k_min=5, k_max=30, k_step=5, k_fine_window=0, k_seeds=1,
        lda_method="batch", lda_max_iter=30, lda_tol=0.001, lda_doc_prior=None, lda_word_prior=None,
    ),
}

DEFAULTS = {**BASE, **PROFILES["publication"], "profile": "publication", "config_version": CONFIG_VERSION}

# Configuraciones de la versión 1 (sin config_version): valores con los que se ejecutaban, para reproducirlas
LEGACY_V1 = dict(
    profile="legacy-v1", n_boot=200, boot_method="percentile", n_perm=0, n_stab_lda=3, n_stab_bert=3,
    k_min=5, k_max=30, k_step=5, k_fine_window=0, k_seeds=1, lda_method="online", lda_max_iter=20,
    lda_select_max_iter=10, trend_test="ols", umap_metric="euclidean", period_mode="width",
    min_cluster_size_rule="v1",
)


def resolve_config(cfg: dict | None) -> dict:
    """Configuración efectiva: perfil + valores del usuario. Sin config_version (versión 1) se completan los
    parámetros nuevos con los valores que usaba la versión 1, para que un config.json antiguo se reproduzca."""
    cfg = dict(cfg or {})
    version = int(cfg.get("config_version") or 1)
    if version < 2:
        base = {**BASE, **PROFILES["publication"], **LEGACY_V1}
    else:
        base = {**BASE, **PROFILES.get(cfg.get("profile") or "publication", PROFILES["publication"])}
        base["profile"] = cfg.get("profile") or "publication"
    out = {**base, **{k: v for k, v in cfg.items() if v is not None or k in ("k", "year_min", "year_max")}}
    out["config_version"] = CONFIG_VERSION if version >= 2 else 1
    out["n_boot"] = int(min(max(int(out["n_boot"]), 0), 5000))
    return out


def estimate_runtime(n_docs: int, cfg: dict, nnz_per_doc: float = 70.0, browser: bool = True) -> dict:
    """Estimación del tiempo de ejecución (minutos). Calibrada con scikit-learn 1.6 en un equipo de referencia:
    un ajuste LDA batch de 100 iteraciones tarda ~45 s con 3 600 documentos (255 000 entradas no nulas) y en
    Pyodide (navegador, un solo hilo) es BROWSER_FACTOR veces más lento; UMAP en el navegador ~1.3 s por cada
    1 000 documentos. Con el criterio de convergencia relativo, los ajustes batch se detienen de media en ~75 % de
    max_iter. Con n_workers > 1 los ajustes de cada fase (selección gruesa, fina, modelo final + réplicas) se
    reparten en rondas; cada proceso rinde ~75 % de uno solo cuando todos los núcleos están ocupados."""
    c = resolve_config(cfg)
    nnz = n_docs * nnz_per_doc
    per_iter = 45.0 / 100 / 255_000  # s por iteración y entrada no nula (batch)
    online_factor = 0.22 * 100 / 20  # online, por iteración, respecto a batch
    batch = c["lda_method"] != "online"
    it_cost = per_iter * (1.0 if batch else online_factor)
    conv = 0.75 if batch and c.get("lda_tol") else 1.0
    fit = lambda k, iters: nnz * iters * conv * it_cost * (1 + max(k - 10, 0) / 120)  # noqa: E731
    workers = max(1, int(c.get("n_workers") or 1))
    rounds = lambda jobs: math.ceil(jobs / workers) * (1 / 0.75 if workers > 1 else 1.0)  # noqa: E731
    lda_s = 0.0
    on = set(c["analyses"])
    if "lda" in on:
        if not c.get("k"):
            ks = list(range(c["k_min"], c["k_max"] + 1, c["k_step"]))
            n_fine = 2 * c["k_fine_window"] if c["k_fine_window"] else 0
            kmid = (c["k_min"] + c["k_max"]) / 2
            f = fit(kmid, c.get("lda_select_max_iter") or c["lda_max_iter"])
            lda_s += (rounds(len(ks) * c["k_seeds"]) + rounds(n_fine * c["k_seeds"])) * f
        kf = c.get("k") or 15
        lda_s += rounds(1 + (c["n_stab_lda"] if "stability" in on else 0)) * fit(kf, c["lda_max_iter"])
    bert_s = 0.0
    if "bertopic" in on:
        runs = 2 + (c["n_stab_bert"] if "stability" in on else 0)
        bert_s = n_docs * (0.012 if "MiniLM" in c["embedding_model"] else 0.06)  # embeddings
        bert_s += runs * (1.3 * n_docs / 1000 + 3 * n_docs / 1000)  # UMAP (JS) + HDBSCAN (Pyodide)
    other = 60 + n_docs * 0.01 + (c["n_boot"] / 1000) * n_docs * 0.004
    factor = BROWSER_FACTOR if browser else 1.0
    lda_factor = (BROWSER_FACTOR_SERIAL if workers == 1 else BROWSER_FACTOR) if browser else 1.0
    total = lda_factor * lda_s + bert_s + factor * other
    return {"minutes": round(total / 60, 1), "lda_minutes": round(lda_factor * lda_s / 60, 1),
            "bertopic_minutes": round(bert_s / 60, 1), "profile": c["profile"], "browser": browser,
            "n_workers": workers}


def planned_seeds(c: dict, umap_runs: int) -> dict:
    """Semillas derivadas de la global (las mismas que usará el análisis)."""
    g = int(c["seed"])
    return {"global": g, "k_selection": [g + 1000 + j for j in range(int(c.get("k_seeds") or 1))],
            "lda_final": g, "lda_stability": list(range(g + 1, g + 1 + int(c.get("n_stab_lda") or 0))),
            "umap_5d": [g + r for r in range(umap_runs)], "umap_2d": g, "bootstrap": g, "permutation": g + 7,
            "validation_sheet": g}


# Medido con 9 250 documentos (K = 15, 30 iteraciones): ~95 s por ajuste en el worker principal (3.5 veces
# Python nativo) y ~70 s por ajuste con 3 workers auxiliares trabajando a la vez (2.0 × 1/0.75).
BROWSER_FACTOR = 2.0
BROWSER_FACTOR_SERIAL = 3.5


def _versions():
    import importlib.metadata as md
    v = {"python": platform.python_version()}
    for p in ["pandas", "numpy", "scikit-learn", "scipy", "networkx", "matplotlib"]:
        try:
            v[p] = md.version(p)
        except Exception:
            pass
    return v


_SMALL = {"and", "of", "the", "in", "for", "with", "on", "to", "a", "an", "by", "&", "from", "at", "via"}


def short_label(name, maxlen=28):
    """Nombre corto para las figuras: el nombre completo si cabe; si no, su primera parte (antes de «and», «:»,
    «,» o «(») recortada por palabras hasta maxlen caracteres."""
    s = " ".join(str(name or "").split())
    if len(s) <= maxlen:
        return s
    parts = [p.strip() for p in re.split(r"\s+(?:and|&)\s+|\s*[:;,(]\s*|\s+[-\u2013\u2014]\s+", s) if p.strip()]
    cand = parts[0] if parts and (len(parts[0].split()) >= 2 or len(parts) == 1) else s
    out = []
    for w in cand.split():
        if out and len(" ".join(out + [w])) > maxlen:
            break
        out.append(w)
    while len(out) > 1 and out[-1].lower().strip(",;:") in _SMALL:
        out.pop()
    return " ".join(out).rstrip(",;:-\u2013 ")


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
        self._kw_cache = None
        self.frames, fmts, errors = [], {}, []
        expanded = []
        self._zip_members = {}
        for name, raw in files:
            if name.lower().endswith(".zip"):
                with zipfile.ZipFile(_io.BytesIO(raw)) as z:
                    inner = [(n.split("/")[-1], z.read(n)) for n in z.namelist()
                             if n.lower().endswith((".txt", ".csv", ".tsv")) and not n.startswith("__MACOSX")]
                expanded += inner
                self._zip_members[name] = [n for n, _ in inner]
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
        self.merged, self.merge_stats, self.dup_links = bio.merge(self.frames, return_links=True)
        self.audit = bio.field_audit(self.frames)
        self.input_meta = [{"file": n, "md5": hashlib.md5(r).hexdigest(), "bytes": len(r)} for n, r in files]
        for m_ in self.input_meta:  # registros por archivo (para .zip, la suma de sus archivos internos)
            m_["records"] = int(sum(len(fr) for fr in self.frames
                                    if fr["FILE"].iloc[0] == m_["file"] or m_["file"].lower().endswith(".zip")
                                    and fr["FILE"].iloc[0] in getattr(self, "_zip_members", {}).get(m_["file"], [])))
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
            "audit": self.audit,
            "top_keywords": self.top_keywords(50),
            "palette_auto": self.palette_choice({}),
            "palettes": PAL.catalogue(),
            "input_files": self.input_meta,
            # registros sin dato de idioma o tipo, por base de origen (no son «otros idiomas»)
            "unknown": {f: {r["Source"]: int(r["Unknown"]) for _, r in bio.status_by_db(m, f).iterrows()
                            if r["Unknown"]} for f in ("LA", "DT")},
        }

    # ------------------------------------------------------------ paleta de colores
    def palette_choice(self, cfg) -> dict:
        """Paleta efectiva: automática (según las ecuaciones/términos de búsqueda y las palabras clave más
        frecuentes) o la elegida por el usuario. La variante depende del corpus si no se fija."""
        cfg = cfg or {}
        q = [cfg.get("search_scopus", ""), cfg.get("search_wos", ""), " ".join(cfg.get("query_terms") or [])]
        if getattr(self, "_kw_cache", None) is None:
            self._kw_cache = self.top_keywords(50)["keywords"]
        kws = [k for k, _ in self._kw_cache]
        wts = [w for _, w in self._kw_cache]
        key, variant, scores = PAL.choose(q, kws, wts)
        auto = (cfg.get("palette") or "auto") == "auto"
        if not auto:
            key = cfg["palette"] if cfg["palette"] in PAL.PALETTES else "classic"
        if cfg.get("palette_variant") not in (None, ""):
            variant = int(cfg["palette_variant"])
        p = PAL.Palette(key, variant)
        return {**p.info(), "auto": auto, "topic": PAL.PALETTES[p.key]["topic"], "scores": scores}

    def set_palette(self, key="auto", variant=None) -> dict:
        """Cambia la paleta después del análisis (solo se vuelven a dibujar las figuras)."""
        self.cfg["palette"], self.cfg["palette_variant"] = key or "auto", variant
        self.palette = self.palette_choice(self.cfg)
        return self.palette

    def top_keywords(self, n=50, synonyms_text=""):
        """Palabras clave más frecuentes tras la normalización automática (y los sinónimos, si se indican), para
        detectar variantes antes del análisis."""
        m = self.merged
        field = "DE" if (m["DE"] != "").mean() >= (m["ID"] != "").mean() else "ID"
        d = biblio.normalize_keywords(m[[field, "ID" if field == "DE" else "DE"]].rename(columns={}),
                                      biblio.parse_synonyms(synonyms_text))
        return {"field": field, "keywords": biblio.keyword_freq(d, field, n).values.tolist()}

    def quality_checks(self) -> list[dict]:
        """Controles de calidad antes de mostrar resultados: [{level: red|amber|green, check, detail}]."""
        out = []
        add = lambda lvl, chk, det: out.append({"level": lvl, "check": chk, "detail": det})  # noqa: E731
        ms = self.merge_stats
        dfb = self.dfb if getattr(self, "dfb", None) is not None else self.df
        for db, n in ms.get("records_by_db", {}).items():
            excl = int((dfb["DB_SOURCES"] == db).sum())
            if n and excl == 0 and len(ms.get("records_by_db", {})) > 1:
                add("red", f"0 registros exclusivos de {db} en el corpus final",
                    f"{db} aportó {n:,} registros, pero ninguno exclusivo llegó al corpus. Revise los filtros "
                    "(idioma, tipo de documento) y la auditoría de campos.")
        unique = ms.get("unique") or 1
        for st in self.excl_steps:
            name, n_, kind = st if len(st) == 3 else (st[0], st[1], "excluded")
            if kind in ("excluded", "modelling") and n_ > 0.2 * unique:
                add("red" if kind == "excluded" else "amber", f"Exclusión por encima del 20 %: {name}",
                    f"{n_:,} de {unique:,} registros únicos ({100 * n_ / unique:.1f} %).")
        for low in getattr(self, "audit", {}).get("critical_low", []):
            add("red", f"Campo crítico {low['field']} por debajo del 90 % en {low['db']}",
                f"Disponible en el {low['pct']} % de los registros. Revise la exportación.")
        if self.lda is not None:
            empty = int((np.bincount(self.lda.dominant, minlength=self.lda.k) == 0).sum())
            if empty:
                add("amber", f"{empty} tema(s) LDA sin documentos dominantes", "Considere un K menor.")
            low_st = [self.lda.ids[i] for i, v in enumerate(self.lda.stab) if np.isfinite(v) and v < 0.2]
            if low_st:
                add("amber", "Temas LDA con estabilidad < 0.2", ", ".join(low_st))
        if self.bert is not None and self.bert.outlier_rate > 0.30:
            add("amber", f"Outliers de BERTopic: {100 * self.bert.outlier_rate:.1f} % (> 30 %)",
                "Reduzca min_cluster_size o min_samples, o use un modelo de embeddings más grande.")
        if not self.cfg.get("query_terms"):
            add("amber", "Términos de la búsqueda vacíos",
                "No se eliminaron del texto; pegue las ecuaciones de búsqueda o escriba los términos.")
        if getattr(self, "audit", None) and all(v.get("CR", 0) == 0 for v in self.audit["by_db"].values()):
            add("amber", "Sin referencias citadas (CR)",
                "No es posible el análisis de co-citación ni de acoplamiento bibliográfico.")
        if not any(o["level"] in ("red", "amber") for o in out):
            add("green", "Sin alertas", "Todos los controles se superaron.")
        return out

    def estimate(self, cfg, n_docs=None):
        n = n_docs or int((self.merged["AB"].str.split().str.len().fillna(0) >= 20).sum())
        out = estimate_runtime(n, cfg, browser=True)
        out["local_minutes_1core"] = estimate_runtime(n, cfg, browser=False)["minutes"]
        out["n_docs"] = n
        return out

    # ------------------------------------------------- 2. filtros y texto
    def prepare(self, cfg):
        self.cfg = resolve_config(cfg)
        c = self.cfg
        if not c["query_terms"] and (c.get("search_scopus") or c.get("search_wos")):
            c["query_terms"] = bio.extract_query_terms(c.get("search_scopus", ""), c.get("search_wos", ""))
            c["query_terms_source"] = "search equations"
        elif c["query_terms"]:
            c.setdefault("query_terms_source", "user")
        self.palette = self.palette_choice(c)
        F.configure(tiff=bool(c["export_tiff"]), cluster_shape=c["cluster_shape"],
                    palette=self.palette["key"], variant=self.palette["variant"])
        dfb, dfm, self.excl_steps = bio.apply_filters(
            self.merged, c["year_min"], c["year_max"], c["doc_types"], c["languages"],
            keep_unknown_type=c["keep_unknown_type"], keep_unknown_language=c["keep_unknown_language"],
            keep_unknown_year=c["keep_unknown_year"])
        # idioma y tipo de documento: elegible / no elegible / sin dato, por base de origen
        self.field_status = {f: bio.status_by_db(self.merged, f, sel)
                             for f, sel in (("LA", c["languages"]), ("DT", c["doc_types"]))}
        self.has_countries = self.availability["C1"] > 0.3 and self.on("countries")
        syn = biblio.parse_synonyms(c.get("synonyms", ""))
        out = []
        for d in (dfb, dfm):
            src = c.get("citation_source", "max")  # por defecto, el máximo entre bases
            if src in ("WoS", "Scopus") and f"TC_{src}" in d:
                d = d.assign(TC=d[f"TC_{src}"])
            if self.availability["C1"] > 0.3:
                d = biblio.add_country_columns(d)
            d, kmap = biblio.normalize_keywords(d, syn, return_map=True)
            out.append(d)
        self.dfb, self.df = out          # corpus bibliométrico y corpus de modelado de temas
        self.keyword_map = kmap
        self.synonyms_used = syn
        self.corpus = Corpus(self.df, c["extra_stopwords"], c["min_df"], c["max_df"], c["query_terms"], syn)
        self.lda = self.bert = self.ksel = None
        return {"documents": len(self.df), "bibliometric_documents": len(self.dfb),
                "vocabulary": len(self.corpus.vocab), "phrases": len(self.corpus.phrases),
                "excluded": self.excl_steps}

    def texts(self):
        return self.corpus.raw

    # --------------------------------------------------------------- 3. LDA
    def lda_params(self):
        c = self.cfg
        return dict(method=c["lda_method"], max_iter=c["lda_max_iter"], tol=c["lda_tol"],
                    doc_prior=c["lda_doc_prior"], word_prior=c["lda_word_prior"])

    # LDA en paralelo en el navegador: el worker principal pide los ajustes pendientes y los reparte entre
    # workers auxiliares; el resultado es idéntico al de la ejecución en serie (cada ajuste con su semilla).
    def lda_parallel_begin(self):
        self._broker = JobBroker()
        return corpus_payload(self.corpus)

    def lda_parallel_step(self, progress=None):
        try:
            return {"done": True, **self.run_lda(progress, executor=self._broker)}
        except NeedJobs as e:
            return {"done": False, "jobs": e.specs}

    def lda_planned_jobs(self):
        c = self.cfg
        n_stab = c["n_stab_lda"] if self.on("stability") else 0
        if c["k"]:
            return 1 + n_stab
        ks = range(c["k_min"], c["k_max"] + 1, c["k_step"])
        fine = 2 * c["k_fine_window"] if c["k_fine_window"] else 0
        return (len(ks) + fine) * max(1, int(c["k_seeds"])) + 1 + n_stab

    def lda_put(self, spec_json, result):
        self._broker.put(spec_json, result)

    def lda_run_local(self, spec_json):
        """Respaldo: ejecuta un ajuste en el worker principal (si un worker auxiliar falla)."""
        self._broker.put(spec_json, run_job_spec(self.corpus, json.loads(spec_json)))

    def run_lda(self, progress=None, executor=None):
        """executor: opcional (p. ej., ProcessPoolExecutor en el script local) para ajustar en paralelo."""
        c = self.cfg
        say = progress or (lambda p, m: None)
        ks = list(range(c["k_min"], c["k_max"] + 1, c["k_step"]))
        params = self.lda_params()
        sel_params = {**params, "max_iter": c.get("lda_select_max_iter") or params["max_iter"]}
        n_stab = c["n_stab_lda"] if self.on("stability") else 0
        if c["k"]:
            self.ksel = None
            k = int(c["k"])
        else:
            self.ksel = select_k(self.corpus, ks, seed=c["seed"], n_seeds=c["k_seeds"],
                                 fine_window=c["k_fine_window"], params=sel_params, executor=executor,
                                 progress=lambda p, m: say(0.75 * p, m))
            k = int(self.ksel.loc[self.ksel["Score"].idxmax(), "K"])
        say(0.78, f"LDA final (K={k})")
        self.lda = LDAResult(self.corpus, k, seed=c["seed"], n_stab=n_stab, params=params, executor=executor,
                             progress=lambda p, m: say(0.8 + 0.2 * p, m))
        self.seeds = self.seed_registry()
        return {"k": k, "iterations": self.lda.n_iter}

    def bert_nr_topics(self):
        c = self.cfg
        if c["bert_nr_topics"] == "lda":
            return self.lda.k if self.lda is not None else None
        return c["bert_nr_topics"] or None

    def n_umap_runs(self):
        return 1 + (self.cfg["n_stab_bert"] if self.on("stability") else 0)

    def umap_settings(self):
        """Parámetros de UMAP que usa el navegador (y el script); se registran en la configuración."""
        c = self.cfg
        runs = self.n_umap_runs()
        return {"metric": c["umap_metric"], "n_neighbors": int(c["umap_neighbors"]),
                "n_components": int(c["umap_components"]), "min_dist": float(c["umap_min_dist"]),
                "seeds": [int(c["seed"]) + r for r in range(runs)], "map_2d": {"n_components": 2, "min_dist": 0.1,
                                                                                "seed": int(c["seed"])}}

    # ---------------------------------------------------------- 4. BERTopic
    def run_bertopic(self, reduced_list, xy, emb_name):
        c = self.cfg
        self.emb_name = emb_name
        mcs = c["min_cluster_size"]
        if not mcs and c.get("min_cluster_size_rule") == "v1":
            mcs = max(10, int(round(self.corpus.n_docs * 0.005)))
        self.bert = BERTResult(self.corpus, reduced_list, xy, emb_name, nr_topics=self.bert_nr_topics(),
                               min_cluster_size=mcs, min_samples=c.get("min_samples"),
                               outlier_threshold=c.get("outlier_threshold", 0.0),
                               umap_seeds=self.umap_settings()["seeds"])
        self.seeds = self.seed_registry()
        out = {"k": self.bert.k, "native": self.bert.native_topics,
               "outliers": round(100 * self.bert.outlier_rate, 1),
               "outliers_after": round(100 * self.bert.outlier_rate_after, 1)}
        if self.bert.outlier_rate > 0.30:
            out["warning"] = (f"El {out['outliers']} % de los documentos quedó sin grupo en HDBSCAN (> 30 %). "
                              "Considere reducir min_cluster_size o min_samples, o usar un modelo de embeddings "
                              "más grande; los outliers se reasignaron por similitud c-TF-IDF.")
        return out

    def seed_registry(self) -> dict:
        """Todas las semillas usadas (global y derivadas)."""
        c = self.cfg
        out = {"global": int(c["seed"])}
        if self.ksel is not None:
            out["k_selection"] = list(self.ksel.attrs.get("seeds", []))
        if self.lda is not None:
            out["lda_final"] = int(self.lda.seed)
            out["lda_stability"] = list(getattr(self.lda, "stab_seeds", []))
        if self.bert is not None or self.on("bertopic"):
            u = self.umap_settings()
            out["umap_5d"] = u["seeds"]
            out["umap_2d"] = u["map_2d"]["seed"]
        out["bootstrap"] = int(c["seed"])
        out["permutation"] = int(c["seed"]) + 7
        out["validation_sheet"] = int(c["seed"])
        return out

    def topic_labels(self):
        out = {}
        for name, r in (("lda", self.lda), ("bert", self.bert)):
            if r is not None:
                out[name] = [{"id": i, "terms": ", ".join(w[:8]), "label": l.split(":", 1)[1].strip(), "short": s}
                             for i, w, l, s in zip(r.ids, r.top10, r.labels, self._shorts(r))]
        return out

    def set_labels(self, labels):
        """labels: {"lda": [nombres], "lda_short": [nombres cortos para las figuras], "bert": ..., "bert_short": ...}.
        Si falta el nombre corto (o está vacío) se calcula automáticamente."""
        for name, r in (("lda", self.lda), ("bert", self.bert)):
            if r is not None and name in labels:
                full = [" ".join(str(l).split()) or "Unnamed topic" for l in labels[name]]
                r.labels = [f"{i}: {l}" for i, l in zip(r.ids, full)]
                short = list(labels.get(f"{name}_short") or [])
                short += [""] * (len(full) - len(short))
                r.short = [" ".join(str(s or "").split()) or short_label(l) for s, l in zip(short, full)]
                r.custom_labels = True

    @staticmethod
    def _shorts(r):
        names = [l.split(":", 1)[1].strip() if ":" in l else l for l in r.labels]
        sh = getattr(r, "short", None)
        return list(sh) if sh and len(sh) == len(names) else [short_label(n) for n in names]

    def _fig_labels(self, r):
        """Nombres de los temas en las figuras: código + nombre corto (los completos van en tablas y texto)."""
        if getattr(r, "custom_labels", False):
            return [f"{i}: {s}" for i, s in zip(r.ids, self._shorts(r))]
        return list(r.labels)

    def propose_titles(self, meta=None):
        from .report import propose_titles
        return propose_titles(self, meta or {})

    def _bip_names(self, r):
        """Nombre de los vectores en los biplots: código + nombre corto si los autores lo asignaron."""
        if getattr(r, "custom_labels", False):
            return [f"{i} {s}" for i, s in zip(r.ids, self._shorts(r))]
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
        pal = getattr(s, "palette", None) or {}
        F.configure(tiff=bool(s.cfg.get("export_tiff", True)), cluster_shape=s.cfg.get("cluster_shape", "hull"),
                    palette=pal.get("key", "classic"), variant=pal.get("variant", 0))
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
        # Los nombres de los expertos se aplican siempre; los que parecen en español se señalan para traducirlos
        if any("best" in x for rows in per_model.values() for x in rows):
            new = {}
            for name, r in (("lda", self.lda), ("bert", self.bert)):
                if r is not None:
                    cur = [l.split(":", 1)[1].strip() if ":" in l else l for l in r.labels]
                    new[name] = [s_.get("best", c) for s_, c in zip(per_model[name], cur)]
            self.set_labels(new)
            applied = True
        return {"raters": v["raters"], "alpha": {k: (None if np.isnan(x) else round(x, 3))
                                                   for k, x in v["alpha"].items()},
                "fleiss_kappa": {k: (None if np.isnan(x) else round(x, 3)) for k, x in v["fleiss_kappa"].items()},
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
        pal = getattr(self, "palette", None) or {}
        F.configure(tiff=bool(final and self.cfg.get("export_tiff", True)),
                    cluster_shape=self.cfg.get("cluster_shape", "hull"),
                    palette=pal.get("key", "classic"), variant=pal.get("variant", 0))
        self.final = final
        out = Path(out_dir)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        for p in list((out / "figures").glob("*")):
            p.unlink()
        fp = lambda n: str(out / "figures" / n)  # noqa: E731
        c, df, lda, bert = self.cfg, self.df, self.lda, self.bert
        # corpus bibliométrico (todos los registros elegibles, también sin resumen) y corpus de modelado (df)
        dfb = self.dfb if getattr(self, "dfb", None) is not None else df
        dfby = dfb[dfb["PY"].notna()]  # análisis que necesitan el año
        T, FIG, X = {}, {}, {}
        ms = self.merge_stats

        FIG["prisma"] = F.prisma(fp("Fig01_PRISMA_flow.png"), ms["records_by_db"], ms["records_total"],
                                 ms["duplicates_removed"], ms["unique"], self.excl_steps, len(df), len(dfb))
        info = biblio.main_information(dfb, self.has_countries)
        T["T1_main_information"] = pd.DataFrame(list(info.items()), columns=["Description", "Results"])
        ap = biblio.annual_production(dfby)
        T["annual_production"] = ap
        X.update(info=info, ap=ap)

        if self.on("biblio"):
            FIG["annual"] = F.annual_production(fp("Fig02_annual_production.png"), ap)
            src = biblio.top_sources(dfb, 15)
            T["T2_top_sources"] = src
            FIG["sources"] = F.barh(fp("Fig03_top_sources.png"), src["Source"], src["Articles"],
                                    "Most relevant sources", "Documents")
            brad = biblio.bradford(dfb)
            T["bradford"] = brad
            au = biblio.top_authors(dfb, 15)
            T["T3_top_authors"] = au
            FIG["authors"] = F.barh(fp("Fig04_top_authors.png"), au["Author"], au["Articles"],
                                    "Most productive authors", "Documents", color=F.GREEN)
            T["lotka"] = biblio.lotka(dfb)
            X["bradford_zone1"] = int((brad["Zone"] == "Zone 1").sum())
            X["n_sources"] = int(len(brad))

        if self.has_countries:
            cp = biblio.country_production(dfb, 20)
            T["T4_countries"] = cp
            FIG["countries"] = F.country_scp_mcp(fp("Fig05_country_production.png"), cp)
            col = biblio.country_collaboration(dfb)
            T["country_collaboration"] = col
            part = Counter(itertools.chain.from_iterable(dfb["COUNTRIES"]))
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
            T["T5_keywords"] = biblio.keyword_freq(dfb, kw_field, 30)
            nodes, edges = biblio.keyword_cooccurrence(dfb, kw_field, 60)
            comms = F.network(fp("Fig07_keyword_cooccurrence.png"), nodes, edges,
                              "Keyword co-occurrence network", top_nodes=50, min_weight=2)
            if comms:
                FIG["kw"] = fp("Fig07_keyword_cooccurrence.png")
                X["kw_clusters"] = [cm[:6] for cm in comms]
        X["kw_field"] = kw_field

        width = int(c["period_width"])
        mode = c.get("period_mode", "width")
        periods, plabels, pwarn = make_periods_mode(df["PY"], mode, width, c.get("n_periods", 5),
                                                    c.get("custom_periods", ""))
        span = int(df["PY"].max() - df["PY"].min()) + 1
        while mode == "width" and len(plabels) < 4 and width > 2 and span >= 8:  # ≥ 4 periodos para el HJ-Biplot
            width -= 1
            periods, plabels, pwarn = make_periods_mode(df["PY"], mode, width)
        X["period_width"] = width
        X["period_mode"] = mode
        X["periods"] = plabels
        X["period_warnings"] = pwarn
        T["periods"] = pd.DataFrame({"Period": plabels, "Documents": [int((periods == p_).sum()) for p_ in plabels]})
        C = None
        if lda is not None:
            if self.ksel is not None:
                FIG["ksel"] = F.k_selection(fp("Fig08_LDA_K_selection.png"), self.ksel, lda.k)
                T["lda_k_selection"] = self.ksel.round(4)
            T["T6_LDA_topics"] = lda.topic_table()
            FIG["lda_words"] = F.topic_words(fp("Fig09_LDA_topics.png"),
                                             [np.sort(r)[::-1] for r in lda.beta], lda.top10,
                                             self._fig_labels(lda), "LDA topics: top terms (word probability)",
                                             color=F.col("primary"))  # un color: los colores por tema son de BERTopic
            ml = pd.DataFrame(lda.theta, columns=lda.labels).groupby(periods.values).mean().reindex(plabels)
            T["lda_by_period"] = ml.round(4).reset_index(names="Period")
            X["ml"] = ml
            if self.on("temporal"):
                FIG["lda_time"] = F.topic_heatmap_time(fp("Fig10_LDA_topics_by_period.png"), ml,
                                                       self._fig_labels(lda), "LDA topic prevalence by period")
            if self.on("hjbiplot") and ml.dropna().shape[0] >= 3:  # periodos sin documentos se omiten
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
                                              self._fig_labels(bert), "BERTopic topics: top terms (c-TF-IDF)")
            if bert.xy is not None:
                FIG["bert_map"] = F.doc_map(fp("Fig13_BERTopic_document_map.png"), bert.xy,
                                            bert.topics_raw, bert.labels,
                                            "BERTopic document map (UMAP projection)")
            mb = pd.DataFrame(bert.theta, columns=bert.labels).groupby(periods.values).mean().reindex(plabels)
            T["bert_by_period"] = mb.round(4).reset_index(names="Period")
            if self.on("temporal"):
                FIG["bert_time"] = F.topic_heatmap_time(fp("Fig14_BERTopic_topics_by_period.png"), mb,
                                                        self._fig_labels(bert), "BERTopic topic prevalence by period",
                                                        cmap="seq2")
            if self.on("hjbiplot") and mb.dropna().shape[0] >= 3:
                hj = HJBiplot(mb.rename(columns=dict(zip(bert.labels, self._bip_names(bert)))))
                hj.col_clusters, hj.k_cols, hj.sil_cols = F.biplot_clusters(hj)[:3]
                FIG["hj_bert"] = F.biplot(fp("Fig15_HJBiplot_BERTopic_periods.png"), hj,
                                          "HJ-Biplot: BERTopic topics and periods",
                                          col_clusters=hj.col_clusters)
                T["hj_bert_inertia"] = hj.summary()
                X["hj_b"] = hj

        trends_l = trends_b = None
        # el año en curso suele estar incompleto (la exportación se hizo durante el año): se excluye
        import datetime as _dt
        this_year = _dt.date.today().year
        excl = [this_year] if c.get("exclude_incomplete_year", True) and int(df["PY"].max()) >= this_year else []
        X["trend_excluded_years"] = excl
        if lda is not None:
            trends_l, _ = compare.topic_trends(lda.theta, df["PY"], c["min_docs_year"], excl, c["alpha"], c["trend_test"])
            T["lda_trends"] = trends_l.assign(Topic=lda.labels)[["Topic"] + list(trends_l.columns)].round(5)
        if bert is not None:
            trends_b, _ = compare.topic_trends(bert.theta, df["PY"], c["min_docs_year"], excl, c["alpha"], c["trend_test"])
            tb = trends_b.assign(Topic=bert.labels)[["Topic"] + list(trends_b.columns)]
            # sensibilidad: tendencias solo con los documentos que el agrupamiento asignó (sin reasignar outliers)
            ok = bert.topics_raw >= 0
            if ok.any() and (~ok).any():
                th_raw = np.zeros((int(ok.sum()), bert.k))
                th_raw[np.arange(int(ok.sum())), bert.topics_raw[ok]] = 1.0
                tr_raw, _ = compare.topic_trends(th_raw, df["PY"].values[ok], c["min_docs_year"], excl, c["alpha"],
                                                c["trend_test"])
                tb["Slope (clustered only)"] = tr_raw["Slope"].values
                tb["Trend (clustered only)"] = tr_raw["Trend"].values
                X["trends_bert_clustered"] = tr_raw
            T["bert_trends"] = tb.round(5)
        X["trends_lda"], X["trends_bert"] = trends_l, trends_b
        tr0 = trends_l if trends_l is not None else trends_b
        if tr0 is not None:
            X["min_docs_year_used"] = tr0.attrs.get("min_docs_year", c["min_docs_year"])
            X["trend_years"] = tr0.attrs.get("n_years")
            X["trend_year_range"] = tr0.attrs.get("years")

        if lda is not None and bert is not None and self.on("compare"):
            C = compare.compare(lda, bert, df["PY"], c["min_docs_year"], c["tau_strong"],
                                c["tau_partial"], excl, titles=df["TI"].values, n_perm=int(c.get("n_perm") or 0),
                                seed=int(c["seed"]) + 7, trend_test=c["trend_test"], alpha=c["alpha"],
                                n_boot=int(c["n_boot"]))
            T["topic_best_matches"] = C["best_matches"]
            T["tau_sensitivity"] = C["threshold_sensitivity"]
            T["T8_model_metrics"] = C["metrics"]
            T["T9_topic_alignment"] = C["matches"].drop(columns=["_i", "_j"])
            T["doc_agreement"] = pd.DataFrame(list(C["agreement"].items()), columns=["Measure", "Value"])
            FIG["sim"] = F.similarity_heatmap(fp("Fig16_LDA_vs_BERTopic_similarity.png"), C["S"],
                                              C["matches"], "Topic similarity between LDA and BERTopic",
                                              null=C.get("null"))
            FIG["cont"] = F.contingency(fp("Fig17_document_assignment_agreement.png"), C["contingency"],
                                        "Document-level agreement between models")
        X["C"] = C
        if self.on("temporal") and (trends_l is not None or trends_b is not None):
            _tr = F.trends(fp("Fig18_hot_cold_topics.png"), trends_l, trends_b,
                                     self._fig_labels(lda) if lda else None,
                                     self._fig_labels(bert) if bert else None)
            if _tr:
                FIG["trends"] = _tr

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
            inf_t, inf = ST.inference_table(self.corpus, lda, bert, B=int(c["n_boot"]), seed=c["seed"],
                                            method=c.get("boot_method", "percentile"))
            T["T8b_inference"] = inf_t
            X["inference"] = inf
            X["tri_ci"] = ST.tri_ci(C["matches"]) if self.on("stability") else None

        # ---- citas
        if self.on("citations"):
            mc = CIT.most_cited(dfby, 10)
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
                nodes, edges, ctab = CIT.cocitation(dfb, 40)
                T["T13_cocited_references"] = ctab
                comms = F.network(fp("Fig21_cocitation_network.png"), nodes, edges, "Co-citation network",
                                  top_nodes=40, min_weight=2)
                if comms:
                    FIG["cocit"] = fp("Fig21_cocitation_network.png")
                    X["cocit_clusters"] = comms
                bn, be, btab = CIT.bibliographic_coupling(dfb, 40)
                if len(btab):
                    T["T14_bibliographic_coupling"] = btab
                    comms_b = F.network(fp("Fig21b_bibliographic_coupling.png"), bn, be,
                                        "Bibliographic coupling network", top_nodes=40, min_weight=2,
                                        normalize="association", size_label="Citations")
                    if comms_b:
                        FIG["coupling"] = fp("Fig21b_bibliographic_coupling.png")
                        X["coupling_clusters"] = comms_b
            X["cr_available"] = self.availability.get("CR", 0)

        # ---- mapa temático y evolución
        if self.on("thematic") and kw_field:
            kwl = dfb[kw_field].map(biblio.split_list).tolist()
            tm = EVO.callon_map(kwl)
            if len(tm) >= 3:
                T["T11_thematic_map"] = tm.round(3)
                FIG["callon"] = EVO.plot_callon(fp("Fig22_thematic_map.png"), tm,
                                                "Thematic map (Callon centrality and density)")
            evp, evl = make_periods(dfby["PY"], max(int(X["period_width"]), (int(dfby["PY"].max()) - int(dfby["PY"].min()) + 1) // 4))
            layers, links = EVO.evolution(dfby, kw_field, evp, evl)
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

        # trazabilidad de la fusión: registros originales vinculados a cada documento con duplicados
        links = getattr(self, "dup_links", None)
        if links is not None and len(links):
            T["merge_linked_records"] = links
        if "CONFLICTS" in self.merged:
            conf = self.merged[self.merged["CONFLICTS"] != ""]
            if len(conf):
                T["merge_conflicts"] = conf[["TI", "PY", "DI", "DB_SOURCES", "N_RECORDS", "CONFLICTS"]].reset_index(drop=True)
        self.tables, self.figs, self.facts = T, FIG, X
        with pd.ExcelWriter(out / "tables" / "all_tables.xlsx") as xw:
            for name, t in T.items():
                t.to_excel(xw, sheet_name=name[:31], index=False)
        docs = df[["UT", "DI", "TI", "SO", "PY", "DB_SOURCES"]
                  + [c_ for c_ in ("N_RECORDS", "FIELDS_FROM_OTHER", "CONFLICTS") if c_ in df]].copy()
        if lda is not None:
            docs["LDA_dominant"] = [lda.ids[i] for i in lda.dominant]
        if bert is not None:
            docs["BERTopic_original"] = [bert.ids[i] if i >= 0 else "Outlier" for i in bert.topics_raw]
            docs["BERTopic"] = [bert.ids[i] if i >= 0 else "Unassigned" for i in bert.topics]
            if getattr(bert, "assign_sim", None) is not None:
                docs["BERTopic_assignment"] = np.where(bert.reassigned, "Outlier reassigned (most similar c-TF-IDF)",
                                                       "HDBSCAN cluster")
                docs["BERTopic_similarity"] = np.round(bert.assign_sim, 4)
                docs["BERTopic_margin_to_2nd"] = np.round(bert.assign_margin, 4)
                docs["BERTopic_low_confidence"] = bert.low_confidence
        docs.to_csv(out / "tables" / "documents_topics.csv", index=False)
        rc = self.effective_config()
        rc["quality_checks"] = self.quality_checks()
        (out / "run_config.json").write_text(json.dumps(rc, indent=2, default=str, ensure_ascii=False),
                                             encoding="utf-8")
        return [{"key": k, "path": v} for k, v in FIG.items()]

    def table_json(self, name, n=50):
        t = self.tables.get(name)
        return None if t is None else json.loads(t.head(n).to_json(orient="split", force_ascii=False))

    def script_package(self, meta=None, cfg=None) -> bytes:
        """Código y configuración exacta del análisis (material suplementario). Con cfg (antes de ejecutar el
        análisis) se usa la configuración elegida en la interfaz."""
        from .script_export import build_package
        return build_package(self, meta, cfg)

    def effective_config(self, meta=None, cfg=None) -> dict:
        """Configuración efectiva para run_config.json y config.json: versión, perfil y todos los parámetros
        (los automáticos ya resueltos), archivos de entrada con MD5 y nº de registros, ecuaciones de búsqueda,
        disponibilidad de campos por base y todas las semillas."""
        meta = dict(meta or {})
        c = dict(resolve_config(cfg) if cfg else (getattr(self, "cfg", None) or resolve_config({"config_version": 2})))
        if self.lda is not None:  # valores automáticos resueltos (no null)
            c["lda_effective"] = {**self.lda.params, "k": self.lda.k, "iterations_final_model": self.lda.n_iter}
        if self.bert is not None:
            c["bertopic_effective"] = {"min_cluster_size": self.bert.min_cluster_size,
                                       "min_samples": self.bert.min_samples,
                                       "outlier_threshold": self.bert.outlier_threshold,
                                       "nr_topics": self.bert.k, "native_topics": self.bert.native_topics,
                                       "outliers_before_pct": round(100 * self.bert.outlier_rate, 1),
                                       "outliers_after_pct": round(100 * getattr(self.bert, "outlier_rate_after", 0), 1)}
        if getattr(self, "corpus", None) is not None and c.get("query_terms"):
            c["query_terms_effective"] = list(c["query_terms"])
        emb = (getattr(self, "emb_name", "") or "").lower()
        kind = "lsa" if "lsa" in emb else "transformer"
        labels = {}
        for name, r in (("lda", self.lda), ("bert", self.bert)):
            if r is not None and getattr(r, "custom_labels", False):
                labels[name] = [l.split(":", 1)[1].strip() if ":" in l else l for l in r.labels]
                labels[name + "_short"] = self._shorts(r)
        files = list(getattr(self, "input_files", []) or [])
        if not files and getattr(self, "merged", None) is not None and "FILE" in self.merged:
            files = sorted(self.merged["FILE"].dropna().unique().tolist())
        audit = getattr(self, "audit", None)
        return {
            "config_version": c.get("config_version", CONFIG_VERSION),
            "bibliotopic_version": __version__,
            "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "generator": "BiblioTopic (https://github.com/CarlosSalasMacias/BiblioTopic)",
            "profile": c.get("profile"),
            "input_files": files,
            "input_files_detail": getattr(self, "input_meta", None) or [{"file": f} for f in files],
            "search": {"scopus": c.get("search_scopus", ""), "wos": c.get("search_wos", ""),
                       "date": c.get("search_date") or meta.get("search_date", ""),
                       "wos_indexes": c.get("wos_indexes", []),
                       "query_terms": c.get("query_terms", []), "query_terms_source": c.get("query_terms_source", "")},
            "field_availability_by_database": audit["by_db"] if audit else None,
            "figure_palette": {k: v for k, v in (getattr(self, "palette", None) or {}).items() if k != "scores"},
            "parameters": c,
            "seeds": self.seed_registry() if not cfg and (self.lda is not None or self.bert is not None)
            else planned_seeds(c, self.n_umap_runs() if not cfg else 1 + (c["n_stab_bert"] if "stability" in c["analyses"] else 0)),
            "lda_k_selected": self.lda.k if self.lda is not None else None,
            "embedding": {"kind": kind, "model": c.get("embedding_model"), "used_in_browser": getattr(self, "emb_name", "")},
            "umap": self.umap_settings() if getattr(self, "cfg", None) else None,
            "merge": self.merge_stats,
            "topic_labels": labels,
            "ai_texts": {kk: v for kk, v in (getattr(self, "ai", None) or {}).items() if kk in ("discussion", "labels_used")},
            "document": {kk: meta.get(kk, "") for kk in ("topic_name", "title", "query", "search_date", "composite",
                                                          "ai_model", "ai_title")},
            "sections": meta.get("sections"),
            "versions_used": self.versions,
        }

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
