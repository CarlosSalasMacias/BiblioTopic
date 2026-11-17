"""Lectura de exportaciones de Web of Science y Scopus, normalización y fusión."""
from __future__ import annotations

import csv
import html
import io
import re
import unicodedata

import pandas as pd

# Esquema común (etiquetas estilo WoS / bibliometrix)
FIELDS = ["AU", "AF", "TI", "SO", "PY", "AB", "DE", "ID", "C1", "RP", "TC",
          "DI", "DT", "LA", "CR", "WC", "UT", "DB"]

SCOPUS_MAP = {
    "Authors": "AU", "Author full names": "AF", "Title": "TI", "Source title": "SO",
    "Year": "PY", "Abstract": "AB", "Author Keywords": "DE", "Index Keywords": "ID",
    "Authors with affiliations": "C1", "Correspondence Address": "RP",
    "Cited by": "TC", "DOI": "DI", "Document Type": "DT",
    "Language of Original Document": "LA", "References": "CR", "EID": "UT",
}

csv.field_size_limit(10**9)


# ---------------------------------------------------------------- lectura
def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def detect_format(text: str) -> str:
    head = text.lstrip("﻿")[:3000]
    first = head.split("\n", 1)[0]
    if first.startswith("PT\t") or "\tAU\t" in first and "\tTI\t" in first:
        return "wos_tab"
    if head.startswith("FN ") or re.search(r"^PT [JBSP]\s*$", head, re.M):
        return "wos_plain"
    if "Source title" in first or ("Authors" in first and "Title" in first and "," in first):
        return "scopus_csv"
    raise ValueError("Formato no reconocido. Use WoS (Tab-delimited o Plain text) o Scopus (CSV).")


def _read_wos_tab(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text.lstrip("﻿")), sep="\t", quoting=csv.QUOTE_NONE,
                     dtype=str, keep_default_na=False, index_col=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def _read_wos_plain(text: str) -> pd.DataFrame:
    multi_newline = {"AU", "AF", "C1", "CR", "C3", "RI", "OI", "BA", "BF"}
    recs, rec, tag = [], {}, None
    for line in text.lstrip("﻿").splitlines():
        if line.startswith("ER"):
            if rec:
                recs.append(rec)
            rec, tag = {}, None
            continue
        if not line.strip() or line.startswith(("FN ", "VR ", "EF")):
            continue
        if line[:2].strip() and line[2:3] == " ":
            tag = line[:2]
            val = line[3:].strip()
            rec[tag] = val if tag not in rec else rec[tag] + "; " + val
        elif tag:
            sep = "; " if tag in multi_newline else " "
            rec[tag] = rec.get(tag, "") + sep + line.strip()
    return pd.DataFrame(recs).fillna("")


def _read_scopus_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text.lstrip("﻿")), dtype=str, keep_default_na=False)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    out = pd.DataFrame({v: df[k] if k in df.columns else "" for k, v in SCOPUS_MAP.items()})
    if not (out["C1"] != "").any() and "Affiliations" in df.columns:
        out["C1"] = df["Affiliations"]
    out["AB"] = out["AB"].replace({"[No abstract available]": ""})
    return out


def read_file(name: str, raw: bytes) -> tuple[pd.DataFrame, str]:
    text = _decode(raw)
    fmt = detect_format(text)
    if fmt == "wos_tab":
        df = _read_wos_tab(text)
    elif fmt == "wos_plain":
        df = _read_wos_plain(text)
    else:
        df = _read_scopus_csv(text)
    df = normalize(df, "Scopus" if fmt == "scopus_csv" else "WoS")
    df["FILE"] = name
    return df, fmt


# ----------------------------------------------------------- normalización
_COPYRIGHT = re.compile(
    r"(©|\(c\)\s*\d{4}|copyright\s*(©|\(c\))?\s*(\d{4}|the\s+author)|\d{4}\s+(the\s+)?authors?\.?\s+published"
    r"|published by elsevier|all rights reserved|this is an open access article distributed under"
    r"|this article is protected by copyright).*$", re.I | re.S)
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9:_-]*(\s[^<>]*)?/?>")


def strip_markup(s: str) -> str:
    """Quita etiquetas HTML/XML (<italic>, <sup>, <inf>, <sub>, <scp>...) y decodifica entidades (&amp;, &lt;...)."""
    s = str(s or "")
    if "<" in s or "&" in s:
        s = html.unescape(_TAG.sub("", html.unescape(s)))
    return re.sub(r"[ \t]+", " ", s).strip()


def clean_abstract(ab: str) -> str:
    ab = strip_markup(ab)
    if ab.lower().startswith("[no abstract"):
        return ""
    return _COPYRIGHT.sub("", ab).strip()


# ------------------------------------------------- tipos de documento e idiomas
DOC_TYPES = ["Article", "Review", "Conference paper", "Book chapter", "Book", "Other"]


def canonical_doctype(raw: str) -> str:
    """Vocabulario común para WoS y Scopus (sin distinguir mayúsculas; acepta etiquetas múltiples de WoS
    como «Article; Proceedings Paper» o «Article; Early Access»). Vacío = no disponible."""
    toks = [t.strip().lower() for t in re.split(r"[;,]", str(raw or "")) if t.strip()]
    if not toks:
        return ""
    if any(t in ("proceedings paper", "conference paper") for t in toks):
        return "Conference paper"
    if any(t in ("review", "systematic review") for t in toks):
        return "Review"
    if any(t in ("book chapter", "chapter") for t in toks):
        return "Book chapter"
    if "book" in toks:
        return "Book"
    if "article" in toks or toks == ["early access"]:
        return "Article"
    return "Other"  # editorial material, letter, note, erratum, correction, meeting abstract, data paper...


def split_languages(raw: str) -> list[str]:
    """«English; Spanish» -> ['English', 'Spanish'] (Scopus da varios idiomas cuando hay resumen traducido)."""
    return [t.strip().title() for t in re.split(r"[;,]", str(raw or "")) if t.strip()]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_author(a: str) -> str:
    """'Dai, XY' / 'Dai X.Y.' / 'Dai, X.-Y.' -> 'DAI XY'."""
    a = _strip_accents(re.sub(r"\(\d+\)", "", a)).strip().strip(".")
    if not a:
        return ""
    if "," in a:
        sur, ini = a.split(",", 1)
    else:
        parts = a.rsplit(" ", 1)
        sur, ini = (parts[0], parts[1]) if len(parts) == 2 else (a, "")
    ini = re.sub(r"[^A-Za-z]", "", ini)
    if len(ini) > 3 and ini[1:].islower():  # nombre completo -> inicial
        ini = ini[0]
    return f"{sur.strip().upper()} {ini.upper()}".strip()


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _strip_accents(str(t)).lower())


def normalize(df: pd.DataFrame, db: str) -> pd.DataFrame:
    out = pd.DataFrame({f: df[f] if f in df.columns else "" for f in FIELDS})
    out = out.fillna("").astype(str)
    out["DB"] = db
    if db == "WoS" and "EA" in df.columns:  # early access sin PY
        ea_year = df["EA"].astype(str).str.extract(r"(\d{4})")[0].fillna("")
        out.loc[out["PY"].str.strip() == "", "PY"] = ea_year
    out["PY"] = pd.to_numeric(out["PY"].str.extract(r"(\d{4})")[0], errors="coerce").astype("Int64")
    out["TC"] = pd.to_numeric(out["TC"], errors="coerce")
    out["AB"] = out["AB"].map(clean_abstract)
    out["DI"] = out["DI"].str.strip().str.lower().str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
    out["DT_RAW"] = out["DT"].str.strip()
    out["DT"] = out["DT"].map(canonical_doctype)
    out["LA"] = out["LA"].map(lambda s_: "; ".join(split_languages(s_)))
    for f_ in ("TI", "SO"):
        out[f_] = out[f_].map(strip_markup)
    sep = ";"
    out["AU"] = out["AU"].map(lambda s: "; ".join(x for x in (norm_author(a) for a in s.split(sep)) if x))
    for f in ("DE", "ID"):
        out[f] = out[f].map(lambda s: "; ".join(k for k in (strip_markup(x).lower() for x in s.split(";")) if k))
    out["TI"] = out["TI"].str.strip()
    return out


# ----------------------------------------------------------------- fusión
def _completeness(row) -> int:
    return sum(bool(str(row[f]).strip()) and str(row[f]) not in ("nan", "<NA>") for f in FIELDS)


def _filled(v) -> bool:
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    return str(v).strip() not in ("", "nan", "<NA>")


def _doi_key(d: str) -> str:
    """DOI comparable: sin ceros a la izquierda en los números (v014 = v14, n01 = n1)."""
    return re.sub(r"(?<!\d)0+(\d)", r"\1", str(d).strip().lower())


def _doi_compatible(a: set, b: set) -> bool:
    """Dos conjuntos de DOI son del mismo documento si algún par coincide o uno es prefijo truncado del otro."""
    return any(x == y or x.startswith(y) or y.startswith(x) for x in a for y in b)


def _dedup_groups(raw: pd.DataFrame) -> list[int]:
    """Grupos de duplicados por componentes conexos (unión-búsqueda): dos registros son el mismo documento si
    comparten DOI, o si tienen el mismo título normalizado (> 15 caracteres) y años iguales o contiguos (las
    bases pueden diferir en un año por la publicación anticipada) o algún año ausente. La unión es transitiva,
    de modo que los grupos no dependen del orden de los archivos."""
    n = len(raw)
    parent = list(range(n))
    dois = [{_doi_key(d)} if d else set() for d in raw["DI"]]  # DOI(s) de cada grupo

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b, check_doi=False):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # por título solo se unen grupos cuyos DOI son compatibles: con DOI distintos son obras distintas
        # (volúmenes de una serie, versión de congreso y de revista...), salvo variantes del mismo DOI
        if check_doi and dois[ra] and dois[rb] and not _doi_compatible(dois[ra], dois[rb]):
            return
        lo, hi = min(ra, rb), max(ra, rb)
        parent[hi] = lo
        dois[lo] |= dois[hi]

    first_doi = {}
    for i, doi in enumerate(raw["DI"]):
        if doi:
            k = _doi_key(doi)
            if k in first_doi:
                union(first_doi[k], i)
            else:
                first_doi[k] = i
    by_title = {}
    for i, t in enumerate(raw["_tnorm"]):
        if len(t) > 15:
            by_title.setdefault(t, []).append(i)
    years = raw["PY"].tolist()
    for idx in by_title.values():
        if len(idx) < 2:
            continue
        known = sorted((i for i in idx if pd.notna(years[i])), key=lambda i: int(years[i]))
        unknown = [i for i in idx if pd.isna(years[i])]
        for a, b in zip(known, known[1:]):
            if int(years[b]) - int(years[a]) <= 1:
                union(a, b, check_doi=True)
        for u in unknown:  # sin año: se une al primero con el mismo título
            union(u, known[0] if known else idx[0], check_doi=True)
    return [find(i) for i in range(n)]


FILL_FIELDS = [f for f in FIELDS if f not in ("DB", "TC")]


def merge(frames: list[pd.DataFrame], return_links: bool = False):
    """Une los lotes y elimina duplicados conservando la información de todos los registros vinculados.

    - Agrupación: DOI o título normalizado + año (±1), con unión transitiva (_dedup_groups).
    - Registro base: el más completo (empates: más citas, base y UT, para que no dependa del orden).
    - Campos complementarios: cada campo vacío del registro base se completa con el primer registro del grupo
      que lo tenga; la procedencia se anota en FIELDS_FROM_OTHER (p. ej., «LA←WoS»).
    - Citas: el máximo del grupo. Conflictos (DOI, año o idioma distintos) en CONFLICTS.
    Con return_links=True devuelve además la tabla de registros originales vinculados a cada documento."""
    raw = pd.concat(frames, ignore_index=True)
    if "FILE" not in raw:
        raw["FILE"] = ""
    stats = {"records_by_db": raw["DB"].value_counts().to_dict(), "records_total": len(raw)}
    raw["_tnorm"] = raw["TI"].map(norm_title)
    raw["_score"] = raw.apply(_completeness, axis=1)
    raw["_g"] = _dedup_groups(raw)
    raw["_tc"] = raw["TC"].fillna(-1)
    raw = raw.sort_values(["_g", "_score", "_tc", "DB", "UT"], ascending=[True, False, False, True, True],
                          kind="mergesort")
    raw["_rank"] = raw.groupby("_g").cumcount()
    merged = raw[raw["_rank"] == 0].set_index("_g").copy()
    size = raw.groupby("_g").size()
    merged["DB_SOURCES"] = raw.groupby("_g")["DB"].agg(lambda s_: "+".join(sorted(set(s_), reverse=True)))
    merged["TC"] = raw.groupby("_g")["TC"].max()  # por defecto el máximo entre bases (se documenta)
    for db_ in ("WoS", "Scopus"):
        merged[f"TC_{db_}"] = raw[raw["DB"] == db_].groupby("_g")["TC"].max()
    merged["N_RECORDS"] = size
    merged["FIELDS_FROM_OTHER"] = ""
    merged["CONFLICTS"] = ""
    multi = raw[raw["_g"].isin(size[size > 1].index)]
    n_filled = 0
    for g, grp in multi.groupby("_g", sort=False):
        rows = grp.to_dict("records")
        base, others = rows[0], rows[1:]
        prov = []
        for f in FILL_FIELDS:
            if not _filled(base[f]):
                for r in others:
                    if _filled(r[f]):
                        merged.at[g, f] = r[f]
                        prov.append(f"{f}\u2190{r['DB']}")
                        break
        n_filled += len(prov)
        conf = []
        dois = {r["DI"] for r in rows if _filled(r["DI"])}
        if len(dois) > 1:
            conf.append("DOI: " + " | ".join(sorted(dois)))
        yrs = {int(r["PY"]) for r in rows if _filled(r["PY"])}
        if len(yrs) > 1:
            conf.append("Year: " + " | ".join(map(str, sorted(yrs))))
        las = {str(r["LA"]).strip() for r in rows if _filled(r["LA"])}
        if len(las) > 1:
            conf.append("Language: " + " | ".join(sorted(las)))
        merged.at[g, "FIELDS_FROM_OTHER"] = "; ".join(prov)
        merged.at[g, "CONFLICTS"] = "; ".join(conf)
    stats["duplicates_removed"] = len(raw) - len(merged)
    stats["unique"] = len(merged)
    stats["overlap"] = merged["DB_SOURCES"].value_counts().to_dict()
    stats["fields_completed_from_duplicates"] = int(n_filled)
    stats["groups_with_conflicts"] = int((merged["CONFLICTS"] != "").sum())
    merged = merged.reset_index(drop=False).rename(columns={"_g": "_group"})
    doc_id = {g: i for i, g in enumerate(merged["_group"])}
    links = None
    if return_links:
        links = raw.assign(DOC=raw["_g"].map(doc_id), KEPT_AS_BASE=raw["_rank"] == 0)[
            ["DOC", "KEPT_AS_BASE", "DB", "FILE", "UT", "DI", "PY", "TI", "LA", "DT", "TC"]]
        links = links[links["DOC"].map(merged["N_RECORDS"]) > 1].reset_index(drop=True)
    merged = merged.drop(columns=[c for c in merged.columns if c.startswith("_")]).reset_index(drop=True)
    return (merged, stats, links) if return_links else (merged, stats)


def field_availability(df: pd.DataFrame) -> dict:
    av = {}
    for f in ["AB", "C1", "RP", "DE", "ID", "TC", "DT", "LA", "CR", "DI"]:
        s = df[f]
        av[f] = float((s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str) != "<NA>")
                       & (s.astype(str) != "nan")).mean())
    return av


UNKNOWN = ""  # valor de un campo sin dato (p. ej., exportaciones de Scopus sin columna de idioma)


def _known(s: pd.Series) -> pd.Series:
    t = s.fillna("").astype(str).str.strip()
    return (t != "") & (t != "nan") & (t != "<NA>")


def status_by_db(df: pd.DataFrame, field: str, eligible=None) -> pd.DataFrame:
    """Recuento por base de origen: valor elegible, no elegible y sin dato (desconocido)."""
    src = df["DB_SOURCES"] if "DB_SOURCES" in df else df["DB"]
    known = _known(df[field])
    elig = _eligible(df[field], field, eligible) if eligible else known
    out = pd.DataFrame({"Source": src, "Eligible": known & elig, "Not eligible": known & ~elig,
                        "Unknown": ~known})
    return out.groupby("Source")[["Eligible", "Not eligible", "Unknown"]].sum().astype(int).reset_index()


def _eligible(values: pd.Series, field: str, allowed) -> pd.Series:
    allowed = {str(a).strip().lower() for a in allowed}
    if field == "LA":  # basta con que uno de los idiomas del registro esté admitido
        return values.map(lambda v: any(x.lower() in allowed for x in split_languages(v)))
    return values.fillna("").astype(str).str.strip().str.lower().isin(allowed)


MIN_ABSTRACT_WORDS = 20


def apply_filters(df: pd.DataFrame, year_min=None, year_max=None, doc_types=None, languages=None,
                  keep_unknown_type=True, keep_unknown_language=True, keep_unknown_year=True):
    """Criterios de elegibilidad en dos niveles; ningún registro se descarta por un campo ausente.

    1) Corpus bibliométrico: registros que cumplen año, tipo de documento e idioma. Para cada criterio hay tres
       estados (cumple, no cumple, dato no disponible); los registros sin dato se conservan salvo que el usuario
       los excluya explícitamente (keep_unknown_* = False).
    2) Corpus de modelado de temas: además requiere resumen (≥ 20 palabras) y año de publicación. Los registros
       sin resumen siguen en los análisis bibliométricos.

    Devuelve (corpus bibliométrico, corpus de modelado, pasos). Cada paso es (motivo, n, tipo) con tipo
    «excluded» (excluido del corpus), «kept» (sin dato, conservado) o «modelling» (fuera solo del modelado)."""
    steps = []
    d = df.copy()
    py_known = d["PY"].notna()
    if year_min is not None or year_max is not None:
        lo = year_min if year_min is not None else int(d["PY"].min())
        hi = year_max if year_max is not None else int(d["PY"].max())
        out = py_known & ~d["PY"].between(lo, hi)
        steps.append((f"Outside {lo}-{hi}", int(out.sum()), "excluded"))
        d = d[~out]
    n_unk = int(d["PY"].isna().sum())
    if n_unk:
        steps.append(("Year not available", n_unk, "kept" if keep_unknown_year else "excluded"))
        if not keep_unknown_year:
            d = d[d["PY"].notna()]
    for field, allowed, keep_unknown, name in (("DT", doc_types, keep_unknown_type, "document type"),
                                               ("LA", languages, keep_unknown_language, "language")):
        if not allowed:
            continue
        known = _known(d[field])
        bad = known & ~_eligible(d[field], field, allowed)
        steps.append((f"Non-eligible {name}", int(bad.sum()), "excluded"))
        if (~known).any():
            steps.append((f"{name.capitalize()} not available", int((~known).sum()),
                          "kept" if keep_unknown else "excluded"))
        drop = bad if keep_unknown else (bad | ~known)
        d = d[~drop]
    dfb = d.reset_index(drop=True)
    words = dfb["AB"].fillna("").str.split().str.len().fillna(0)
    no_ab = words == 0
    short = (words > 0) & (words < MIN_ABSTRACT_WORDS)
    no_py = dfb["PY"].isna() & ~no_ab & ~short
    steps.append(("No abstract", int(no_ab.sum()), "modelling"))
    steps.append((f"Abstract shorter than {MIN_ABSTRACT_WORDS} words", int(short.sum()), "modelling"))
    if no_py.any():
        steps.append(("Year not available", int(no_py.sum()), "modelling"))
    dfm = dfb[~(no_ab | short | no_py)].reset_index(drop=True)
    return dfb, dfm, steps


# ------------------------------------------------------------ auditoría de campos
AUDIT_FIELDS = ["AB", "DE", "ID", "LA", "DT", "PY", "DI", "TC", "C1", "RP", "CR"]
CRITICAL_FIELDS = ["LA", "DT", "AB", "RP/C1", "DE"]
CRITICAL_MIN = 0.90


def _availability(d: pd.DataFrame) -> dict:
    out = {f: float(_known(d[f].astype(str).where(d[f].notna(), "")).mean()) if f in d else 0.0
           for f in AUDIT_FIELDS}
    rp = _known(d["RP"].astype(str)) if "RP" in d else pd.Series(False, index=d.index)
    c1 = _known(d["C1"].astype(str)) if "C1" in d else pd.Series(False, index=d.index)
    out["RP/C1"] = float((rp | c1).mean()) if len(d) else 0.0
    return out


def field_audit(frames: list[pd.DataFrame]) -> dict:
    """Disponibilidad (%) de los campos por base y por archivo, y campos críticos por debajo del 90 %."""
    raw = pd.concat(frames, ignore_index=True)
    by_db = {db: _availability(g) for db, g in raw.groupby("DB")}
    by_file = {f"{fn} ({g['DB'].iloc[0]})": {**_availability(g), "_n": int(len(g))} for fn, g in raw.groupby("FILE")}
    low = [{"db": db, "field": f, "pct": round(100 * v[f], 1)} for db, v in by_db.items()
           for f in CRITICAL_FIELDS if v[f] < CRITICAL_MIN]
    pct = lambda d: {k: round(100 * v, 1) if not k.startswith("_") else v for k, v in d.items()}  # noqa: E731
    return {"by_db": {k: pct(v) for k, v in by_db.items()}, "by_file": {k: pct(v) for k, v in by_file.items()},
            "critical": CRITICAL_FIELDS, "critical_min": int(100 * CRITICAL_MIN), "critical_low": low}


# ------------------------------------------------------ términos de la ecuación de búsqueda
_FIELD_TAG = re.compile(r"\b(TITLE-ABS-KEY|TITLE-ABS|TITLE|ABS|KEY|AUTHKEY|INDEXTERMS|ALL|TS|TI|AB|AK|KP|ALL)\s*=?\s*",
                        re.I)
_LIMITS = re.compile(r"\b(AND\s+(NOT\s+)?)?(LIMIT-TO|EXCLUDE|PUBYEAR|DOCTYPE|LANGUAGE|SUBJAREA|SRCTYPE|PUBSTAGE|"
                     r"OA|AFFILCOUNTRY|EXACTKEYWORD|PY|DT|LA|WC|SU|CU|OG|SO|DOP)\b\s*(=|>|<|>=|<=)?\s*"
                     r"(\((?:[^()]|\([^()]*\))*\)|\d{4}|\"[^\"]*\")", re.I)
_OPS = {"and", "or", "not", "near", "same", "pre", "w", "and not"}


def extract_query_terms(*equations: str) -> list[str]:
    """Términos de las ecuaciones de búsqueda (Scopus/WoS) sin operadores, comodines, etiquetas de campo ni
    límites; se conservan las expresiones compuestas entre comillas. Se omiten los términos excluidos (NOT)."""
    terms = []
    for eq in equations:
        if not eq or not eq.strip():
            continue
        q = _LIMITS.sub(" ", eq)
        q = _FIELD_TAG.sub(" ", q)
        # quitar los grupos excluidos: NOT (…) o AND NOT (…), y NOT término
        while True:
            m = re.search(r"\bNOT\s*\(", q, re.I)
            if not m:
                break
            depth, i = 0, m.end() - 1
            while i < len(q):
                depth += q[i] == "("
                depth -= q[i] == ")"
                if depth == 0:
                    break
                i += 1
            q = q[:m.start()] + " " + q[i + 1:]
        q = re.sub(r"\bNOT\s+(\"[^\"]*\"|\S+)", " ", q, flags=re.I)
        for ph in re.findall(r"[\"{]([^\"}]+)[\"}]", q):
            t = re.sub(r"[*?$]", "", ph).strip().lower()
            if t:
                terms.append(re.sub(r"\s+", " ", t))
        q = re.sub(r"[\"{][^\"}]*[\"}]", " ", q)
        for tok in re.split(r"[\s()]+", q):
            t = re.sub(r"[*?$]", "", tok).strip().lower()
            t = re.sub(r"^(w|pre|near)/\d+$", "", t)
            if t and t not in _OPS and not re.fullmatch(r"[\d/=<>-]+", t) and len(t) > 1:
                terms.append(t)
    return list(dict.fromkeys(terms))
