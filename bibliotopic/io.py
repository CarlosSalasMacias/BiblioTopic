"""Lectura de exportaciones de Web of Science y Scopus, normalización y fusión."""
from __future__ import annotations

import csv
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
    r"(©|\(c\)\s*\d{4}|copyright\s*(©|\(c\))?\s*\d{4}|\d{4}\s+(the\s+)?authors?\.?\s+published"
    r"|published by elsevier|all rights reserved).*$", re.I | re.S)


def clean_abstract(ab: str) -> str:
    ab = (ab or "").strip()
    if ab.lower().startswith("[no abstract"):
        return ""
    return _COPYRIGHT.sub("", ab).strip()


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
    out["DT"] = out["DT"].str.split(";").str[0].str.strip().str.title()
    out["LA"] = out["LA"].str.split(";").str[0].str.strip().str.title()
    sep = ";"
    out["AU"] = out["AU"].map(lambda s: "; ".join(x for x in (norm_author(a) for a in s.split(sep)) if x))
    for f in ("DE", "ID"):
        out[f] = out[f].map(lambda s: "; ".join(k.strip().lower() for k in s.split(";") if k.strip()))
    out["TI"] = out["TI"].str.strip()
    return out


# ----------------------------------------------------------------- fusión
def _completeness(row) -> int:
    return sum(bool(str(row[f]).strip()) and str(row[f]) not in ("nan", "<NA>") for f in FIELDS)


def merge(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Une los lotes, elimina duplicados (DOI y título+año) y registra el solapamiento."""
    raw = pd.concat(frames, ignore_index=True)
    stats = {"records_by_db": raw["DB"].value_counts().to_dict(), "records_total": len(raw)}
    raw["_tkey"] = raw["TI"].map(norm_title) + "_" + raw["PY"].astype(str)
    raw["_score"] = raw.apply(_completeness, axis=1)
    raw["_dbsrc"] = raw["DB"]

    # agrupar por DOI o por título+año (componentes conectados simples)
    key = {}
    group_id = []
    gid = 0
    for doi, tk in zip(raw["DI"], raw["_tkey"]):
        g = key.get(("d", doi)) if doi else None
        if g is None and len(tk) > 15:
            g = key.get(("t", tk))
        if g is None:
            g, gid = gid, gid + 1
        if doi:
            key[("d", doi)] = g
        if len(tk) > 15:
            key[("t", tk)] = g
        group_id.append(g)
    raw["_g"] = group_id

    dbs = raw.groupby("_g")["DB"].agg(lambda s: "+".join(sorted(set(s), reverse=True)))
    tcmax = raw.groupby("_g")["TC"].max()
    raw = raw.sort_values(["_g", "_score"], ascending=[True, False])
    merged = raw.drop_duplicates("_g").copy()
    merged["DB_SOURCES"] = merged["_g"].map(dbs)
    merged["TC"] = merged["_g"].map(tcmax)
    stats["duplicates_removed"] = len(raw) - len(merged)
    stats["unique"] = len(merged)
    stats["overlap"] = merged["DB_SOURCES"].value_counts().to_dict()
    merged = merged.drop(columns=[c for c in merged.columns if c.startswith("_")])
    return merged.reset_index(drop=True), stats


def field_availability(df: pd.DataFrame) -> dict:
    av = {}
    for f in ["AB", "C1", "RP", "DE", "ID", "TC", "DT", "LA", "CR", "DI"]:
        s = df[f]
        av[f] = float((s.notna() & (s.astype(str).str.strip() != "") & (s.astype(str) != "<NA>")
                       & (s.astype(str) != "nan")).mean())
    return av


def apply_filters(df: pd.DataFrame, year_min=None, year_max=None, doc_types=None,
                  languages=None) -> tuple[pd.DataFrame, list]:
    """Filtros de elegibilidad; devuelve el corpus y los pasos para el diagrama PRISMA."""
    steps = []
    d = df.copy()
    if year_min is not None or year_max is not None:
        lo = year_min if year_min is not None else int(d["PY"].min())
        hi = year_max if year_max is not None else int(d["PY"].max())
        m = d["PY"].between(lo, hi)
        steps.append((f"Outside {lo}-{hi}", int((~m).sum())))
        d = d[m]
    if doc_types and (d["DT"] != "").any():
        m = d["DT"].isin(doc_types)
        steps.append(("Other document types", int((~m).sum())))
        d = d[m]
    if languages and (d["LA"] != "").any():
        m = d["LA"].isin(languages)
        steps.append(("Other languages", int((~m).sum())))
        d = d[m]
    m = d["AB"].str.split().str.len().fillna(0) >= 20
    steps.append(("Missing/short abstract (<20 words)", int((~m).sum())))
    d = d[m]
    return d.reset_index(drop=True), steps
