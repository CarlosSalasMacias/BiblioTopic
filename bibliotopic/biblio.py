"""Indicadores bibliométricos (equivalentes a los de bibliometrix) en Python."""
from __future__ import annotations

import itertools
import re
from collections import Counter

import numpy as np
import pandas as pd

try:
    import pycountry
    _PYC = True
except ImportError:  # pragma: no cover
    _PYC = False

ALIASES = {
    "usa": "USA", "united states": "USA", "united states of america": "USA", "u.s.a.": "USA",
    "peoples r china": "China", "people's republic of china": "China", "pr china": "China",
    "china": "China", "hong kong": "China", "macau": "China",
    "england": "United Kingdom", "scotland": "United Kingdom", "wales": "United Kingdom",
    "north ireland": "United Kingdom", "northern ireland": "United Kingdom", "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "south korea": "South Korea", "korea": "South Korea", "republic of korea": "South Korea",
    "korea, republic of": "South Korea", "russia": "Russia", "russian federation": "Russia",
    "viet nam": "Vietnam", "vietnam": "Vietnam", "turkiye": "Turkey", "turkey": "Turkey",
    "iran": "Iran", "czech republic": "Czech Republic", "czechia": "Czech Republic",
    "cote ivoire": "Cote d'Ivoire", "cote d'ivoire": "Cote d'Ivoire", "ivory coast": "Cote d'Ivoire",
    "taiwan": "Taiwan", "u arab emirates": "United Arab Emirates", "bolivia": "Bolivia",
    "venezuela": "Venezuela", "tanzania": "Tanzania", "dem rep congo": "DR Congo",
    "democratic republic of the congo": "DR Congo", "democratic republic congo": "DR Congo",
    "frg": "Germany", "west germany": "Germany", "fed rep ger": "Germany", "ussr": "Russia", "rep congo": "Congo", "laos": "Laos",
    "papua n guinea": "Papua New Guinea", "trinid & tobago": "Trinidad and Tobago",
    "bosnia & herceg": "Bosnia and Herzegovina", "syria": "Syria", "brunei": "Brunei",
    "micronesia": "Micronesia", "moldova": "Moldova", "netherlands": "Netherlands",
    "the netherlands": "Netherlands",
}
if _PYC:
    for c in pycountry.countries:
        for attr in ("name", "common_name", "official_name"):
            v = getattr(c, attr, None)
            if v:
                ALIASES.setdefault(v.lower(), getattr(c, "common_name", None) or c.name)


def _country_of(segment: str) -> str | None:
    s = re.sub(r"[\.\s]+$", "", segment.strip()).lower()
    s = re.sub(r"\s+", " ", s)
    if s in ALIASES:
        return ALIASES[s]
    if re.search(r"\busa$", s):  # "CA 95616 USA"
        return "USA"
    s2 = re.sub(r"[\d\-]+", "", s).strip()
    return ALIASES.get(s2)


def countries_in(address_field: str) -> list[str]:
    """Países presentes en un campo C1 (WoS o Scopus); uno por dirección."""
    if not address_field:
        return []
    txt = re.sub(r"\[[^\]]*\]", "|", address_field)  # quita listas de autores WoS
    out = []
    for addr in re.split(r"[;|]", txt):
        segs = [p for p in addr.split(",") if p.strip()]
        for seg in reversed(segs):
            c = _country_of(seg)
            if c:
                out.append(c)
                break
    return out


def corresponding_country(rp: str) -> str | None:
    if not rp:
        return None
    rp = re.sub(r"email:.*$", "", rp, flags=re.I)
    for seg in reversed([p for p in re.split(r"[,;]", rp) if p.strip()]):
        c = _country_of(seg)
        if c:
            return c
    return None


def add_country_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["COUNTRIES"] = d["C1"].map(lambda s: sorted(set(countries_in(s))))
    d["RP_COUNTRY"] = d["RP"].map(corresponding_country)
    # si no hay autor de correspondencia, se usa el país de la primera dirección
    first = d["C1"].map(lambda s: (countries_in(s) or [None])[0])
    d["RP_COUNTRY"] = d["RP_COUNTRY"].fillna(first)
    return d


# --------------------------------------------------------------- indicadores
def split_list(s: str) -> list[str]:
    return [x.strip() for x in str(s).split(";") if x.strip()]


def main_information(df: pd.DataFrame, has_countries: bool) -> dict:
    py = df["PY"].dropna().astype(int)
    first, last = int(py.min()), int(py.max())
    yearly = py.value_counts().sort_index()
    # crecimiento hasta el último año completo (excluye el año en curso y los de acceso anticipado)
    last_full = min(last, pd.Timestamp.now().year - 1)
    n_first = yearly.iloc[0]
    n_last = yearly.get(last_full, 0)
    cagr = (((n_last / n_first) ** (1 / max(last_full - first, 1)) - 1) * 100
            if n_first and n_last and last_full > first else np.nan)
    authors = df["AU"].map(split_list)
    n_auth = authors.map(len)
    all_auth = Counter(itertools.chain.from_iterable(authors))
    info = {
        "Timespan": f"{first}-{last}",
        "Sources (journals, books, etc.)": df["SO"].str.upper().nunique(),
        "Documents": len(df),
        f"Annual growth rate {first}-{last_full} (%)": round(cagr, 2),
        "Document average age (years)": round(float((pd.Timestamp.now().year - py).mean()), 2),
        "Authors": len(all_auth),
        "Single-authored documents": int((n_auth == 1).sum()),
        "Co-authors per document": round(float(n_auth.mean()), 2),
    }
    if df["TC"].notna().any():
        info["Average citations per document"] = round(float(df["TC"].mean()), 2)
    if (df["DE"] != "").any():
        info["Author keywords (DE)"] = len(set(itertools.chain.from_iterable(df["DE"].map(split_list))))
    if (df["ID"] != "").any():
        info["Keywords Plus / Index keywords (ID)"] = len(
            set(itertools.chain.from_iterable(df["ID"].map(split_list))))
    if has_countries:
        multi = df["COUNTRIES"].map(len) > 1
        info["International co-authorship (%)"] = round(100 * float(multi.mean()), 2)
    return info


def annual_production(df: pd.DataFrame) -> pd.DataFrame:
    t = df.groupby("PY").agg(Articles=("TI", "size"))
    if df["TC"].notna().any():
        t["MeanTC"] = df.groupby("PY")["TC"].mean()
        now = pd.Timestamp.now().year
        t["MeanTCperYear"] = t["MeanTC"] / (now - t.index.astype(int) + 1)
    full = range(int(t.index.min()), int(t.index.max()) + 1)
    t = t.reindex(full).fillna({"Articles": 0})
    t.index.name = "Year"
    return t.reset_index()


def top_sources(df: pd.DataFrame, n=10) -> pd.DataFrame:
    s = df["SO"].str.upper().str.strip().value_counts()
    out = s.head(n).rename_axis("Source").reset_index(name="Articles")
    out["% of corpus"] = (100 * out["Articles"] / len(df)).round(2)
    return out


def bradford(df: pd.DataFrame) -> pd.DataFrame:
    s = df["SO"].str.upper().str.strip().value_counts().rename_axis("Source").reset_index(name="Freq")
    s["Rank"] = np.arange(1, len(s) + 1)
    s["cumFreq"] = s["Freq"].cumsum()
    tot = s["Freq"].sum()
    s["Zone"] = np.where(s["cumFreq"] <= tot / 3, "Zone 1",
                         np.where(s["cumFreq"] <= 2 * tot / 3, "Zone 2", "Zone 3"))
    return s


PARTICLES = {"de", "da", "do", "dos", "das", "del", "della", "la", "le", "van", "von", "der", "den", "di", "du",
             "y", "e"}


def pretty_author(a: str) -> str:
    """'PARTELLI FL' -> 'Partelli FL'; 'DE LA HOZ-M J' -> 'De la Hoz-M J'."""
    parts = a.split()
    if not parts:
        return a
    ini = parts[-1] if len(parts) > 1 and parts[-1].isupper() and len(parts[-1]) <= 3 else ""
    sur = parts[:-1] if ini else parts
    out = []
    for i, w in enumerate(sur):
        lw = w.lower()
        if i > 0 and lw in PARTICLES:
            out.append(lw)
        else:
            out.append("-".join(x[:1].upper() + x[1:] for x in lw.split("-")))
    return " ".join(out + ([ini] if ini else []))


def top_authors(df: pd.DataFrame, n=10) -> pd.DataFrame:
    auth = df["AU"].map(split_list)
    c = Counter(itertools.chain.from_iterable(auth))
    frac = Counter()
    for a in auth:
        for x in a:
            frac[x] += 1 / len(a)
    rows = [(pretty_author(a), k, round(frac[a], 2)) for a, k in c.most_common(n)]
    return pd.DataFrame(rows, columns=["Author", "Articles", "Articles fractionalized"])


def lotka(df: pd.DataFrame) -> pd.DataFrame:
    c = Counter(itertools.chain.from_iterable(df["AU"].map(split_list)))
    dist = Counter(c.values())
    t = pd.DataFrame(sorted(dist.items()), columns=["Documents written", "N. of authors"])
    t["Proportion of authors"] = (t["N. of authors"] / t["N. of authors"].sum()).round(4)
    return t


def country_production(df: pd.DataFrame, n=20) -> pd.DataFrame:
    """Producción por país del autor de correspondencia (SCP/MCP) y por participación."""
    d = df[df["RP_COUNTRY"].notna()]
    mcp = d["COUNTRIES"].map(len) > 1
    t = pd.DataFrame({"Country": d["RP_COUNTRY"], "MCP": mcp})
    g = t.groupby("Country").agg(Articles=("MCP", "size"), MCP=("MCP", "sum"))
    g["SCP"] = g["Articles"] - g["MCP"]
    g["MCP ratio"] = (g["MCP"] / g["Articles"]).round(3)
    part = Counter(itertools.chain.from_iterable(df["COUNTRIES"]))
    g["Participation"] = g.index.map(lambda c: part.get(c, 0))
    if df["TC"].notna().any():
        g["Total citations"] = d.groupby("RP_COUNTRY")["TC"].sum()
        g["Average citations"] = (g["Total citations"] / g["Articles"]).round(2)
    g = g.sort_values("Articles", ascending=False).head(n)
    return g.reset_index()


def country_collaboration(df: pd.DataFrame) -> pd.DataFrame:
    edges = Counter()
    for cs in df["COUNTRIES"]:
        for a, b in itertools.combinations(sorted(set(cs)), 2):
            edges[(a, b)] += 1
    return pd.DataFrame([(a, b, w) for (a, b), w in edges.items()],
                        columns=["From", "To", "Weight"]).sort_values("Weight", ascending=False)


def norm_keyword(k: str, synonyms=None) -> str:
    """Minúsculas, sin puntuación final ni abreviaturas de autoridad botánica ('coffea arabica l.')."""
    k = re.sub(r"\s+", " ", k.lower()).strip(" .;,")
    k = re.sub(r"(\s+[a-z]\.?)+$", "", k) if len(k.split()) > 1 else k
    k = k.replace("‐", "-").strip()
    return (synonyms or {}).get(k, k)


def parse_synonyms(text: str) -> dict:
    """'arabica coffee = coffea arabica; robusta = coffea canephora' -> dict."""
    out = {}
    for part in re.split(r"[;\n]", text or ""):
        if "=" in part:
            a, b = part.split("=", 1)
            for x in a.split(","):
                if x.strip():
                    out[norm_keyword(x)] = norm_keyword(b)
    return out


def normalize_keywords(df: pd.DataFrame, synonyms=None) -> pd.DataFrame:
    d = df.copy()
    for f in ("DE", "ID"):
        d[f] = d[f].map(lambda s: "; ".join(dict.fromkeys(
            k for k in (norm_keyword(x, synonyms) for x in split_list(s)) if k)))
    return d


def keyword_freq(df: pd.DataFrame, field="DE", n=30) -> pd.DataFrame:
    c = Counter(itertools.chain.from_iterable(df[field].map(split_list)))
    return pd.DataFrame(c.most_common(n), columns=["Keyword", "Occurrences"])


def keyword_cooccurrence(df: pd.DataFrame, field="DE", top=50) -> tuple[list, Counter]:
    lists = df[field].map(lambda s: sorted(set(split_list(s))))
    freq = Counter(itertools.chain.from_iterable(lists))
    keep = {k for k, _ in freq.most_common(top)}
    edges = Counter()
    for ks in lists:
        ks = [k for k in ks if k in keep]
        for a, b in itertools.combinations(ks, 2):
            edges[(a, b)] += 1
    return [(k, freq[k]) for k in keep], edges
