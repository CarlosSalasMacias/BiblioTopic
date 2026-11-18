"""Análisis de citas: documentos más citados (global y por tema) y red de co-citación."""
from __future__ import annotations

import itertools
import re
from collections import Counter

import numpy as np
import pandas as pd

from .biblio import pretty_author, split_list


def _surname(a: str) -> str:
    parts = pretty_author(a).split()
    if len(parts) > 1 and parts[-1].isupper() and len(parts[-1]) <= 3:
        parts = parts[:-1]
    return " ".join(parts)


def short_ref(row) -> str:
    au = split_list(row["AU"])
    first = _surname(au[0]) if au else "Anon."
    etal = " et al." if len(au) > 2 else (f" y {_surname(au[1])}" if len(au) == 2 else "")
    return f"{first}{etal} ({int(row['PY'])})"


def full_ref(row) -> str:
    """Referencia APA aproximada a partir de los metadatos del corpus."""
    au = [pretty_author(a) for a in split_list(row["AU"])]
    def fmt(a):
        parts = a.split()
        if len(parts) >= 2 and parts[-1].isupper():
            return f"{' '.join(parts[:-1])}, {'. '.join(parts[-1])}."
        return a
    names = [fmt(a) for a in au]
    if len(names) > 20:
        names = names[:19] + ["...", names[-1]]
    auth = ", ".join(names[:-1]) + (", & " if len(names) > 1 else "") + names[-1] if names else "Anon."
    doi = f" https://doi.org/{row['DI']}" if str(row.get("DI", "")).strip() else ""
    return f"{auth} ({int(row['PY'])}). {str(row['TI']).strip().rstrip('.')}. {str(row['SO']).title()}.{doi}"


def most_cited(df: pd.DataFrame, n=10) -> pd.DataFrame:
    if not df["TC"].notna().any():
        return None
    d = df.assign(TC=df["TC"].fillna(0)).sort_values("TC", ascending=False).head(n)
    age = (pd.Timestamp.now().year - d["PY"].astype(int) + 1).clip(lower=1)
    return pd.DataFrame({"Document": [short_ref(r) for _, r in d.iterrows()], "Title": d["TI"].values,
                         "Source": d["SO"].str.title().values, "Citations": d["TC"].astype(int).values,
                         "Citations per year": (d["TC"] / age).round(2).values, "DOI": d["DI"].values})


def cited_by_topic(df: pd.DataFrame, assign, ids, labels, n=3):
    """Documentos más citados dentro de cada tema (asignación dominante)."""
    if not df["TC"].notna().any():
        return None
    rows = []
    tc = df["TC"].fillna(0).values
    for k, (tid, lab) in enumerate(zip(ids, labels)):
        idx = np.where(assign == k)[0]
        for i in idx[np.argsort(-tc[idx])][:n]:
            r = df.iloc[i]
            rows.append({"Topic": lab, "Document": short_ref(r), "Title": r["TI"], "Citations": int(tc[i]),
                         "_row": int(i)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ co-citación
ANON = {"anonymous", "anon", "anonym", "no author", "no authors"}


def ref_key(ref: str) -> str | None:
    """'Aria M, 2017, J INFORMETR, V11' (WoS) o 'Aria, M., ..., (2017) Journal...' (Scopus) -> 'Aria M, 2017'.
    Los apellidos con partícula (De, Da, Van, Von, Le...) se conservan completos («De Souza J, 2013», no «De,
    2013») y las referencias anónimas se agrupan por fuente («Anonymous [FAOSTAT], 2014»); sin fuente identificable
    se excluyen (None)."""
    ref = ref.strip()
    if not ref:
        return None
    m = re.search(r"\((\d{4})\)", ref) or re.search(r",\s*(\d{4})\s*,", ref) or re.search(r"\b(19|20)\d{2}\b", ref)
    if not m:
        return None
    year = m.group(1) if len(m.group(1)) == 4 else m.group(0)
    segs = [x.strip() for x in ref.split(",")]
    first = re.sub(r"[^A-Za-zÀ-ÿ\-\s']", "", segs[0]).strip()
    if not first:
        return None
    if first.lower().strip("[] ") in ANON:
        # fuente: segmento que sigue al año (WoS: «Anonymous, 2014, FAOSTAT DATABASE»)
        src = ""
        for i_, sg in enumerate(segs):
            if sg == year and i_ + 1 < len(segs):
                src = segs[i_ + 1]
                break
        src = re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9À-ÿ\-\s]", "", src)).strip().upper()[:30]
        return f"Anonymous [{src}], {year}" if src else None
    parts = first.split()
    wos_style = len(parts) > 1 and parts[-1].isupper() and len(parts[-1]) <= 3 and not first.isupper() or (
        first.isupper() and len(parts) > 1 and len(parts[-1]) <= 3)
    if wos_style:  # «De Souza J» / «DE SOUZA J»: apellido = todo menos las iniciales finales
        sur, ini = " ".join(parts[:-1]), parts[-1]
    else:  # Scopus: «De Souza, J.»
        sur = " ".join(parts)
        ini = re.sub(r"[^A-Z]", "", segs[1])[:2] if len(segs) > 1 else ""
    sur = " ".join(w.capitalize() if not ("-" in w) else "-".join(x.capitalize() for x in w.split("-"))
                   for w in sur.split())
    return f"{sur} {ini[:2]}, {year}".replace("  ", " ")


def cocitation(df: pd.DataFrame, top=40):
    refs = df["CR"].fillna("").map(lambda s: sorted({k for k in (ref_key(x) for x in s.split(";")) if k}))
    freq = Counter(itertools.chain.from_iterable(refs))
    keep = {k for k, _ in freq.most_common(top)}
    edges = Counter()
    for rs in refs:
        rs = [r for r in rs if r in keep]
        for a, b in itertools.combinations(rs, 2):
            edges[(a, b)] += 1
    table = pd.DataFrame(freq.most_common(20), columns=["Cited reference", "Local citations"])
    return [(k, freq[k]) for k in keep], edges, table


def bibliographic_coupling(df: pd.DataFrame, top=40, min_refs=1):
    """Acoplamiento bibliográfico (Kessler, 1963) entre los documentos más citados del corpus: dos documentos
    están acoplados si comparten referencias citadas; el peso es el número de referencias compartidas.
    Devuelve (nodos [(documento, citas)], aristas {(a, b): peso}, tabla de los pares más acoplados)."""
    refs = df["CR"].fillna("").map(lambda s: {k for k in (ref_key(x) for x in s.split(";")) if k})
    d = df.assign(_refs=refs, _tc=df["TC"].fillna(0))
    d = d[d["_refs"].map(len) >= min_refs].sort_values("_tc", ascending=False).head(top)
    names = [short_ref(r) for _, r in d.iterrows()]
    nodes = [(nm, int(tc)) for nm, tc in zip(names, d["_tc"])]
    edges = Counter()
    rows = []
    R = list(d["_refs"])
    for i, j in itertools.combinations(range(len(R)), 2):
        w = len(R[i] & R[j])
        if w:
            edges[(names[i], names[j])] = w
            rows.append((names[i], names[j], w, round(w / np.sqrt(len(R[i]) * len(R[j])), 3)))
    table = pd.DataFrame(sorted(rows, key=lambda x: -x[2])[:20],
                         columns=["Document A", "Document B", "Shared references", "Coupling strength (cosine)"])
    return nodes, edges, table
