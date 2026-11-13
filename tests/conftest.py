"""Corpus sintéticos pequeños con resultados esperados conocidos."""
import csv
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WOS_COLS = ["PT", "AU", "AF", "TI", "SO", "LA", "DT", "DE", "ID", "AB", "C1", "RP", "TC", "PY", "DI", "UT"]
SCOPUS_COLS = ["Authors", "Author full names", "Title", "Year", "Source title", "Cited by", "DOI", "Abstract",
               "Author Keywords", "Index Keywords", "Document Type", "EID"]  # sin columna de idioma

POOLS = {
    "rust": "leaf rust fungicide hemileia pathogen spore infection disease resistance epidemic".split(),
    "soil": "soil nitrogen fertilizer organic matter nutrient compost biochar phosphorus potassium".split(),
    "shade": "shade tree agroforestry canopy biodiversity bird forest habitat carbon".split(),
}


def abstract(pool, rng, n=40):
    words = POOLS[pool]
    return " ".join(rng.choice(words, n)) + " results show the effect in coffee plantations"


def wos_text(rows):
    out = io.StringIO()
    w = csv.writer(out, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
    w.writerow(WOS_COLS)
    for r in rows:
        w.writerow([r.get(c, "") for c in WOS_COLS])
    return out.getvalue().encode("utf-8")


def scopus_text(rows):
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(SCOPUS_COLS)
    for r in rows:
        w.writerow([r.get(c, "") for c in SCOPUS_COLS])
    return out.getvalue().encode("utf-8")


def make_corpus(n_per_pool=30, years=range(2014, 2023), seed=0):
    """Registros WoS con idioma y Scopus sin idioma; 6 documentos están en ambas bases."""
    rng = np.random.default_rng(seed)
    wos, scopus = [], []
    i = 0
    yrs = list(years)
    for pool in POOLS:
        for _ in range(n_per_pool):
            y = yrs[i % len(yrs)]
            ti = f"Study number {i} on {pool} management in coffee systems"
            ab = abstract(pool, rng)
            doi = f"10.1000/test.{i}"
            if i % 2 == 0:
                wos.append({"PT": "J", "AU": "Perez, J; Lopez, A", "TI": ti, "SO": "AGRO J", "LA": "English",
                            "DT": "Article", "AB": ab, "TC": str(i % 7), "PY": str(y), "DI": doi,
                            "UT": f"WOS:{i}", "DE": f"{pool}; coffee", "C1": "[Perez, J] Univ, Quito, Ecuador."})
            if i % 2 == 1 or i % 10 == 0:  # los i % 10 == 0 también están en WoS
                scopus.append({"Authors": "Perez J.; Lopez A.", "Title": ti, "Year": str(y), "Source title": "Agro J",
                               "Cited by": str(i % 5), "DOI": doi, "Abstract": ab, "Author Keywords": f"{pool}; coffee",
                               "Document Type": "Article", "EID": f"2-s2.0-{i}"})
            i += 1
    return wos_text(wos), scopus_text(scopus)


@pytest.fixture(scope="session")
def corpus_files():
    w, s = make_corpus()
    return [("scopus.csv", s), ("savedrecs.txt", w)]
