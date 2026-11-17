"""Importadores: el caso que falló en un análisis real. Ningún registro debe perderse por un campo ausente."""
import csv
import io

import pandas as pd

from bibliotopic import biblio, io as bio, pipeline

AB = ("Coffee plantations under shade trees showed lower leaf rust incidence and higher yield stability across "
      "several farms and seasons in the region. ")


def scopus_csv(rows, with_language=True):
    cols = ["Authors", "Title", "Year", "Source title", "Cited by", "DOI", "Abstract", "Author Keywords",
            "Index Keywords", "Document Type", "Correspondence Address", "Authors with affiliations", "EID"]
    if with_language:
        cols.append("Language of Original Document")
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([r.get(c, "") for c in cols])
    return out.getvalue().encode("utf-8")


def wos_tab(rows):
    cols = ["PT", "AU", "TI", "SO", "LA", "DT", "DE", "ID", "AB", "C1", "RP", "TC", "PY", "DI", "UT"]
    out = io.StringIO()
    w = csv.writer(out, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\", lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([r.get(c, "") for c in cols])
    return out.getvalue().encode("utf-8")


def sc(i, **kw):
    r = {"Authors": "Perez J.", "Title": f"Scopus only study {i} on shaded coffee systems", "Year": "2020",
         "Source title": "Agro J", "Cited by": "3", "DOI": f"10.9/s{i}", "Abstract": AB * 2,
         "Author Keywords": "<italic>Coffea arabica</italic>; agro-forestry", "Document Type": "Article",
         "Correspondence Address": "J. Perez; Universidad Técnica, Portoviejo, Ecuador; email: j@x.ec",
         "Authors with affiliations": "Perez J., Universidad Técnica, Portoviejo, Ecuador", "EID": f"2-s2.0-{i}",
         "Language of Original Document": "English"}
    r.update(kw)
    return r


def wo(i, **kw):
    r = {"PT": "J", "AU": "Perez, J", "TI": f"WoS study {i} on coffee rust", "SO": "AGRO J", "LA": "English",
         "DT": "Article", "DE": "coffea arabica; agroforestry", "AB": AB * 2, "C1": "[Perez, J] Univ, Quito, Ecuador.",
         "RP": "Perez, J (corresponding author), Univ, Quito, Ecuador.", "TC": "5", "PY": "2021", "DI": f"10.9/w{i}",
         "UT": f"WOS:{i}"}
    r.update(kw)
    return r


def load(files, **cfg):
    S = pipeline.Session()
    out = S.load(files)
    S.prepare({"config_version": 2, "analyses": ["biblio", "countries", "keywords"], **cfg})
    return S, out


def test_scopus_without_language_keeps_all_exclusive_records():
    scopus = scopus_csv([sc(i) for i in range(30)], with_language=False)
    wos = wos_tab([wo(i) for i in range(20)])
    S, out = load([("scopus.csv", scopus), ("savedrecs.txt", wos)], languages=["English"])
    assert out["unknown"]["LA"] == {"Scopus": 30}
    assert (S.df["DB_SOURCES"] == "Scopus").sum() == 30  # antes: 0 (todos excluidos «por idioma»)
    assert ("Language not available", 30, "kept") in S.excl_steps
    low = {(x["db"], x["field"]) for x in out["audit"]["critical_low"]}
    assert ("Scopus", "LA") in low  # la auditoría lo marca en rojo
    assert not [q for q in S.quality_checks() if "0 registros exclusivos" in q["check"]]


def test_scopus_with_language_and_multiple_languages():
    scopus = scopus_csv([sc(0, **{"Language of Original Document": "English; Spanish"}),
                         sc(1, **{"Language of Original Document": "Portuguese"})])
    S, _ = load([("scopus.csv", scopus)], languages=["English"])
    assert list(S.df["TI"].str.contains("study 0")) == [True]  # se conserva si alguno de sus idiomas es admitido


def test_doc_type_harmonization():
    wos = wos_tab([wo(1, DT="Article; Proceedings Paper"), wo(2, DT="Article; Early Access"),
                   wo(3, DT="Proceedings Paper"), wo(4, DT="Book Chapter"), wo(5, DT="Editorial Material"),
                   wo(6, DT="Review; Early Access")])
    scopus = scopus_csv([sc(7, **{"Document Type": "Conference paper"}), sc(8, **{"Document Type": "Book chapter"}),
                         sc(9, **{"Document Type": "ARTICLE"})])
    S = pipeline.Session()
    S.load([("scopus.csv", scopus), ("savedrecs.txt", wos)])
    dt = dict(zip(S.merged["UT"], S.merged["DT"]))
    assert dt["WOS:1"] == dt["WOS:3"] == dt["2-s2.0-7"] == "Conference paper"
    assert dt["WOS:2"] == dt["2-s2.0-9"] == "Article"
    assert dt["WOS:4"] == dt["2-s2.0-8"] == "Book chapter"
    assert dt["WOS:5"] == "Other" and dt["WOS:6"] == "Review"


def test_markup_and_copyright_cleanup():
    scopus = scopus_csv([sc(0, Abstract=AB * 2 + "© 2020 Elsevier B.V. All rights reserved.",
                            Title="Effect of <i>Hemileia vastatrix</i> on CO<sub>2</sub> uptake &amp; yield")])
    S = pipeline.Session()
    S.load([("scopus.csv", scopus)])
    r = S.merged.iloc[0]
    assert "<" not in r["TI"] and "&amp;" not in r["TI"] and "Hemileia vastatrix" in r["TI"]
    assert "Elsevier" not in r["AB"] and "©" not in r["AB"]
    assert "coffea arabica" in r["DE"].split("; ")


def test_duplicates_merge_fields_and_citations_per_base():
    doi = "10.9/dup"
    scopus = scopus_csv([sc(0, DOI=doi, Title="Shared study on coffee rust and shade", **{"Cited by": "12"})],
                        with_language=False)
    wos = wos_tab([wo(0, DI=doi, TI="Shared study on coffee rust and shade", TC="7", LA="English")])
    S = pipeline.Session()
    out = S.load([("scopus.csv", scopus), ("savedrecs.txt", wos)])
    r = S.merged.iloc[0]
    assert len(S.merged) == 1 and r["DB_SOURCES"] == "WoS+Scopus"
    assert r["TC"] == 12 and r["TC_WoS"] == 7 and r["TC_Scopus"] == 12
    assert r["LA"] == "English"  # completado desde WoS
    assert out["input_files"][0]["md5"] and out["input_files"][0]["records"] == 1


def test_country_source_and_fallback():
    d = pd.DataFrame({"C1": ["[A] Univ, Quito, Ecuador.", "[B] Univ, Lima, Peru."],
                      "RP": ["A (corresponding author), Univ, Quito, Ecuador.", ""]})
    out = biblio.add_country_columns(d)
    assert list(out["RP_COUNTRY"]) == ["Ecuador", "Peru"]
    assert list(out["RP_COUNTRY_SOURCE"]) == ["Corresponding address", "First author affiliation"]


def test_query_terms_from_equations():
    t = bio.extract_query_terms('TITLE-ABS-KEY ( ( coffee OR "coffea arabica" OR coffea* ) AND NOT ( caffeine ) ) '
                                'AND PUBYEAR > 1999 AND ( LIMIT-TO ( DOCTYPE , "ar" ) )',
                                'TS=(coffee* OR "coffea canephora") NOT TS=(espresso)')
    assert set(t) == {"coffee", "coffea arabica", "coffea", "coffea canephora"}


def test_keyword_normalization_and_synonyms():
    d = pd.DataFrame({"DE": ["Agro-forestry; Fertilisers; colour; Coffee plantations", "agroforestry; arabica coffee"],
                      "ID": ["", ""]})
    syn = biblio.parse_synonyms("term,canonical\narabica coffee,coffea arabica\n")
    out = biblio.normalize_keywords(d, syn)
    ks = set("; ".join(out["DE"]).split("; "))
    assert "fertilizer" in ks and "color" in ks and "coffee plantation" in ks and "coffea arabica" in ks
    assert len({k for k in ks if k.replace("-", "").replace(" ", "") == "agroforestry"}) == 1


def test_config_version_compatibility():
    old = pipeline.resolve_config({"k_min": 5})  # config.json de la versión 1 (sin config_version)
    assert old["config_version"] == 1 and old["lda_method"] == "online" and old["n_stab_lda"] == 3
    new = pipeline.resolve_config({"config_version": 2})
    assert new["profile"] == "publication" and new["n_boot"] == 1000 and new["lda_method"] == "batch"
    assert new["n_stab_lda"] == 10 and new["k_max"] == 40 and new["k_seeds"] == 3
    quick = pipeline.resolve_config({"config_version": 2, "profile": "quick"})
    assert quick["lda_method"] == "batch" and quick["lda_max_iter"] == 30 and quick["n_boot"] == 200
    assert pipeline.resolve_config({"config_version": 2, "n_boot": 99999})["n_boot"] == 5000
