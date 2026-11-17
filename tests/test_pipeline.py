"""Extremo a extremo con un corpus sintético: tablas y texto del borrador deben coincidir."""
import numpy as np
import pytest
from sklearn.decomposition import PCA

from bibliotopic import pipeline, report
from bibliotopic.bert_model import lsa_embeddings


@pytest.fixture(scope="module")
def session(corpus_files, tmp_path_factory):
    S = pipeline.Session()
    out = S.load(corpus_files)
    assert out["ok"]
    S.prepare({"k": 3, "n_boot": 20, "min_cluster_size": 5, "min_df": 2, "n_stab_lda": 1, "n_stab_bert": 1,
               "export_tiff": False, "composite": False})
    S.run_lda()
    e = lsa_embeddings(S.texts())
    red = [PCA(5, random_state=r).fit_transform(e) for r in range(S.n_umap_runs())]
    S.run_bertopic(red, PCA(2).fit_transform(e), "LSA")
    d = tmp_path_factory.mktemp("out")
    S.render(str(d))
    report.build(S, str(d / "doc.docx"), {"topic_name": "coffee"}, None)
    return S, out, d


def test_unknown_language_records_kept(session):
    S, out, _ = session
    assert out["unknown"]["LA"].get("Scopus", 0) > 0  # Scopus sin columna de idioma
    assert all(kind != "excluded" for name, n, kind in S.excl_steps if name.endswith("not available"))
    assert (S.df["DB_SOURCES"] == "Scopus").sum() > 0  # los registros exclusivos de Scopus se conservan


def test_merge_stats_consistent(session):
    S, out, _ = session
    st = out["stats"]
    assert st["records_total"] - st["duplicates_removed"] == st["unique"] == len(S.merged)
    assert st["overlap"].get("WoS+Scopus", 0) == 9  # i % 10 == 0 con i par -> en ambas bases


def test_documents_table_traceability(session):
    _, _, d = session
    import pandas as pd
    docs = pd.read_csv(d / "tables" / "documents_topics.csv")
    for col in ("BERTopic_original", "BERTopic", "BERTopic_assignment", "BERTopic_similarity",
                "BERTopic_low_confidence", "N_RECORDS"):
        assert col in docs
    reassigned = docs["BERTopic_original"] == "Outlier"
    assert (docs.loc[~reassigned, "BERTopic_original"] == docs.loc[~reassigned, "BERTopic"]).all()


def test_report_matches_tables(session):
    import docx
    S, out, d = session
    text = "\n".join(p.text for p in docx.Document(str(d / "doc.docx")).paragraphs)
    st = out["stats"]
    assert f"Se eliminaron {st['duplicates_removed']} duplicados" in text
    assert f"integrado por {len(S.df)} documentos" in text
    assert "ecuación de búsqueda (" not in text  # no se indicaron términos de búsqueda: no se afirma
    assert "no informaban el idioma" in text
    n_all, n_ok = len(S.bert.topics), int((S.bert.topics_raw >= 0).sum())
    assert f"(n = {n_all})" in text and f"(n = {n_ok})" in text
    assert "Stable" not in text


def test_tri_expert_validation_in_report(session, tmp_path, monkeypatch):
    """Con valoraciones de expertos, el TRI se correlaciona con ellas (tabla T12b) y el borrador lo informa."""
    import functools
    import pandas as pd
    from docx import Document
    from bibliotopic import stats as ST
    S, _, _ = session
    rng = np.random.default_rng(0)
    rows = [{"model": m, "topic": t, "interpretability": rng.uniform(1, 5), "coherence": rng.uniform(1, 5),
             "model_precision": rng.uniform(0, 1), "intrusion_answers": 3}
            for m, r in (("LDA", S.lda), ("BERTopic", S.bert)) for t in r.ids]
    table = pd.DataFrame([{"Model": m, "Interpretability (1-5)": "3.00 [2.00, 4.00]", "Coherence (1-5)": "3.00 [2.00, 4.00]",
                           "Word-intrusion precision": "0.50 [0.20, 0.80]"} for m in ("LDA", "BERTopic")])
    old = S.validation
    S.validation = {"per_topic": pd.DataFrame(rows), "table": table, "tests": {}, "alpha": {}, "fleiss_kappa": {},
                    "n_raters": 3, "raters": ["a", "b", "c"]}
    monkeypatch.setattr(ST, "tri_expert_correlation", functools.partial(ST.tri_expert_correlation, min_pairs=2))
    try:
        S.render(str(tmp_path))
        assert "T12b_TRI_vs_experts" in S.tables
        assert list(S.tables["T12b_TRI_vs_experts"]["Predictor"]) == ["TRI", "Cosine", "Mean stability"]
        report.build(S, str(tmp_path / "doc.docx"), {"topic_name": "coffee"}, None)
        txt = "\n".join(p.text for p in Document(str(tmp_path / "doc.docx")).paragraphs)
        assert "primera validación empírica del TRI" in txt and "correlación del TRI con la interpretabilidad" in txt
        assert "Magsarjav" in txt and "Velden" in txt
    finally:
        S.validation = old


def test_title_topic_ignores_not_terms(session):
    """El tema de los títulos propuestos nunca es un término excluido con NOT en la ecuación de búsqueda."""
    from bibliotopic.report import _topic_phrase
    S, _, _ = session
    old = dict(S.cfg)
    try:
        S.cfg.update(query_terms=[], search_scopus='TITLE-ABS-KEY ( coffee ) AND NOT TITLE-ABS-KEY ( "coffee shop*" )',
                     search_wos="")
        t = _topic_phrase(S, {"query": 'Scopus: TITLE-ABS-KEY ( coffee ) AND NOT TITLE-ABS-KEY ( "coffee shop*" )'})
        assert "shop" not in t and t == "coffee"
    finally:
        S.cfg.clear(); S.cfg.update(old)
