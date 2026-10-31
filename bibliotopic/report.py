"""Borrador del manuscrito (Métodos y Resultados) en español; título, figuras y tablas en inglés."""
from __future__ import annotations

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

REFS = {
    "aria": "Aria, M., & Cuccurullo, C. (2017). bibliometrix: An R-tool for comprehensive science mapping analysis. Journal of Informetrics, 11(4), 959–975. https://doi.org/10.1016/j.joi.2017.08.007",
    "arun": "Arun, R., Suresh, V., Veni Madhavan, C. E., & Narasimha Murthy, M. N. (2010). On finding the natural number of topics with latent Dirichlet allocation: Some observations. In Advances in Knowledge Discovery and Data Mining (PAKDD 2010), LNCS 6118 (pp. 391–402). Springer.",
    "blei": "Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. Journal of Machine Learning Research, 3, 993–1022.",
    "blondel": "Blondel, V. D., Guillaume, J.-L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment, 2008(10), P10008.",
    "bouma": "Bouma, G. (2009). Normalized (pointwise) mutual information in collocation extraction. Proceedings of the Biennial GSCL Conference, 31–40.",
    "bradford": "Bradford, S. C. (1934). Sources of information on specific subjects. Engineering, 137, 85–86.",
    "campello": "Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. In Advances in Knowledge Discovery and Data Mining (PAKDD 2013), LNCS 7819 (pp. 160–172). Springer.",
    "cao": "Cao, J., Xia, T., Li, J., Zhang, Y., & Tang, S. (2009). A density-based method for adaptive LDA model selection. Neurocomputing, 72(7–9), 1775–1781.",
    "deveaud": "Deveaud, R., SanJuan, E., & Bellot, P. (2014). Accurate and effective latent concept modeling for ad hoc information retrieval. Document Numérique, 17(1), 61–84.",
    "dieng": "Dieng, A. B., Ruiz, F. J. R., & Blei, D. M. (2020). Topic modeling in embedding spaces. Transactions of the Association for Computational Linguistics, 8, 439–453.",
    "gabriel": "Gabriel, K. R. (1971). The biplot graphic display of matrices with application to principal component analysis. Biometrika, 58(3), 453–467.",
    "galindo": "Galindo, M. P. (1986). Una alternativa de representación simultánea: HJ-Biplot. Qüestiió, 10(1), 13–23.",
    "griffiths": "Griffiths, T. L., & Steyvers, M. (2004). Finding scientific topics. Proceedings of the National Academy of Sciences, 101(suppl. 1), 5228–5235.",
    "grootendorst": "Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.",
    "hoffman": "Hoffman, M., Bach, F., & Blei, D. (2010). Online learning for latent Dirichlet allocation. Advances in Neural Information Processing Systems, 23, 856–864.",
    "hubert": "Hubert, L., & Arabie, P. (1985). Comparing partitions. Journal of Classification, 2(1), 193–218.",
    "kuhn": "Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1–2), 83–97.",
    "lotka": "Lotka, A. J. (1926). The frequency distribution of scientific productivity. Journal of the Washington Academy of Sciences, 16(12), 317–323.",
    "mcinnes17": "McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. Journal of Open Source Software, 2(11), 205.",
    "mcinnes18": "McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426.",
    "mimno": "Mimno, D., Wallach, H. M., Talley, E., Leenders, M., & McCallum, A. (2011). Optimizing semantic coherence in topic models. Proceedings of EMNLP 2011, 262–272.",
    "montes23": "Montes-Escobar, K., De la Hoz-M, J., Barreiro-Linzán, M. D., Fonseca-Restrepo, C., Lapo-Palacios, M. Á., Verduga-Alcívar, D. A., & Salas-Macias, C. A. (2023). Trends in agroforestry research from 1993 to 2022: A topic model using latent Dirichlet allocation and HJ-Biplot. Mathematics, 11(10), 2250. https://doi.org/10.3390/math11102250",
    "abril24": "Abril-Espin, E., Montes-Escobar, K., de la Hoz-M, J., Garcés-Fiallos, F. R., & Salas-Macias, C. A. (2024). Exploring global trends in scientific research on Rubus glaucus Benth.: A comprehensive analysis integrating bibliometrics, LDA, and HJ-Biplot. Journal of Berry Research, 14(4), 287–300. https://doi.org/10.1177/18785093241284955",
    "salas26": "Salas-Macías, C. A., Montes-Escobar, K., Zamora-Ledezma, E., Fonseca-Restrepo, C., & de la Hoz-M, J. (2026). The knowledge architecture of carbon markets: A 30-year global view. Social Sciences & Humanities Open, 13, 102827.",
    "montes26": "Montes-Escobar, K., Jarre-Castro, E., de la Hoz-M, J., Salas-Macias, C. A., Macias-Barberán, J., Intriago-Flor, F., Fonseca-Restrepo, C., & Pérez-Parra, J. C. (2026). Microplastics bioaccumulation and trophic transfer research (2010–2025): A bibliometric analysis using LDA and HJ-Biplot. Microplastics, 5(4), 213. https://doi.org/10.3390/microplastics5040213",
    "page": "Page, M. J., McKenzie, J. E., Bossuyt, P. M., Boutron, I., Hoffmann, T. C., Mulrow, C. D., et al. (2021). The PRISMA 2020 statement: An updated guideline for reporting systematic reviews. BMJ, 372, n71. https://doi.org/10.1136/bmj.n71",
    "pedregosa": "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "reimers": "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. Proceedings of EMNLP-IJCNLP 2019, 3982–3992.",
    "vaneck": "van Eck, N. J., & Waltman, L. (2009). How to normalize cooccurrence data? An analysis of some well-known similarity measures. Journal of the American Society for Information Science and Technology, 60(8), 1635–1651. https://doi.org/10.1002/asi.21075",
    "ward": "Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. Journal of the American Statistical Association, 58(301), 236–244. https://doi.org/10.1080/01621459.1963.10500845",
    "rousseeuw": "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7",
    "strehl": "Strehl, A., & Ghosh, J. (2002). Cluster ensembles: A knowledge reuse framework for combining multiple partitions. Journal of Machine Learning Research, 3, 583–617.",
}


def f(x, d=2):
    """Número con punto decimal y separador de miles con espacio fino."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n.d."
    if isinstance(x, (int, np.integer)) or float(x).is_integer() and d == 0:
        return f"{int(x):,}".replace(",", " ")
    return f"{x:,.{d}f}".replace(",", " ")


def enum_es(items):
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " y " + items[-1]


def pl(n, sing, plur):
    return f"{n} {sing if n == 1 else plur}"


def _short(label):
    return label.split(":", 1)[1].strip() if ":" in label else label


class Writer:
    def __init__(self, title):
        self.doc = Document()
        st = self.doc.styles["Normal"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(12)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        for s in self.doc.sections:
            s.left_margin = s.right_margin = Cm(2.5)
        self.fig_n, self.tab_n = 0, 0
        self.fig_no, self.tab_no = {}, {}
        p = self.doc.add_paragraph()
        r = p.add_run(title)
        r.bold = True
        r.font.size = Pt(15)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def note(self, text):
        p = self.doc.add_paragraph()
        r = p.add_run(text)
        r.italic = True
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        return p

    def h(self, text, level=1):
        self.doc.add_heading(text, level=level)

    def p(self, *chunks):
        """chunks: str o ('texto', 'hl') para resaltar lo que los autores deben revisar."""
        par = self.doc.add_paragraph()
        par.paragraph_format.first_line_indent = Cm(0.75)
        par.paragraph_format.line_spacing = 1.5
        par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        for c in chunks:
            if isinstance(c, tuple):
                r = par.add_run(c[0])
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            else:
                par.add_run(c)
        return par

    def reserve(self, figs, tabs):
        for k in figs:
            self.fig_n += 1
            self.fig_no[k] = self.fig_n
        for k in tabs:
            self.tab_n += 1
            self.tab_no[k] = self.tab_n

    def F(self, k):
        return f"Figura {self.fig_no[k]}"

    def T(self, k):
        return f"Tabla {self.tab_no[k]}"

    def figure(self, key, path, caption, width=15.5):
        self.doc.add_picture(path, width=Cm(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        p = self.doc.add_paragraph()
        r = p.add_run(f"Figure {self.fig_no[key]}. ")
        r.bold = True
        p.add_run(caption).font.size = Pt(10)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    def table(self, key, df: pd.DataFrame, caption, note=None, font=8.5):
        p = self.doc.add_paragraph()
        r = p.add_run(f"Table {self.tab_no[key]}. ")
        r.bold = True
        p.add_run(caption)
        t = self.doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Table Grid"
        for i, c in enumerate(df.columns):
            cell = t.rows[0].cells[i]
            cell.text = str(c)
            for run in cell.paragraphs[0].runs:
                run.bold = True
                run.font.size = Pt(font)
            sh = OxmlElement("w:shd")
            sh.set(qn("w:fill"), "DCE6F0")
            cell._tc.get_or_add_tcPr().append(sh)
        for row in df.itertuples(index=False):
            cells = t.add_row().cells
            for i, v in enumerate(row):
                if isinstance(v, float):
                    v = "" if np.isnan(v) else (f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}")
                cells[i].text = str(v)
                for run in cells[i].paragraphs[0].runs:
                    run.font.size = Pt(font)
        if note:
            q = self.doc.add_paragraph()
            q.add_run(note).font.size = Pt(9)
        self.doc.add_paragraph()


def _cluster_text(hj, attr="col_clusters", idx=None):
    lab = getattr(hj, attr, None)
    if lab is None:
        return ""
    names = list(hj.cols.index if attr == "col_clusters" else hj.rows.index)
    groups = {}
    for n_, k in zip(names, lab):
        groups.setdefault(int(k), []).append(str(n_))
    return "; ".join(f"conglomerado {k + 1}: {', '.join(v)}" for k, v in sorted(groups.items()))


def _biplot_associations(hj, top=3):
    """Para cada fila (periodo/país), columnas (temas) más alineadas en el plano 1-2."""
    R = hj.rows.iloc[:, :2].values
    C = hj.cols.iloc[:, :2].values
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    cos = Rn @ Cn.T
    return {hj.rows.index[i]: [hj.cols.index[j] for j in np.argsort(-cos[i])[:top]]
            for i in range(len(R))}


SECTIONS = {
    "methods": "Materiales y métodos",
    "results": "Resultados",
    "tables": "Tablas (en inglés)",
    "figures": "Figuras (en inglés)",
    "references": "Referencias de los métodos",
}

CITE_NAMES = {
    "aria": "Aria y Cuccurullo, 2017", "arun": "Arun et al., 2010", "blei": "Blei et al., 2003",
    "blondel": "Blondel et al., 2008", "bouma": "Bouma, 2009", "bradford": "Bradford, 1934",
    "campello": "Campello et al., 2013", "cao": "Cao et al., 2009", "deveaud": "Deveaud et al., 2014",
    "dieng": "Dieng et al., 2020", "gabriel": "Gabriel, 1971", "galindo": "Galindo, 1986",
    "griffiths": "Griffiths y Steyvers, 2004", "grootendorst": "Grootendorst, 2022",
    "hoffman": "Hoffman et al., 2010", "hubert": "Hubert y Arabie, 1985", "kuhn": "Kuhn, 1955",
    "lotka": "Lotka, 1926", "mcinnes18": "McInnes et al., 2018", "mimno": "Mimno et al., 2011",
    "montes23": "Montes-Escobar et al., 2023", "abril24": "Abril-Espin et al., 2024",
    "salas26": "Salas-Macías et al., 2026", "montes26": "Montes-Escobar et al., 2026",
    "page": "Page et al., 2021", "pedregosa": "Pedregosa et al., 2011",
    "reimers": "Reimers y Gurevych, 2019", "strehl": "Strehl y Ghosh, 2002",
    "vaneck": "van Eck y Waltman, 2009", "ward": "Ward, 1963", "rousseeuw": "Rousseeuw, 1987",
}

FIG_CAPTIONS = {
    "prisma": "Flow diagram of record identification, deduplication and screening (adapted from PRISMA 2020).",
    "annual": "Annual scientific production.",
    "sources": "Most relevant sources by number of documents.",
    "authors": "Most productive authors.",
    "countries": "Most productive countries by corresponding author; SCP: single-country publications; MCP: multiple-country publications.",
    "collab_map": "Country collaboration map. Fill: number of documents (log scale); lines: strongest international co-authorship links.",
    "collab": "Country collaboration network. Node size: number of documents; edge width: number of co-authored documents; colours: Louvain communities; layout based on association strength.",
    "kw": "Keyword co-occurrence network. Node size: occurrences; edge width: co-occurrences; colours: Louvain communities; layout based on association strength.",
    "ksel": "Selection of the number of LDA topics (K) with four normalised metrics. Dashed line: selected K.",
    "lda_words": "Top terms of the LDA topics (word–topic probability).",
    "lda_time": "Mean prevalence of LDA topics by period.",
    "hj_lda": "HJ-Biplot of periods (squares, joined chronologically) and LDA topics (vectors). Colours and shaded hulls: topic clusters (Ward linkage, number of clusters selected by silhouette).",
    "bert_words": "Top terms of the BERTopic topics (c-TF-IDF weight).",
    "bert_map": "Two-dimensional UMAP projection of document embeddings coloured by BERTopic topic; grey: outliers.",
    "bert_time": "Prevalence of BERTopic topics by period.",
    "hj_bert": "HJ-Biplot of periods (squares, joined chronologically) and BERTopic topics (vectors). Colours and shaded hulls: topic clusters (Ward linkage, silhouette).",
    "sim": "Cosine similarity between LDA and BERTopic topics; white boxes mark the optimal one-to-one matching.",
    "cont": "Document-level agreement: proportion of documents of each LDA dominant topic assigned to each BERTopic topic.",
    "trends": "Linear trend of the annual prevalence of each topic. Red: hot (p < 0.05, positive slope); blue: cold (p < 0.05, negative slope).",
    "hj_country": "HJ-Biplot of the most productive countries (points, coloured by country cluster) and LDA topics (vectors, coloured by topic cluster).",
    "strategic": "Strategic map of LDA topics: growth (slope) versus impact (citations per year); bubble size: prevalence.",
}
FIG_ORDER = list(FIG_CAPTIONS)

TAB_CAPTIONS = {
    "T1_main_information": ("Main information about the corpus.", None),
    "T2_top_sources": ("Most relevant sources.", None),
    "T4_countries": ("Most productive countries (corresponding author).",
                     "SCP: single-country publications; MCP: multiple-country publications."),
    "T6_LDA_topics": ("Topics identified by LDA.",
                      "Prevalence: mean document–topic probability. Stability: mean cosine similarity "
                      "with matched topics across random seeds."),
    "T7_BERTopic_topics": ("Topics identified by BERTopic.", "Prevalence computed after outlier reassignment."),
    "T8_model_metrics": ("Comparison of model quality metrics.", None),
    "T9_topic_alignment": ("Alignment between LDA and BERTopic topics (Hungarian matching).",
                           "TRI: Topic Robustness Index = cosine × √(stability_LDA × stability_BERTopic)."),
}
TAB_ORDER = list(TAB_CAPTIONS)


def build(R, path, meta, sections=None):
    """meta: topic_name, title, query, search_date, embedding_tool.  sections: subconjunto de SECTIONS."""
    sections = set(sections or SECTIONS)
    cfg, df, lda, bert, T, FIG, X = R.cfg, R.df, R.lda, R.bert, R.tables, R.figs, R.facts
    C, info, ap = X.get("C"), X["info"], X["ap"]
    ms = R.merge_stats
    stab_on = R.on("stability")
    topic = meta.get("topic_name") or "[TOPIC]"
    models = " and ".join(m for m, r in (("LDA", lda), ("BERTopic", bert)) if r is not None)
    title = meta.get("title") or (
        f"Mapping the scientific landscape of {topic}: bibliometrics"
        + (f" and {'a comparative ' if lda is not None and bert is not None else ''}topic modelling "
           f"using {models}" if models else " analysis"))
    W = Writer(title)
    W.note("Borrador generado automáticamente a partir de los datos. Los textos resaltados en amarillo "
           "requieren revisión de los autores expertos en la temática (en particular, las etiquetas "
           "de los temas). Título, figuras y tablas en inglés; texto en español.")
    W.reserve([k for k in FIG_ORDER if k in FIG], [k for k in TAB_ORDER if k in T])
    used = set()

    def cite(*keys):
        used.update(keys)
        return "(" + "; ".join(CITE_NAMES[k] for k in keys) + ")"

    def ref(*keys):
        """'(Tabla 2; Figura 3)' solo con los elementos que existen."""
        parts = [W.T(k) if k in W.tab_no else W.F(k) for k in keys if k in W.tab_no or k in W.fig_no]
        return f" ({'; '.join(parts)})" if parts else ""

    def tl_(labels):
        return "; ".join(f"{l.split(':')[0]}: {_short(l)}" for l in labels)

    def num():
        n = 0
        while True:
            n += 1
            yield n

    by_db = ms["records_by_db"]
    dbs_txt = enum_es([{"WoS": "Web of Science Core Collection (WoS)", "Scopus": "Scopus"}[k] for k in by_db])
    py = df["PY"].astype(int)
    n = len(df)
    trends_l, trends_b = X.get("trends_lda"), X.get("trends_bert")

    # ================================================================ MÉTODOS
    if "methods" in sections:
        s = num()
        W.h("2. Materiales y métodos", 1)
        W.h(f"2.{next(s)}. Fuente de datos y estrategia de búsqueda", 2)
        q = meta.get("query") or "[ecuación de búsqueda]"
        date = meta.get("search_date") or "[fecha]"
        W.p(f"La información se obtuvo de {dbs_txt}, bases de datos de referencia para estudios "
            "bibliométricos por su cobertura multidisciplinar y la calidad de sus metadatos. La búsqueda se "
            "realizó el ", (date, "hl"), " mediante la siguiente ecuación booleana aplicada a título, resumen "
            "y palabras clave: ", (q, "hl"), ". Los registros se exportaron con todos sus campos disponibles ("
            + "; ".join(f"{k}: {f(v, 0)} registros" for k, v in by_db.items()) + ").")

        W.h(f"2.{next(s)}. Depuración y fusión del corpus", 2)
        es = {"Other document types": "otros tipos de documento", "Other languages": "otros idiomas",
              "Missing/short abstract (<20 words)": "registros sin resumen o con menos de 20 palabras"}
        excl_txt = enum_es([f"{es.get(name, 'registros fuera del periodo ' + name.split()[-1])} "
                            f"(n = {f(k, 0)})" for name, k in R.excl_steps if k])
        filt = []
        if cfg.get("doc_types") and R.availability["DT"] > 0:
            filt.append("tipos de documento " + enum_es(cfg["doc_types"]))
        if cfg.get("languages") and R.availability["LA"] > 0:
            filt.append("idioma " + enum_es(cfg["languages"]))
        W.p(("Los registros de ambas fuentes se normalizaron a un esquema común y se fusionaron. "
             if len(by_db) > 1 else "Los registros de los distintos lotes se normalizaron y se unificaron. ")
            + "Los duplicados se identificaron primero por coincidencia del DOI y, en ausencia de este, por "
            "coincidencia del título normalizado (minúsculas, sin signos ni tildes) y el año de publicación; de "
            "cada grupo de duplicados se conservó el registro con más campos completos y el mayor número de "
            f"citas. Se eliminaron {f(ms['duplicates_removed'], 0)} duplicados, quedando {f(ms['unique'], 0)} "
            "registros únicos. Como criterios de elegibilidad se exigió "
            + (enum_es(filt) + " y " if filt else "")
            + f"un resumen de al menos 20 palabras. Se excluyeron {excl_txt or 'ningún registro'}, de modo que "
            f"el corpus final quedó integrado por {f(n, 0)} documentos publicados entre {py.min()} y {py.max()}. "
            f"El proceso se resume en un diagrama de flujo adaptado de PRISMA 2020 {cite('page')}{ref('prisma')}.")

        if R.on("biblio") or R.has_countries or X.get("kw_field"):
            W.h(f"2.{next(s)}. Análisis bibliométrico", 2)
            extra = []
            if R.has_countries:
                extra.append("la producción por país del autor de correspondencia, distinguiendo publicaciones "
                             "de un solo país (SCP) y de colaboración internacional (MCP), el mapa mundial de "
                             "colaboración y la red de colaboración entre países")
            if X.get("kw_field"):
                extra.append("la red de co-ocurrencia de las palabras clave más frecuentes")
            if R.has_countries or X.get("kw_field"):
                extra[-1] += (". En las redes, los vínculos se normalizaron mediante la fuerza de asociación "
                              f"{cite('vaneck')}, la disposición se obtuvo con el algoritmo de Fruchterman-Reingold "
                              "y las comunidades se identificaron con el algoritmo de Louvain " + cite("blondel"))
            W.p("Se calcularon los indicadores descriptivos habituales del análisis de mapeo científico "
                f"{cite('aria')}: tasa de crecimiento anual compuesta, edad media de los documentos, autores, "
                "coautores por documento, fuentes más relevantes, zonas de Bradford " + cite("bradford")
                + " y distribución de la productividad de autores según la ley de Lotka " + cite("lotka")
                + (". Además, se analizaron " + enum_es(extra) if extra else "")
                + ". Los indicadores se implementaron en Python siguiendo las definiciones de bibliometrix.")

        if lda is not None or bert is not None:
            W.h(f"2.{next(s)}. Preprocesamiento del texto", 2)
            W.p("Para el modelado de temas se concatenó el título y el resumen de cada documento. El texto se "
                "convirtió a minúsculas y se eliminaron números, signos de puntuación, las palabras vacías del "
                "inglés, un listado de términos académicos genéricos (p. ej., study, results, analysis) y los "
                "términos de la propia ecuación de búsqueda"
                + (f" ({enum_es(cfg['query_terms'])})" if cfg.get("query_terms") else "")
                + ", que por estar presentes en casi todos los documentos no aportan capacidad discriminante. "
                "Los términos se llevaron a su forma singular y se identificaron expresiones compuestas "
                "(bigramas) con información mutua puntual normalizada (NPMI) ≥ 0.35 y al menos 10 apariciones "
                f"{cite('bouma')}, lo que generó {f(len(R.corpus.phrases), 0)} términos compuestos. Se retuvieron "
                f"los términos presentes en al menos {cfg['min_df']} documentos y en no más del "
                f"{int(100 * cfg['max_df'])} % de ellos, con un vocabulario final de "
                f"{f(len(R.corpus.vocab), 0)} términos"
                + (". El mismo vocabulario se utilizó en ambos modelos para que la comparación fuera directa."
                   if lda is not None and bert is not None else "."))

        if lda is not None:
            W.h(f"2.{next(s)}. Modelado de temas con LDA", 2)
            ksel_txt = ""
            if R.ksel is not None:
                ks = R.ksel["K"]
                ksel_txt = (f"El número de temas (K) se evaluó entre {ks.min()} y {ks.max()} (paso "
                            f"{cfg['k_step']}) con cuatro métricas complementarias: CaoJuan2009 {cite('cao')} y "
                            f"Arun2010 {cite('arun')}, a minimizar, y Deveaud2014 {cite('deveaud')} y la "
                            "coherencia NPMI de las diez palabras principales de cada tema "
                            f"{cite('bouma')}, a maximizar. Las métricas se normalizaron entre 0 y 1 y se "
                            "seleccionó el K con el mayor promedio. ")
            else:
                ksel_txt = f"El número de temas se fijó en K = {lda.k}. "
            W.p(f"La asignación latente de Dirichlet (LDA) {cite('blei')} representa cada documento como una "
                "mezcla de temas y cada tema como una distribución de probabilidad sobre el vocabulario. El "
                f"modelo se estimó por inferencia variacional en línea {cite('hoffman')} con scikit-learn "
                f"{cite('pedregosa')}. " + ksel_txt
                + (f"Para evaluar la estabilidad, el modelo final se reestimó con {cfg['n_stab_lda']} semillas "
                   "aleatorias distintas; cada tema se emparejó con su homólogo mediante el algoritmo húngaro "
                   f"{cite('kuhn')} y se calculó la similitud coseno media entre réplicas como índice de "
                   "estabilidad." if stab_on else ""))

        if bert is not None:
            W.h(f"2.{next(s)}. Modelado de temas con BERTopic", 2)
            lsa = "lsa" in R.emb_name.lower()
            emb_txt = ("una representación semántica latente (TF-IDF + SVD) [REEMPLAZAR si se usó un modelo de "
                       "lenguaje]" if lsa else
                       f"el modelo de lenguaje preentrenado {R.emb_name} {cite('reimers')}")
            W.p(f"BERTopic {cite('grootendorst')} agrupa documentos semánticamente similares a partir de "
                f"representaciones vectoriales densas. Los títulos y resúmenes se codificaron con {emb_txt}; "
                f"la dimensionalidad se redujo a cinco componentes con UMAP {cite('mcinnes18')} (15 vecinos) y "
                f"los documentos se agruparon con HDBSCAN {cite('campello')} (tamaño mínimo de grupo = "
                f"{bert.min_cluster_size}). Cada tema se describió mediante TF-IDF basado en clases "
                "(c-TF-IDF) con reducción de palabras frecuentes"
                + (", sobre el mismo vocabulario usado en LDA" if lda is not None else "") + ". "
                + (f"HDBSCAN identificó {bert.native_topics} grupos, que se fusionaron por similitud c-TF-IDF "
                   f"(agrupamiento jerárquico con enlace promedio) hasta K = {bert.k}"
                   + (" para comparar ambos modelos con igual granularidad" if lda is not None else "") + ". "
                   if bert.native_topics != bert.k else "")
                + "Los documentos no asignados (outliers) se reasignaron al tema con mayor similitud c-TF-IDF "
                "para los análisis por documento."
                + (f" La estabilidad se evaluó con {bert.n_stab} semillas de UMAP distintas, emparejando los "
                   "temas con el algoritmo húngaro." if bert.n_stab else "")
                + f" El procedimiento se ejecutó en el navegador mediante {meta.get('embedding_tool', 'transformers.js y umap-js')}.")

        if (R.on("temporal") or R.on("hjbiplot")) and (lda is not None or bert is not None):
            W.h(f"2.{next(s)}. Dinámica temporal" + (" y HJ-Biplot" if R.on("hjbiplot") else ""), 2)
            txt = ""
            if R.on("temporal"):
                txt += (f"La evolución de cada tema se estimó mediante la regresión lineal de su prevalencia media "
                        f"anual sobre el año de publicación {cite('griffiths')}, considerando los años con al "
                        f"menos {cfg['min_docs_year']} documentos; los temas con pendiente significativa "
                        "(p < 0.05) se clasificaron como emergentes (hot) o en declive (cold). ")
            if R.on("hjbiplot"):
                txt += (f"Para analizar conjuntamente temas y tiempo se agruparon los años en periodos "
                        f"({enum_es(X['periods'])}) y se construyó una matriz periodo × tema con la prevalencia "
                        f"media, a la que se aplicó un HJ-Biplot {cite('galindo', 'gabriel')}. Este método "
                        "representa filas y columnas en un mismo sistema de ejes con la máxima calidad de "
                        "representación; los ángulos agudos entre vectores de temas indican correlación "
                        "positiva, y la proximidad entre un periodo y la dirección de un tema indica una mayor "
                        "prevalencia de ese tema en dicho periodo. Para identificar grupos de temas con "
                        "comportamiento similar se aplicó un análisis de conglomerados jerárquico con el método "
                        f"de Ward {cite('ward')} sobre las coordenadas de los temas en el plano factorial; el "
                        f"número de conglomerados se eligió con el coeficiente de silueta {cite('rousseeuw')}."
                        + (" El mismo procedimiento se aplicó a una matriz país × tema con los países más "
                           "productivos." if "hj_country" in FIG else "")
                        + " Este enfoque se ha empleado previamente en estudios bibliométricos con LDA "
                        f"{cite('montes23', 'abril24', 'salas26', 'montes26')}.")
            W.p(txt)

        if C is not None:
            W.h(f"2.{next(s)}. Comparación entre LDA y BERTopic", 2)
            W.p("Ambos modelos se compararon en tres niveles. (i) Calidad intrínseca: coherencia NPMI y UMass "
                f"{cite('bouma', 'mimno')}, diversidad de temas (proporción de términos únicos entre las 25 "
                f"palabras principales de todos los temas) {cite('dieng')}"
                + (", estabilidad entre semillas" if stab_on else "") + " y proporción de outliers. "
                "(ii) Correspondencia entre temas: se calculó la similitud coseno entre la distribución "
                "palabra-tema de LDA y el vector c-TF-IDF de BERTopic y se obtuvo el emparejamiento óptimo uno a "
                f"uno con el algoritmo húngaro {cite('kuhn')}; también se informa el índice de Jaccard entre las "
                "diez palabras principales. (iii) Coincidencia a nivel de documento: índice de Rand ajustado "
                f"(ARI) {cite('hubert')} e información mutua normalizada (NMI) {cite('strehl')} entre el tema "
                "dominante de LDA y el tema asignado por BERTopic. "
                + ("Con estos elementos se propone un Índice de Robustez Temática (TRI) para cada par "
                   "emparejado: TRI = cos(LDA, BERTopic) × √(estabilidad LDA × estabilidad BERTopic). "
                   if stab_on else "")
                + f"Un par se consideró convergente cuando la similitud fue ≥ {cfg['tau_strong']}, parcial "
                f"cuando estuvo entre {cfg['tau_partial']} y {cfg['tau_strong']}, y divergente en otro caso."
                + (" Finalmente, se evaluó la concordancia en la clasificación temporal (emergente, estable o en "
                   "declive) de los temas emparejados." if R.on("temporal") else ""))

        W.h(f"2.{next(s)}. Reproducibilidad", 2)
        v = R.versions
        W.p("Todos los análisis se realizaron con la aplicación web BiblioTopic, que ejecuta Python "
            f"{v.get('python')} en el navegador (Pyodide) con pandas {v.get('pandas')}, scikit-learn "
            f"{v.get('scikit-learn')}, SciPy {v.get('scipy')}, networkx {v.get('networkx')} y matplotlib "
            f"{v.get('matplotlib')}. Se fijó la semilla aleatoria {cfg['seed']}. La configuración completa, las "
            "tablas intermedias y la asignación de temas por documento se proporcionan como material "
            "suplementario.")

    # ============================================================== RESULTADOS
    if "results" in sections:
        W.h("3. Resultados", 1)
        sec = num()
        W.h(f"3.{next(sec)}. Descripción general del corpus", 2)
        peak = ap.loc[ap["Articles"].idxmax()]
        recent = py[py >= py.max() - 9]
        W.p(f"El corpus final incluyó {f(n, 0)} documentos publicados en "
            f"{info['Sources (journals, books, etc.)']} fuentes por {f(info['Authors'], 0)} autores entre "
            f"{py.min()} y {py.max()}{ref('T1_main_information')}. La producción creció a una tasa anual "
            f"compuesta de {f(next(v for k, v in info.items() if k.startswith('Annual growth')))} % "
            f"({next(k for k in info if k.startswith('Annual growth')).split()[3]}), con {f(info['Co-authors per document'])} "
            f"coautores por documento y {f(info['Single-authored documents'], 0)} documentos de autoría única. "
            f"El año más productivo fue {int(peak['Year'])} ({f(peak['Articles'], 0)} documentos). La mitad del "
            f"corpus se publicó a partir de {int(np.median(py))} y el {f(100 * len(recent) / n, 1)} % en los "
            f"últimos diez años ({py.max() - 9}-{py.max()}){ref('annual')}."
            + (f" El promedio de citas por documento fue de {f(info['Average citations per document'])}."
               if "Average citations per document" in info else ""))
        ov = ms.get("overlap", {})
        if "WoS+Scopus" in ov:
            W.p(f"Del total de registros únicos, {f(ov.get('WoS+Scopus', 0), 0)} estaban indexados en ambas "
                f"bases, {f(ov.get('WoS', 0), 0)} solo en WoS y {f(ov.get('Scopus', 0), 0)} solo en Scopus, lo "
                "que justifica el uso combinado de ambas fuentes.")

        if "T2_top_sources" in T:
            W.h(f"3.{next(sec)}. Fuentes y autores", 2)
            src = T["T2_top_sources"]
            W.p("Las fuentes con más publicaciones fueron "
                + enum_es([f"{s_.title()} (n = {a})" for s_, a in src[["Source", "Articles"]].values[:3]])
                + f"{ref('T2_top_sources', 'sources')}. Según la ley de Bradford, {X['bradford_zone1']} de las "
                f"{f(X['n_sources'], 0)} fuentes ({f(100 * X['bradford_zone1'] / X['n_sources'], 1)} %) "
                "concentraron un tercio de la producción (zona núcleo). Los autores más productivos fueron "
                + enum_es([f"{a} ({k})" for a, k in T["T3_top_authors"][["Author", "Articles"]].values[:5]])
                + f"{ref('authors')}. En cuanto a la ley de Lotka, el "
                f"{f(100 * T['lotka']['Proportion of authors'].iloc[0], 1)} % de los autores publicó un solo "
                "documento.")

        if R.has_countries:
            W.h(f"3.{next(sec)}. Países y colaboración internacional", 2)
            cp = T["T4_countries"]
            top_mcp = cp.head(10).sort_values("MCP ratio", ascending=False)[["Country", "MCP ratio"]].values[:3]
            W.p("Los países con mayor producción, según el país del autor de correspondencia, fueron "
                + enum_es([f"{c} (n = {a})" for c, a in cp[["Country", "Articles"]].values[:5]])
                + f"{ref('T4_countries', 'countries', 'collab_map')}. El "
                f"{f(info.get('International co-authorship (%)', 0))} % de los documentos tuvo coautoría "
                "internacional. Entre los diez principales, la mayor proporción de colaboración internacional "
                "(MCP) correspondió a " + enum_es([f"{c} ({f(100 * r, 1)} %)" for c, r in top_mcp]) + ".")
            cl = X.get("collab_clusters", [])
            if "collab" in FIG and cl:
                W.p((f"La red de colaboración{ref('collab')} mostró {len(cl)} comunidades; las principales "
                     f"estuvieron encabezadas por {enum_es([', '.join(c[:4]) for c in cl[:3]])}.")
                    if len(cl) > 1 else
                    f"La red de colaboración{ref('collab')} formó una única comunidad articulada en torno a "
                    f"{', '.join(cl[0][:4])}.")
        elif R.on("countries"):
            W.note("[Países y colaboración: los archivos no contienen afiliaciones (campo C1). Exporte desde WoS "
                   "con 'Full Record and Cited References' y desde Scopus con 'Affiliations'.]")

        if "kw" in FIG:
            W.h(f"3.{next(sec)}. Palabras clave", 2)
            kf, kc = T["T5_keywords"], X["kw_clusters"]
            W.p("Las palabras clave más frecuentes fueron "
                + enum_es([f"{k} ({o})" for k, o in kf.values[:8]])
                + f". La red de co-ocurrencia{ref('kw')} se organizó en {len(kc)} "
                f"{'grupos' if len(kc) > 1 else 'grupo'}; los principales reunieron "
                + "; ".join(f"«{', '.join(c[:4])}»" for c in kc[:3]) + ".")

        last_p = X["periods"][-1]
        first_p = X["periods"][0]
        if lda is not None:
            W.h(f"3.{next(sec)}. Temas identificados con LDA", 2)
            order = np.argsort(-lda.prevalence)
            txt = (f"El criterio combinado de las cuatro métricas indicó K = {lda.k} como número óptimo de "
                   f"temas{ref('ksel')}; " if R.ksel is not None else f"Con K = {lda.k} temas, ")
            txt += (f"la coherencia NPMI media fue {f(np.nanmean(lda.npmi), 3)} y la diversidad "
                    f"{f(lda.diversity, 3)}"
                    + (f"; la estabilidad media entre semillas fue {f(np.nanmean(lda.stab), 3)}" if stab_on else "")
                    + ". ")
            if R.ksel is not None and lda.k == R.ksel["K"].max():
                txt += ("La métrica combinada alcanzó su máximo en el límite superior del intervalo evaluado; "
                        "se recomienda ampliar el intervalo de K. ")
            txt += f"Los temas, sus términos principales y su prevalencia se presentan a continuación{ref('T6_LDA_topics', 'lda_words')}. "
            W.p(txt, ("[Revisar las etiquetas de los temas.]", "hl"))
            W.p("Los temas más prevalentes fueron " + "; ".join(
                f"{lda.labels[i]} ({f(100 * lda.prevalence[i], 1)} %), caracterizado por "
                f"{', '.join(lda.top10[i][:6])}" for i in order[:3])
                + ". Los menos prevalentes fueron "
                + enum_es([f"{lda.labels[i]} ({f(100 * lda.prevalence[i], 1)} %)" for i in order[-2:]]) + ".")
            if R.on("temporal") and trends_l is not None:
                hot = [lda.labels[i] for i in range(lda.k) if trends_l["Trend"][i] == "Hot"]
                cold = [lda.labels[i] for i in range(lda.k) if trends_l["Trend"][i] == "Cold"]
                W.p(f"Respecto a la dinámica temporal{ref('lda_time', 'trends')}, "
                    + (f"{pl(len(hot), 'tema mostró', 'temas mostraron')} una tendencia creciente significativa ({tl_(hot)})"
                       if hot else "ningún tema mostró una tendencia creciente significativa")
                    + " y " + (f"{len(cold)} una tendencia decreciente ({tl_(cold)})"
                               if cold else "ninguno una tendencia decreciente") + ".")
            if "hj_l" in X:
                hj = X["hj_l"]
                a = _biplot_associations(hj)
                W.p(f"El HJ-Biplot periodo × tema de LDA{ref('hj_lda')} explicó el "
                    f"{f(100 * (hj.inertia[0] + hj.inertia[1]), 1)} % de la inercia en los dos primeros ejes "
                    f"(eje 1: {f(100 * hj.inertia[0], 1)} %; eje 2: {f(100 * hj.inertia[1], 1)} %). El periodo "
                    f"más antiguo ({first_p}) se asoció con {enum_es(a[first_p])}, mientras que el más reciente "
                    f"({last_p}) se asoció con {enum_es(a[last_p])}."
                    + (f" El análisis de conglomerados identificó {hj.k_cols} grupos de temas (silueta = "
                       f"{f(hj.sil_cols, 2)}): {_cluster_text(hj)}." if getattr(hj, "col_clusters", None) is not None else ""))

        if bert is not None:
            W.h(f"3.{next(sec)}. Temas identificados con BERTopic", 2)
            ob = np.argsort(-bert.prevalence)
            W.p(f"HDBSCAN identificó {pl(bert.native_topics, 'grupo semántico', 'grupos semánticos')}"
                + (f", que se redujeron a {bert.k} temas" if bert.native_topics != bert.k else "")
                + f"; el {f(100 * bert.outlier_rate, 1)} % de los documentos quedó inicialmente sin asignar "
                f"(outliers){ref('T7_BERTopic_topics', 'bert_words', 'bert_map')}. La coherencia NPMI media "
                f"fue {f(np.nanmean(bert.npmi), 3)} y la diversidad {f(bert.diversity, 3)}"
                + (f"; la estabilidad media fue {f(np.nanmean(bert.stab), 3)}" if bert.n_stab else "")
                + ". Los temas de mayor tamaño fueron " + "; ".join(
                    f"{bert.labels[i]} ({f(100 * bert.prevalence[i], 1)} %; {', '.join(bert.top10[i][:6])})"
                    for i in ob[:3]) + ".")
            txt = ""
            if R.on("temporal") and trends_b is not None:
                hot = [bert.labels[i] for i in range(bert.k) if trends_b["Trend"][i] == "Hot"]
                cold = [bert.labels[i] for i in range(bert.k) if trends_b["Trend"][i] == "Cold"]
                txt += (f"En el tiempo{ref('bert_time')}, "
                        + (f"{pl(len(hot), 'tema fue emergente', 'temas fueron emergentes')} ({tl_(hot)})" if hot else "ningún tema fue emergente")
                        + " y " + (f"{pl(len(cold), 'estuvo', 'estuvieron')} en declive ({tl_(cold)})" if cold
                                   else "ninguno estuvo en declive") + ". ")
            if "hj_b" in X:
                hjb = X["hj_b"]
                ab = _biplot_associations(hjb)
                txt += (f"El HJ-Biplot correspondiente{ref('hj_bert')} explicó el "
                        f"{f(100 * (hjb.inertia[0] + hjb.inertia[1]), 1)} % de la inercia; el periodo {last_p} "
                        f"se asoció con {enum_es(ab[last_p])}."
                        + (f" Los temas se organizaron en {hjb.k_cols} conglomerados (silueta = {f(hjb.sil_cols, 2)}): "
                           f"{_cluster_text(hjb)}." if getattr(hjb, "col_clusters", None) is not None else ""))
            if txt:
                W.p(txt)

        if C is not None:
            W.h(f"3.{next(sec)}. Comparación entre LDA y BERTopic", 2)
            mt = C["metrics"].set_index("Metric")
            ag = C["agreement"]
            m = C["matches"]
            rob, par, div = (m[m["Agreement"] == a] for a in ("Robust", "Partial", "Divergent"))

            def cmp(metric, name):
                a, b = mt.loc[metric, "LDA"], mt.loc[metric, "BERTopic"]
                who = "LDA" if a > b else "BERTopic"
                return f"{who} obtuvo mayor {name} ({f(a, 3)} en LDA frente a {f(b, 3)} en BERTopic)"
            parts = [cmp("Mean NPMI coherence", "coherencia NPMI"), cmp("Topic diversity (top-25)", "diversidad")]
            if stab_on:
                parts.append(cmp("Mean topic stability (cosine across seeds)", "estabilidad entre semillas"))
            W.p(f"La comparación de la calidad de ambos modelos{ref('T8_model_metrics')} mostró que "
                + enum_es(parts) + ".")

            def pairs(df_, k=4):
                return "; ".join(
                    f"{r['LDA topic']} – {r['BERTopic topic']} (coseno = {f(r['Cosine'], 2)}"
                    + (f"; TRI = {f(r['TRI'], 2)}" if stab_on else "")
                    + f"; términos compartidos: {r['Shared terms'] or '—'})"
                    for _, r in df_.sort_values("Cosine", ascending=False).head(k).iterrows())
            W.p(f"El emparejamiento óptimo entre temas{ref('sim', 'T9_topic_alignment')} identificó "
                f"{pl(len(rob), 'par convergente', 'pares convergentes')} (coseno ≥ {cfg['tau_strong']}), "
                f"{pl(len(par), 'parcial', 'parciales')} y {pl(len(div), 'divergente', 'divergentes')}. "
                + (f"Los pares convergentes fueron {pairs(rob)}. Estos temas, detectados de forma independiente "
                   "por un modelo probabilístico basado en frecuencias y por otro basado en semántica "
                   "contextual, constituyen el núcleo temático más sólido del campo. " if len(rob) else
                   f"Ningún par alcanzó el umbral de convergencia; los de mayor correspondencia fueron "
                   f"{pairs(par)}. " if len(par) else "")
                + ("Por el contrario, " + enum_es([f"{r['LDA topic'].split(':')[0]} y "
                                                   f"{r['BERTopic topic'].split(':')[0]}"
                                                   for _, r in div.head(3).iterrows()])
                   + " no encontraron un equivalente claro en el otro modelo, lo que sugiere temas sensibles al "
                   "método que deben interpretarse con cautela." if len(div) else ""))
            tag = ag.get("Trend agreement in matched topics", np.nan)
            W.p(f"A nivel de documento{ref('cont')}, la coincidencia entre la asignación de LDA y la de "
                f"BERTopic fue ARI = {f(ag['ARI (all documents)'], 3)} y NMI = {f(ag['NMI (all documents)'], 3)} "
                f"(excluyendo outliers: ARI = {f(ag['ARI (non-outliers)'], 3)}; NMI = "
                f"{f(ag['NMI (non-outliers)'], 3)})."
                + (f" En los temas emparejados, ambos modelos coincidieron en la clasificación temporal en el "
                   f"{f(100 * tag, 1)} % de los casos." if R.on("temporal") and not np.isnan(tag) else ""))

        if "hj_country" in FIG:
            hjc = X["hj_country"]
            ac = _biplot_associations(hjc, 2)
            W.h(f"3.{next(sec)}. Especialización temática de los países", 2)
            W.p(f"El HJ-Biplot país × tema{ref('hj_country')} explicó el "
                f"{f(100 * (hjc.inertia[0] + hjc.inertia[1]), 1)} % de la inercia. "
                + "; ".join(f"{c} se asoció con {enum_es(ac[c])}" for c in list(hjc.rows.index[:4])) + "."
                + (f" Los países se agruparon en {hjc.k_rows} conglomerados ({_cluster_text(hjc, 'row_clusters')})."
                   if getattr(hjc, "row_clusters", None) is not None else ""))
        if "strategic" in FIG:
            W.p(f"El mapa estratégico{ref('strategic')} cruza el crecimiento y el impacto (citas por año) de "
                "cada tema de LDA: los temas del cuadrante superior derecho combinan crecimiento e impacto "
                "elevados y representan frentes de investigación consolidados en expansión.")

    # ============================================================== TABLAS / FIGURAS
    if "tables" in sections and W.tab_no:
        W.doc.add_page_break()
        W.h("Tables", 1)
        for k in W.tab_no:
            t = T[k]
            if k == "T9_topic_alignment":
                cols = ["LDA topic", "BERTopic topic", "Cosine", "Jaccard top-10", "TRI", "Agreement",
                        "Trend LDA", "Trend BERTopic"]
                t = t[[c for c in cols if c in t.columns]]
            if k == "T2_top_sources":
                t = t.head(10)
            W.table(k, t, *TAB_CAPTIONS[k])
    if "figures" in sections and W.fig_no:
        W.doc.add_page_break()
        W.h("Figures", 1)
        for k in W.fig_no:
            cap = FIG_CAPTIONS[k]
            if k == "annual" and "MeanTCperYear" in ap:
                cap = "Annual scientific production and mean citations per year."
            W.figure(k, FIG[k], cap)
    if "references" in sections and used:
        W.h("Referencias (métodos)", 1)
        for k in sorted(used, key=lambda x: REFS[x]):
            p = W.doc.add_paragraph(REFS[k])
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
    W.doc.save(path)
    return path
