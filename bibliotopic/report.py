"""Borrador del manuscrito (Métodos y Resultados) en español; título, figuras y tablas en inglés."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

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
    "efron": "Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall/CRC.",
    "mann": "Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. The Annals of Mathematical Statistics, 18(1), 50–60.",
    "chang": "Chang, J., Boyd-Graber, J., Gerrish, S., Wang, C., & Blei, D. M. (2009). Reading tea leaves: How humans interpret topic models. Advances in Neural Information Processing Systems, 22, 288–296.",
    "hayes": "Hayes, A. F., & Krippendorff, K. (2007). Answering the call for a standard reliability measure for coding data. Communication Methods and Measures, 1(1), 77–89. https://doi.org/10.1080/19312450709336664",
    "callon": "Callon, M., Courtial, J. P., & Laville, F. (1991). Co-word analysis as a tool for describing the network of interactions between basic and technological research: The case of polymer chemistry. Scientometrics, 22(1), 155–205.",
    "cobo": "Cobo, M. J., López-Herrera, A. G., Herrera-Viedma, E., & Herrera, F. (2011). An approach for detecting, quantifying, and visualizing the evolution of a research field: A practical application to the Fuzzy Sets Theory field. Journal of Informetrics, 5(1), 146–166. https://doi.org/10.1016/j.joi.2010.10.002",
    "small": "Small, H. (1973). Co-citation in the scientific literature: A new measure of the relationship between two documents. Journal of the American Society for Information Science, 24(4), 265–269.",
    "montazeri": "Montazeri, A., Mohammadi, S., Hesari, P. M., Ghaemi, M., Riazi, H., & Sheikhi-Mobarakeh, Z. (2023). Preliminary guideline for reporting bibliometric reviews of the biomedical literature (BIBLIO): A minimum requirements. Systematic Reviews, 12, 239. https://doi.org/10.1186/s13643-023-02410-2",
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
        st.font.color.rgb = RGBColor(0, 0, 0)
        # Títulos y subtítulos en negro y en la misma fuente (los estilos por defecto de Word son azules)
        sizes = {"Title": 15, "Heading 1": 14, "Heading 2": 12, "Heading 3": 12, "Heading 4": 12}
        for name, size in sizes.items():
            hs = self.doc.styles[name]
            hs.font.name = "Times New Roman"
            hs.font.size = Pt(size)
            hs.font.bold = True
            hs.font.italic = name in ("Heading 3", "Heading 4")
            hs.font.color.rgb = RGBColor(0, 0, 0)
            rpr = hs.element.get_or_add_rPr()
            fonts = rpr.find(qn("w:rFonts"))
            if fonts is None:
                fonts = OxmlElement("w:rFonts")
                rpr.append(fonts)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                fonts.set(qn(a), "Times New Roman")
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                fonts.attrib.pop(qn(a), None)
            if name == "Title":
                pbdr = hs.element.get_or_add_pPr().find(qn("w:pBdr"))
                if pbdr is not None:
                    pbdr.getparent().remove(pbdr)
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

    def reserve_figs(self, keys, composites):
        """Numera figuras; las que forman parte de una compuesta comparten número con letra (3a, 3b)."""
        panel_of = {k: name for name, ks in composites.items() for k in ks}
        self.fig_entries, done = [], set()
        for k in keys:
            name = panel_of.get(k)
            if name:
                if name in done:
                    continue
                done.add(name)
                self.fig_n += 1
                for i, pk in enumerate(composites[name]):
                    self.fig_no[pk] = f"{self.fig_n}{chr(97 + i)}"
                self.fig_no[name] = str(self.fig_n)
                self.fig_entries.append(name)
            else:
                self.fig_n += 1
                self.fig_no[k] = str(self.fig_n)
                self.fig_entries.append(k)

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
    "discussion": "Discusión (borrador con IA; requiere clave de API)",
    "checklist": "Lista de verificación de reporte y declaración de disponibilidad",
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
    "efron": "Efron y Tibshirani, 1993", "mann": "Mann y Whitney, 1947", "chang": "Chang et al., 2009",
    "hayes": "Hayes y Krippendorff, 2007", "callon": "Callon et al., 1991", "cobo": "Cobo et al., 2011",
    "small": "Small, 1973", "montazeri": "Montazeri et al., 2023",
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
    "hj_lda": "HJ-Biplot of periods (squares, joined chronologically) and LDA topics (vectors). Colours and {shapes}: topic clusters (Ward linkage; number of clusters selected by silhouette among non-overlapping solutions).",
    "bert_words": "Top terms of the BERTopic topics (c-TF-IDF weight).",
    "bert_map": "Two-dimensional UMAP projection of document embeddings coloured by BERTopic topic; grey: outliers.",
    "bert_time": "Prevalence of BERTopic topics by period.",
    "hj_bert": "HJ-Biplot of periods (squares, joined chronologically) and BERTopic topics (vectors). Colours and {shapes}: topic clusters (Ward linkage; number of clusters selected by silhouette among non-overlapping solutions).",
    "sim": "Cosine similarity between LDA and BERTopic topics; white boxes mark the optimal one-to-one matching.",
    "cont": "Document-level agreement: proportion of documents of each LDA dominant topic assigned to each BERTopic topic.",
    "trends": "Linear trend of the annual prevalence of each topic. Red: hot (p < 0.05, positive slope); blue: cold (p < 0.05, negative slope).",
    "hj_country": "HJ-Biplot of the most productive countries (points, coloured by country cluster) and LDA topics (vectors, coloured by topic cluster); {shapes} enclose the country clusters (Ward linkage; number of clusters selected by silhouette among non-overlapping solutions).",
    "strategic": "Strategic map of LDA topics: growth (slope) versus impact (citations per year); bubble size: prevalence.",
    "callon": "Thematic map of keyword clusters (Callon centrality and density, rank-scaled); bubble size: occurrences.",
    "sankey": "Thematic evolution of keyword clusters across periods; flow width proportional to the inclusion index.",
    "cocit": "Co-citation network of the most cited references. Node size: local citations; colours: Louvain communities.",
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
    "T8b_inference": ("Statistical comparison of LDA and BERTopic (95% confidence intervals).",
                      "Coherence and agreement: percentile bootstrap over documents; stability: t-based CI over "
                      "topics; tests: two-sided Mann-Whitney U on per-topic values (r = |Z|/√N)."),
    "T10_most_cited": ("Most cited documents in the corpus.", None),
    "T11_thematic_map": ("Thematic clusters of author keywords (Callon centrality and density).", None),
    "T12_expert_validation": ("Expert validation of the topics (mean and 95% CI across topics).",
                              "Word-intrusion precision: proportion of raters identifying the intruder word."),
    "T13_cocited_references": ("Most cited references (local citations).", None),
}
TAB_ORDER = ["T1_main_information", "T2_top_sources", "T4_countries", "T10_most_cited", "T13_cocited_references",
             "T11_thematic_map", "T6_LDA_topics", "T7_BERTopic_topics", "T8_model_metrics", "T8b_inference",
             "T9_topic_alignment", "T12_expert_validation"]


def build(R, path, meta, sections=None):
    """meta: topic_name, title, query, search_date, embedding_tool.  sections: subconjunto de SECTIONS."""
    sections = set(sections or SECTIONS)
    cfg, df, lda, bert, T, FIG, X = R.cfg, R.df, R.lda, R.bert, R.tables, R.figs, R.facts
    C, info, ap = X.get("C"), X["info"], X["ap"]
    ms = R.merge_stats
    stab_on = R.on("stability")
    topic = meta.get("topic_name") or "[TOPIC]"
    models = " and ".join(m for m, r in (("LDA", lda), ("BERTopic", bert)) if r is not None)
    title = meta.get("title") or propose_titles(R, meta)[0]
    W = Writer(title)
    W.note("Borrador generado automáticamente a partir de los datos. Los textos resaltados en amarillo "
           "requieren revisión de los autores expertos en la temática (en particular, las etiquetas "
           "de los temas). Título, figuras y tablas en inglés; texto en español.")
    comps = X.get("composites", {}) if meta.get("composite", True) else {}
    W.reserve([], [k for k in TAB_ORDER if k in T])
    W.reserve_figs([k for k in FIG_ORDER if k in FIG], comps)
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
                        f"número de conglomerados se eligió con el coeficiente de silueta {cite('rousseeuw')}, entre las "
                        "soluciones cuyos grupos no se superponen en el plano factorial; cada conglomerado se "
                        + ("delimitó con una elipse que contiene a todos sus elementos." if cfg.get("cluster_shape") == "ellipse"
                           else "delimitó con su envolvente convexa.")
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

        if "T8b_inference" in T:
            W.h(f"2.{next(s)}. Inferencia estadística", 2)
            W.p("Para cuantificar la incertidumbre de la comparación se calcularon intervalos de confianza del "
                f"95 % mediante bootstrap percentil {cite('efron')} con {cfg['n_boot']} remuestreos de documentos "
                "para la coherencia NPMI media de cada modelo, su diferencia y los índices ARI y NMI; para la "
                "estabilidad media se usaron intervalos basados en la distribución t sobre los temas. Las "
                "diferencias entre modelos en los valores por tema se contrastaron con la prueba U de "
                f"Mann-Whitney bilateral {cite('mann')}, informando el tamaño del efecto r = |Z|/√N."
                + (" Para el TRI medio de los pares emparejados se obtuvo también un intervalo bootstrap."
                   if X.get("tri_ci") else ""))
        if "T10_most_cited" in T or "cocit" in FIG:
            W.h(f"2.{next(s)}. Análisis de citas", 2)
            W.p("Se identificaron los documentos más citados del corpus (citas totales y citas por año) y los más "
                "citados dentro de cada tema"
                + (". Además, a partir de las referencias citadas se construyó una red de co-citación "
                   f"{cite('small')} con las 40 referencias más citadas localmente; las referencias se "
                   "normalizaron como primer autor y año." if "cocit" in FIG else "."))
        if "callon" in FIG or "sankey" in FIG:
            W.h(f"2.{next(s)}. Mapa temático y evolución temática", 2)
            W.p("Las palabras clave de autor se agruparon mediante el algoritmo de Louvain sobre la red de "
                f"co-ocurrencia ponderada por el índice de equivalencia {cite('callon')}. Para cada grupo se "
                "calcularon la centralidad (intensidad de sus vínculos con otros grupos) y la densidad (intensidad "
                "de sus vínculos internos), y se representaron en un mapa temático de cuatro cuadrantes: temas "
                "motores, nicho, emergentes o en declive, y básicos."
                + (" La evolución temática se analizó dividiendo el periodo en subperiodos, agrupando las palabras "
                   "clave de cada uno y enlazando los grupos de subperiodos consecutivos mediante el índice de "
                   f"inclusión {cite('cobo')}, representado en un diagrama de Sankey." if "sankey" in FIG else ""))
        if R.validation is not None:
            V = R.validation
            W.h(f"2.{next(s)}. Validación de los temas por expertos", 2)
            W.p(f"La interpretabilidad de los temas se validó con {pl(V['n_raters'], 'evaluador experto', 'evaluadores expertos')} "
                "mediante una hoja de evaluación generada por la aplicación, en la que los temas de ambos modelos "
                "se presentaron en orden aleatorio y sin indicar el modelo de origen. Cada evaluador calificó la "
                "interpretabilidad y la coherencia de cada tema en una escala de 1 a 5 a partir de sus diez "
                "términos principales y tres documentos representativos, y realizó la prueba de la palabra "
                f"intrusa {cite('chang')}, en la que debía identificar el término ajeno entre cinco términos "
                "principales del tema y uno de otro tema. El acuerdo entre evaluadores se estimó con el alfa de "
                f"Krippendorff {cite('hayes')} (nivel de intervalo para las calificaciones y nominal para la "
                "palabra intrusa).")
        if R.ai.get("discussion") or R.ai.get("labels_used") or meta.get("ai_title"):
            W.h(f"2.{next(s)}. Uso de inteligencia artificial", 2)
            W.p("Se utilizó un modelo de lenguaje (", (meta.get("ai_model") or "Claude", "hl"), ", Anthropic) "
                "como asistente para "
                + enum_es([x for x, on in (("proponer nombres para los temas a partir de sus términos y "
                                            "documentos representativos", R.ai.get("labels_used")),
                                           ("redactar un primer borrador de la discusión", R.ai.get("discussion")),
                                           ("proponer el título del artículo", meta.get("ai_title")))
                           if on])
                + ". Todas las propuestas fueron revisadas, corregidas y aprobadas por los autores, que asumen "
                "la responsabilidad del contenido."
                + (" Las citas del borrador de la discusión se restringieron a documentos del propio corpus."
                   if R.ai.get("discussion") else ""))

        W.h(f"2.{next(s)}. Reproducibilidad", 2)
        v = R.versions
        W.p("Todos los análisis se realizaron con la aplicación web BiblioTopic, que ejecuta Python "
            f"{v.get('python')} en el navegador (Pyodide) con pandas {v.get('pandas')}, scikit-learn "
            f"{v.get('scikit-learn')}, SciPy {v.get('scipy')}, networkx {v.get('networkx')} y matplotlib "
            f"{v.get('matplotlib')}. Se fijó la semilla aleatoria {cfg['seed']}. El script de Python que reproduce "
            "el análisis, con su configuración completa (parámetros, semilla y nombres de los temas), las "
            "tablas intermedias y la asignación de temas por documento se proporcionan como material "
            "suplementario. El código de BiblioTopic está disponible en ",
            ("https://github.com/CarlosSalasMacias/BiblioTopic (DOI de Zenodo: [completar])", "hl"), ".")

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
            inf = X.get("inference")
            if inf:
                ch, d = inf["coh"], inf["coh"]["difference"]
                sig = "estadísticamente significativa" if d["p"] < 0.05 else "no significativa"
                txt = (f"Los intervalos de confianza{ref('T8b_inference')} confirmaron estas diferencias: la "
                       f"coherencia NPMI media fue {f(ch['LDA']['mean'], 3)} (IC 95 %: {f(ch['LDA']['lo'], 3)} a "
                       f"{f(ch['LDA']['hi'], 3)}) en LDA y {f(ch['BERTopic']['mean'], 3)} ({f(ch['BERTopic']['lo'], 3)} "
                       f"a {f(ch['BERTopic']['hi'], 3)}) en BERTopic; la diferencia ({f(d['mean'], 3)}; IC 95 %: "
                       f"{f(d['lo'], 3)} a {f(d['hi'], 3)}; p = {f(d['p'], 3)}) fue {sig}. ")
                tc = inf["t_coh"]
                if not np.isnan(tc["p"]):
                    txt += (f"La comparación de la coherencia por tema con la prueba U de Mann-Whitney dio "
                            f"U = {f(tc['U'], 1)}, p = {f(tc['p'], 3)} (r = {f(tc['r'], 2)}). ")
                ts = inf["t_stab"]
                if stab_on and not np.isnan(ts["p"]):
                    txt += (f"La estabilidad media fue {f(inf['stab_lda'][0], 3)} (IC 95 %: {f(inf['stab_lda'][1], 3)} "
                            f"a {f(inf['stab_lda'][2], 3)}) en LDA y {f(inf['stab_bert'][0], 3)} "
                            f"({f(inf['stab_bert'][1], 3)} a {f(inf['stab_bert'][2], 3)}) en BERTopic (U = "
                            f"{f(ts['U'], 1)}, p = {f(ts['p'], 3)}). ")
                if X.get("tri_ci"):
                    m_, lo_, hi_ = X["tri_ci"]
                    txt += f"El TRI medio de los pares emparejados fue {f(m_, 3)} (IC 95 %: {f(lo_, 3)} a {f(hi_, 3)})."
                W.p(txt)

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

        if "T10_most_cited" in T:
            W.h(f"3.{next(sec)}. Documentos más citados", 2)
            mc = T["T10_most_cited"]
            W.p(f"Los documentos más citados{ref('T10_most_cited')} fueron "
                + enum_es([f"{r['Document']}, «{r['Title']}» ({r['Citations']} citas)" for _, r in mc.head(3).iterrows()])
                + "."
                + (" Dentro de los temas de LDA, los trabajos más influyentes fueron: " + "; ".join(
                    f"{g}: {', '.join(d['Document'].head(2))}" for g, d in X["cited_lda"].groupby("Topic", sort=False))
                   + "." if X.get("cited_lda") is not None and len(X["cited_lda"]) else ""))
            if "cocit" in FIG:
                cc = X.get("cocit_clusters", [])
                W.p(f"La red de co-citación{ref('T13_cocited_references', 'cocit')} se organizó en "
                    f"{pl(len(cc), 'comunidad', 'comunidades')}, encabezadas por "
                    + enum_es([", ".join(c[:2]) for c in cc[:4]]) + ".")
        if "callon" in FIG or "sankey" in FIG:
            W.h(f"3.{next(sec)}. Mapa temático y evolución temática", 2)
            if "T11_thematic_map" in T:
                tm = T["T11_thematic_map"]
                byq = {q: tm[tm["Quadrant"] == q]["Label"].tolist() for q in tm["Quadrant"].unique()}
                names = {"Motor themes": "temas motores", "Niche themes": "temas nicho",
                         "Emerging or declining themes": "temas emergentes o en declive", "Basic themes": "temas básicos"}
                W.p(f"El mapa temático{ref('callon', 'T11_thematic_map')} identificó "
                    f"{pl(len(tm), 'grupo temático', 'grupos temáticos')}. "
                    + " ".join(f"Como {names[q]} se ubicaron «" + "»; «".join(v) + "»." for q, v in byq.items() if v)
                    + " ", ("[Interpretar el significado de cada cuadrante para el campo de estudio.]", "hl"))
            if "sankey" in FIG and X.get("evolution"):
                layers, links = X["evolution"]
                strong = sorted(links, key=lambda l: -l["inclusion"])[:3]
                W.p(f"La evolución temática{ref('sankey')} mostró la continuidad y transformación de los grupos de "
                    f"palabras clave entre {layers[0][0]['period']} y {layers[-1][0]['period']}. Los vínculos más fuertes "
                    "fueron " + enum_es([f"«{layers[l['t']][l['i']]['top']}» → «{layers[l['t'] + 1][l['j']]['top']}» "
                                         f"(inclusión = {f(l['inclusion'], 2)})" for l in strong]) + ".")
        if R.validation is not None and "T12_expert_validation" in T:
            V = R.validation
            W.h(f"3.{next(sec)}. Validación por expertos", 2)
            al = V["alpha"]
            tv = V["table"].set_index("Model")
            txt = (f"Participaron {pl(V['n_raters'], 'evaluador', 'evaluadores')}{ref('T12_expert_validation')}. "
                   + "; ".join(f"{'E' if i == 0 else 'e'}n {m}, la interpretabilidad media fue {tv.loc[m, 'Interpretability (1-5)']}, la "
                               f"coherencia {tv.loc[m, 'Coherence (1-5)']} y la precisión en la prueba de la palabra "
                               f"intrusa {tv.loc[m, 'Word-intrusion precision']}" for i, m in enumerate(tv.index)) + ". ")
            if V["tests"]:
                t_ = V["tests"].get("interpretability", {})
                if t_ and not np.isnan(t_.get("p", np.nan)):
                    txt += (f"La diferencia en interpretabilidad entre modelos fue U = {f(t_['U'], 1)}, "
                            f"p = {f(t_['p'], 3)}. ")
            txt += ("El acuerdo entre evaluadores (alfa de Krippendorff) fue "
                    + enum_es([f"{f(al[k], 2)} para {n}" for k, n in (("interpretability", "la interpretabilidad"),
                                                                       ("coherence", "la coherencia"),
                                                                       ("intrusion", "la palabra intrusa"))
                               if al.get(k) is not None and not np.isnan(al[k])]) + ".")
            W.p(txt)

    # ============================================================== DISCUSIÓN (IA)
    corpus_refs = []
    if "discussion" in sections and R.ai.get("discussion"):
        import re as _re
        W.h("4. Discusión", 1)
        W.note("[Borrador generado con asistencia de IA a partir de los resultados. Debe ser revisado, "
               "contrastado con la literatura y reescrito por los autores. Solo cita documentos del corpus.]")
        refmap = {r["id"]: r for r in getattr(R, "ai_refs", [])}

        def sub(m):
            ids = [x.strip() for x in m.group(1).split(",")]
            parts = []
            for i in ids:
                if i in refmap:
                    if i not in corpus_refs:
                        corpus_refs.append(i)
                    parts.append(refmap[i]["cite"].replace(" (", ", ").rstrip(")"))
            return "(" + "; ".join(parts) + ")" if parts else ""
        for para in [x.strip() for x in R.ai["discussion"].split("\n") if x.strip()]:
            if para.startswith("#"):
                W.h(para.lstrip("# ").strip(), 2)
                continue
            txt = _re.sub(r"\[(R\d+(?:\s*,\s*R\d+)*)\]", sub, para)
            parts = _re.split(r"(\[cita externa\])", txt)
            W.p(*[(x, "hl") if x == "[cita externa]" else x for x in parts if x])

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
        def cap_of(k):
            if k == "annual" and "MeanTCperYear" in ap:
                return "Annual scientific production and mean citations per year."
            cap = FIG_CAPTIONS.get(k, "")
            if k == "trends" and lda is not None and bert is not None:
                cap += " Left: LDA; right: BERTopic."
            shapes = "ellipses" if cfg.get("cluster_shape") == "ellipse" else "convex hulls"
            return cap.replace("{shapes}", shapes)
        for k in W.fig_entries:
            if k in comps:
                cap = " ".join(f"({chr(97 + i)}) {cap_of(pk)}" for i, pk in enumerate(comps[k]))
            else:
                cap = cap_of(k)
            W.figure(k, FIG[k], cap)
    if "checklist" in sections:
        used.add("montazeri")
    if "references" in sections and used:
        W.h("Referencias (métodos)", 1)
        for k in sorted(used, key=lambda x: REFS[x]):
            p = W.doc.add_paragraph(REFS[k])
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
    if corpus_refs:
        from .citations import full_ref
        W.h("Referencias del corpus citadas en la discusión", 1)
        rows = {r["id"]: r for r in R.ai_refs}
        for rid in sorted(corpus_refs, key=lambda i: rows[i]["cite"]):
            p = W.doc.add_paragraph(full_ref(R.df.iloc[rows[rid]["row"]]))
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.first_line_indent = Cm(-0.75)
    if "checklist" in sections:
        _checklist(W, R, meta, FIG, T)
    W.doc.save(path)
    return path


def _checklist(W, R, meta, FIG, T):
    """Lista de verificación basada en los requisitos mínimos de BIBLIO y PRISMA-S + declaración de datos."""
    W.doc.add_page_break()
    W.h("Material para el envío", 1)
    W.note("[Lista orientativa basada en BIBLIO (Montazeri et al., 2023) y PRISMA 2020. Verifique cada ítem "
           "con la versión oficial de la guía y con las instrucciones de la revista.]")
    q = bool(meta.get("query"))
    d = bool(meta.get("search_date"))
    items = [
        ("Title identifies the study as a bibliometric / topic-modelling analysis", "Title", True),
        ("Databases searched and rationale", "2.1", True),
        ("Full Boolean search strategy and searched fields", "2.1", q),
        ("Date of the search", "2.1", d),
        ("Export format and fields retrieved", "2.1", True),
        ("Deduplication procedure across databases", "2.2", True),
        ("Eligibility criteria (years, document types, languages)", "2.2", True),
        ("Flow diagram of records", "Figure 1", "prisma" in FIG),
        ("Software, versions and random seeds", "Reproducibility", True),
        ("Definition of bibliometric indicators", "2.3", R.on("biblio")),
        ("Text preprocessing", "Methods", R.lda is not None or R.bert is not None),
        ("Topic-model parameters and selection of the number of topics", "Methods", R.lda is not None),
        ("Validation of topics (coherence, stability, experts)", "Methods / Results",
         R.validation is not None or R.on("stability")),
        ("Uncertainty of comparisons (confidence intervals, tests)", "Results", "T8b_inference" in T),
        ("Use of AI tools disclosed", "Methods", bool(R.ai)),
        ("Limitations (database coverage, language, document types)", "Discussion", False),
        ("Data and code availability", "Statement below", True),
    ]
    import pandas as pd
    t = pd.DataFrame([{"Item": i, "Reported in": w, "Status": "Done" if ok else "To complete by authors"}
                      for i, w, ok in items])
    W.tab_n += 1
    W.tab_no["checklist"] = W.tab_n
    W.table("checklist", t, "Reporting checklist (adapted from BIBLIO and PRISMA 2020).")
    W.h("Declaración de disponibilidad de datos y código", 2)
    dbs = " y ".join(R.merge_stats["records_by_db"])
    W.p(f"Los registros bibliográficos se obtuvieron de {dbs} y están sujetos a las condiciones de licencia de "
        "dichas bases, por lo que no pueden redistribuirse; la ecuación de búsqueda, los identificadores de los "
        "documentos (DOI/UT/EID), la asignación de temas por documento, las tablas intermedias y el script de "
        "Python con la configuración completa del análisis se proporcionan como material suplementario. El código de "
        "BiblioTopic es de acceso abierto en https://github.com/CarlosSalasMacias/BiblioTopic y está archivado "
        "en Zenodo (", ("DOI: [completar]", "hl"), ").")


# ------------------------------------------------------------------ título
def _topic_phrase(R, meta):
    """Tema del estudio: el indicado por el usuario; si no, la frase de la ecuación de búsqueda o las
    palabras clave más frecuentes del corpus."""
    import re
    t = (meta.get("topic_name") or "").strip()
    if t:
        return t
    q = meta.get("query") or ""
    phrases = [p.strip() for p in re.findall(r'"([^"]{3,60})"', q) if p.strip()]
    if phrases:
        return phrases[0]
    terms = [w for w in (R.cfg.get("query_terms") or []) if w.strip()]
    if terms:
        return terms[0]
    kw = R.tables.get("T5_keywords")
    if kw is not None and len(kw):
        return " and ".join(str(k) for k in kw["Keyword"].values[:2])
    return "[topic]"


def _cap(s):
    return s[:1].upper() + s[1:] if s else s


def propose_titles(R, meta=None):
    """Títulos en inglés acordes con el tema del corpus y los análisis realizados (de más a menos sobrio)."""
    meta = meta or {}
    topic = _topic_phrase(R, meta)
    df, lda, bert, X = R.df, R.lda, R.bert, R.facts
    y0 = int(df["PY"].min())
    y1 = int(X.get("last_full_year") or df["PY"].max())
    span = f"{y0}–{y1}"
    years = y1 - y0 + 1
    both = lda is not None and bert is not None
    models = "LDA and BERTopic" if both else ("LDA" if lda is not None else "BERTopic" if bert is not None else "")
    extras = []
    if X.get("hj_l") is not None or X.get("hj_b") is not None:
        extras.append("HJ-Biplot")
    if "sankey" in R.figs or "callon" in R.figs:
        extras.append("thematic evolution")
    out = []
    short = len(topic.split()) <= 3 and not re.search(r"\b(in|of|for|on|and|with)\b", topic)
    lead = f"{_cap(topic)} research" if short else f"Research on {topic}"   # al inicio del título
    mid = f"{topic} research" if short else f"research on {topic}"          # dentro del título
    if both:
        out.append(f"{lead} ({span}): a bibliometric analysis and comparison of LDA and BERTopic topic models")
        out.append(f"Thematic structure and evolution of {mid}: evidence from LDA and BERTopic")
    elif models:
        out.append(f"{lead} ({span}): a bibliometric and {models} topic-modelling analysis")
        out.append(f"Thematic structure and evolution of {mid}: a {models} topic-modelling approach")
    else:
        out.append(f"{lead} ({span}): a bibliometric analysis")
        out.append(f"Mapping the intellectual and social structure of {mid}")
    # Con nombres de temas asignados por los autores: de qué tema a qué tema ha evolucionado el campo
    r = lda if lda is not None and getattr(lda, "custom_labels", False) else (
        bert if bert is not None and getattr(bert, "custom_labels", False) else None)
    tr = None if r is None else X.get("trends_lda" if r is lda else "trends_bert")
    if tr is not None and len(tr):
        names = [l.split(":", 1)[1].strip() if ":" in l else l for l in r.labels]
        sl = tr["Slope"].values
        hot = [i for i in range(len(names)) if tr["Trend"].iloc[i] == "Hot"]
        cold = [i for i in range(len(names)) if tr["Trend"].iloc[i] == "Cold"]
        if hot and cold:
            h, c = max(hot, key=lambda i: sl[i]), min(cold, key=lambda i: sl[i])
            out.append(f"From {names[c].lower()} to {names[h].lower()}: {years} years of {mid} "
                       f"mapped with bibliometrics and topic modelling")
        elif hot:
            h = max(hot, key=lambda i: sl[i])
            out.append(f"The rise of {names[h].lower()} in {mid}: a bibliometric and "
                       f"topic-modelling analysis ({span})")
    tl = ["bibliometrics"] + (["topic modelling"] if models else []) + extras[:1]
    tools = tl[0] if len(tl) == 1 else ", ".join(tl[:-1]) + " and " + tl[-1]
    out.append(f"Mapping {years} years of {mid} with {tools}")
    if X.get("hj_country") is not None:
        out.append(f"Who studies what in {topic}? Countries, collaboration and research topics ({span})")
    seen, res = set(), []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower())
            res.append(t)
    return res
