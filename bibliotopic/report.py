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


def _biplot_associations(hj, top=3):
    """Para cada fila (periodo/país), columnas (temas) más alineadas en el plano 1-2."""
    R = hj.rows.iloc[:, :2].values
    C = hj.cols.iloc[:, :2].values
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    cos = Rn @ Cn.T
    return {hj.rows.index[i]: [hj.cols.index[j] for j in np.argsort(-cos[i])[:top]]
            for i in range(len(R))}


def build(R, path, meta):
    """meta: topic_name, query, databases, search_date."""
    cfg, df, lda, bert, T, FIG, X = R.cfg, R.df, R.lda, R.bert, R.tables, R.figs, R.facts
    C, info, ap = X["C"], X["info"], X["ap"]
    ms = R.merge_stats
    topic = meta.get("topic_name") or "[TOPIC]"
    title = meta.get("title") or (f"Mapping the scientific landscape of {topic}: bibliometrics and a "
                                  "comparative topic modelling approach using LDA and BERTopic")
    W = Writer(title)
    W.note("Borrador generado automáticamente a partir de los datos. Los textos resaltados en amarillo "
           "requieren revisión de los autores expertos en la temática (en particular, las etiquetas "
           "de los temas). Título, figuras y tablas en inglés; texto en español.")

    fig_order = [k for k in ["prisma", "annual", "sources", "authors", "countries", "collab", "kw",
                             "ksel", "lda_words", "lda_time", "hj_lda", "bert_words", "bert_map",
                             "bert_time", "hj_bert", "sim", "cont", "trends", "hj_country",
                             "strategic"] if k in FIG]
    tab_order = [k for k in ["T1_main_information", "T2_top_sources", "T4_countries",
                             "T6_LDA_topics", "T7_BERTopic_topics", "T8_model_metrics",
                             "T9_topic_alignment"] if k in T]
    W.reserve(fig_order, tab_order)
    used = set()

    def cite(*keys):
        used.update(keys)
        names = {"aria": "Aria y Cuccurullo, 2017", "arun": "Arun et al., 2010",
                 "blei": "Blei et al., 2003", "blondel": "Blondel et al., 2008",
                 "bouma": "Bouma, 2009", "bradford": "Bradford, 1934",
                 "campello": "Campello et al., 2013", "cao": "Cao et al., 2009",
                 "deveaud": "Deveaud et al., 2014", "dieng": "Dieng et al., 2020",
                 "gabriel": "Gabriel, 1971", "galindo": "Galindo, 1986",
                 "griffiths": "Griffiths y Steyvers, 2004", "grootendorst": "Grootendorst, 2022",
                 "hoffman": "Hoffman et al., 2010", "hubert": "Hubert y Arabie, 1985",
                 "kuhn": "Kuhn, 1955", "lotka": "Lotka, 1926", "mcinnes17": "McInnes et al., 2017",
                 "mcinnes18": "McInnes et al., 2018", "mimno": "Mimno et al., 2011",
                 "montes23": "Montes-Escobar et al., 2023", "abril24": "Abril-Espin et al., 2024",
                 "salas26": "Salas-Macías et al., 2026", "montes26": "Montes-Escobar et al., 2026",
                 "page": "Page et al., 2021", "pedregosa": "Pedregosa et al., 2011",
                 "reimers": "Reimers y Gurevych, 2019", "strehl": "Strehl y Ghosh, 2002"}
        return "(" + "; ".join(names[k] for k in keys) + ")"

    by_db = ms["records_by_db"]
    dbs_txt = enum_es([{"WoS": "Web of Science Core Collection (WoS)", "Scopus": "Scopus"}[k]
                       for k in by_db])
    py = df["PY"].astype(int)
    n = len(df)

    # ================================================================ MÉTODOS
    def tl_(labels):
        return "; ".join(f"{l.split(':')[0]}: {_short(l)}" for l in labels)

    W.h("2. Materiales y métodos", 1)
    W.h("2.1. Fuente de datos y estrategia de búsqueda", 2)
    q = meta.get("query") or "[ecuación de búsqueda]"
    date = meta.get("search_date") or "[fecha]"
    W.p(f"La información se obtuvo de {dbs_txt}, bases de datos de referencia para estudios "
        "bibliométricos por su cobertura multidisciplinar y la calidad de sus metadatos. La búsqueda se "
        f"realizó el ", (date, "hl"), " mediante la siguiente ecuación booleana aplicada a título, resumen "
        "y palabras clave: ", (q, "hl"), ". Los registros se exportaron con todos sus campos "
        f"disponibles ({'; '.join(f'{k}: {f(v, 0)} registros' for k, v in by_db.items())}).")
    W.h("2.2. Depuración y fusión del corpus", 2)
    es = {"Other document types": "otros tipos de documento", "Other languages": "otros idiomas",
          "Missing/short abstract (<20 words)": "registros sin resumen o con menos de 20 palabras"}
    excl_txt = enum_es([f"{es.get(name, 'registros fuera del periodo ' + name.split()[-1])} "
                        f"(n = {f(k, 0)})" for name, k in R.excl_steps if k])
    filt = []
    if cfg.get("doc_types") and R.availability["DT"] > 0:
        filt.append("tipos de documento " + enum_es(cfg["doc_types"]))
    if cfg.get("languages") and R.availability["LA"] > 0:
        filt.append("idioma " + enum_es(cfg["languages"]))
    W.p("Los registros de ambas fuentes se normalizaron a un esquema común (autores, título, fuente, año, "
        "resumen, palabras clave, afiliaciones, citas, DOI) y se fusionaron. Los duplicados se "
        "identificaron primero por coincidencia del DOI y, en ausencia de este, por coincidencia del título "
        "normalizado (minúsculas, sin signos ni tildes) y el año de publicación; de cada grupo de duplicados "
        "se conservó el registro con más campos completos y el mayor número de citas. "
        f"Se eliminaron {f(ms['duplicates_removed'], 0)} duplicados, quedando {f(ms['unique'], 0)} registros "
        "únicos. Como criterios de elegibilidad se exigió "
        + (enum_es(filt) + " y " if filt else "")
        + f"un resumen de al menos 20 palabras. Se excluyeron {excl_txt or 'ningún registro'}, de modo que "
        f"el corpus final quedó integrado por {f(n, 0)} documentos publicados entre {py.min()} y {py.max()}. "
        f"El proceso se resume en un diagrama de flujo adaptado de PRISMA 2020 {cite('page')} "
        f"({W.F('prisma')}).")

    W.h("2.3. Análisis bibliométrico", 2)
    extra = []
    if R.has_countries:
        extra.append("la producción por país del autor de correspondencia, distinguiendo publicaciones de un "
                     "solo país (SCP) y de colaboración internacional (MCP), y la red de colaboración entre "
                     "países, cuyas comunidades se identificaron con el algoritmo de Louvain " + cite("blondel"))
    if X.get("kw_field"):
        extra.append("la red de co-ocurrencia de las palabras clave más frecuentes")
    W.p("Se calcularon los indicadores descriptivos habituales del análisis de mapeo científico "
        f"{cite('aria')}: tasa de crecimiento anual compuesta, edad media de los documentos, autores, "
        "coautores por documento, fuentes más relevantes, zonas de Bradford " + cite("bradford")
        + " y distribución de la productividad de autores según la ley de Lotka " + cite("lotka")
        + (". Además, se analizaron " + enum_es(extra) if extra else "")
        + ". Los indicadores se implementaron en Python siguiendo las definiciones de bibliometrix.")

    W.h("2.4. Preprocesamiento del texto", 2)
    W.p("Para el modelado de temas se concatenó el título y el resumen de cada documento. El texto se "
        "convirtió a minúsculas y se eliminaron números, signos de puntuación, las palabras vacías del inglés, "
        "un listado de términos académicos genéricos (p. ej., study, results, analysis) y los términos de "
        "la propia ecuación de búsqueda"
        + (f" ({enum_es(cfg['query_terms'])})" if cfg.get("query_terms") else "")
        + ", que por estar presentes en casi todos los documentos no aportan capacidad discriminante. "
        "Los términos se llevaron a su forma singular y se identificaron expresiones compuestas "
        f"(bigramas) con información mutua puntual normalizada (NPMI) ≥ 0.35 y al menos 10 apariciones "
        f"{cite('bouma')}, lo que generó {f(len(R.corpus.phrases), 0)} términos compuestos. Finalmente se "
        f"retuvieron los términos presentes en al menos {cfg['min_df']} documentos y en no más del "
        f"{int(100 * cfg['max_df'])} % de ellos, con un vocabulario final de {f(len(R.corpus.vocab), 0)} "
        "términos. El mismo vocabulario se utilizó en ambos modelos para que la comparación fuera directa.")

    W.h("2.5. Modelado de temas con LDA", 2)
    ks = R.ksel["K"]
    W.p(f"La asignación latente de Dirichlet (LDA) {cite('blei')} representa cada documento como una "
        "mezcla de temas y cada tema como una distribución de probabilidad sobre el vocabulario. El modelo se "
        f"estimó por inferencia variacional en lote {cite('hoffman')} con scikit-learn {cite('pedregosa')}. "
        f"El número de temas (K) se evaluó entre {ks.min()} y {ks.max()} (paso {cfg['k_step']}) con cuatro "
        f"métricas complementarias: CaoJuan2009 {cite('cao')} y Arun2010 {cite('arun')}, a minimizar, y "
        f"Deveaud2014 {cite('deveaud')} y la coherencia NPMI de las diez palabras principales de cada tema "
        f"{cite('bouma')}, a maximizar. Las métricas se normalizaron entre 0 y 1 y se seleccionó el K con "
        "el mayor promedio" + (" (valor fijado por los autores: K = " + str(cfg['k']) + ")" if cfg.get("k") else "")
        + f". Para evaluar la estabilidad, el modelo final se reestimó con {cfg['n_stab_lda']} semillas "
        "aleatorias distintas; cada tema se emparejó con su homólogo mediante el algoritmo húngaro "
        f"{cite('kuhn')} y se calculó la similitud coseno media entre réplicas como índice de estabilidad.")

    W.h("2.6. Modelado de temas con BERTopic", 2)
    emb_txt = (f"el modelo de lenguaje preentrenado {R.emb_name} {cite('reimers')}"
               if "fallback" not in R.emb_name.lower() else
               "una representación semántica latente (TF-IDF + SVD) [REEMPLAZAR por el modelo de embeddings usado]")
    W.p(f"BERTopic {cite('grootendorst')} agrupa documentos semánticamente similares a partir de "
        f"representaciones vectoriales densas. Los títulos y resúmenes se codificaron con {emb_txt}; la "
        f"dimensionalidad se redujo a cinco componentes con UMAP {cite('mcinnes18')} (15 vecinos, distancia "
        "coseno) y los documentos se agruparon con HDBSCAN "
        f"{cite('campello', 'mcinnes17')} (tamaño mínimo de grupo = {bert.min_cluster_size}). Cada tema se "
        "describió mediante TF-IDF basado en clases (c-TF-IDF) sobre el mismo vocabulario usado en LDA. "
        + (f"HDBSCAN identificó {bert.native_topics} grupos; para comparar ambos modelos con igual "
           f"granularidad, los temas se fusionaron jerárquicamente hasta K = {bert.k}. "
           if bert.native_topics != bert.k else "")
        + "Los documentos no asignados (outliers) se reasignaron al tema con mayor similitud c-TF-IDF para "
        f"los análisis por documento. La estabilidad se evaluó como en LDA, con {cfg['n_stab_bert']} "
        "semillas de UMAP distintas.")

    W.h("2.7. Dinámica temporal y HJ-Biplot", 2)
    W.p(f"La evolución de cada tema se estimó mediante la regresión lineal de su prevalencia media anual "
        f"sobre el año de publicación {cite('griffiths')}, considerando los años con al menos "
        f"{cfg['min_docs_year']} documentos; los temas con pendiente significativa (p < 0.05) se clasificaron "
        "como emergentes (hot) o en declive (cold). Para analizar conjuntamente temas y tiempo se agruparon "
        f"los años en periodos ({enum_es(X['periods'])}) y se construyó una matriz periodo × tema con la "
        f"prevalencia media, a la que se aplicó un HJ-Biplot {cite('galindo', 'gabriel')}. Este método permite "
        "representar filas y columnas en un mismo sistema de ejes con la máxima calidad de representación; "
        "los ángulos agudos entre vectores de temas indican correlación positiva, y la proximidad entre un "
        "periodo y la dirección de un tema indica una mayor prevalencia de ese tema en dicho periodo."
        + (" El mismo procedimiento se aplicó a una matriz país × tema con los países más productivos."
           if R.has_countries else "")
        + f" Este enfoque ha sido empleado previamente en estudios bibliométricos con LDA "
        f"{cite('montes23', 'abril24', 'salas26', 'montes26')}.")

    W.h("2.8. Comparación entre LDA y BERTopic", 2)
    W.p("Ambos modelos se compararon en tres niveles. (i) Calidad intrínseca: coherencia NPMI y UMass "
        f"{cite('bouma', 'mimno')}, diversidad de temas (proporción de términos únicos entre las 25 "
        f"palabras principales de todos los temas) {cite('dieng')}, estabilidad entre semillas y proporción "
        "de outliers. (ii) Correspondencia entre temas: se calculó la similitud coseno entre la distribución "
        "palabra-tema de LDA y el vector c-TF-IDF de BERTopic y se obtuvo el emparejamiento óptimo uno a uno "
        f"con el algoritmo húngaro {cite('kuhn')}; también se informa el índice de Jaccard entre las diez "
        f"palabras principales. (iii) Coincidencia a nivel de documento: índice de Rand ajustado (ARI) "
        f"{cite('hubert')} e información mutua normalizada (NMI) {cite('strehl')} entre el tema dominante "
        "de LDA y el tema asignado por BERTopic. Con estos elementos se propone un Índice de Robustez "
        "Temática (TRI) para cada par emparejado: TRI = cos(LDA, BERTopic) × √(estabilidad LDA × "
        "estabilidad BERTopic). Un par se consideró convergente (robusto) cuando la similitud fue ≥ "
        f"{cfg['tau_strong']}, parcial cuando estuvo entre {cfg['tau_partial']} y {cfg['tau_strong']}, y "
        "divergente en otro caso. Finalmente, se evaluó la concordancia en la clasificación temporal "
        "(emergente, estable o en declive) de los temas emparejados.")

    W.h("2.9. Reproducibilidad", 2)
    v = R.versions
    W.p(f"Todos los análisis se realizaron en Python {v.get('python')} con pandas {v.get('pandas')}, "
        f"scikit-learn {v.get('scikit-learn')}, BERTopic {v.get('bertopic')}, umap-learn "
        f"{v.get('umap-learn')}, hdbscan {v.get('hdbscan')}, sentence-transformers "
        f"{v.get('sentence-transformers')} y networkx {v.get('networkx')}, mediante la aplicación "
        f"BiblioTopic desarrollada por los autores. Se fijó la semilla aleatoria {cfg['seed']}. La "
        "configuración completa, las tablas intermedias y la asignación de temas por documento se "
        "proporcionan como material suplementario.")

    # ============================================================== RESULTADOS
    W.h("3. Resultados", 1)
    sec = iter(range(1, 20))
    W.h(f"3.{next(sec)}. Descripción general del corpus", 2)
    peak = ap.loc[ap["Articles"].idxmax()]
    recent = py[py >= py.max() - 9]
    half_year = int(np.median(py))
    W.p(f"El corpus final incluyó {f(n, 0)} documentos publicados en {info['Sources (journals, books, etc.)']} "
        f"fuentes por {f(info['Authors'], 0)} autores entre {py.min()} y {py.max()} ({W.T('T1_main_information')}). "
        f"La producción creció a una tasa anual compuesta de {f(info['Annual growth rate (%)'])} %, con "
        f"{f(info['Co-authors per document'])} coautores por documento y "
        f"{f(info['Single-authored documents'], 0)} documentos de autoría única. El año más productivo fue "
        f"{int(peak['Year'])} ({f(peak['Articles'], 0)} documentos). La mitad del corpus se publicó a partir de "
        f"{half_year} y el {f(100 * len(recent) / n, 1)} % en los últimos diez años "
        f"({py.max() - 9}-{py.max()}), lo que refleja un interés creciente en la temática ({W.F('annual')})."
        + (f" El promedio de citas por documento fue de {f(info['Average citations per document'])}."
           if "Average citations per document" in info else ""))
    if "WoS+Scopus" in ms.get("overlap", {}):
        ov = ms["overlap"]
        W.p(f"Del total de registros únicos, {f(ov.get('WoS+Scopus', 0), 0)} estaban indexados en ambas bases, "
            f"{f(ov.get('WoS', 0), 0)} solo en WoS y {f(ov.get('Scopus', 0), 0)} solo en Scopus, lo que "
            "justifica el uso combinado de ambas fuentes.")

    W.h(f"3.{next(sec)}. Fuentes y autores", 2)
    src = T["T2_top_sources"]
    W.p(f"Las fuentes con más publicaciones fueron {enum_es([f'{s.title()} (n = {a})' for s, a in src[['Source', 'Articles']].values[:3]])} "
        f"({W.T('T2_top_sources')}; {W.F('sources')}). Según la ley de Bradford, {X['bradford_zone1']} de las "
        f"{f(X['n_sources'], 0)} fuentes ({f(100 * X['bradford_zone1'] / X['n_sources'], 1)} %) concentraron "
        "un tercio de la producción (zona núcleo). "
        f"Los autores más productivos fueron {enum_es([f'{a} ({k})' for a, k in T['T3_top_authors'][['Author', 'Articles']].values[:5]])} "
        f"({W.F('authors')}). En cuanto a la ley de Lotka, el "
        f"{f(100 * T['lotka']['Proportion of authors'].iloc[0], 1)} % de los autores publicó un solo documento.")

    if R.has_countries:
        W.h(f"3.{next(sec)}. Países y colaboración internacional", 2)
        cp = T["T4_countries"]
        W.p(f"Los países con mayor producción, según el país del autor de correspondencia, fueron "
            f"{enum_es([f'{c} (n = {a})' for c, a in cp[['Country', 'Articles']].values[:5]])} "
            f"({W.T('T4_countries')}; {W.F('countries')}). El {f(info.get('International co-authorship (%)', 0))} % "
            "de los documentos tuvo coautoría internacional. Entre los diez principales, la mayor proporción "
            f"de colaboración internacional (MCP) correspondió a "
            f"{enum_es([f'{c} ({f(100 * r, 1)} %)' for c, r in cp.head(10).sort_values('MCP ratio', ascending=False)[['Country', 'MCP ratio']].values[:3]])}.")
        if "collab" in FIG:
            cl = X.get("collab_clusters", [])
            W.p((f"La red de colaboración ({W.F('collab')}) mostró {len(cl)} comunidades; las principales "
                 f"estuvieron encabezadas por {enum_es([', '.join(c[:4]) for c in cl[:3]])}.") if len(cl) > 1 else
                f"La red de colaboración ({W.F('collab')}) formó una única comunidad articulada en torno a "
                f"{', '.join(cl[0][:4]) if cl else '—'}.")
    else:
        W.note("[Países y colaboración: el archivo de entrada no contiene afiliaciones (campo C1). Para incluir "
               "esta sección, exporte desde WoS el contenido 'Full Record and Cited References' y desde "
               "Scopus incluya 'Affiliations'.]")
    if "kw" in FIG:
        W.h(f"3.{next(sec)}. Palabras clave", 2)
        kf = T["T5_keywords"]
        W.p(f"Las palabras clave más frecuentes fueron {enum_es([f'{k} ({o})' for k, o in kf.values[:8]])}. "
            f"La red de co-ocurrencia ({W.F('kw')}) se organizó en {len(X['kw_clusters'])} "
            f"{'grupos' if len(X['kw_clusters']) > 1 else 'grupo'}; los "
            "principales reunieron " + "; ".join(f"«{', '.join(c[:4])}»" for c in X["kw_clusters"][:3]) + ".")

    # ---- LDA
    W.h(f"3.{next(sec)}. Temas identificados con LDA", 2)
    ksel = R.ksel.set_index("K")
    tl = C["trends_lda"]
    order = np.argsort(-lda.prevalence)
    hot_l = [lda.labels[i] for i in range(lda.k) if tl["Trend"][i] == "Hot"]
    cold_l = [lda.labels[i] for i in range(lda.k) if tl["Trend"][i] == "Cold"]
    W.p(f"El criterio combinado de las cuatro métricas indicó K = {lda.k} como número óptimo de temas "
        f"({W.F('ksel')}); para este valor, la coherencia NPMI media fue {f(np.nanmean(lda.npmi), 3)}, la "
        f"diversidad {f(lda.diversity, 3)} y la estabilidad media entre semillas {f(np.nanmean(lda.stab), 3)}. "
        + ("La métrica combinada alcanzó su máximo en el límite superior del intervalo evaluado; se recomienda "
           "ampliar el intervalo de K. " if lda.k == ksel.index.max() else "")
        + f"Los temas, sus términos principales y su prevalencia se presentan en la {W.T('T6_LDA_topics')} y "
        f"la {W.F('lda_words')}. ", ("[Etiquetas provisionales generadas a partir de los tres términos "
                                    "principales: los autores deben asignar un nombre interpretativo a cada tema.]", "hl"))
    W.p("Los temas más prevalentes fueron " + "; ".join(
        f"{lda.ids[i]} ({f(100 * lda.prevalence[i], 1)} %), caracterizado por {', '.join(lda.top10[i][:6])}"
        for i in order[:3]) + ". Los menos prevalentes fueron " + enum_es(
        [f"{lda.ids[i]} ({f(100 * lda.prevalence[i], 1)} %)" for i in order[-2:]]) + ".")
    W.p(f"Respecto a la dinámica temporal ({W.F('lda_time')}; {W.F('trends')}), "
        + (f"{len(hot_l)} temas mostraron una tendencia creciente significativa ({tl_(hot_l)})"
           if hot_l else "ningún tema mostró una tendencia creciente significativa")
        + " y " + (f"{len(cold_l)} una tendencia decreciente ({tl_(cold_l)})"
                   if cold_l else "ninguno una tendencia decreciente") + ".")
    hj = X["hj_l"]
    assoc = _biplot_associations(hj)
    first_p, last_p = X["periods"][0], X["periods"][-1]
    W.p(f"El HJ-Biplot periodo × tema de LDA ({W.F('hj_lda')}) explicó el "
        f"{f(100 * (hj.inertia[0] + hj.inertia[1]), 1)} % de la inercia en los dos primeros ejes "
        f"(eje 1: {f(100 * hj.inertia[0], 1)} %; eje 2: {f(100 * hj.inertia[1], 1)} %). El periodo más "
        f"antiguo ({first_p}) se asoció con {enum_es(assoc[first_p])}, mientras que el más reciente "
        f"({last_p}) se asoció con {enum_es(assoc[last_p])}, lo que evidencia el desplazamiento del foco de "
        "investigación a lo largo del tiempo.")

    # ---- BERTopic
    W.h(f"3.{next(sec)}. Temas identificados con BERTopic", 2)
    tb = C["trends_bert"]
    ob = np.argsort(-bert.prevalence)
    hot_b = [bert.labels[i] for i in range(bert.k) if tb["Trend"][i] == "Hot"]
    cold_b = [bert.labels[i] for i in range(bert.k) if tb["Trend"][i] == "Cold"]
    W.p(f"HDBSCAN identificó {bert.native_topics} grupos semánticos"
        + (f", que se redujeron a {bert.k} temas" if bert.native_topics != bert.k else "")
        + f"; el {f(100 * bert.outlier_rate, 1)} % de los documentos quedó inicialmente sin asignar "
        f"(outliers) ({W.T('T7_BERTopic_topics')}; {W.F('bert_words')}"
        + (f"; {W.F('bert_map')}" if "bert_map" in FIG else "") + "). La coherencia NPMI media fue "
        f"{f(np.nanmean(bert.npmi), 3)}, la diversidad {f(bert.diversity, 3)} y la estabilidad media "
        f"{f(np.nanmean(bert.stab), 3)}. Los temas de mayor tamaño fueron " + "; ".join(
            f"{bert.ids[i]} ({f(100 * bert.prevalence[i], 1)} %; {', '.join(bert.top10[i][:6])})"
            for i in ob[:3]) + ".")
    hjb = X["hj_b"]
    ab = _biplot_associations(hjb)
    W.p(f"En el tiempo ({W.F('bert_time')}), "
        + (f"{len(hot_b)} temas fueron emergentes ({tl_(hot_b)})" if hot_b
           else "ningún tema fue emergente")
        + " y " + (f"{len(cold_b)} estuvieron en declive ({tl_(cold_b)})"
                   if cold_b else "ninguno estuvo en declive")
        + f". El HJ-Biplot correspondiente ({W.F('hj_bert')}) explicó el "
        f"{f(100 * (hjb.inertia[0] + hjb.inertia[1]), 1)} % de la inercia; el periodo {last_p} se asoció "
        f"con {enum_es(ab[last_p])}.")

    # ---- comparación
    W.h(f"3.{next(sec)}. Comparación entre LDA y BERTopic", 2)
    mt = C["metrics"].set_index("Metric")
    ag = C["agreement"]
    m = C["matches"]
    rob = m[m["Agreement"] == "Robust"]
    par = m[m["Agreement"] == "Partial"]
    div = m[m["Agreement"] == "Divergent"]
    better = lambda metric: ("LDA" if mt.loc[metric, "LDA"] > mt.loc[metric, "BERTopic"] else "BERTopic")  # noqa: E731
    W.p(f"La {W.T('T8_model_metrics')} resume la calidad de ambos modelos. {better('Mean NPMI coherence')} "
        f"obtuvo mayor coherencia NPMI ({f(mt.loc['Mean NPMI coherence', 'LDA'], 3)} en LDA frente a "
        f"{f(mt.loc['Mean NPMI coherence', 'BERTopic'], 3)} en BERTopic), {better('Topic diversity (top-25)')} "
        f"mayor diversidad ({f(mt.loc['Topic diversity (top-25)', 'LDA'], 3)} frente a "
        f"{f(mt.loc['Topic diversity (top-25)', 'BERTopic'], 3)}) y {better('Mean topic stability (cosine across seeds)')} "
        f"mayor estabilidad entre semillas ({f(mt.loc['Mean topic stability (cosine across seeds)', 'LDA'], 3)} "
        f"frente a {f(mt.loc['Mean topic stability (cosine across seeds)', 'BERTopic'], 3)}).")
    W.p(f"El emparejamiento óptimo entre temas ({W.F('sim')}; {W.T('T9_topic_alignment')}) identificó "
        f"{len(rob)} pares convergentes (coseno ≥ {cfg['tau_strong']}), {len(par)} parciales y {len(div)} "
        "divergentes. "
        + ("Los temas convergentes con mayor índice de robustez fueron " + "; ".join(
            f"{r['LDA topic'].split(':')[0]}–{r['BERTopic topic'].split(':')[0]} (coseno = {f(r['Cosine'], 2)}; "
            f"TRI = {f(r['TRI'], 2)}; términos compartidos: {r['Shared terms'] or '—'})"
            for _, r in rob.sort_values('TRI', ascending=False).head(4).iterrows()) + ". "
           if len(rob) else "")
        + ("Estos temas, detectados de forma independiente por un modelo probabilístico basado en "
           "frecuencias y por otro basado en semántica contextual, constituyen el núcleo temático más sólido "
           "del campo. " if len(rob) else
           "Ningún par alcanzó el umbral de convergencia; los pares con mayor correspondencia fueron " + "; ".join(
               f"{r['LDA topic'].split(':')[0]}–{r['BERTopic topic'].split(':')[0]} (coseno = {f(r['Cosine'], 2)}; "
               f"términos compartidos: {r['Shared terms'] or '—'})"
               for _, r in par.sort_values('TRI', ascending=False).head(4).iterrows())
           + ", que representan los temas más consistentes entre ambos enfoques. ")
        + ("Por el contrario, " + enum_es([f"{r['LDA topic'].split(':')[0]} y {r['BERTopic topic'].split(':')[0]}"
                                           for _, r in div.head(3).iterrows()])
           + " no encontraron un equivalente claro en el otro modelo, lo que sugiere temas sensibles al método "
           "que deben interpretarse con cautela." if len(div) else ""))
    W.p(f"A nivel de documento ({W.F('cont')}), la coincidencia entre la asignación de LDA y la de BERTopic "
        f"fue ARI = {f(ag['ARI (all documents)'], 3)} y NMI = {f(ag['NMI (all documents)'], 3)} "
        f"(excluyendo outliers: ARI = {f(ag['ARI (non-outliers)'], 3)}; NMI = {f(ag['NMI (non-outliers)'], 3)}). "
        + (f"En los temas emparejados (convergentes y parciales), ambos modelos coincidieron en la "
           f"clasificación temporal en el {f(100 * ag['Trend agreement in matched topics'], 1)} % de los casos."
           if not np.isnan(ag["Trend agreement in matched topics"]) else ""))
    if R.has_countries and "hj_country" in FIG:
        hjc = X["hj_country"]
        ac = _biplot_associations(hjc, 2)
        top3 = list(hjc.rows.index[:4])
        W.h(f"3.{next(sec)}. Especialización temática de los países", 2)
        W.p(f"El HJ-Biplot país × tema ({W.F('hj_country')}) explicó el "
            f"{f(100 * (hjc.inertia[0] + hjc.inertia[1]), 1)} % de la inercia. " + "; ".join(
                f"{c} se asoció con {enum_es(ac[c])}" for c in top3) + ".")
    if "strategic" in FIG:
        W.p(f"El mapa estratégico ({W.F('strategic')}) cruza el crecimiento y el impacto (citas por año) de "
            "cada tema de LDA: los temas del cuadrante superior derecho combinan crecimiento e impacto "
            "elevados y representan frentes de investigación consolidados en expansión.")

    # ============================================================== FIGURAS / TABLAS
    W.doc.add_page_break()
    W.h("Tables", 1)
    tcap = {
        "T1_main_information": ("Main information about the corpus.", None),
        "T2_top_sources": ("Most relevant sources.", None),
        "T4_countries": ("Most productive countries (corresponding author).",
                         "SCP: single-country publications; MCP: multiple-country publications."),
        "T6_LDA_topics": ("Topics identified by LDA.",
                          "Labels are provisional. Prevalence: mean document–topic probability. "
                          "Stability: mean cosine similarity with matched topics across random seeds."),
        "T7_BERTopic_topics": ("Topics identified by BERTopic.",
                               "Prevalence computed after outlier reassignment."),
        "T8_model_metrics": ("Comparison of model quality metrics.", None),
        "T9_topic_alignment": ("Alignment between LDA and BERTopic topics (Hungarian matching).",
                               "TRI: Topic Robustness Index = cosine × √(stability_LDA × stability_BERTopic)."),
    }
    for k in tab_order:
        t = T[k]
        if k == "T9_topic_alignment":
            t = t[["LDA topic", "BERTopic topic", "Cosine", "Jaccard top-10", "TRI", "Agreement",
                   "Trend LDA", "Trend BERTopic"]]
        if k == "T2_top_sources":
            t = t.head(10)
        W.table(k, t, *tcap[k])
    W.doc.add_page_break()
    W.h("Figures", 1)
    fcap = {
        "prisma": "Flow diagram of record identification, deduplication and screening (adapted from PRISMA 2020).",
        "annual": "Annual scientific production" + (" and mean citations per year." if "MeanTCperYear" in ap else "."),
        "sources": "Most relevant sources by number of documents.",
        "authors": "Most productive authors.",
        "countries": "Most productive countries by corresponding author; SCP: single-country publications; MCP: multiple-country publications.",
        "collab": "Country collaboration network. Node size: number of documents; colours: Louvain communities.",
        "kw": "Keyword co-occurrence network. Node size: occurrences; colours: Louvain communities.",
        "ksel": "Selection of the number of LDA topics (K) with four normalised metrics. Dashed line: selected K.",
        "lda_words": "Top terms of the LDA topics (word–topic probability).",
        "lda_time": "Mean prevalence of LDA topics by period.",
        "hj_lda": "HJ-Biplot of periods (points) and LDA topics (vectors).",
        "bert_words": "Top terms of the BERTopic topics (c-TF-IDF weight).",
        "bert_map": "Two-dimensional UMAP projection of document embeddings coloured by BERTopic topic; grey: outliers.",
        "bert_time": "Prevalence of BERTopic topics by period.",
        "hj_bert": "HJ-Biplot of periods (points) and BERTopic topics (vectors).",
        "sim": "Cosine similarity between LDA and BERTopic topics; white boxes mark the optimal one-to-one matching.",
        "cont": "Document-level agreement: proportion of documents of each LDA dominant topic assigned to each BERTopic topic.",
        "trends": "Linear trend of the annual prevalence of each topic. Red: hot (p < 0.05, positive slope); blue: cold (p < 0.05, negative slope).",
        "hj_country": "HJ-Biplot of the most productive countries (points) and LDA topics (vectors).",
        "strategic": "Strategic map of LDA topics: growth (slope) versus impact (citations per year); bubble size: prevalence.",
    }
    for k in fig_order:
        W.figure(k, FIG[k], fcap[k])

    W.h("Referencias (métodos)", 1)
    for k in sorted(used, key=lambda x: REFS[x]):
        p = W.doc.add_paragraph(REFS[k])
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.first_line_indent = Cm(-0.75)
    W.doc.save(path)
    return path
