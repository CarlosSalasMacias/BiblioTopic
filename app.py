"""BiblioTopic — interfaz Streamlit.  Ejecutar con:  streamlit run app.py"""
from __future__ import annotations

import io
import pickle
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from bibliotopic import io as bio, pipeline, report

st.set_page_config(page_title="BiblioTopic", page_icon="📚", layout="wide")
OUT_ROOT = Path("resultados")

st.title("📚 BiblioTopic")
st.caption("Bibliometría + LDA + BERTopic + HJ-Biplot sobre corpus de Web of Science y Scopus, "
           "con borrador de Métodos y Resultados.")

ss = st.session_state

# --------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("1. Datos del estudio")
    topic_name = st.text_input("Tema del estudio (en inglés, para el título)", "", placeholder="e.g., Coffea arabica")
    title = st.text_input("Título del artículo (opcional, en inglés)", "")
    query = st.text_area("Ecuación de búsqueda booleana", "", placeholder='TS=("Coffea arabica" OR "arabica coffee")')
    search_date = st.text_input("Fecha de la búsqueda", date.today().strftime("%d/%m/%Y"))

    st.header("2. Preprocesamiento")
    qterms = st.text_input("Términos de la búsqueda a excluir del texto (coma)", "", placeholder="coffee, coffea")
    extra_sw = st.text_area("Palabras vacías adicionales (coma)", "")
    c1, c2 = st.columns(2)
    min_df = c1.number_input("Frecuencia doc. mínima", 2, 50, 5)
    max_df = c2.slider("Frecuencia doc. máxima", 0.2, 0.9, 0.5, 0.05)

    st.header("3. LDA")
    c1, c2, c3 = st.columns(3)
    kmin = c1.number_input("K mín", 2, 50, 4)
    kmax = c2.number_input("K máx", 3, 80, 20)
    kstep = c3.number_input("Paso", 1, 10, 2)
    kfix = st.number_input("Fijar K (0 = automático)", 0, 80, 0)
    n_stab_lda = st.slider("Réplicas de estabilidad LDA", 0, 10, 4)

    st.header("4. BERTopic")
    emb_model = st.selectbox("Modelo de embeddings", [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "sentence-transformers/allenai-specter",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"])
    nr_mode = st.radio("Número de temas", ["Igual que LDA (comparación justa)", "Automático (HDBSCAN)"])
    mcs = st.number_input("Tamaño mínimo de grupo (0 = auto)", 0, 500, 0)
    n_stab_bert = st.slider("Réplicas de estabilidad BERTopic", 0, 6, 3)

    st.header("5. Tiempo y comparación")
    pwidth = st.number_input("Amplitud de periodos (años)", 2, 15, 5)
    tau_s = st.slider("Umbral de convergencia (coseno)", 0.2, 0.9, 0.5, 0.05)
    tau_p = st.slider("Umbral de convergencia parcial", 0.1, 0.7, 0.3, 0.05)
    seed = st.number_input("Semilla aleatoria", 0, 99999, 42)

# --------------------------------------------------------------- pestañas
tabs = st.tabs(["📥 Datos", "📊 Bibliometría", "🧩 LDA", "🤖 BERTopic", "⚖️ Comparación",
                "📝 Etiquetas y borrador"])

with tabs[0]:
    st.subheader("Cargar exportaciones")
    st.markdown("Suba **todos los lotes** de WoS (*Tab-delimited* o *Plain text*, contenido **Full Record "
                "and Cited References**) y/o de Scopus (*CSV* con todos los campos).")
    ups = st.file_uploader("Archivos", type=["txt", "csv", "tsv"], accept_multiple_files=True)
    if ups:
        files = [(u.name, u.getvalue()) for u in ups]
        if ss.get("file_names") != [n for n, _ in files]:
            frames = []
            errs = []
            for n, raw in files:
                try:
                    frames.append(bio.read_file(n, raw)[0])
                except Exception as e:
                    errs.append(f"{n}: {e}")
            for e in errs:
                st.error(e)
            if frames:
                merged, stats = bio.merge(frames)
                ss.update(files=files, file_names=[n for n, _ in files], merged=merged,
                          mstats=stats, avail=bio.field_availability(merged))
        if "merged" in ss:
            m, s, av = ss.merged, ss.mstats, ss.avail
            a, b, c, d = st.columns(4)
            a.metric("Registros cargados", f"{s['records_total']:,}")
            b.metric("Duplicados eliminados", f"{s['duplicates_removed']:,}")
            c.metric("Registros únicos", f"{s['unique']:,}")
            d.metric("Con resumen", f"{(m['AB'].str.len() > 0).sum():,}")
            st.caption("Registros por base: " + ", ".join(f"{k} = {v:,}" for k, v in s["records_by_db"].items())
                       + " · Registros únicos por origen: "
                       + ", ".join(f"{k} = {v:,}" for k, v in s["overlap"].items()))
            avdf = pd.DataFrame({"Campo": list(av), "Completitud (%)": [round(100 * v, 1) for v in av.values()]})
            st.dataframe(avdf, hide_index=True, width=420)
            missing = [k for k in ["C1", "DE", "TC"] if av[k] < 0.3]
            if missing:
                st.warning("Faltan campos clave: " + ", ".join(missing) + ". Sin C1 no hay análisis de países "
                           "ni colaboración; sin DE no hay red de palabras clave; sin TC no hay citas. En WoS "
                           "exporte con 'Full Record and Cited References'.")
            yrs = m["PY"].dropna().astype(int)
            ss.yr = st.slider("Periodo a analizar", int(yrs.min()), int(yrs.max()),
                              (int(yrs.min()), int(yrs.max())))
            dts = sorted(x for x in m["DT"].unique() if x)
            las = sorted(x for x in m["LA"].unique() if x)
            ss.dts = st.multiselect("Tipos de documento", dts,
                                    [x for x in dts if x in ("Article", "Review")]) if dts else None
            ss.las = st.multiselect("Idiomas", las, ["English"] if "English" in las else las) if las else None

            if st.button("▶️ Ejecutar análisis completo", type="primary"):
                cfg = dict(year_min=ss.yr[0], year_max=ss.yr[1], doc_types=ss.dts or None,
                           languages=ss.las or None,
                           query_terms=[t.strip() for t in qterms.split(",") if t.strip()],
                           extra_stopwords=[t.strip() for t in extra_sw.split(",") if t.strip()],
                           min_df=int(min_df), max_df=float(max_df), k_min=int(kmin), k_max=int(kmax),
                           k_step=int(kstep), k=int(kfix) or None, seed=int(seed),
                           n_stab_lda=int(n_stab_lda), n_stab_bert=int(n_stab_bert),
                           embed_model=emb_model,
                           bert_nr_topics="lda" if nr_mode.startswith("Igual") else None,
                           min_cluster_size=int(mcs) or None, period_width=int(pwidth),
                           tau_strong=float(tau_s), tau_partial=float(tau_p))
                bar = st.progress(0.0, "Iniciando…")
                t0 = time.time()

                def cb(p, msg):
                    bar.progress(min(p, 1.0), f"{msg} · {time.time() - t0:.0f} s")
                out = OUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
                R = pipeline.compute(ss.files, cfg, cb)
                pipeline.render(R, out, cb)
                bar.progress(1.0, f"Listo en {time.time() - t0:.0f} s")
                ss.R, ss.out = R, out
                with open(out / "project.pkl", "wb") as fh:
                    pickle.dump(R, fh)
                st.success(f"Análisis terminado. Resultados en {out.resolve()}")
    st.divider()
    proj = st.file_uploader("…o reabrir un proyecto guardado (project.pkl)", type=["pkl"])
    if proj is not None and st.button("Abrir proyecto"):
        ss.R = pickle.loads(proj.getvalue())
        ss.out = OUT_ROOT / ("reabierto_" + time.strftime("%Y%m%d_%H%M%S"))
        pipeline.render(ss.R, ss.out)
        st.success("Proyecto cargado.")

R = ss.get("R")


def show_figs(keys, cols=2):
    ks = [k for k in keys if k in R.figs]
    for i in range(0, len(ks), cols):
        cs = st.columns(cols)
        for c, k in zip(cs, ks[i:i + cols]):
            c.image(R.figs[k], width="stretch")


if R is None:
    for t in tabs[1:]:
        with t:
            st.info("Cargue los archivos y ejecute el análisis en la pestaña «Datos».")
else:
    with tabs[1]:
        st.dataframe(R.tables["T1_main_information"], hide_index=True)
        show_figs(["prisma", "annual", "sources", "authors", "countries", "collab", "kw"])
        for k in ["T2_top_sources", "T3_top_authors", "T4_countries", "T5_keywords"]:
            if k in R.tables:
                st.markdown(f"**{k}**")
                st.dataframe(R.tables[k], hide_index=True)
    with tabs[2]:
        st.dataframe(R.tables["lda_k_selection"], hide_index=True)
        show_figs(["ksel", "lda_words", "lda_time", "hj_lda", "hj_country", "strategic"])
        st.dataframe(R.tables["T6_LDA_topics"], hide_index=True)
    with tabs[3]:
        if "fallback" in R.emb_name.lower():
            st.warning("No se pudo descargar el modelo de embeddings; se usó LSA. Revise su conexión a "
                       "internet y vuelva a ejecutar para obtener BERTopic con embeddings de lenguaje.")
        st.write(f"Embeddings: **{R.emb_name}** · grupos HDBSCAN: **{R.bert.native_topics}** · "
                 f"outliers: **{100 * R.bert.outlier_rate:.1f} %**")
        show_figs(["bert_words", "bert_map", "bert_time", "hj_bert"])
        st.dataframe(R.tables["T7_BERTopic_topics"], hide_index=True)
    with tabs[4]:
        st.dataframe(R.tables["T8_model_metrics"], hide_index=True)
        st.dataframe(R.tables["doc_agreement"], hide_index=True)
        st.dataframe(R.tables["T9_topic_alignment"], hide_index=True)
        show_figs(["sim", "cont", "trends"])
    with tabs[5]:
        st.markdown("Asigne un **nombre interpretativo (en inglés)** a cada tema. Se usará en figuras, "
                    "tablas y en el borrador.")
        c1, c2 = st.columns(2)

        def editor(col, res, key):
            t = pd.DataFrame({"Topic": res.ids, "Top terms": [", ".join(w[:8]) for w in res.top10],
                              "Label": [l.split(":", 1)[1].strip() for l in res.labels]})
            return col.data_editor(t, key=key, hide_index=True, disabled=["Topic", "Top terms"],
                                   width="stretch")
        el = editor(c1, R.lda, "ed_lda")
        eb = editor(c2, R.bert, "ed_bert")
        if st.button("📝 Aplicar etiquetas y generar borrador", type="primary"):
            R.lda.labels = [f"{i}: {l}" for i, l in zip(el["Topic"], el["Label"])]
            R.bert.labels = [f"{i}: {l}" for i, l in zip(eb["Topic"], eb["Label"])]
            with st.spinner("Regenerando figuras y borrador…"):
                pipeline.render(R, ss.out)
                meta = dict(topic_name=topic_name, title=title, query=query, search_date=search_date)
                docx_path = ss.out / "Borrador_Metodos_Resultados.docx"
                report.build(R, str(docx_path), meta)
                with open(ss.out / "project.pkl", "wb") as fh:
                    pickle.dump(R, fh)
            st.success("Borrador generado.")
            ss.docx = docx_path
        if ss.get("docx"):
            st.download_button("⬇️ Descargar borrador (.docx)", ss.docx.read_bytes(),
                               file_name=ss.docx.name)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for p in Path(ss.out).rglob("*"):
                    if p.is_file():
                        z.write(p, p.relative_to(ss.out))
            st.download_button("⬇️ Descargar todo (figuras, tablas, configuración) .zip",
                               buf.getvalue(), file_name="BiblioTopic_resultados.zip")
