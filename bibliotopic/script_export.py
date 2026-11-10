"""Paquete de reproducibilidad: el código y la configuración exacta del análisis realizado en la aplicación.

Contenido del .zip:
  analisis_bibliotopic.py   script principal, comentado paso a paso
  config.json               parámetros, semilla, nombres de temas y metadatos del análisis
  bibliotopic/              código de análisis (el mismo que se ejecutó en el navegador)
  requirements.txt          librerías con las versiones utilizadas
  LEEME.md                  instrucciones
  datos/                    carpeta vacía para las exportaciones de Scopus y WoS (no se incluyen)
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import zipfile
from pathlib import Path

SCRIPT = r'''"""Análisis bibliométrico y de modelado de temas (LDA y BERTopic) generado con BiblioTopic.

Reproduce el análisis realizado en la aplicación web con los mismos parámetros (config.json).

Uso:
    pip install -r requirements.txt
    python analisis_bibliotopic.py            # lee las exportaciones de la carpeta datos/

Los resultados (figuras, tablas y borrador en Word) se guardan en la carpeta resultados/.
"""
import json
import sys
from pathlib import Path

import numpy as np

from bibliotopic import pipeline, report
from bibliotopic.bert_model import lsa_embeddings

BASE = Path(__file__).resolve().parent
CONFIG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
DATOS = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "datos"
SALIDA = BASE / "resultados"


def progreso(p, msg):
    print(f"  [{p:4.0%}] {msg}", flush=True)


# 1. Lectura y fusión de las exportaciones de Scopus (CSV) y Web of Science (texto delimitado por tabulaciones).
#    Se eliminan duplicados por DOI y, en su defecto, por título + año.
#    El orden de lectura importa (decide qué registro se conserva entre duplicados y el orden de los documentos):
#    se usa el mismo de la aplicación, Scopus primero y después los lotes de WoS.
def es_scopus(p):
    if p.suffix.lower() == ".csv":
        return True
    if p.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(p) as z:
            return any(n.lower().endswith(".csv") for n in z.namelist())
    return False


ORDEN = {n: i for i, n in enumerate(CONFIG.get("input_files") or [])}
archivos = sorted((p for p in DATOS.iterdir() if p.suffix.lower() in (".csv", ".txt", ".tsv", ".zip")),
                  key=lambda p: (ORDEN.get(p.name, len(ORDEN)), 0 if es_scopus(p) else 1, p.name))
if not archivos:
    sys.exit(f"No hay exportaciones en {DATOS}. Copie allí los archivos de Scopus y WoS "
             f"(en el análisis original: {', '.join(CONFIG['input_files']) or 'no registrados'}).")
S = pipeline.Session()
resumen = S.load([(p.name, p.read_bytes()) for p in archivos])
print("Registros:", resumen["stats"])

# 2. Filtros de elegibilidad (años, tipos de documento, idiomas, resumen) y preprocesamiento del texto
#    (minúsculas, palabras vacías, singularización, bigramas por NPMI, vocabulario por frecuencia documental).
print("Preprocesamiento:", S.prepare(CONFIG["parameters"]))

# 3. LDA: selección de K con CaoJuan2009, Arun2010, Deveaud2014 y NPMI; modelo final y estabilidad entre semillas.
if S.on("lda"):
    print("LDA:", S.run_lda(progreso))

# 4. BERTopic: embeddings de los documentos -> UMAP (5D) -> HDBSCAN -> c-TF-IDF con el mismo vocabulario que LDA.
if S.on("bertopic"):
    from umap import UMAP
    textos = S.texts()
    if CONFIG["embedding"]["kind"] == "minilm":
        # Mismo modelo que en el navegador (all-MiniLM-L6-v2), aquí mediante sentence-transformers.
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(
            textos, normalize_embeddings=True, show_progress_bar=True)
        nombre = "all-MiniLM-L6-v2 (sentence-transformers)"
    else:
        emb, nombre = lsa_embeddings(textos), "LSA fallback (TF-IDF + SVD)"
    semilla = S.cfg["seed"]
    # Una proyección UMAP por réplica de estabilidad (semilla + r) y una proyección 2D para el mapa de documentos.
    reducciones = [UMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=semilla + r).fit_transform(emb)
                   for r in range(S.n_umap_runs())]
    xy = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=semilla).fit_transform(emb)
    print("BERTopic:", S.run_bertopic([np.asarray(r) for r in reducciones], np.asarray(xy), nombre))

# 5. Nombres de los temas asignados por los autores en la aplicación.
if CONFIG.get("topic_labels"):
    S.set_labels(CONFIG["topic_labels"])

# 6. Validación por expertos (opcional): hojas completadas por los evaluadores en la carpeta validacion/.
hojas = sorted((BASE / "validacion").glob("*.xlsx")) if (BASE / "validacion").exists() else []
if hojas:
    print("Validación:", S.load_validation([(p.name, p.read_bytes()) for p in hojas])["alpha"])

# 7. Textos de la asistencia con IA usados en el documento (si los hubo).
if CONFIG.get("ai_texts"):
    S.set_ai_text(CONFIG["ai_texts"])

# 8. Figuras (PNG, PDF y TIFF 600 dpi), tablas (Excel/CSV) y borrador del artículo en Word.
S.render(str(SALIDA), final=True)
report.build(S, str(SALIDA / "Borrador_Metodos_Resultados.docx"), CONFIG["document"], CONFIG.get("sections"))
print("Resultados en:", SALIDA)
'''

README = """# Script del análisis — BiblioTopic

Este paquete contiene el código y la configuración exacta con que se realizó el análisis en la aplicación web
BiblioTopic ({date}). Sirve como material suplementario del artículo y permite reproducir los resultados.

## Contenido

| Archivo | Descripción |
|---|---|
| `analisis_bibliotopic.py` | Script principal, comentado paso a paso. |
| `config.json` | Parámetros del análisis, semilla aleatoria, nombres de los temas y datos del documento. |
| `bibliotopic/` | Código de análisis (el mismo que se ejecutó en el navegador). |
| `requirements.txt` | Librerías de Python con las versiones utilizadas. |
| `datos/` | Copie aquí las exportaciones de Scopus y Web of Science. |

Por razones de licencia, los registros de Scopus y Web of Science **no se incluyen**. Archivos usados en el análisis
original: {files}.

## Cómo ejecutarlo

1. Instale Python 3.11 o 3.12.
2. Instale las librerías:

       pip install -r requirements.txt

3. Copie las exportaciones de Scopus (CSV) y Web of Science (.txt) en la carpeta `datos/`.
4. Opcional: copie las hojas de validación completadas por los evaluadores en una carpeta `validacion/`.
5. Ejecute:

       python analisis_bibliotopic.py

Los resultados se guardan en `resultados/`.

## Parámetros principales

{params}

## Reproducibilidad

- Con los mismos archivos, la semilla indicada y las versiones de `requirements.txt`, el filtrado, la bibliometría
  y LDA dan los mismos resultados que en la aplicación. El script lee los archivos en el mismo orden que la
  aplicación (Scopus primero y después los lotes de WoS); conserve los nombres originales de los archivos.
- En BERTopic, el navegador usa transformers.js y umap-js; el script usa sentence-transformers y umap-learn con el
  mismo modelo ({embedding}) y los mismos parámetros. Las implementaciones de UMAP no son idénticas, por lo que la
  composición exacta de los grupos de BERTopic puede variar ligeramente; los temas principales deberían coincidir.
- Si el número de temas se eligió automáticamente, la selección se repite con el mismo procedimiento. Para fijarlo,
  cambie `"k"` en `config.json` (en el análisis original se obtuvo K = {k}).
"""


def _params_md(c, k):
    rows = [
        ("Años", f"{c.get('year_min') or 'todos'} – {c.get('year_max') or 'todos'}"),
        ("Tipos de documento", ", ".join(c.get("doc_types") or []) or "todos"),
        ("Idiomas", ", ".join(c.get("languages") or []) or "todos"),
        ("Análisis", ", ".join(c.get("analyses") or [])),
        ("K de LDA", f"fijo = {c['k']}" if c.get("k") else f"automático entre {c.get('k_min')} y {c.get('k_max')} "
                                                            f"(paso {c.get('k_step')}); seleccionado K = {k}"),
        ("Frecuencia documental", f"mínima {c.get('min_df')}, máxima {c.get('max_df')}"),
        ("Semilla aleatoria", c.get("seed")),
        ("Remuestreos bootstrap", c.get("n_boot")),
    ]
    return "\n".join(f"- **{a}:** {b}" for a, b in rows)


def build_package(S, meta=None) -> bytes:
    meta = dict(meta or {})
    pkg = Path(__file__).resolve().parent
    emb = (S.emb_name or "").lower()
    kind = "minilm" if "minilm" in emb else ("lsa" if emb else "minilm")
    labels = {}
    for name, r in (("lda", S.lda), ("bert", S.bert)):
        if r is not None and getattr(r, "custom_labels", False):
            labels[name] = [l.split(":", 1)[1].strip() if ":" in l else l for l in r.labels]
            labels[name + "_short"] = S._shorts(r)
    k = S.lda.k if S.lda is not None else None
    params = {key: v for key, v in S.cfg.items()}
    config = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "generator": "BiblioTopic (https://github.com/CarlosSalasMacias/BiblioTopic)",
        "input_files": list(getattr(S, "input_files", []) or []),
        "parameters": params,
        "lda_k_selected": k,
        "embedding": {"kind": kind, "used_in_browser": S.emb_name or ""},
        "topic_labels": labels,
        "ai_texts": {kk: v for kk, v in (getattr(S, "ai", None) or {}).items() if kk in ("discussion", "labels_used")},
        "document": {kk: meta.get(kk, "") for kk in ("topic_name", "title", "query", "search_date", "composite",
                                                      "ai_model", "ai_title")},
        "sections": meta.get("sections"),
        "versions_used": S.versions,
    }
    pins = {"pandas", "numpy", "scikit-learn", "scipy", "networkx", "matplotlib"}
    req = [f"{p}=={S.versions[p]}" for p in sorted(pins) if S.versions.get(p)]
    req += ["openpyxl>=3.1", "python-docx>=1.1", "pycountry>=23.12", "adjustText>=1.0", "Pillow>=10",
            "umap-learn>=0.5.5"]
    if kind == "minilm":
        req.append("sentence-transformers>=2.7")
    readme = README.format(
        date=_dt.date.today().isoformat(),
        files=", ".join(config["input_files"]) or "no registrados",
        params=_params_md(S.cfg, k),
        embedding="all-MiniLM-L6-v2" if kind == "minilm" else "LSA (TF-IDF + SVD)",
        k=k if k is not None else "—",
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("BiblioTopic_script/analisis_bibliotopic.py", SCRIPT)
        z.writestr("BiblioTopic_script/config.json", json.dumps(config, indent=2, ensure_ascii=False, default=str))
        z.writestr("BiblioTopic_script/requirements.txt", "\n".join(req) + "\n")
        z.writestr("BiblioTopic_script/LEEME.md", readme)
        z.writestr("BiblioTopic_script/datos/LEEME.txt",
                   "Copie aquí las exportaciones de Scopus (CSV) y Web of Science (.txt).\n")
        for p in sorted(pkg.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json"):
                z.write(p, "BiblioTopic_script/bibliotopic/" + str(p.relative_to(pkg)))
    return buf.getvalue()
