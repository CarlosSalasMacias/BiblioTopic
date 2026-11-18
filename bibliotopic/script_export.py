"""Paquete de reproducibilidad: el código y la configuración exacta del análisis realizado en la aplicación.

Contenido del .zip:
  analisis_bibliotopic.py   script principal, comentado paso a paso
  config.json               configuración efectiva (config_version, perfil, parámetros, semillas, archivos y MD5)
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

Reproduce el análisis de la aplicación web con la configuración de config.json.

Uso:
    pip install -r requirements.txt
    python analisis_bibliotopic.py                 # lee las exportaciones de la carpeta datos/
    python analisis_bibliotopic.py --nucleos 8     # ajustes de LDA en paralelo (selección de K y estabilidad)
    python analisis_bibliotopic.py otra/carpeta    # otra carpeta de datos

Los resultados (figuras, tablas, run_config.json y borrador en Word) se guardan en resultados/.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from bibliotopic import pipeline, report
from bibliotopic.bert_model import lsa_embeddings

BASE = Path(__file__).resolve().parent
CONFIG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
ap = argparse.ArgumentParser()
ap.add_argument("datos", nargs="?", default=str(BASE / "datos"))
ap.add_argument("--nucleos", type=int, default=1, help="procesos para los ajustes de LDA (por defecto 1)")
args = ap.parse_args()
DATOS, SALIDA = Path(args.datos), BASE / "resultados"
PARAMS = dict(CONFIG["parameters"])
# config.json sin config_version (versión 1 de BiblioTopic): se reproducen los valores de esa versión
PARAMS.setdefault("config_version", CONFIG.get("config_version", 1))


def progreso(p, msg):
    print(f"  [{p:4.0%}] {msg}", flush=True)


# 1. Lectura y fusión de las exportaciones de Scopus (CSV) y Web of Science (texto delimitado por tabulaciones).
#    Duplicados por DOI o por título + año (±1), con agrupación transitiva y campos complementados.
#    Se usa el mismo orden de archivos que en la aplicación y se comprueba el MD5 registrado de cada archivo.
def es_scopus(p):
    if p.suffix.lower() == ".csv":
        return True
    if p.suffix.lower() == ".zip":
        import zipfile
        with zipfile.ZipFile(p) as z:
            return any(n.lower().endswith(".csv") for n in z.namelist())
    return False


REG = {f["file"]: f for f in CONFIG.get("input_files_detail") or []}
ORDEN = {n: i for i, n in enumerate(CONFIG.get("input_files") or [])}
archivos = sorted((p for p in DATOS.iterdir() if p.suffix.lower() in (".csv", ".txt", ".tsv", ".zip")),
                  key=lambda p: (ORDEN.get(p.name, len(ORDEN)), 0 if es_scopus(p) else 1, p.name))
if not archivos:
    sys.exit(f"No hay exportaciones en {DATOS}. Copie allí los archivos de Scopus y WoS "
             f"(en el análisis original: {', '.join(CONFIG.get('input_files') or []) or 'ver config.json'}).")
for p in archivos:
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    if p.name in REG and REG[p.name].get("md5") and REG[p.name]["md5"] != md5:
        print(f"AVISO: {p.name} no coincide con el archivo original (MD5 distinto); los resultados pueden variar.")
S = pipeline.Session()
resumen = S.load([(p.name, p.read_bytes()) for p in archivos])
print("Registros:", resumen["stats"])

# 2. Criterios de elegibilidad (sin descartar registros por campos ausentes) y preprocesamiento del texto.
print("Preprocesamiento:", S.prepare(PARAMS))

# 3. LDA: selección de K en dos etapas (5 métricas promediadas en varias semillas), modelo final y estabilidad.
if S.on("lda"):
    if args.nucleos > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.nucleos) as ex:
            print("LDA:", S.run_lda(progreso, executor=ex))
    else:
        print("LDA:", S.run_lda(progreso))

# 4. BERTopic: embeddings -> UMAP (coseno, 5D; una proyección por semilla de estabilidad) -> HDBSCAN -> c-TF-IDF.
if S.on("bertopic"):
    from umap import UMAP
    textos = S.texts()
    modelo = PARAMS.get("embedding_model") or "Xenova/all-MiniLM-L6-v2"
    if CONFIG["embedding"]["kind"] != "lsa":
        # Mismo modelo que en el navegador, aquí mediante sentence-transformers
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("sentence-transformers/" + modelo.split("/")[-1]).encode(
            textos, normalize_embeddings=True, show_progress_bar=True)
        nombre = modelo.split("/")[-1] + " (sentence-transformers)"
    else:
        emb, nombre = lsa_embeddings(textos), "LSA fallback (TF-IDF + SVD)"
    us = S.umap_settings()
    reducciones = [UMAP(n_neighbors=us["n_neighbors"], n_components=us["n_components"], min_dist=us["min_dist"],
                        metric=us["metric"], random_state=s).fit_transform(emb) for s in us["seeds"]]
    xy = UMAP(n_neighbors=us["n_neighbors"], n_components=2, min_dist=us["map_2d"]["min_dist"],
              metric=us["metric"], random_state=us["map_2d"]["seed"]).fit_transform(emb)
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

# 8. Figuras (PNG, PDF y TIFF 600 dpi), tablas, run_config.json, controles de calidad y borrador en Word.
S.render(str(SALIDA), final=True)
for c in S.quality_checks():
    print(f"  [{c['level']}] {c['check']}: {c['detail']}")
report.build(S, str(SALIDA / "Borrador_Metodos_Resultados.docx"), CONFIG["document"], CONFIG.get("sections"))
print("Resultados en:", SALIDA)
'''

README = """# Script del análisis — BiblioTopic {version}

Este paquete contiene el código y la configuración exacta con que se realizó (o se realizará) el análisis en la
aplicación web BiblioTopic ({date}). Sirve como material suplementario del artículo y permite reproducir los
resultados.

## Contenido

| Archivo | Descripción |
|---|---|
| `analisis_bibliotopic.py` | Script principal, comentado paso a paso. |
| `config.json` | Configuración efectiva (versión {config_version}, perfil «{profile}»): parámetros, semillas, archivos de entrada con su MD5, ecuaciones de búsqueda y disponibilidad de campos por base. |
| `bibliotopic/` | Código de análisis (el mismo que se ejecutó en el navegador). |
| `requirements.txt` | Librerías de Python con las versiones utilizadas. |
| `datos/` | Copie aquí las exportaciones de Scopus y Web of Science. |

Por razones de licencia, los registros de Scopus y Web of Science **no se incluyen**.

## Archivos de entrada del análisis original

{files}

## Cómo ejecutarlo

1. Instale Python 3.11 o 3.12.
2. Instale las librerías:

       pip install -r requirements.txt

3. Copie las exportaciones de Scopus (CSV) y Web of Science (.txt) en la carpeta `datos/`, con sus nombres
   originales (el script comprueba el MD5 de cada archivo y avisa si no coincide).
4. Opcional: copie las hojas de validación completadas por los evaluadores en una carpeta `validacion/`.
5. Ejecute, con tantos núcleos como tenga su equipo para acelerar la selección de K y la estabilidad de LDA:

       python analisis_bibliotopic.py --nucleos 8

Los resultados se guardan en `resultados/`. Tiempo estimado con un núcleo: {local_minutes} (con N núcleos,
aproximadamente esa cifra dividida entre N en la parte de LDA).

## Parámetros principales

{params}

## Reproducibilidad

- Con los mismos archivos, las semillas indicadas y las versiones de `requirements.txt`, la depuración, la
  bibliometría y LDA dan los mismos resultados que en la aplicación. El script lee los archivos en el mismo orden
  que la aplicación (Scopus primero y después los lotes de WoS).
- En BERTopic, el navegador usa transformers.js y umap-js; el script usa sentence-transformers y umap-learn con el
  mismo modelo ({embedding}), la misma métrica (coseno) y las mismas semillas. Las implementaciones de UMAP no son
  idénticas, por lo que la composición exacta de los grupos puede variar ligeramente.
- Un `config.json` sin el campo `config_version` (versión 1 de BiblioTopic) se ejecuta con los valores de esa
  versión para reproducir análisis anteriores.
- Si el número de temas se eligió automáticamente, la selección se repite con el mismo procedimiento. Para fijarlo,
  cambie `"k"` en `config.json` (en el análisis original se obtuvo K = {k}).
"""


def _params_md(c, k):
    yrs = f"{c.get('year_min') or 'todos'} – {c.get('year_max') or 'todos'}"
    rows = [
        ("Perfil", c.get("profile")),
        ("Años", yrs + (" (las tendencias excluyen el año en curso)" if c.get("exclude_incomplete_year", True) else "")),
        ("Tipos de documento", ", ".join(c.get("doc_types") or []) or "todos"),
        ("Idiomas", ", ".join(c.get("languages") or []) or "todos"),
        ("Registros sin idioma, tipo o año", "se conservan" if all(c.get(x, True) for x in (
            "keep_unknown_language", "keep_unknown_type", "keep_unknown_year")) else "algunos se excluyen (ver config.json)"),
        ("Análisis", ", ".join(c.get("analyses") or [])),
        ("K de LDA", f"fijo = {c['k']}" if c.get("k") else
         f"automático: búsqueda gruesa {c.get('k_min')}–{c.get('k_max')} (paso {c.get('k_step')}) y fina ±"
         f"{c.get('k_fine_window')}, con {c.get('k_seeds')} semillas por K; seleccionado K = {k if k else '—'}"),
        ("LDA", f"inferencia {c.get('lda_method')}, máximo {c.get('lda_max_iter')} iteraciones; priors α y η = 1/K "
                "salvo que se indique otro valor"),
        ("Estabilidad", f"LDA {c.get('n_stab_lda')} semillas; BERTopic {c.get('n_stab_bert')} semillas de UMAP"),
        ("Bootstrap / permutaciones", f"{c.get('n_boot')} remuestreos ({c.get('boot_method')}) / {c.get('n_perm')}"),
        ("Tendencias", {"mann_kendall_hr": "Mann-Kendall corregido por autocorrelación (Hamed-Rao) con pendiente de Sen",
                        "mann_kendall": "Mann-Kendall con pendiente de Sen"}.get(c.get("trend_test"), "regresión lineal")),
        ("Frecuencia documental", f"mínima {c.get('min_df')}, máxima {c.get('max_df')}"),
        ("Semilla global", c.get("seed")),
    ]
    return "\n".join(f"- **{a}:** {b}" for a, b in rows)


def build_package(S, meta=None, cfg=None) -> bytes:
    """cfg: configuración elegida en la interfaz antes de ejecutar el análisis (opcional)."""
    from .pipeline import __version__, estimate_runtime
    meta = dict(meta or {})
    pkg = Path(__file__).resolve().parent
    config = S.effective_config(meta, cfg)
    kind = config["embedding"]["kind"]
    c = config["parameters"]
    pins = {"pandas", "numpy", "scikit-learn", "scipy", "networkx", "matplotlib"}
    req = [f"{p}=={S.versions[p]}" for p in sorted(pins) if S.versions.get(p)]
    req += ["openpyxl>=3.1", "python-docx>=1.1", "pycountry>=23.12", "adjustText>=1.0", "Pillow>=10",
            "umap-learn>=0.5.5"]
    if kind != "lsa":
        req.append("sentence-transformers>=2.7")
    files = config.get("input_files_detail") or []
    files_md = ("\n".join(f"- `{f['file']}`: {f.get('records', '?')} registros; MD5 `{f.get('md5') or 'no registrado'}`"
                          for f in files)
                or "\n".join(f"- `{n}`" for n in config.get("input_files") or []) or "- (no registrados)")
    n_docs = len(S.df) if getattr(S, "df", None) is not None else int(
        (S.merged["AB"].str.split().str.len().fillna(0) >= 20).sum())
    est = estimate_runtime(n_docs, c, browser=False)["minutes"]
    readme = README.format(
        version=__version__, date=_dt.date.today().isoformat(), config_version=config["config_version"],
        profile=c.get("profile"), files=files_md, params=_params_md(c, config.get("lda_k_selected")),
        embedding=(c.get("embedding_model") or "all-MiniLM-L6-v2").split("/")[-1] if kind != "lsa"
        else "LSA (TF-IDF + SVD)",
        k=config.get("lda_k_selected") or "—",
        local_minutes=f"~{est:.0f} min" if est < 90 else f"~{est / 60:.1f} h")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("BiblioTopic_script/analisis_bibliotopic.py", SCRIPT)
        z.writestr("BiblioTopic_script/config.json", json.dumps(config, indent=2, ensure_ascii=False, default=str))
        z.writestr("BiblioTopic_script/requirements.txt", "\n".join(req) + "\n")
        z.writestr("BiblioTopic_script/LEEME.md", readme)
        z.writestr("BiblioTopic_script/datos/LEEME.txt",
                   "Copie aquí las exportaciones de Scopus (CSV) y Web of Science (.txt) con sus nombres originales.\n")
        for p in sorted(pkg.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json"):
                z.write(p, "BiblioTopic_script/bibliotopic/" + str(p.relative_to(pkg)))
    return buf.getvalue()
