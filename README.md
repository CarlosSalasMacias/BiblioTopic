# BiblioTopic

Aplicación web para analizar corpus de **Web of Science** y **Scopus** y obtener un borrador de
*Materiales y métodos* y *Resultados*.

**Abrir la aplicación:** https://carlossalasmacias.github.io/BiblioTopic/

Todo se ejecuta en el navegador: los archivos no se suben a ningún servidor y no hace falta instalar nada.
Se recomienda Chrome o Edge actualizados en una computadora con al menos 8 GB de RAM.

## Qué hace

1. **Fusión del corpus**: une el CSV de Scopus y los lotes de WoS, elimina duplicados (DOI y título + año)
   y genera un diagrama de flujo tipo PRISMA.
2. **Bibliometría**: producción anual, fuentes, Bradford, Lotka, autores, países (SCP/MCP), red de colaboración
   internacional y red de palabras clave.
3. **LDA** con selección de K por cuatro métricas (CaoJuan2009, Arun2010, Deveaud2014, NPMI).
4. **BERTopic**: embeddings de lenguaje (all-MiniLM-L6-v2) → UMAP → HDBSCAN → c-TF-IDF, con el mismo
   vocabulario que LDA.
5. **Dinámica temporal** (temas emergentes y en declive) y **HJ-Biplot** (periodos × temas, países × temas).
6. **Comparación LDA vs BERTopic**: coherencia, diversidad, estabilidad entre semillas, emparejamiento de temas,
   ARI/NMI e **Índice de Robustez Temática (TRI)**.
7. **Documento Word** con los apartados elegidos (texto en español; título, figuras y tablas en inglés),
   y un .zip con todas las figuras (PNG 300 dpi) y tablas (Excel).

## Uso

1. Suba el CSV de Scopus y/o uno o varios lotes de WoS (también acepta .zip).
2. Marque los análisis que quiera realizar (y, si lo desea, ajuste las opciones avanzadas).
3. Pulse **Ejecutar análisis**. Con ~5 000 documentos tarda entre 15 y 40 minutos según el equipo;
   no cierre la pestaña.
4. Ponga nombre a cada tema y pulse **Aplicar nombres**.
5. Elija los apartados del documento y descárguelo.

### Cómo exportar los datos

- **Web of Science:** *Export → Tab delimited file*, **Record Content: Full Record and Cited References**
  (de 1 000 en 1 000 registros; suba todos los lotes).
- **Scopus:** *Export → CSV* con todos los campos. Si el archivo es muy grande puede omitir *References*.

## Estructura

| Carpeta | Contenido |
|---|---|
| `index.html`, `css/`, `js/` | Interfaz web. `js/worker.js` carga Python (Pyodide), transformers.js y umap-js. |
| `bibliotopic/` | Motor de análisis en Python (el mismo código se ejecuta en el navegador y en local). |
| `run_cli.py` | Ejecución local opcional sin navegador (`pip install -r requirements.txt`). |

## Publicación en GitHub Pages

*Settings → Pages → Build and deployment → Source: Deploy from a branch*, rama `main`, carpeta `/ (root)`.
