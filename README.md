# BiblioTopic

Aplicación web de código abierto para analizar corpus de **Web of Science** y **Scopus** y obtener figuras,
tablas y un borrador del artículo (Métodos, Resultados y, opcionalmente, Discusión).

**Abrir la aplicación:** https://carlossalasmacias.github.io/BiblioTopic/

Todo se ejecuta en el navegador: los archivos no se suben a ningún servidor y no hace falta instalar nada.
Se recomienda Chrome o Edge actualizados y una computadora con al menos 8 GB de RAM.

## Qué hace

**Datos**
- Fusiona el CSV de Scopus y los lotes de WoS (también en .zip), elimina duplicados (DOI y título + año) y genera
  un diagrama de flujo tipo PRISMA.
- Normaliza palabras clave y admite una lista de sinónimos.
- Permite guardar el proyecto en un archivo `.bibliotopic` y reabrirlo más tarde.

**Bibliometría**
- Producción anual, fuentes, Bradford, Lotka y autores.
- Países (SCP/MCP), mapa mundial de colaboración y red de colaboración.
- Red de palabras clave, al estilo bibliometrix.
- Documentos más citados (globales y por tema) y red de co-citación (cuando la exportación trae las referencias citadas).
- Mapa temático de Callon (temas motores, nicho, emergentes y básicos) y evolución temática en diagrama de Sankey.

**Modelado de temas**
- **LDA:** selección de K por cuatro métricas (CaoJuan2009, Arun2010, Deveaud2014, NPMI) y estabilidad entre semillas.
- **BERTopic:** embeddings de lenguaje (all-MiniLM-L6-v2), UMAP, HDBSCAN y c-TF-IDF, con el mismo vocabulario que LDA.
- **Dinámica temporal:** temas emergentes y en declive.
- **HJ-Biplot** de periodos × temas y de países × temas, con conglomerados de Ward (k elegido por silueta),
  mostrados como envolventes o elipses.

**Comparación LDA vs BERTopic**
- Coherencia, diversidad, estabilidad, emparejamiento húngaro de temas, ARI/NMI e Índice de Robustez Temática (TRI).
- Intervalos de confianza del 95 % por bootstrap y pruebas U de Mann-Whitney con tamaño del efecto.

**Validación por expertos**
- Hoja de Excel ciega respecto al modelo: calificación de 1 a 5 y prueba de la palabra intrusa.
- Lectura de las hojas completadas y cálculo del alfa de Krippendorff.

**Asistencia con IA (opcional, con su propia clave de API de Anthropic)**
- Propuesta de nombres para los temas.
- Borrador de la Discusión en español que solo cita documentos del propio corpus; donde hace falta literatura
  externa deja la marca `[cita externa]`.

**Salidas**
- Documento Word con los apartados elegidos: texto en español; título, figuras y tablas en inglés.
- Lista de verificación de reporte y declaración de disponibilidad de datos y código.
- Figuras a doble columna (175 mm) en TIFF 600 dpi, PNG 300 dpi y PDF vectorial, con figuras compuestas
  con paneles (a), (b), (c).
- Todas las tablas en Excel.

## Uso

1. Suba el CSV de Scopus y/o uno o varios lotes de WoS.
2. Marque los análisis (y, si lo desea, ajuste las opciones avanzadas) y pulse **Ejecutar análisis**.
3. Ponga nombre a cada tema (o pida propuestas a la IA) y pulse **Aplicar nombres**.
4. Opcional: descargue la hoja de validación, envíela a los evaluadores y cargue las hojas completadas.
5. Elija los apartados del documento y descargue el Word y el .zip de figuras y tablas.
6. Use **Guardar proyecto** para continuar otro día sin repetir el análisis.

### Cómo exportar los datos

- **Web of Science:** *Export → Tab delimited file*, **Record Content: Full Record and Cited References**
  (de 1 000 en 1 000 registros; suba todos los lotes o un .zip).
- **Scopus:** *Export → CSV* con todos los campos. Para la co-citación incluya *References*.

## Cómo citar

Si usa BiblioTopic, cítelo con el DOI de Zenodo (ver «Cite this repository» en GitHub o el archivo
`CITATION.cff`).

### Publicar una versión con DOI en Zenodo

1. En https://zenodo.org/account/settings/github/ active el repositorio `CarlosSalasMacias/BiblioTopic`.
2. En GitHub cree una *release* (por ejemplo `v1.0.0`) desde *Releases → Draft a new release*.
3. Zenodo archiva la versión y asigna un DOI; añada ese DOI a este README y al texto de disponibilidad.

## Estructura

| Carpeta | Contenido |
|---|---|
| `index.html`, `css/`, `js/` | Interfaz web. `js/worker.js` carga Python (Pyodide), transformers.js y umap-js; `js/ai.js` llama a la API de Claude. |
| `bibliotopic/` | Motor de análisis en Python (el mismo código se ejecuta en el navegador y en local). |
| `run_cli.py` | Ejecución local opcional sin navegador (`pip install -r requirements.txt`). |

## Licencia

MIT. Los mapas usan Natural Earth (dominio público).
