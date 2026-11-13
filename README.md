# BiblioTopic

Aplicación web de código abierto para analizar corpus de **Web of Science** y **Scopus** y obtener figuras,
tablas y un borrador del artículo (Métodos, Resultados y, opcionalmente, Discusión).

**Abrir la aplicación:** https://carlossalasmacias.github.io/BiblioTopic/

Todo se ejecuta en el navegador: los archivos no se suben a ningún servidor y no hace falta instalar nada.
Se recomienda Chrome o Edge actualizados y una computadora con al menos 8 GB de RAM.

## Qué hace

**Datos**
- Fusiona el CSV de Scopus y los lotes de WoS (también en .zip) y elimina duplicados: DOI o título normalizado con
  el mismo año o uno contiguo, con agrupación transitiva (no depende del orden de los archivos) y sin unir por título
  registros con DOI distintos. Los campos vacíos se completan con los de los duplicados, con la procedencia de cada
  campo, los registros originales vinculados y los conflictos (DOI, año, idioma) en tablas propias.
- Idioma y tipo de documento con tres estados: elegible, no elegible y **sin dato** (p. ej., Scopus sin columna de
  idioma). Los registros sin dato no se cuentan como «otro idioma»; se conservan por defecto y la decisión es
  explícita en la interfaz y en el informe, con su recuento por base.
- Diagrama de flujo tipo PRISMA.
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
- **Dinámica temporal:** pendiente de la prevalencia anual con IC 95 %, valores p ajustados por Benjamini-Hochberg,
  exclusión del año en curso (incompleto) y cuatro estados: creciente, decreciente, sin tendencia significativa y no
  estimable (menos de 3 años con datos suficientes). Son descripciones exploratorias del corpus recuperado.
- **Trazabilidad de BERTopic:** por documento se guardan la asignación original del agrupamiento, la final, la
  similitud con el tema, el motivo y una marca de baja confianza; prevalencia y tendencias también sin reasignar
  los outliers (análisis de sensibilidad).
- **HJ-Biplot** de periodos × temas y de países × temas, con conglomerados de Ward (k elegido por silueta),
  mostrados como envolventes convexas o elipses (una sola forma por figura y sin superposición).

**Comparación LDA vs BERTopic**
- Coherencia, diversidad, estabilidad, emparejamiento húngaro de temas (con términos compartidos, documento
  representativo de cada tema y marca de correspondencia no convincente), e Índice de Robustez Temática (TRI,
  no disponible cuando falta la estabilidad).
- ARI/NMI para dos poblaciones, cada una con su n e intervalo: todos los documentos (outliers reasignados) y solo
  los documentos agrupados originalmente.
- Intervalos de confianza del 95 % por bootstrap y pruebas U de Mann-Whitney con tamaño del efecto. Los intervalos
  bootstrap están condicionados a los temas ajustados; la variación al reentrenar se aproxima con la estabilidad
  entre semillas.

**Validación por expertos**
- Hoja de Excel ciega respecto al modelo: calificación de 1 a 5 y prueba de la palabra intrusa.
- Lectura de las hojas completadas y cálculo del alfa de Krippendorff.

**Asistencia con IA (opcional, con su propia clave de API de Anthropic)**
- Propuesta de nombres para los temas y de títulos para el artículo.
- Borrador de la Discusión en español que solo cita documentos del propio corpus; donde hace falta literatura
  externa deja la marca `[cita externa]`.

**Salidas**
- Documento Word con los apartados elegidos: texto en español; título, figuras y tablas en inglés; texto en negro
  y figuras sin título (la descripción va en la leyenda).
- Lista de verificación de reporte y declaración de disponibilidad de datos y código.
- Figuras a doble columna (175 mm) en TIFF 600 dpi, PNG 300 dpi y PDF vectorial, con figuras compuestas
  con paneles (a), (b), (c).
- Todas las tablas en Excel.
- **Script del análisis** (.zip): el código en Python, la configuración exacta (parámetros, semilla, nombres de los
  temas), las versiones de las librerías e instrucciones para reproducir el análisis fuera del navegador. Se
  descarga con su propio botón y también va dentro del .zip de resultados, listo para el material suplementario.

## Uso

1. Suba el CSV de Scopus y/o uno o varios lotes de WoS.
2. Marque los análisis (y, si lo desea, ajuste las opciones avanzadas) y pulse **Ejecutar análisis**.
3. Ponga nombre a cada tema (o pida propuestas a la IA) y pulse **Aplicar nombres**.
4. Opcional: descargue la hoja de validación, envíela a los evaluadores y cargue las hojas completadas.
5. Elija el título (la aplicación propone varios), los apartados del documento y descargue el Word y el .zip de figuras y tablas.
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
| `tests/` | Pruebas automáticas con corpus sintéticos pequeños (`pip install pytest` y `pytest`): importación, idiomas sin dato, deduplicación, denominadores, outliers, tendencias no estimables y coherencia entre tablas y texto del borrador. Se ejecutan en GitHub Actions en cada cambio. |

## Licencia

MIT. Los mapas usan Natural Earth (dominio público).
