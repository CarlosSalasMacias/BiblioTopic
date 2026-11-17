# BiblioTopic

Aplicación web de código abierto para analizar corpus de **Web of Science** y **Scopus** y obtener figuras,
tablas y un borrador del artículo (Métodos, Resultados y, opcionalmente, Discusión).

**Abrir la aplicación:** https://carlossalasmacias.github.io/BiblioTopic/

Todo se ejecuta en el navegador: los archivos no se suben a ningún servidor y no hace falta instalar nada.
Se recomienda Chrome o Edge actualizados y una computadora con al menos 8 GB de RAM.

**Versión 2.0.0** (para publicación en revistas Q1). Los cambios, su justificación y la tabla de valores por
defecto nuevos frente a los anteriores están en [CHANGELOG.md](CHANGELOG.md). Las configuraciones de la versión 1
(`config.json` sin `config_version`) siguen ejecutándose con sus valores originales.

## Perfiles

| Perfil | Para qué | Valores principales |
|---|---|---|
| **Publicación** (por defecto) | Resultados para el artículo | bootstrap 1000 (opción BCa), 1000 permutaciones, estabilidad con 10 semillas (LDA) y 10 semillas de UMAP (BERTopic), K en dos etapas (5–40 paso 5 y ±5 paso 1) con 3 semillas por K y 5 métricas, LDA batch hasta 100 iteraciones con priors 1/K, UMAP coseno |
| **Exploración rápida** | Primera mirada al corpus | bootstrap 200, 3 semillas, K 5–30 paso 5 con 1 semilla, LDA online 20 iteraciones |

Antes de ejecutar, la aplicación estima el tiempo en el navegador. Con corpus grandes (~9 000 documentos) el perfil
Publicación tarda ~4–5 h en el navegador; en ese caso conviene descargar el script (botón **Descargar script con
esta configuración**, disponible antes de ejecutar) y correrlo en una computadora con varios núcleos:

```bash
python analisis_bibliotopic.py --nucleos 8
```

(~20–30 min con 8 núcleos). Ningún valor se rebaja automáticamente.

## Qué hace

**Datos**
- Fusiona el CSV de Scopus y los lotes de WoS (también en .zip) y elimina duplicados: DOI o título normalizado con
  el mismo año o uno contiguo, con agrupación transitiva (no depende del orden de los archivos) y sin unir por título
  registros con DOI distintos. Los campos vacíos se completan con los de los duplicados, con la procedencia de cada
  campo, los registros originales vinculados y los conflictos (DOI, año, idioma) en tablas propias.
- **Ningún registro se descarta por un campo ausente.** Idioma, tipo de documento y año tienen tres estados:
  elegible, no elegible y **sin dato** (p. ej., Scopus sin columna de idioma). Los registros sin dato se conservan
  por defecto; excluirlos es una opción explícita (desactivada), con su recuento por base en el informe.
- Dos corpus: **bibliométrico** (todos los registros elegibles, aunque no tengan resumen o año) y **de modelado**
  (resumen de al menos 20 palabras y año).
- **Auditoría de campos** al cargar (AB, DE, ID, LA, DT, PY, DI, TC, C1, RP, CR), por base y por archivo. Si LA, DT,
  AB, RP/C1 o DE están por debajo del 90 % en alguna base, un aviso rojo bloquea el análisis hasta confirmarlo.
- Tipos de documento armonizados entre bases: Article, Review, Conference paper, Book chapter, Book, Other
  («Article; Early Access» = Article). Idiomas múltiples de Scopus separados (se conserva el registro si alguno
  está admitido).
- Citas de WoS y de Scopus por separado (`TC_WoS`, `TC_Scopus`); por defecto se usa el máximo.
- Limpieza de etiquetas HTML/XML y de avisos de copyright en títulos, resúmenes y palabras clave.
- Ecuaciones de búsqueda de cada base, fecha e índices de WoS: se guardan en `run_config.json` y en el informe, y
  de ellas se obtienen automáticamente los términos de búsqueda que se eliminan del texto (editables).
- Diagrama de flujo tipo PRISMA con excluidos, conservados sin dato y no utilizables para el modelado.
- Normaliza palabras clave (singular, guion/espacio, británico/estadounidense) y admite un CSV de sinónimos
  `término,forma_canónica` que se aplica a las palabras clave y al vocabulario de los modelos. Muestra las 50
  palabras clave más frecuentes antes del análisis.
- Permite guardar el proyecto en un archivo `.bibliotopic` y reabrirlo más tarde.

**Bibliometría**
- Producción anual, fuentes, Bradford, Lotka y autores.
- Países (SCP/MCP), mapa mundial de colaboración y red de colaboración.
- Red de palabras clave, al estilo bibliometrix.
- Documentos más citados (globales y por tema) y red de co-citación (cuando la exportación trae las referencias citadas).
- Mapa temático de Callon (temas motores, nicho, emergentes y básicos) y evolución temática en diagrama de Sankey.

**Modelado de temas**
- **LDA:** selección de K en dos etapas (gruesa y fina) con varias semillas por K y cinco métricas (CaoJuan2009,
  Arun2010, Deveaud2014, NPMI y estabilidad de Greene et al., 2014); ajuste batch con priors explícitos y
  estabilidad por tema (media ± DE entre semillas).
- **BERTopic:** embeddings de lenguaje (all-MiniLM-L6-v2 u, opcionalmente, all-mpnet-base-v2), UMAP con distancia
  coseno, HDBSCAN (`min_cluster_size` = máx(15, 0.5 % N)) y c-TF-IDF, con el mismo vocabulario que LDA; umbral
  configurable para reasignar outliers, con el % antes y después y aviso si supera el 30 %.
- **Dinámica temporal:** Mann-Kendall con pendiente de Sen (y regresión lineal), valores q de Benjamini-Hochberg,
  exclusión del año en curso (incompleto) y cuatro estados: creciente, decreciente, sin tendencia significativa y no
  estimable. Periodos de ancho fijo, con un número similar de documentos o personalizados, con aviso si el primer
  periodo es demasiado amplio. Son descripciones exploratorias del corpus recuperado.
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
- Mejor correspondencia de cada tema (muchos a uno), línea base nula del coseno por permutación (percentil 95 en
  la figura), IC 95 % del TRI por par y sensibilidad a los umbrales τ.
- Intervalos de confianza del 95 % por bootstrap (percentil o BCa) y pruebas U de Mann-Whitney con tamaño del
  efecto. Cada métrica tiene una sola estimación puntual (datos completos) en tablas y texto; el bootstrap solo
  aporta el intervalo. Los intervalos
  bootstrap están condicionados a los temas ajustados; la variación al reentrenar se aproxima con la estabilidad
  entre semillas.

**Validación por expertos**
- Hoja de Excel ciega respecto al modelo: calificación de 1 a 5 y prueba de la palabra intrusa.
- Lectura de las hojas completadas: alfa de Krippendorff, kappa de Fleiss y precisión de la palabra intrusa por tema.

**Controles de calidad**
- Panel con semáforo antes de los resultados (bases sin registros exclusivos, exclusiones altas, campos críticos
  incompletos, temas inestables o vacíos, outliers > 30 %, términos de búsqueda vacíos, referencias ausentes).
- Acoplamiento bibliográfico además de la co-citación cuando hay referencias citadas; si no las hay, se indica.

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
- **`run_config.json`**: versión, perfil, todos los parámetros efectivos (valores automáticos ya resueltos), archivos
  de entrada con MD5 y número de registros, ecuaciones y fecha de búsqueda, disponibilidad de campos por base y
  todas las semillas.
- **Script del análisis** (.zip): el código en Python (con `--nucleos N` para paralelizar), la configuración exacta
  (parámetros, semillas, nombres de los temas), las versiones de las librerías, la lista de archivos de entrada con
  su MD5 (el script la comprueba) e instrucciones para reproducir el análisis fuera del navegador. Se
  descarga con su propio botón y también va dentro del .zip de resultados, listo para el material suplementario.

## Uso

1. Suba el CSV de Scopus y/o uno o varios lotes de WoS. Revise la auditoría de campos y, si hay un aviso rojo,
   vuelva a exportar o confirme que desea continuar.
2. Pegue las ecuaciones de búsqueda y la fecha, elija el perfil, revise la estimación de tiempo y las palabras
   clave más frecuentes (y, si lo desea, cargue un CSV de sinónimos o ajuste las opciones avanzadas). Pulse
   **Ejecutar análisis**.
3. Ponga nombre a cada tema (o pida propuestas a la IA) y pulse **Aplicar nombres**.
4. Opcional: descargue la hoja de validación, envíela a los evaluadores y cargue las hojas completadas.
5. Elija el título (la aplicación propone varios), los apartados del documento y descargue el Word y el .zip de figuras y tablas.
6. Use **Guardar proyecto** para continuar otro día sin repetir el análisis.

### Cómo exportar los datos

- **Web of Science:** *Export → Tab delimited file*, **Record Content: Full Record and Cited References**
  (de 1 000 en 1 000 registros; suba todos los lotes o un .zip).
- **Scopus:** *Export → CSV* marcando todos los grupos de campos. Compruebe en particular *Language of Original
  Document*, *Document Type*, *Abstract*, *Author Keywords*, *Index Keywords*, *Affiliations*, *Correspondence
  Address* y, para la co-citación, *References*. Sin *Language of Original Document* los registros de Scopus
  quedan como «idioma sin dato» (se conservan, pero la auditoría lo marca en rojo).

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
| `tests/` | Pruebas automáticas con corpus sintéticos pequeños (`pip install pytest` y `pytest`): importadores (Scopus con y sin idioma, WoS con etiquetas múltiples; ningún registro se pierde por un campo ausente), tipos de documento, limpieza de etiquetas, citas por base, país de correspondencia, términos de búsqueda, sinónimos, compatibilidad con `config_version`, deduplicación, Mann-Kendall, permutaciones, BCa, kappa de Fleiss y coherencia entre tablas y texto del borrador. Se ejecutan en GitHub Actions en cada cambio. |

## Licencia

MIT. Los mapas usan Natural Earth (dominio público).
