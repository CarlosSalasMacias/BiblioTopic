# Registro de cambios

## 2.1.0 — Revisión metodológica (2026-11-18)

Cambios posteriores a la 2.0.0, a partir de las observaciones de árbitros. `run_config.json` registra la versión con la que se ejecutó cada análisis.

### Periodo de análisis

- Tras cargar y fusionar los archivos, un bloque visible «Periodo de análisis» (antes «Año desde/hasta» estaba oculto en
  las opciones avanzadas) muestra los registros por año, sugiere un año de inicio (primer año desde el que la
  producción es continua y cada año tiene al menos máx(2, 1 % del máximo anual) registros) e indica cuántos registros
  quedan fuera. El periodo se aplica a todos los análisis, figura en el PRISMA y en Métodos como criterio de
  elegibilidad («año de publicación entre … y …»).

### Proyectos guardados

- Al abrir un proyecto (`.bibliotopic`) el formulario recupera su configuración (periodo, K, semilla, réplicas,
  pruebas, tipos de documento, idiomas, análisis…) y el resumen del corpus (bloque del periodo incluido). Así se puede
  cambiar una opción, por ejemplo excluir el año siguiente, y pulsar «Ejecutar análisis» sin volver a cargar los
  archivos; el resto de opciones se mantienen. Antes el formulario quedaba con los valores por defecto.

### Análisis largos: descarga del modelo y punto de control tras LDA

- La descarga de la librería de embeddings y del modelo de lenguaje tiene un límite: si no llega ningún dato durante
  3 minutos se reintenta una vez y, si vuelve a ocurrir, el análisis se detiene con un mensaje claro. Antes podía
  quedarse esperando indefinidamente con «quedan menos de 2 min».
- Si la tarjeta gráfica falla al calcular los embeddings (WebGPU: «Device was lost», DXGI_ERROR_DEVICE_HUNG en
  Windows) o un lote no termina en 3 minutos, los embeddings se repiten en la CPU con el mismo modelo, con un aviso.
  Antes el análisis quedaba esperando indefinidamente.
- Punto de control al terminar LDA (además del que ya había tras BERTopic). Si BERTopic falla o se cuelga, al
  recargar la página aparece «Continuar el análisis», que ejecuta BERTopic, la comparación y las figuras con la misma
  configuración sin repetir LDA.
- Al abrir un proyecto guardado con LDA ajustado pero sin BERTopic, el análisis continúa desde ese punto.
- Un análisis nuevo borra el punto de control anterior, para no recuperar por error un análisis distinto.

### Correcciones de esta versión

- La leyenda de la figura de tendencias nombra la prueba que realmente clasificó los temas (Mann-Kendall con corrección de Hamed-Rao, Mann-Kendall clásico u OLS); antes decía siempre Hamed-Rao.
- Valor p frente a la nula re-emparejada: su mínimo es 1/(n_perm + 1), como en la nula por par (los cosenos emparejados de una misma permutación no son independientes). Ya no puede redondearse a 0.
- Co-citación: las variantes de un mismo autor se unen («DaMatta FM» y «DaMatta, Fábio M.» → «Damatta FM»), y los apellidos se comparan sin tildes, porque WoS los escribe sin ellas.

### Corrección: la última etapa se quedaba bloqueada en el navegador

| Cambio | Justificación |
|---|---|
| El bootstrap de la coherencia NPMI y de la concordancia ARI/NMI se calcula con pesos por documento (veces que sale cada documento en la réplica) sobre matrices dispersas, en lugar de copiar y multiplicar matrices densas en cada réplica. Los resultados son idénticos (comprobado: mismas estimaciones e intervalos; ARI/NMI iguales a scikit-learn hasta 10⁻¹⁶). | En Python nativo la multiplicación densa usa BLAS optimizado y tarda segundos, pero el navegador no lo tiene: con ~7 600 documentos y 1000 réplicas, «Generando figuras y tablas» pasaba de ~1 min a más de 30 min y, en portátiles, parecía bloqueado. Ahora es ~30 veces más rápido también en Python. |
| Barra de **progreso total** con tiempo transcurrido y **tiempo restante** (estimado al empezar y corregido con la velocidad real del equipo); avance dentro de la última etapa (figura a figura, permutaciones, bootstrap). | Antes solo se veía el avance de la etapa actual y la última etapa no informaba nada. |
| **Punto de control automático** al terminar LDA y BERTopic (se guarda en el navegador). Si algo falla después, al recargar la página aparece «Recuperar el último análisis», que solo repite las figuras y tablas. | Un fallo al final ya no obliga a repetir horas de modelado. |
| Aviso si el análisis no informa avances durante 10 min, errores visibles en el panel de progreso y **tiempo por etapa** al terminar. | Distinguir un análisis lento de uno detenido. |

### Revisión metodológica sugerida por un árbitro (tendencias, ARI/NMI y línea base nula)

| Cambio | Justificación |
|---|---|
| La figura de tendencias y el mapa estratégico muestran la **pendiente de Sen** con su IC (la de OLS solo si se elige la regresión); la leyenda indica la prueba usada. | Antes se dibujaba la pendiente lineal coloreada según Mann-Kendall: estimador y prueba no coincidían. |
| **Mann-Kendall corregido por autocorrelación (Hamed y Rao, 1998)**, por defecto para clasificar las tendencias (el clásico y OLS siguen como opciones); se informan ambos p y el factor de corrección. Verificado con `pymannkendall`. | Las prevalencias anuales están autocorrelacionadas y el Mann-Kendall clásico da falsos positivos. |
| IC de **ARI y NMI con jackknife delete-d** (submuestras sin reemplazo de la mitad de los documentos; Shao y Wu, 1989), centrado en la estimación. | El bootstrap con reemplazo duplica documentos, que coinciden en ambas particiones e inflan la concordancia: la estimación quedaba fuera de su intervalo. |
| **Línea base nula re-emparejada**: en cada permutación se repite el emparejamiento húngaro y cada par se compara con la distribución de los cosenos emparejados bajo la nula (columnas «Re-matched null p» y «Above re-matched null»). | La nula por par no tiene en cuenta que el algoritmo húngaro elige las parejas más similares; la re-emparejada es más estricta. |

No se automatiza la sensibilidad a otro K de LDA (p. ej., K = número de temas de BERTopic): duplicaría el tiempo de LDA y se hace con «Fijar K». La sensibilidad a `min_cluster_size` sí se automatizó (abajo).

### Segunda revisión: sensibilidad, comparabilidad y redacción

| Cambio | Justificación |
|---|---|
| **Sensibilidad a `min_cluster_size`** (tabla `bert_mcs_sensitivity`): se repite HDBSCAN sobre la misma proyección UMAP con el valor elegido, ½ y ¼; para cada uno, grupos nativos, temas, % de outliers, NPMI, estabilidad (hasta 5 réplicas) y ARI/NMI con LDA. Opción en BERTopic (activada). | Con un 33 % de outliers, un árbitro pedirá ver qué ocurre con grupos más pequeños sin tener que repetir el análisis. |
| Reasignaciones de **baja confianza** en la tabla de métricas (% de outliers reasignados y de baja confianza) y por tema en la tabla de BERTopic. | Se calculaban por documento, pero no aparecían en ninguna tabla. |
| `bert_trends`: el análisis «solo documentos agrupados» incluye pendiente de Sen con IC, q de la prueba usada y la prueba. | Antes solo tenía la pendiente OLS, distinta de la prueba principal. |
| **Submuestreo en las réplicas de estabilidad** (opción; 0 = solo semilla): cada réplica de LDA y de BERTopic usa además un submuestreo sin reemplazo de documentos. El texto indica qué varía en cada modelo. | En BERTopic solo cambia la semilla de UMAP (embeddings y HDBSCAN son deterministas), en LDA se reestima el modelo: el submuestreo añade la misma perturbación de datos a ambos. La fórmula del TRI no cambia. |
| **Sensibilidad de tendencias** (tablas `lda_trend_sensitivity`, `bert_trend_sensitivity`): clasificación con solo los años con ≥ N documentos (por defecto 30) y con una regresión ponderada por documentos por año. Se omite si el umbral más exigente no deja 3 años. | Los años con pocos documentos dan medias ruidosas. |
| Terminología opcional **creciente/decreciente** (Increasing/Decreasing) en lugar de emergente/en declive (Hot/Cold), en tablas y texto. | Petición de revisores: «emergente» sugiere novedad, no solo aumento. |
| El texto de tendencias informa la **pendiente y la q de la prueba con la que se clasificó** (Sen y Hamed-Rao por defecto), en puntos porcentuales por año. | Antes citaba la pendiente y la q de OLS aunque la clasificación fuera de Mann-Kendall. |
| **Concordancia de tendencias**: desglose (creciente en ambos, sin tendencia en ambos, decreciente en ambos, discordantes) y **kappa de Cohen**; se aclara que se calcula sobre los pares convergentes o parciales. | El porcentaje solo no distingue acuerdo por azar. |
| Correspondencias muchos a uno con su propio **TRI** («Best-match TRI») y el número de temas que comparten la misma mejor correspondencia. | Permite valorar las correspondencias que el emparejamiento uno a uno descarta. |
| Validación por expertos: **alfa de Krippendorff** en la misma tabla (T12), aviso de **efecto techo** cuando todos los temas de un modelo tienen la puntuación máxima, y la estabilidad del par se identifica como **media geométrica** en T12b. | Claridad de la tabla y de la interpretación. |
| Línea base nula: el texto indica que el menor p posible es 1/(permutaciones + 1) y se informa como p ≤ ese valor. | Con 1000 permutaciones, «p < 0.001» no es alcanzable. |
| Co-citación: apellidos con partícula (De, Da, Van, Von, Le…) completos y referencias anónimas agrupadas por fuente («Anonymous [FAOSTAT], 2014»); sin fuente se excluyen. | Aparecían nodos como «De, 2013» y «Anonymous, 2014». |
| Periodos: opción para que los periodos de ancho fijo empiecen en el primer año con ≥ N documentos (los anteriores forman un periodo inicial). El texto describe el primer periodo cuando agrupa muchos años y advierte que el último incluye el año incompleto. | El primer periodo agrupaba 72 años sin explicarlo. |
| Informe: tablas y figuras **numeradas por orden de primera cita** (dos pasadas); Resultados distingue el corpus bibliométrico y el de modelado con su n en cada apartado. | Requisito de Elsevier; antes la descripción general daba el n del corpus de modelado junto a estadísticas del bibliométrico. |

### Índice de Robustez Temática (TRI): fundamentación y primera validación

| Cambio | Justificación |
|---|---|
| Métodos sitúa el TRI en la literatura: combina la estabilidad frente a la reinicialización (Greene et al., 2014; Belford et al., 2018; Hosseiny Marani y Baumer, 2023) con la comparación entre métodos sin verdad de referencia habitual en cienciometría (Gläser et al., 2017; Velden et al., 2017), y explica por qué la repetibilidad sola no basta (Magsarjav et al., 2025). | Una búsqueda bibliográfica (Consensus y SciSpace, ~300 registros) no encontró un índice por par de temas que combine la similitud entre modelos de distinta naturaleza con la estabilidad entre semillas; sí trabajos que usan cada criterio por separado. |
| Con la validación por expertos cargada, nueva tabla «TRI frente a la valoración de los expertos»: correlación de Spearman, sobre los pares emparejados, entre el TRI y la interpretabilidad, la coherencia y la precisión de la palabra intrusa del par, comparada con la del coseno solo y la estabilidad sola. Texto en Métodos y Resultados. | Primera validación empírica del índice, sin depender de umbrales. |
| El TRI de cada par se informa con 3 decimales en Resultados. | Regla de decimales para índices. |
| Cada ρ de la tabla «TRI frente a la valoración de los expertos» lleva su IC 95 % (bootstrap percentil sobre los pares), y se añaden las diferencias ρ(TRI) − ρ(estabilidad) y ρ(TRI) − ρ(coseno) con bootstrap pareado e IC 95 %; Métodos y Resultados lo informan e indican si la diferencia es distinta de cero. | Las correlaciones comparten datos y el TRI contiene la estabilidad: no son independientes y no deben compararse solo por su valor puntual (en el corpus del café, 0.517 frente a 0.615 con unos 26 pares). |

### Corrección metodológica: intervalo de confianza de la coherencia NPMI

| Cambio | Justificación |
|---|---|
| El IC 95 % de la coherencia NPMI media de cada modelo y de su diferencia se obtiene por bootstrap **sobre los temas** (cada tema es una observación; se remuestrean los temas de cada modelo de forma independiente). BCa usa el jackknife que omite un tema; su valor p es el menor nivel al que el intervalo BCa excluye el 0. ARI y NMI siguen remuestreando documentos. | Al remuestrear documentos, duplicar unos y omitir otros altera las co-ocurrencias y sesga a la baja el NPMI de los términos poco frecuentes: en el corpus del café, el NPMI de BERTopic (0.218) quedaba fuera de su propio IC [0.153, 0.202] y el BCa daba un intervalo degenerado. Con temas como unidad, el intervalo contiene la estimación ([0.149, 0.292]) y es coherente con la prueba U de Mann-Whitney, que también compara temas. Los intervalos son más anchos: reflejan la incertidumbre real con el número de temas disponible. |

### Paleta de colores de las figuras

| Cambio | Justificación |
|---|---|
| Una sola paleta por artículo para todas las figuras (barras, mapas de calor, mapa de colaboración, redes, biplots, mapa temático, Sankey, PRISMA). Los temas de BERTopic conservan su color en el gráfico de términos y en el mapa de documentos. | Antes cada figura usaba su propio esquema (viridis, magma, Blues, tab20…) y un mismo tema podía cambiar de color. |
| Seis paletas (Tierra, Océano, Salud, Tecnología, Sociedad, Clásica), con dos variantes cada una, y elección **automática según la temática** (ecuaciones y términos de búsqueda y las 50 palabras clave más frecuentes). La variante depende del propio corpus. Selector en la interfaz con vista previa; al cambiarla después del análisis solo se vuelven a dibujar las figuras. Se registra en `run_config.json` (`figure_palette`). | Dos artículos distintos no deben tener figuras idénticas. |
| Todas las paletas están validadas: 8 colores categóricos en orden fijo con separación suficiente para daltonismo (protanopía y deuteranopía, ΔE ≥ 8 en OKLab entre colores contiguos) y con visión normal (ΔE ≥ 15), y escalas secuenciales de un solo tono con luminosidad monótona. | Muchas revistas exigen figuras legibles para lectores con daltonismo. |

No cambia el tiempo de análisis: los colores solo intervienen al dibujar.


## 2.0.0 — Versión para publicación (revistas Q1)

Dos objetivos: que la importación no sesgue el corpus sin avisar y que los valores por defecto sean los adecuados
para publicar. Las configuraciones anteriores siguen funcionando: los `config.json` sin `config_version` se
ejecutan con los valores de la versión 1 (perfil `legacy-v1`).

### 1. Importación y depuración

| Cambio | Justificación |
|---|---|
| Ningún filtro descarta registros por un campo ausente. Idioma, tipo de documento y año tienen tres estados (cumple, no cumple, **no disponible**); los no disponibles se conservan por defecto y hay opciones explícitas para excluirlos (desactivadas). | En un análisis real, el CSV de Scopus sin `Language of Original Document` hizo que el filtro de idioma eliminara los 4 943 registros exclusivos de Scopus sin aviso. |
| Dos corpus: **bibliométrico** (todos los registros elegibles, también sin resumen o sin año) y **de modelado** (requiere resumen ≥ 20 palabras y año). | Un registro sin resumen no puede modelarse, pero sí cuenta en la bibliometría; antes se perdía en ambos. |
| PRISMA con categorías separadas: excluidos por criterio, «no disponible» (conservados) y «no utilizables para el modelado». | Transparencia del flujo de registros. |
| Auditoría de campos al cargar (AB, DE, ID, LA, DT, PY, DI, TC, C1, RP, CR), por base y por archivo. Si LA, DT, AB, RP/C1 o DE están por debajo del 90 % en alguna base, aviso rojo que bloquea el análisis hasta confirmarlo. Instrucciones de exportación detalladas. | `field_availability` mostraba LA = 0.53, pero nadie lo vio: el aviso debe ser visible y requerir una decisión. |
| Tipos de documento armonizados (sin distinguir mayúsculas, con etiquetas múltiples de WoS): Article, Review, Conference paper, Book chapter, Book, Other. «Article; Early Access» = Article; «Article; Proceedings Paper» = Conference paper. La interfaz usa este vocabulario y el valor original se conserva en `DT_RAW`. | «Conference paper»/«Book chapter» (Scopus) no coincidían con «Proceedings Paper»/«Book Chapter» (WoS). |
| Idiomas múltiples de Scopus («English; Spanish»): se conserva el registro si alguno está admitido; nota en el informe sobre los resúmenes en inglés de Scopus. | Antes solo se miraba el primer idioma. |
| Fusión por campos: campos vacíos completados desde la otra base, con procedencia (`FIELDS_FROM_OTHER`), conflictos (`CONFLICTS`) y tabla de registros vinculados. Citas `TC_WoS` y `TC_Scopus` por separado; por defecto se usa el máximo (opción: WoS o Scopus), documentado en el informe. | El registro conservado perdía LA, RP, ID, CR o DI que sí traía la otra base. |
| País del autor de correspondencia desde RP (WoS) o `Correspondence Address` (Scopus) y, en su defecto, desde la afiliación del primer autor, con el origen marcado (`RP_COUNTRY_SOURCE`) y el % sin país en el informe. | Trazabilidad del país asignado. |
| Limpieza de etiquetas HTML/XML y entidades (`<italic>`, `<sup>`, `<inf>`, `&amp;`) en títulos, resúmenes y palabras clave; más patrones de copyright. | «<italic>coffea arabica</italic>» aparecía como palabra clave distinta. |
| Ecuaciones de búsqueda de Scopus y WoS, fecha e índices de WoS en la interfaz; se guardan en `run_config.json` y en el informe. `query_terms` se rellena automáticamente desde las ecuaciones (sin operadores, comodines, etiquetas ni términos excluidos con NOT; con las expresiones compuestas) y es editable. Las expresiones compuestas se eliminan del texto como secuencia. | `query_terms` quedaba vacío aunque Métodos decía que se habían eliminado; ahora la frase solo aparece si hubo términos. |
| Normalización automática de palabras clave (minúsculas, singular, guion/espacio, británico/estadounidense) y archivo de sinónimos CSV `término,forma_canónica`, aplicado a las palabras clave y al vocabulario de los modelos. Se muestran las 50 palabras clave más frecuentes antes del análisis. | Variantes del mismo concepto fragmentaban redes y temas. |
| Año en curso: incluido en los descriptivos; excluido de las regresiones de tendencia y de la tasa de crecimiento, y así se indica en el informe. | Un año incompleto sesga tendencias y crecimiento. |

### 2. Perfiles y valores por defecto

Dos perfiles: **Publicación** (por defecto) y **Exploración rápida**, con estimación del tiempo de ejecución en el
navegador. Ningún valor del perfil Publicación se rebaja automáticamente: si el tiempo estimado supera 60 min, la
interfaz lo dice y propone ejecutar el script localmente en paralelo o explorar antes con el perfil rápido.

| Parámetro | Antes (v1) | Publicación (v2, por defecto) | Exploración rápida |
|---|---|---|---|
| Bootstrap | 200, percentil | **1000** (máx. 5000), percentil; opción **BCa** | 200 |
| Permutaciones (línea base nula del coseno) | — | **1000** | 200 |
| Estabilidad LDA | 3 semillas | **10 semillas**, media ± DE por tema y global | 3 |
| Estabilidad BERTopic | 3 semillas de UMAP | **10 semillas de UMAP**, media ± DE | 3 |
| Selección de K | 5–30 paso 5, 1 semilla, 4 métricas | **5–40 paso 5 + fina ±5 paso 1, 3 semillas por K, 5 métricas** (+ estabilidad de Greene et al., 2014); todas las curvas | 5–30 paso 5, 1 semilla |
| LDA, inferencia | online, 20 iteraciones (10 en la selección) | **batch**, máx. **100** iteraciones con criterio de convergencia relativo (cambio de la perplejidad < 0.1 % cada 10 iteraciones) | **batch**, máx. 30 |
| Priors α, η | implícitos (1/K) | **explícitos, 1/K**, registrados | 1/K |
| UMAP | 15 vecinos, 5 dim., min_dist 0, euclídea | 15 vecinos, 5 dim., min_dist 0, **coseno** | igual |
| HDBSCAN `min_cluster_size` | máx(10, 0.5 % N) | **máx(15, 0.5 % N)**; `min_samples` registrado | igual |
| Reasignación de outliers | todos, sin umbral | c-TF-IDF con **umbral configurable** (0 = todos); % antes y después; aviso si > 30 % | igual |
| Embeddings | all-MiniLM-L6-v2 | all-MiniLM-L6-v2; opción **all-mpnet-base-v2** | igual |
| Tendencias | OLS, p < 0.05 sin corrección | OLS **+ Mann-Kendall con pendiente de Sen**, **BH** sobre todos los temas; clasificación por la q de Mann-Kendall | igual |
| Año en curso en tendencias | incluido | **excluido** | excluido |
| Semillas | global | global **y todas las derivadas registradas** | igual |
| Procesos en paralelo (LDA, navegador) | 1 | **automático**: núcleos − 1, hasta 6 (resultado idéntico al de 1 proceso) | igual |

**Viabilidad en el navegador.** Con ~9 000 documentos, la primera versión de este perfil necesitaba ~4–5 h en el
navegador (unos 65 ajustes de LDA de 100 iteraciones en un solo proceso). Se redujo sin rebajar ningún valor:

1. **Criterio de convergencia relativo.** El criterio de scikit-learn es absoluto (`perp_tol` = 0.1 unidades de
   perplejidad, con perplejidades de ~2 000) y nunca se alcanzaba: todos los ajustes llegaban a 100 iteraciones.
   Ahora el ajuste se detiene cuando la perplejidad, evaluada cada 10 iteraciones, cambia menos del 0.1 %. En el
   corpus de prueba (3 587 documentos, K = 5–40) se detuvo entre 49 y 99 iteraciones (media ~70) y los temas
   coinciden con los de 100 iteraciones (coseno medio ≥ 0.9995, mínimo ≥ 0.997). Las iteraciones reales se
   informan en Métodos.
2. **LDA en paralelo en el navegador.** Los ajustes (selección de K, modelo final y réplicas de estabilidad) se
   reparten entre varios workers de Pyodide; cada ajuste conserva su semilla, así que el resultado es idéntico al
   de un solo proceso (comprobado con pruebas automáticas y en el navegador). Por defecto se usan los núcleos
   lógicos menos uno, hasta 6 (cada worker ocupa ~300 MB; 2 como máximo con ≤ 4 GB de RAM); puede cambiarse en
   «Procesos en paralelo». Si un worker no puede iniciarse o falla, su ajuste se repite en el principal.

Medición en el navegador con el corpus real (9 250 documentos, 6 ajustes de LDA): 9.5 min en un solo proceso
frente a 2.5 min con 3 procesos (3.9 veces más rápido). La medición mostró además que la estimación anterior para
un solo proceso era optimista (el worker principal es 3.5 veces más lento que Python nativo, no 2): con un solo
proceso, este perfil habría tardado ~6 h. Estimación corregida con ~9 250 documentos: **~1 h 50 min con 3
procesos, ~1 h 15 min con 5 y ~1 h con 6 o más** (incluidos ~10 min de BERTopic con 10 semillas de UMAP). Para ir más rápido sin cambiar los valores, el script
descargable ejecuta lo mismo con `--nucleos N`. SPECTER no está disponible en el navegador con transformers.js;
puede usarse en el script local.

**Aviso sobre la versión 1 y el perfil rápido anterior.** Al medir se comprobó que LDA *online* con 20
iteraciones (el valor de la versión 1) no converge en corpus de este tamaño: sus temas se parecen poco a los del
modelo convergido (coseno medio 0.44–0.55) y su NPMI es mucho menor (0.03–0.05 frente a 0.11–0.15). Por eso el
perfil «Exploración rápida» usa ahora batch con 30 iteraciones (coseno 0.99 con el modelo convergido) y conviene
repetir con la versión 2 los análisis hechos con la versión 1.

### 3. Comparación LDA–BERTopic y TRI

- Mejor correspondencia de cada tema en el otro modelo (muchos a uno), junto al par húngaro, para detectar pares
  forzados.
- Línea base nula del coseno por permutación (1000 permutaciones de los pesos del vocabulario): valor p por par y
  percentil 95 nulo, marcado en la figura de similitud.
- TRI con IC 95 % por par (remuestreo de las réplicas de estabilidad) y análisis de sensibilidad de τ_fuerte
  (0.40–0.60) y τ_parcial (0.20–0.40).
- Una sola fuente por métrica: la estimación puntual se calcula con todos los datos y es la misma en todas las
  tablas y el texto; el bootstrap solo aporta el intervalo (la media bootstrap se rotula como tal). Corrige NPMI
  0.129 frente a 0.112 y p 0.027 frente a 0.028.
- Valores p como «< 0.001»; tres decimales en índices y uno en porcentajes.

### 4. Dinámica temporal

- Mann-Kendall con pendiente de Sen además de la regresión lineal; BH sobre todos los temas; clasificación por q.
- Periodos de ancho fijo, con un número similar de documentos o personalizados; aviso si el primer periodo agrupa
  demasiados años con pocos documentos (p. ej., 1926–2001).

### 5. Calidad y validación

- Panel de controles con semáforo antes de los resultados (base sin registros exclusivos, exclusión > 20 %, campo
  crítico < 90 %, temas LDA vacíos o con estabilidad < 0.2, outliers > 30 %, `query_terms` vacío, CR vacío); también
  en `run_config.json` y en la consola del script.
- Validación humana: kappa de Fleiss (palabra intrusa y calificaciones) y precisión de intrusión por tema, además del
  alfa de Krippendorff.
- Acoplamiento bibliográfico (Kessler, 1963) además de la co-citación cuando hay CR; si no lo hay, se dice en el
  informe.

### 6. Reproducibilidad e informe

- `run_config.json`: versión de BiblioTopic, `config_version`, perfil, parámetros efectivos (K, priors, iteraciones,
  `min_cluster_size`, `min_samples` resueltos), archivos con MD5 y nº de registros, ecuaciones y fecha, disponibilidad
  de campos por base, todas las semillas y los controles de calidad.
- El LEEME del script lista los archivos usados con su MD5 (antes decía «no registrados»); el script comprueba el MD5.
- Métodos refleja lo ejecutado: sin afirmaciones sobre pasos omitidos y con los valores reales.
- Pruebas automáticas de los importadores (Scopus con y sin LA, WoS con etiquetas múltiples) y de cada función nueva.
