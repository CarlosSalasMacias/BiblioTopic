# Registro de cambios

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
