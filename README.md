# BiblioTopic

Aplicación local para analizar un corpus de **Web of Science** y/o **Scopus**:

1. Fusión de lotes y bases, eliminación de duplicados (DOI y título + año) y diagrama de flujo tipo PRISMA.
2. Bibliometría: producción anual, fuentes, Bradford, Lotka, autores, países (SCP/MCP), red de
   colaboración internacional y red de palabras clave.
3. **LDA** con selección de K por 4 métricas (CaoJuan2009, Arun2010, Deveaud2014, NPMI) y estabilidad entre semillas.
4. **BERTopic** (embeddings + UMAP + HDBSCAN + c-TF-IDF) con el mismo vocabulario que LDA.
5. Dinámica temporal (temas emergentes / en declive) y **HJ-Biplot** (periodos × temas, países × temas).
6. **Comparación LDA vs BERTopic**: coherencia, diversidad, estabilidad, emparejamiento húngaro de temas,
   ARI/NMI por documento e **Índice de Robustez Temática (TRI)**.
7. **Borrador en Word** de *Materiales y métodos* y *Resultados* en español (título, figuras y tablas en inglés).

## Instalación (una sola vez)

Descargue el código con *Code → Download ZIP* en GitHub (o `git clone https://github.com/CarlosSalasMacias/BiblioTopic`).

Requiere **Python 3.10 – 3.13** (https://www.python.org/downloads/; en Windows marque *Add Python to PATH*).

- **Windows:** doble clic en `iniciar_windows.bat`
- **macOS / Linux:** `./iniciar_mac_linux.sh`

La primera vez se instalan las librerías (10–15 min). Después, el mismo archivo abre la aplicación
en el navegador (http://localhost:8501). La primera ejecución de BERTopic descarga el modelo de
embeddings (~90 MB), así que necesita internet.

## Cómo exportar los datos

**Web of Science:** *Export → Tab delimited file* (o *Plain text*), **Record Content: Full Record and
Cited References**. WoS exporta de 1 000 en 1 000 registros: descargue todos los lotes y súbalos juntos.

> Si exporta solo "Author, Title, Source, Abstract" no habrá países, palabras clave ni citas.

**Scopus:** *Export → CSV*, marque todos los campos (en especial *Affiliations*, *Author keywords*,
*Index keywords*, *Abstract*, *Cited by* y *References*).

## Uso

1. Pestaña **Datos**: suba todos los archivos y revise la completitud de los campos.
2. Ajuste los parámetros en la barra lateral (los valores por defecto son razonables).
3. Pulse **Ejecutar análisis completo**. Con ~4 000 documentos tarda entre 10 y 20 min.
4. Revise las pestañas de resultados.
5. En **Etiquetas y borrador**, ponga nombre a cada tema (en inglés) y genere el borrador.
6. Descargue el `.docx` y el `.zip` con todas las figuras (PNG 300 dpi), las tablas (Excel) y la configuración.

Los resultados se guardan en `resultados/<fecha_hora>/`, incluido `project.pkl` para reabrir el proyecto
sin volver a calcular.

## Uso sin interfaz

```bash
python run_cli.py config_ejemplo.json lote1.txt lote2.txt scopus.csv
```

## Notas metodológicas

- Los términos de la ecuación de búsqueda (p. ej., *coffee*) se excluyen del texto porque aparecen en
  casi todos los documentos.
- BERTopic se reduce por defecto al mismo número de temas que LDA para que la comparación sea justa;
  se informa también el número de grupos que HDBSCAN encontró de forma natural.
- Los textos resaltados en amarillo en el borrador deben revisarlos los autores.
