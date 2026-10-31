"""Ejecución local sin navegador (opcional).

    pip install -r requirements.txt
    python run_cli.py config_ejemplo.json lote1.txt lote2.txt scopus.csv

Usa sentence-transformers y umap-learn si están instalados; si no, LSA como respaldo.
"""
import json
import sys
from pathlib import Path

import numpy as np

from bibliotopic import pipeline, report
from bibliotopic.bert_model import lsa_embeddings

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
meta = cfg.pop("meta", {})
sections = cfg.pop("sections", None)
out = Path(cfg.pop("output_dir", "resultados"))
say = lambda p, m: print(f"[{p:4.0%}] {m}", flush=True)  # noqa: E731

S = pipeline.Session()
print(S.load([(Path(p).name, Path(p).read_bytes()) for p in sys.argv[2:]])["stats"])
print(S.prepare(cfg))
if S.on("lda"):
    print(S.run_lda(say))
if S.on("bertopic"):
    from umap import UMAP
    try:
        from sentence_transformers import SentenceTransformer
        emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(
            S.texts(), normalize_embeddings=True, show_progress_bar=True)
        name = "all-MiniLM-L6-v2 (sentence-transformers)"
    except Exception as e:
        print("Embeddings: se usa LSA", e)
        emb, name = lsa_embeddings(S.texts()), "LSA fallback (TF-IDF + SVD)"
    seed = S.cfg["seed"]
    reds = [UMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=seed + r).fit_transform(emb)
            for r in range(S.n_umap_runs())]
    xy = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=seed).fit_transform(emb)
    print(S.run_bertopic([np.asarray(r) for r in reds], xy, name))
S.render(str(out))
report.build(S, str(out / "Borrador_Metodos_Resultados.docx"), meta, sections)
print("Listo:", out.resolve())
