"""Ejecución sin interfaz:  python run_cli.py config.json archivo1.txt archivo2.csv ..."""
import json
import sys
from pathlib import Path

from bibliotopic import pipeline, report

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
meta = cfg.pop("meta", {})
files = [(Path(p).name, Path(p).read_bytes()) for p in sys.argv[2:]]
out = Path(cfg.pop("output_dir", "resultados/cli"))
R = pipeline.compute(files, cfg, progress=lambda p, m: print(f"[{p:4.0%}] {m}", flush=True))
pipeline.render(R, out)
report.build(R, str(out / "Borrador_Metodos_Resultados.docx"), meta)
print("Listo:", out.resolve())
