// BiblioTopic — worker auxiliar para LDA: recibe el corpus (matriz documento-término) y ajusta los modelos que
// le pide el worker principal. Cada ajuste tiene su propia semilla, así que el resultado es idéntico al de la
// ejecución en serie, sea cual sea el número de workers.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";

const PY_FILES = ["__init__.py", "text.py", "lda_model.py"];
let py = null;

function pyBytes(proxy) {
  const buf = proxy.getBuffer("u8");
  try { return buf.data.slice(); } finally { buf.release(); proxy.destroy(); }
}

async function init(base) {
  py = await loadPyodide();
  await py.loadPackage(["numpy", "pandas", "scipy", "scikit-learn"]);
  py.FS.mkdirTree("/home/pyodide/bibliotopic");
  for (const f of PY_FILES) {
    const txt = await (await fetch(new URL(f, base), { cache: "no-cache" })).text();
    py.FS.writeFile(`/home/pyodide/bibliotopic/${f}`, txt);
  }
  await py.runPythonAsync(`
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/pyodide")
from bibliotopic import lda_model as L
_C = None
`);
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      await init(m.base);
      self.postMessage({ type: "ready" });
    } else if (m.type === "corpus") {
      py.globals.set("_payload", m.payload);
      py.runPython("_C = L.corpus_from_payload(_payload.to_bytes()); del _payload");
      self.postMessage({ type: "corpus-ok" });
    } else if (m.type === "job") {
      py.globals.set("_spec", m.spec);
      const bytes = pyBytes(py.runPython("L.run_job_spec(_C, json.loads(_spec))"));
      self.postMessage({ type: "result", id: m.id, bytes }, [bytes.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: "error", id: m.id, error: String(err && err.message || err) });
  }
};
