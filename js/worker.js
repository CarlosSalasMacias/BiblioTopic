// BiblioTopic — worker: Python (Pyodide) + embeddings (transformers.js) + UMAP (umap-js).
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";
import { UMAP } from "https://cdn.jsdelivr.net/npm/umap-js@1.4.0/+esm";

const PY_FILES = ["__init__.py", "io.py", "biblio.py", "text.py", "lda_model.py", "bert_model.py",
  "hjbiplot.py", "compare.py", "figures.py", "report.py", "pipeline.py", "stats.py", "validation.py",
  "citations.py", "evolution.py", "script_export.py"];
const EMB_MODEL = "Xenova/all-MiniLM-L6-v2";
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1";

let py = null;
const post = (type, data = {}, transfer = []) => self.postMessage({ type, ...data }, transfer);
const progress = (p, msg) => post("progress", { p, msg });

// Generador pseudoaleatorio con semilla (mulberry32) para UMAP reproducible
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function init() {
  progress(0.02, "Cargando Python en el navegador…");
  py = await loadPyodide();
  progress(0.25, "Cargando librerías científicas (numpy, pandas, scikit-learn, matplotlib)…");
  await py.loadPackage(["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "networkx",
    "micropip", "lxml"]);
  progress(0.75, "Instalando python-docx, openpyxl, pycountry y adjustText…");
  const micropip = py.pyimport("micropip");
  await micropip.install(["python-docx", "openpyxl", "pycountry", "adjustText"]);
  progress(0.9, "Cargando BiblioTopic…");
  py.FS.mkdirTree("/home/pyodide/bibliotopic");
  const base = new URL("../bibliotopic/", import.meta.url);
  for (const f of PY_FILES) {
    const txt = await (await fetch(new URL(f, base), { cache: "no-cache" })).text();
    py.FS.writeFile(`/home/pyodide/bibliotopic/${f}`, txt);
  }
  py.FS.mkdirTree("/home/pyodide/bibliotopic/data");
  const world = await (await fetch(new URL("data/world_110m.json", base))).text();
  py.FS.writeFile("/home/pyodide/bibliotopic/data/world_110m.json", world);
  await py.runPythonAsync(`
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/pyodide")
import numpy as np
from bibliotopic import pipeline, report
from bibliotopic.bert_model import lsa_embeddings
S = pipeline.Session()
`);
  const analyses = JSON.parse(py.runPython("json.dumps(pipeline.ANALYSES, ensure_ascii=False)"));
  const sections = JSON.parse(py.runPython("json.dumps(report.SECTIONS, ensure_ascii=False)"));
  progress(1, "Listo");
  post("ready", { analyses, sections });
}

function callPy(code, args = {}) {
  for (const [k, v] of Object.entries(args)) py.globals.set(k, v);
  return py.runPythonAsync(code);
}

async function loadFiles(files) {
  progress(0.1, "Leyendo y fusionando archivos…");
  py.globals.set("_files", files.map((f) => [f.name, f.data]));
  const out = await py.runPythonAsync(`
_f = [(str(f[0]), f[1].to_bytes()) for f in _files]
json.dumps(S.load(_f), ensure_ascii=False, default=str)
`);
  progress(1, "Archivos cargados");
  post("loaded", { summary: JSON.parse(out) });
}

// ---------------------------------------------------------------- embeddings
async function embed(texts) {
  try {
    progress(0.0, "Descargando el modelo de lenguaje (MiniLM, ~90 MB la primera vez)…");
    const { pipeline, env } = await import(TRANSFORMERS_URL);
    env.allowLocalModels = false;
    let device = "wasm";
    if (self.navigator?.gpu) {
      try { if (await navigator.gpu.requestAdapter()) device = "webgpu"; } catch (e) { /* wasm */ }
    }
    const extractor = await pipeline("feature-extraction", EMB_MODEL, { device, dtype: "fp32" });
    const n = texts.length, dim = 384, out = new Float32Array(n * dim), bs = 32;
    for (let i = 0; i < n; i += bs) {
      const batch = texts.slice(i, i + bs);
      const t = await extractor(batch, { pooling: "mean", normalize: true });
      out.set(t.data, i * dim);
      progress(i / n, `Embeddings ${Math.min(i + bs, n)} / ${n} (${device})`);
    }
    return { data: out, dim, name: "all-MiniLM-L6-v2 (sentence-transformers, ONNX vía transformers.js)" };
  } catch (e) {
    post("warn", { msg: "No se pudo usar el modelo de lenguaje (" + e.message + "). Se usa LSA (TF-IDF + SVD)." });
    const arr = await py.runPythonAsync(`_e = lsa_embeddings(S.texts()); _e.tobytes()`);
    const dim = py.runPython("_e.shape[1]");
    const u8 = arr.toJs();
    const data = new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength));
    arr.destroy();
    return { data, dim, name: "LSA fallback (TF-IDF + SVD)" };
  }
}

async function umap(rows, nComponents, minDist, seed, label) {
  const u = new UMAP({ nComponents, nNeighbors: 15, minDist, random: rng(seed) });
  const nEpochs = u.initializeFit(rows);
  for (let i = 0; i < nEpochs; i++) {
    u.step();
    if (i % 25 === 0) {
      progress(i / nEpochs, `${label}: época ${i} / ${nEpochs}`);
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  return Float32Array.from(u.getEmbedding().flat());
}

// ------------------------------------------------------------------ análisis
async function run(cfg) {
  const t0 = Date.now();
  const stage = (s) => post("stage", { stage: s, elapsed: (Date.now() - t0) / 1000 });
  stage("Filtrando y preprocesando el texto");
  const prep = JSON.parse(await callPy(
    "json.dumps(S.prepare(json.loads(_cfg)), ensure_ascii=False, default=str)", { _cfg: JSON.stringify(cfg) }));
  post("prepared", { prep });
  const on = (a) => cfg.analyses.includes(a);
  const pyProgress = (p, m) => progress(p, m);
  if (on("lda")) {
    stage("LDA: selección de K y modelo final");
    py.globals.set("_prog", pyProgress);
    await py.runPythonAsync("S.run_lda(lambda p, m: _prog(p, m))");
  }
  if (on("bertopic")) {
    stage("BERTopic: embeddings de los documentos");
    const texts = py.runPython("S.texts()").toJs();
    const emb = await embed(texts);
    const n = texts.length, rows = [];
    for (let i = 0; i < n; i++) rows.push(Array.from(emb.data.subarray(i * emb.dim, (i + 1) * emb.dim)));
    const runs = py.runPython("S.n_umap_runs()");
    const reduced = [];
    for (let r = 0; r < runs; r++) {
      stage(`BERTopic: UMAP ${r + 1} de ${runs}${r ? " (estabilidad)" : ""}`);
      reduced.push(await umap(rows, 5, 0.0, cfg.seed + r, `UMAP ${r + 1}/${runs}`));
    }
    stage("BERTopic: mapa 2D de documentos");
    const xy = await umap(rows, 2, 0.1, cfg.seed, "UMAP 2D");
    stage("BERTopic: HDBSCAN y c-TF-IDF");
    py.globals.set("_red", reduced);
    py.globals.set("_xy", xy);
    py.globals.set("_emb_name", emb.name);
    await py.runPythonAsync(`
_n = len(S.texts())
_reds = [np.frombuffer(r.to_bytes(), dtype=np.float32).reshape(_n, 5) for r in _red]
_xy2 = np.frombuffer(_xy.to_bytes(), dtype=np.float32).reshape(_n, 2)
S.run_bertopic(_reds, _xy2, _emb_name)
`);
  }
  stage("Generando figuras y tablas");
  await render();
  post("done", { elapsed: (Date.now() - t0) / 1000 });
}

async function render() {
  const figs = JSON.parse(await py.runPythonAsync("json.dumps(S.render('/out'))"));
  const out = [];
  const transfer = [];
  for (const f of figs) {
    const data = py.FS.readFile(f.path);
    out.push({ key: f.key, name: f.path.split("/").pop(), data });
    transfer.push(data.buffer);
  }
  const tables = JSON.parse(await py.runPythonAsync(`
json.dumps({k: S.table_json(k) for k in ["T1_main_information", "T2_top_sources", "T3_top_authors",
            "T4_countries", "T5_keywords", "T6_LDA_topics", "T7_BERTopic_topics", "T8_model_metrics",
            "doc_agreement", "T9_topic_alignment", "T8b_inference", "T10_most_cited", "T11_thematic_map",
            "T12_expert_validation"]}, ensure_ascii=False, default=str)`));
  const labels = JSON.parse(py.runPython("json.dumps(S.topic_labels(), ensure_ascii=False)"));
  post("results", { figs: out, tables, labels }, transfer);
}

async function setLabels(labels, aiUsed) {
  await callPy("S.set_labels(json.loads(_lab))", { _lab: JSON.stringify(labels) });
  if (aiUsed) await py.runPythonAsync("S.set_ai_text({'labels_used': True})");
  await render();
}

async function finalRender() {
  if (!py.runPython("bool(getattr(S, 'final', False))")) {
    progress(0.1, "Preparando figuras finales (TIFF 600 dpi y figuras compuestas)…");
    await py.runPythonAsync("S.render('/out', final=True)");
  }
}

async function applyFigOpts(opts) {
  if (!opts) return;
  await callPy(`
_o = json.loads(_opts)
if any(S.cfg.get(k) != v for k, v in _o.items()):
    S.cfg.update(_o)
    S.final = False`, { _opts: JSON.stringify(opts) });
}

async function buildReport(meta, sections, opts) {
  await applyFigOpts(opts);
  await finalRender();
  await callPy(`report.build(S, "/out/Borrador_Metodos_Resultados.docx", json.loads(_meta), json.loads(_sec))`,
    { _meta: JSON.stringify(meta), _sec: JSON.stringify(sections) });
  const data = py.FS.readFile("/out/Borrador_Metodos_Resultados.docx");
  post("file", { name: "Borrador_Metodos_Resultados.docx", data,
    mime: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" }, [data.buffer]);
}

async function bytesOut(code, name, mime) {
  const b = await py.runPythonAsync(code);
  const data = b.toJs();
  b.destroy();
  post("file", { name, data, mime }, [data.buffer]);
}

async function saveProject() {
  progress(0.3, "Guardando el proyecto…");
  await bytesOut("S.save_project()", `BiblioTopic_${new Date().toISOString().slice(0, 10)}.bibliotopic`,
    "application/octet-stream");
}

async function openProject(data) {
  progress(0.1, "Abriendo el proyecto…");
  py.globals.set("_proj", data);
  await py.runPythonAsync("S = pipeline.Session.load_project(_proj.to_bytes())");
  const cfg = JSON.parse(py.runPython("json.dumps(S.cfg, default=str)"));
  post("project", { cfg });
  await render();
  post("done", { elapsed: 0 });
}

async function validationSheet() {
  await bytesOut("S.validation_sheet()", "BiblioTopic_validation_sheet.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
}

async function validationLoad(files) {
  py.globals.set("_vf", files.map((f) => [f.name, f.data]));
  const out = await py.runPythonAsync(
    "json.dumps(S.load_validation([(str(f[0]), f[1].to_bytes()) for f in _vf]), ensure_ascii=False, default=str)");
  post("validation", { result: JSON.parse(out) });
  await render();
}

async function aiContext() {
  const out = await py.runPythonAsync("json.dumps(S.ai_context(), ensure_ascii=False, default=str)");
  post("ai_context", { ctx: JSON.parse(out) });
}

async function aiSet(texts) {
  await callPy("S.set_ai_text(json.loads(_ai))", { _ai: JSON.stringify(texts) });
  post("ai_saved", {});
}

async function proposeTitles(meta) {
  const out = await callPy("json.dumps(S.propose_titles(json.loads(_meta)), ensure_ascii=False)",
    { _meta: JSON.stringify(meta || {}) });
  post("titles", { titles: JSON.parse(out) });
}

async function scriptPackage(meta) {
  await callPy("_b = S.script_package(json.loads(_meta))", { _meta: JSON.stringify(meta || {}) });
  await bytesOut("_b", "BiblioTopic_script.zip", "application/zip");
}

async function zipAll(opts, meta) {
  await applyFigOpts(opts);
  await finalRender();
  py.globals.set("_meta", JSON.stringify(meta || {}));
  const b = await py.runPythonAsync("S.zip_outputs('/out', json.loads(_meta))");
  const data = b.toJs();
  b.destroy();
  post("file", { name: "BiblioTopic_resultados.zip", data, mime: "application/zip" }, [data.buffer]);
}

self.onmessage = async (ev) => {
  const { cmd } = ev.data;
  try {
    if (cmd === "init") await init();
    else if (cmd === "load") await loadFiles(ev.data.files);
    else if (cmd === "run") await run(ev.data.cfg);
    else if (cmd === "labels") await setLabels(ev.data.labels, ev.data.aiUsed);
    else if (cmd === "save_project") await saveProject();
    else if (cmd === "open_project") await openProject(ev.data.data);
    else if (cmd === "val_sheet") await validationSheet();
    else if (cmd === "val_load") await validationLoad(ev.data.files);
    else if (cmd === "ai_context") await aiContext();
    else if (cmd === "ai_set") await aiSet(ev.data.texts);
    else if (cmd === "report") await buildReport(ev.data.meta, ev.data.sections, ev.data.opts);
    else if (cmd === "zip") await zipAll(ev.data.opts, ev.data.meta);
    else if (cmd === "script") await scriptPackage(ev.data.meta);
    else if (cmd === "titles") await proposeTitles(ev.data.meta);
  } catch (e) {
    post("error", { msg: String(e.message || e) });
  }
};
