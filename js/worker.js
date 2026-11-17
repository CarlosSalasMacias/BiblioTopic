// BiblioTopic — worker: Python (Pyodide) + embeddings (transformers.js) + UMAP (umap-js).
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs";
import { UMAP } from "https://cdn.jsdelivr.net/npm/umap-js@1.4.0/+esm";

const PY_FILES = ["__init__.py", "io.py", "biblio.py", "text.py", "lda_model.py", "bert_model.py",
  "hjbiplot.py", "compare.py", "figures.py", "report.py", "pipeline.py", "stats.py", "validation.py",
  "citations.py", "evolution.py", "script_export.py", "palettes.py"];
const EMB_MODEL = "Xenova/all-MiniLM-L6-v2";
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1";

let py = null;
const post = (type, data = {}, transfer = []) => self.postMessage({ type, ...data }, transfer);
// Progreso total del análisis: cada etapa pesa lo que se estima que tarda (S.run_plan); el avance dentro de la
// etapa viene de sus propios mensajes. El tiempo restante se corrige con la velocidad observada en este equipo.
let TRACK = null;
function trackStart(plan) {
  const order = ["prep", "lda", "embed", "umap", "cluster", "render"];
  const w = {};
  for (const k of order) w[k] = Math.max(0, Number(plan[k]) || 0);
  TRACK = { w, total: Object.values(w).reduce((a, b) => a + b, 0) || 1, done: 0, cur: null, frac: 0,
    t0: Date.now(), stageT0: Date.now(), timings: {} };
}
function trackStage(key) {
  if (!TRACK) return;
  if (TRACK.cur) {
    TRACK.done += TRACK.w[TRACK.cur] || 0;
    TRACK.timings[TRACK.cur] = (TRACK.timings[TRACK.cur] || 0) + (Date.now() - TRACK.stageT0) / 1000;
  }
  TRACK.cur = key; TRACK.frac = 0; TRACK.stageT0 = Date.now();
}
function trackOverall(p) {
  if (!TRACK || !TRACK.cur) return null;
  TRACK.frac = Math.max(TRACK.frac, Math.min(1, p || 0));
  const overall = Math.min(0.995, (TRACK.done + (TRACK.w[TRACK.cur] || 0) * TRACK.frac) / TRACK.total);
  const elapsed = (Date.now() - TRACK.t0) / 1000;
  const estRemaining = TRACK.total * (1 - overall);
  // con más del 10 % hecho, se usa la velocidad real (mezclada con la estimación al principio)
  const a = Math.min(1, Math.max(0, (overall - 0.03) / 0.2));
  const observed = overall > 0.03 ? elapsed * (1 - overall) / overall : estRemaining;
  return { overall, elapsed, remaining: a * observed + (1 - a) * estRemaining };
}
const progress = (p, msg) => post("progress", { p, msg, total: trackOverall(p) });

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
const EMB_MODELS = {
  "Xenova/all-MiniLM-L6-v2": { label: "all-MiniLM-L6-v2", size: "~90 MB" },
  "Xenova/all-mpnet-base-v2": { label: "all-mpnet-base-v2", size: "~440 MB" },
};

async function embed(texts, model = EMB_MODEL) {
  const info = EMB_MODELS[model] || { label: model, size: "" };
  try {
    progress(0.0, `Descargando el modelo de lenguaje (${info.label}, ${info.size} la primera vez)…`);
    const { pipeline, env } = await import(TRANSFORMERS_URL);
    env.allowLocalModels = false;
    let device = "wasm";
    if (self.navigator?.gpu) {
      try { if (await navigator.gpu.requestAdapter()) device = "webgpu"; } catch (e) { /* wasm */ }
    }
    const extractor = await pipeline("feature-extraction", model, { device, dtype: "fp32" });
    const n = texts.length, bs = 32;
    let dim = 0, out = null;
    for (let i = 0; i < n; i += bs) {
      const batch = texts.slice(i, i + bs);
      const t = await extractor(batch, { pooling: "mean", normalize: true });
      if (!out) { dim = t.dims[t.dims.length - 1]; out = new Float32Array(n * dim); }
      out.set(t.data, i * dim);
      progress(i / n, `Embeddings ${Math.min(i + bs, n)} / ${n} (${device})`);
    }
    return { data: out, dim, name: `${info.label} (sentence-transformers, ONNX vía transformers.js)` };
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

// distancia coseno (los embeddings llegan normalizados, pero se calcula la general)
function cosineDist(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return 1 - d / (Math.sqrt(na * nb) || 1);
}

async function umap(rows, nComponents, minDist, seed, label, opts = {}) {
  const u = new UMAP({ nComponents, nNeighbors: opts.nNeighbors || 15, minDist, random: rng(seed),
    ...(opts.metric === "cosine" ? { distanceFn: cosineDist } : {}) });
  (opts.onProgress || progress)(0, `${label}: grafo de vecinos (${rows.length.toLocaleString()} documentos)…`);
  const nEpochs = u.initializeFit(rows);
  for (let i = 0; i < nEpochs; i++) {
    u.step();
    if (i % 25 === 0) {
      (opts.onProgress || progress)(i / nEpochs, `${label}: época ${i} / ${nEpochs}`);
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  return Float32Array.from(u.getEmbedding().flat());
}

// ------------------------------------------------- LDA en paralelo (workers auxiliares)
function pyBytes(proxy) {
  const buf = proxy.getBuffer("u8");
  try { return buf.data.slice(); } finally { buf.release(); proxy.destroy(); }
}

class LdaPool {
  constructor(n) { this.n = n; this.workers = []; this.idle = []; this.queue = []; this.pending = new Map(); this.seq = 0; }

  async start() {
    const base = new URL("../bibliotopic/", import.meta.url).href;
    const boot = (w) => new Promise((res, rej) => {
      w.onmessage = (e) => (e.data.type === "ready" ? res(w) : rej(new Error(e.data.error)));
      w.onerror = (e) => rej(e);
      w.postMessage({ type: "init", base });
    });
    const tries = [];
    for (let i = 0; i < this.n; i++) {
      tries.push(boot(new Worker(new URL("./lda_worker.js?v=20261126", import.meta.url), { type: "module" })));
    }
    const res = await Promise.allSettled(tries);
    this.workers = res.filter((r) => r.status === "fulfilled").map((r) => r.value);
    if (!this.workers.length) throw new Error("no se pudo iniciar ningún worker auxiliar");
    for (const w of this.workers) {
      w.onmessage = (e) => this._done(w, e.data);
      w.onerror = (e) => this._crash(w, e);
    }
    return this.workers.length;
  }

  async setCorpus(payload) {
    await Promise.all(this.workers.map((w) => new Promise((res, rej) => {
      w.onmessage = (e) => (e.data.type === "corpus-ok" ? res() : rej(new Error(e.data.error)));
      w.postMessage({ type: "corpus", payload });
    })));
    for (const w of this.workers) w.onmessage = (e) => this._done(w, e.data);
    this.idle = [...this.workers];
  }

  run(spec) {
    return new Promise((resolve, reject) => { this.queue.push({ spec, resolve, reject }); this._pump(); });
  }

  _pump() {
    while (this.idle.length && this.queue.length) {
      const w = this.idle.pop();
      const job = this.queue.shift();
      const id = ++this.seq;
      this.pending.set(w, { id, ...job });
      w.postMessage({ type: "job", id, spec: job.spec });
    }
    if (!this.workers.length) for (const j of this.queue.splice(0)) j.reject(new Error("sin workers"));
  }

  _done(w, msg) {
    const job = this.pending.get(w);
    if (!job) return;
    this.pending.delete(w);
    this.idle.push(w);
    if (msg.type === "result") job.resolve(msg.bytes); else job.reject(new Error(msg.error));
    this._pump();
  }

  _crash(w, e) {  // p. ej., memoria insuficiente: se retira el worker y su ajuste se repite en el principal
    const job = this.pending.get(w);
    this.pending.delete(w);
    this.workers = this.workers.filter((x) => x !== w);
    this.idle = this.idle.filter((x) => x !== w);
    try { w.terminate(); } catch (_) { /* nada */ }
    if (job) job.reject(new Error(e && e.message || "worker auxiliar detenido"));
    this._pump();
  }

  terminate() { for (const w of this.workers) w.terminate(); this.workers = []; this.idle = []; }
}

async function runLdaParallel(pool, nUsed) {
  const payload = pyBytes(await py.runPythonAsync("S.lda_parallel_begin()"));
  await pool.setCorpus(payload);
  const planned = Number(py.runPython("S.lda_planned_jobs()"));
  let done = 0, local = 0;
  for (;;) {
    const step = JSON.parse(await py.runPythonAsync("json.dumps(S.lda_parallel_step(), default=str)"));
    if (step.done) break;
    await Promise.all(step.jobs.map(async (spec) => {
      const sj = JSON.stringify(spec);
      try {
        const bytes = await pool.run(sj);
        py.globals.set("_sj", sj);
        py.globals.set("_rb", bytes);
        py.runPython("S.lda_put(_sj, _rb.to_bytes()); del _rb");
      } catch (err) {
        local += 1;
        py.globals.set("_sj", sj);
        py.runPython("S.lda_run_local(_sj)");
      }
      done += 1;
      const what = spec.kind === "metrics" ? `selección de K: K=${spec.k}, semilla ${spec.seed}`
        : spec.kind === "full" ? `modelo final K=${spec.k}` : `estabilidad: semilla ${spec.seed}`;
      progress(Math.min(done / Math.max(planned, done), 0.99), `LDA (${nUsed} procesos) · ${what} · ${done}/${Math.max(planned, done)}`);
    }));
  }
  py.globals.set("_nw", nUsed);
  py.globals.set("_nl", local);
  py.runPython("S.cfg['n_workers_used'] = int(_nw); S.cfg['lda_jobs_run_locally'] = int(_nl)");
}

// ------------------------------------------------------------------ análisis
async function run(cfg) {
  const t0 = Date.now();
  const stage = (s) => post("stage", { stage: s, elapsed: (Date.now() - t0) / 1000 });
  const nw = Number(cfg.n_workers) || 1;
  // los workers auxiliares arrancan mientras se preprocesa el texto
  const poolP = cfg.analyses.includes("lda") && nw > 1
    ? (async () => { const p = new LdaPool(nw); await p.start(); return p; })().catch((err) => {
      post("warn", { msg: `No se pudieron iniciar los procesos en paralelo (${err.message || err}); LDA se ejecutará en un solo proceso.` });
      return null;
    }) : Promise.resolve(null);
  stage("Filtrando y preprocesando el texto");
  const prep = JSON.parse(await callPy(
    "json.dumps(S.prepare(json.loads(_cfg)), ensure_ascii=False, default=str)", { _cfg: JSON.stringify(cfg) }));
  post("prepared", { prep });
  trackStart(JSON.parse(py.runPython("json.dumps(S.run_plan())")));
  const on = (a) => cfg.analyses.includes(a);
  const pyProgress = (p, m) => progress(p, m);
  if (on("lda")) {
    trackStage("lda");
    stage("LDA: selección de K y modelo final");
    const pool = await poolP;
    if (pool) {
      stage(`LDA: selección de K y modelo final (${pool.workers.length} procesos en paralelo)`);
      try { await runLdaParallel(pool, pool.workers.length); } finally { pool.terminate(); }
    } else {
      py.globals.set("_prog", pyProgress);
      await py.runPythonAsync("S.run_lda(lambda p, m: _prog(p, m)); S.cfg['n_workers_used'] = 1");
    }
  } else {
    (await poolP)?.terminate();
  }
  if (on("bertopic")) {
    trackStage("embed");
    stage("BERTopic: embeddings de los documentos");
    const texts = py.runPython("S.texts()").toJs();
    const emb = await embed(texts, cfg.embedding_model);
    const n = texts.length, rows = [];
    for (let i = 0; i < n; i++) rows.push(Array.from(emb.data.subarray(i * emb.dim, (i + 1) * emb.dim)));
    const us = JSON.parse(py.runPython("json.dumps(S.umap_settings())"));
    const runs = us.seeds.length;
    const reduced = [];
    trackStage("umap");
    const umapProg = (r) => (p, m) => progress((r + p) / (runs + 1), m);  // avance de todas las proyecciones
    for (let r = 0; r < runs; r++) {
      stage(`BERTopic: UMAP ${r + 1} de ${runs}${r ? " (estabilidad)" : ""}`);
      reduced.push(await umap(rows, us.n_components, us.min_dist, us.seeds[r], `UMAP ${r + 1}/${runs}`,
        { metric: us.metric, nNeighbors: us.n_neighbors, onProgress: umapProg(r) }));
    }
    stage("BERTopic: mapa 2D de documentos");
    const xy = await umap(rows, 2, us.map_2d.min_dist, us.map_2d.seed, "UMAP 2D",
      { metric: us.metric, nNeighbors: us.n_neighbors, onProgress: umapProg(runs) });
    trackStage("cluster");
    stage("BERTopic: HDBSCAN y c-TF-IDF");
    progress(0.05, "Agrupando los documentos (HDBSCAN) y extrayendo los términos de cada tema…");
    py.globals.set("_red", reduced);
    py.globals.set("_xy", xy);
    py.globals.set("_emb_name", emb.name);
    await py.runPythonAsync(`
_n = len(S.texts())
_reds = [np.frombuffer(r.to_bytes(), dtype=np.float32).reshape(_n, -1) for r in _red]
_xy2 = np.frombuffer(_xy.to_bytes(), dtype=np.float32).reshape(_n, 2)
_bt = S.run_bertopic(_reds, _xy2, _emb_name)
`);
    const bt = JSON.parse(py.runPython("json.dumps(_bt)"));
    if (bt.warning) post("warn", { msg: bt.warning });
  }
  // punto de control: los modelos ya están ajustados; si algo falla al generar las figuras, el proyecto
  // guardado permite repetir solo esta última etapa
  try {
    progress(1, "Guardando un punto de control del análisis…");
    const bytes = pyBytes(await py.runPythonAsync("S.save_project()"));
    post("checkpoint", { data: bytes }, [bytes.buffer]);
  } catch (e) { /* el punto de control es opcional */ }
  trackStage("render");
  stage("Generando figuras y tablas");
  progress(0.01, "Preparando figuras y tablas…");
  await render(true);
  trackStage(null);
  post("done", { elapsed: (Date.now() - t0) / 1000, timings: TRACK ? TRACK.timings : null });
  TRACK = null;
}

async function render(withProgress = false) {
  py.globals.set("_rprog", (p, m) => progress(p, m));
  const figs = JSON.parse(await py.runPythonAsync(
    withProgress ? "json.dumps(S.render('/out', progress=lambda p, m: _rprog(p, m)))" : "json.dumps(S.render('/out'))"));
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
            "T12_expert_validation", "topic_best_matches", "tau_sensitivity", "T14_bibliographic_coupling", "periods",
            "merge_conflicts"]}, ensure_ascii=False, default=str)`));
  const labels = JSON.parse(py.runPython("json.dumps(S.topic_labels(), ensure_ascii=False)"));
  const quality = JSON.parse(py.runPython("json.dumps(S.quality_checks(), ensure_ascii=False, default=str)"));
  const palette = JSON.parse(py.runPython("json.dumps(getattr(S, 'palette', None) or {}, default=str)"));
  const palettes = JSON.parse(py.runPython("from bibliotopic import palettes as _PAL; json.dumps(_PAL.catalogue())"));
  post("results", { figs: out, tables, labels, quality, palette, palettes }, transfer);
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

async function scriptPackage(meta, cfg) {
  // con cfg (antes de ejecutar): el paquete lleva la configuración elegida en la interfaz
  await callPy("_b = S.script_package(json.loads(_meta), json.loads(_cfg) if _cfg else None)",
    { _meta: JSON.stringify(meta || {}), _cfg: cfg ? JSON.stringify(cfg) : "" });
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
    else if (cmd === "script") await scriptPackage(ev.data.meta, ev.data.cfg);
    else if (cmd === "estimate") {
      const e = JSON.parse(await callPy("json.dumps(S.estimate(json.loads(_cfg)))", { _cfg: JSON.stringify(ev.data.cfg) }));
      post("estimate", { estimate: e });
    }
    else if (cmd === "palette_preview") {
      const p = JSON.parse(await callPy("json.dumps(S.palette_choice(json.loads(_cfg)), default=str)", { _cfg: JSON.stringify(ev.data.cfg) }));
      post("palette", { palette: p });
    }
    else if (cmd === "palette") {  // cambiar la paleta después del análisis: solo se vuelven a dibujar las figuras
      progress(0.2, "Aplicando la paleta a las figuras…");
      await callPy("S.set_palette(json.loads(_pk), _pv)", { _pk: JSON.stringify(ev.data.key), _pv: ev.data.variant });
      await py.runPythonAsync("S.final = False");
      await render();
    }
    else if (cmd === "titles") await proposeTitles(ev.data.meta);
  } catch (e) {
    // De un error de Python se muestra solo el mensaje final (p. ej., «ValueError: …»), no la traza completa
    let msg = String(e.message || e);
    const m = msg.match(/(?:ValueError|KeyError|RuntimeError): ([^\n]+)\s*$/);
    if (m) msg = m[1].replace(/^'|'$/g, "");
    post("error", { msg });
  }
};
