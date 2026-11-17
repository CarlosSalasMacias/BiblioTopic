// BiblioTopic — interfaz (hilo principal). El trabajo pesado ocurre en js/worker.js.
const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.js?v=20261124", import.meta.url), { type: "module" });
const state = { analyses: {}, sections: {}, summary: null, labels: null, busy: false, aiLabels: false, aiModel: "" };

const DEPENDS = { compare: ["lda", "bertopic"], inference: ["compare", "lda", "bertopic"], strategic: ["lda"],
  hjbiplot: [], temporal: [] };
const DEFAULT_OFF_SECTIONS = ["discussion"];
const FIG_TITLES = {
  prisma: "Diagrama de flujo", annual: "Producción anual", sources: "Fuentes", authors: "Autores",
  countries: "Países", collab_map: "Mapa de colaboración", collab: "Red de colaboración entre países", kw: "Palabras clave",
  ksel: "Selección de K (LDA)", lda_words: "Temas LDA", lda_time: "LDA por periodo",
  hj_lda: "HJ-Biplot LDA", bert_words: "Temas BERTopic", bert_map: "Mapa de documentos",
  bert_time: "BERTopic por periodo", hj_bert: "HJ-Biplot BERTopic", sim: "Similitud LDA–BERTopic",
  cont: "Coincidencia por documento", trends: "Temas emergentes y en declive",
  hj_country: "HJ-Biplot países", strategic: "Mapa estratégico", callon: "Mapa temático (Callon)",
  sankey: "Evolución temática (Sankey)", cocit: "Red de co-citación",
};
const TABLE_TITLES = {
  T1_main_information: "Información principal", T8_model_metrics: "Comparación de modelos",
  doc_agreement: "Coincidencia por documento", T9_topic_alignment: "Emparejamiento de temas",
  T4_countries: "Países", T2_top_sources: "Fuentes principales", T8b_inference: "Inferencia estadística (IC 95 %)",
  T10_most_cited: "Documentos más citados", T11_thematic_map: "Mapa temático", T12_expert_validation: "Validación por expertos",
  topic_best_matches: "Mejor correspondencia de cada tema (muchos a uno)", tau_sensitivity: "Sensibilidad a los umbrales τ",
  T14_bibliographic_coupling: "Acoplamiento bibliográfico", periods: "Periodos", merge_conflicts: "Duplicados con datos discrepantes",
};

function toast(msg, ms = 5000) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.hidden = true), ms);
}
const setBar = (el, p) => ($(el).style.width = `${Math.round(100 * Math.max(0, Math.min(1, p)))}%`);
const fmtTime = (s) => (s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);
const fmtDur = (s) => (s < 90 ? "menos de 2 min" : s < 3600 ? `${Math.round(s / 60)} min`
  : `${Math.floor(s / 3600)} h ${String(Math.round((s % 3600) / 60)).padStart(2, "0")} min`);

// ---------------------------------------------- progreso total, reloj y vigilancia del análisis
const RUN = { t0: 0, last: 0, timer: null, eta: null, etaAt: 0 };
function runClockStart() {
  RUN.t0 = RUN.last = Date.now(); RUN.eta = null;
  $("total-box").hidden = false; $("watchdog").hidden = true; $("timings").hidden = true;
  setBar("bar-total", 0); $("tot-pct").textContent = "0 %"; $("tot-eta").textContent = "calculando el tiempo restante…";
  clearInterval(RUN.timer);
  RUN.timer = setInterval(() => {
    const el = (Date.now() - RUN.t0) / 1000, quiet = (Date.now() - RUN.last) / 1000;
    let rem = "";
    if (RUN.eta != null) rem = ` · quedan ~${fmtDur(Math.max(0, RUN.eta - (Date.now() - RUN.etaAt) / 1000))}`;
    $("tot-clock").textContent = `Tiempo transcurrido: ${fmtDur(el)}${rem}. Mantenga esta pestaña abierta y el equipo enchufado.`;
    // sin noticias del análisis durante 10 min: probablemente se detuvo (p. ej., memoria insuficiente)
    if (quiet > 600) {
      $("watchdog").hidden = false;
      $("watchdog").innerHTML = `<b>El análisis no informa avances desde hace ${fmtDur(quiet)}.</b> Puede haberse detenido.
        Compruébelo con Shift + Esc (Administrador de tareas del navegador): si BiblioTopic usa 0 % de CPU, recargue
        la página y use «Recuperar el último análisis» para repetir solo las figuras.`;
    } else $("watchdog").hidden = true;
  }, 1000);
}
function runClockStop() { clearInterval(RUN.timer); RUN.timer = null; }
function runTotal(t) {
  RUN.last = Date.now();
  if (!t) return;
  const pct = Math.round(100 * t.overall);
  setBar("bar-total", t.overall);
  $("tot-pct").textContent = `${pct} %`;
  RUN.eta = t.remaining; RUN.etaAt = Date.now();
  $("tot-eta").textContent = t.overall < 0.03 ? "· tiempo restante estimado al empezar"
    : `· quedan ~${fmtDur(t.remaining)}`;
}
const STAGE_NAMES = { prep: "Preparación", lda: "LDA", embed: "Embeddings", umap: "UMAP", cluster: "HDBSCAN y c-TF-IDF",
  render: "Figuras y tablas" };
function showTimings(t) {
  if (!t) return;
  const parts = Object.entries(t).filter(([, v]) => v >= 1).map(([k, v]) => `${STAGE_NAMES[k] || k}: ${fmtDur(v)}`);
  $("timings").hidden = !parts.length;
  $("timings").textContent = `Tiempo por etapa — ${parts.join(" · ")}`;
}

// ---------------------------------------------- punto de control (IndexedDB) para recuperar un análisis
function idb() {
  return new Promise((res, rej) => {
    const r = indexedDB.open("bibliotopic", 1);
    r.onupgradeneeded = () => r.result.createObjectStore("kv");
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
}
async function idbPut(key, val) {
  try {
    const db = await idb();
    await new Promise((res, rej) => { const tx = db.transaction("kv", "readwrite"); tx.objectStore("kv").put(val, key);
      tx.oncomplete = res; tx.onerror = () => rej(tx.error); });
  } catch (e) { /* almacenamiento no disponible: sin punto de control */ }
}
async function idbGet(key) {
  try {
    const db = await idb();
    return await new Promise((res, rej) => { const r = db.transaction("kv").objectStore("kv").get(key);
      r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
  } catch (e) { return null; }
}
async function showRecover() {
  const ck = await idbGet("checkpoint");
  if (!ck || !ck.data) return;
  const box = $("recover");
  box.hidden = false;
  box.innerHTML = `Hay un análisis guardado automáticamente el ${new Date(ck.when).toLocaleString()} (modelos ya
    ajustados). <button type="button" id="b-recover">Recuperar el último análisis</button>
    <span class="hint">Solo se vuelven a generar las figuras y tablas.</span>`;
  $("b-recover").onclick = () => {
    if (state.busy || !$("env").classList.contains("ready")) return toast("Espere a que el entorno esté listo.");
    busy(true);
    $("run-status").hidden = false;
    $("stage").textContent = "Recuperando el análisis guardado…";
    const data = new Uint8Array(ck.data);
    worker.postMessage({ cmd: "open_project", data }, [data.buffer]);
    box.hidden = true;
  };
}
const csv = (s) => s.split(",").map((x) => x.trim()).filter(Boolean);

function busy(on) {
  state.busy = on;
  for (const id of ["b-load", "b-run", "b-labels", "b-doc", "b-zip", "b-save", "b-val-sheet", "b-ai-labels",
    "b-ai-disc", "b-script", "b-ai-translate", "b-ai-short"]) $(id).disabled = on;
}

// ------------------------------------------------------------------ archivos
let files = { scopus: [], wos: [] };
for (const kind of ["scopus", "wos"]) {
  $(`f-${kind}`).addEventListener("change", (e) => {
    files[kind] = [...e.target.files];
    $(`l-${kind}`).textContent = files[kind].length ? files[kind].map((f) => f.name).join(", ") : "Ningún archivo";
    $("b-load").disabled = !(files.scopus.length || files.wos.length) || state.busy;
  });
}

$("b-load").onclick = async () => {
  const all = [...files.scopus, ...files.wos];
  if (!all.length) return;
  busy(true);
  const payload = await Promise.all(all.map(async (f) => ({ name: f.name, data: new Uint8Array(await f.arrayBuffer()) })));
  worker.postMessage({ cmd: "load", files: payload }, payload.map((p) => p.data.buffer));
};

function showSummary(s) {
  const box = $("summary");
  box.hidden = false;
  if (!s.ok) {
    box.innerHTML = `<div class="warn err">${s.errors.join("<br>")}</div>`;
    return;
  }
  const st = s.stats;
  const by = Object.entries(st.records_by_db).map(([k, v]) => `${k}: ${v.toLocaleString()}`).join(" · ");
  const ov = Object.entries(st.overlap).map(([k, v]) => `${k}: ${v.toLocaleString()}`).join(" · ");
  const av = s.availability;
  const warns = [];
  if (av.C1 < 30) warns.push("Sin afiliaciones (C1): no habrá análisis de países ni colaboración.");
  if (av.DE < 30 && av.ID < 30) warns.push("Sin palabras clave: no habrá red de palabras clave.");
  if (av.TC < 30) warns.push("Sin número de citas: no habrá mapa estratégico ni citas por año.");
  if (st.records_by_db.WoS && av.C1 < 30) warns.push("En WoS exporte con «Full Record and Cited References».");
  for (const [f, nom] of [["LA", "idioma"], ["DT", "tipo de documento"]]) {
    const u = Object.entries((s.unknown || {})[f] || {});
    if (u.length) warns.push(`${u.map(([k, v]) => `${k}: ${v.toLocaleString()}`).join(" · ")} registros sin ${nom} informado. `
      + `Se conservan por defecto (opción «Sin dato» en las opciones de filtrado); desmárquela para excluirlos.`);
  }
  box.innerHTML = `
    ${overlapCard(st)}
    <div class="stats">
      <div class="stat"><b>${st.records_total.toLocaleString()}</b><span>registros cargados</span></div>
      <div class="stat"><b>${st.duplicates_removed.toLocaleString()}</b><span>duplicados eliminados</span></div>
      <div class="stat"><b>${st.unique.toLocaleString()}</b><span>registros únicos</span></div>
      <div class="stat"><b>${s.with_abstract.toLocaleString()}</b><span>con resumen</span></div>
      <div class="stat"><b>${s.year_min}–${s.year_max}</b><span>periodo</span></div>
    </div>
    <p class="hint">Por base: ${by}<br>Origen de los registros únicos: ${ov}</p>
    <p class="hint">Completitud de campos (%): resumen ${av.AB} · afiliaciones ${av.C1} · palabras clave ${av.DE} ·
      citas ${av.TC} · tipo de documento ${av.DT} · DOI ${av.DI}</p>
    ${s.errors.map((e) => `<div class="warn err">${e}</div>`).join("")}
    ${warns.map((w) => `<div class="warn">${w}</div>`).join("")}
    ${auditHtml(s.audit)}
    ${topKwHtml(s.top_keywords)}`;
  state.auditBlock = !!(s.audit && s.audit.critical_low.length);
  if (s.palettes) { fillPalettes(s.palettes); showPalette(s.palette_auto); requestPalette(); }
  // la confirmación aparece también junto al botón «Ejecutar análisis»; ambas casillas están sincronizadas
  $("audit-gate").hidden = !state.auditBlock;
  syncAudit(false);
  if ($("audit-ok")) $("audit-ok").onchange = (e) => syncAudit(e.target.checked);
  state.loaded = true;
  applyProfile($("o-profile").value);
  // opciones dependientes de los datos
  $("o-ymin").value = s.year_min;
  // el año en curso se incluye en los descriptivos; las tendencias y la tasa de crecimiento lo excluyen
  $("o-ymax").value = s.year_max;
  const unk = (f) => Object.values((s.unknown || {})[f] || {}).reduce((a, b) => a + b, 0);
  const unkBy = (f) => Object.entries((s.unknown || {})[f] || {}).map(([k, v]) => `${k}: ${v.toLocaleString()}`).join(" · ");
  chips("o-dt", s.doc_types, ["Article", "Review"], unk("DT"), unkBy("DT"));
  chips("o-la", s.languages, ["English"], unk("LA"), unkBy("LA"));
  $("s-analyses").hidden = false;
}

// Valores del campo + «Sin dato» (registros sin el campo: p. ej., Scopus sin columna de idioma). Los
// registros sin dato no son «otro idioma»: se conservan por defecto y la decisión queda explícita.
// Auditoría de campos por base y por archivo; si un campo crítico está por debajo del 90 % en alguna base,
// el análisis queda bloqueado hasta que el usuario lo confirme.
const FIELD_NAMES = { AB: "Resumen", DE: "Palabras clave de autor", ID: "Index Keywords / Keywords Plus", LA: "Idioma",
  DT: "Tipo de documento", PY: "Año", DI: "DOI", TC: "Citas", C1: "Afiliaciones", RP: "Autor de correspondencia",
  CR: "Referencias citadas", "RP/C1": "Correspondencia o afiliación" };
// Tarjeta de cobertura por base: registros de cada base, únicos y barra solo Scopus / ambas / solo WoS
function overlapCard(st) {
  const by = st.records_by_db || {}, ov = st.overlap || {};
  if (!(by.Scopus && by.WoS)) return "";
  const parts = [["Solo Scopus", ov.Scopus || 0, "sc"], ["Ambas", ov["WoS+Scopus"] || 0, "both"], ["Solo WoS", ov.WoS || 0, "wos"]];
  const tot = parts.reduce((a, p) => a + p[1], 0) || 1;
  const pct = (n) => (100 * n / tot).toFixed(1);
  return `<div class="ovcard">
    <div class="ovnums">
      <div><span>Scopus</span><b>${by.Scopus.toLocaleString()}</b></div>
      <div><span>WoS</span><b>${by.WoS.toLocaleString()}</b></div>
      <div><span>Registros únicos</span><b class="ok">${st.unique.toLocaleString()}</b></div>
    </div>
    <div class="ovbar" role="img" aria-label="${parts.map((p) => `${p[0]}: ${p[1]}`).join(", ")}">
      ${parts.map((p) => `<i class="${p[2]}" style="flex-grow:${p[1]}" title="${p[0]}: ${p[1].toLocaleString()} (${pct(p[1])} %)"></i>`).join("")}</div>
    <div class="ovleg">${parts.map((p) => `<span><i class="${p[2]}"></i>${p[0]}: <b>${p[1].toLocaleString()}</b> (${pct(p[1])} %)</span>`).join("")}</div>
  </div>`;
}

function auditHtml(a) {
  if (!a) return "";
  const fields = ["AB", "DE", "ID", "LA", "DT", "PY", "DI", "TC", "C1", "RP", "CR"];
  const crit = new Set(a.critical);
  const cell = (v, f) => `<td class="${(crit.has(f) || (f === "C1" || f === "RP")) && v < a.critical_min ? "low" : ""}">${v}</td>`;
  const row = (name, v) => `<tr><th>${escH(name)}${v._n ? ` <span class="hint">(${v._n.toLocaleString()})</span>` : ""}</th>${fields.map((f) => cell(v[f], f)).join("")}</tr>`;
  const head = `<tr><th></th>${fields.map((f) => `<th title="${FIELD_NAMES[f]}">${f}</th>`).join("")}</tr>`;
  const low = a.critical_low;
  return `<details class="audit" ${low.length ? "open" : ""}><summary>Auditoría de campos (% de registros con el campo)</summary>
    <table class="audit-t"><thead>${head}</thead><tbody>
      ${Object.entries(a.by_db).map(([db, v]) => row(db, v)).join("")}
      <tr><td colspan="${fields.length + 1}" class="hint">Por archivo</td></tr>
      ${Object.entries(a.by_file).map(([fn, v]) => row(fn, v)).join("")}</tbody></table></details>
    ${low.length ? `<div class="warn err"><b>Campos críticos incompletos:</b>
      ${low.map((l) => `${FIELD_NAMES[l.field] || l.field} (${l.field}) en ${l.db}: ${l.pct} %`).join("; ")}.
      Los registros sin el dato no se excluyen (se conservan y se informan aparte), pero el análisis puede quedar
      sesgado. Lo recomendable es volver a exportar con todos los campos (ver «Cómo exportar»).
      <label class="confirm"><input type="checkbox" id="audit-ok"> Entiendo la advertencia y quiero continuar con estos archivos</label></div>` : ""}`;
}
function topKwHtml(t) {
  if (!t || !t.keywords.length) return "";
  return `<details class="topkw"><summary>Las 50 palabras clave más frecuentes (${t.field}), tras la normalización automática</summary>
    <p class="hint">Revise si hay variantes del mismo concepto y añádalas al archivo de sinónimos (opciones avanzadas).</p>
    <div class="kwlist">${t.keywords.map(([k, n]) => `<span>${escH(k)} <b>${n}</b></span>`).join("")}</div></details>`;
}

function chips(id, counts, defaults, unknown = 0, unknownBy = "") {
  const box = $(id);
  const keys = Object.keys(counts);
  if (!keys.length) {
    box.innerHTML = '<span class="hint">(no disponible en los archivos; no se filtra)</span>';
    return;
  }
  const useDefault = keys.some((k) => defaults.includes(k));
  box.innerHTML = keys.map((k) => `<label><input type="checkbox" value="${k}"
      ${!useDefault || defaults.includes(k) ? "checked" : ""}> ${k} (${counts[k]})</label>`).join("")
    + (unknown ? `<label title="${unknownBy}"><input type="checkbox" value="__unknown__" checked>
      <i>Sin dato</i> (${unknown}${unknownBy ? "; " + unknownBy : ""})</label>` : "");
}
const checked = (id) => [...$(id).querySelectorAll("input:checked")].map((i) => i.value).filter((v) => v !== "__unknown__");
const keepUnknown = (id) => !$(id).querySelector('input[value="__unknown__"]') || $(id).querySelector('input[value="__unknown__"]').checked;

// ------------------------------------------------------------------ análisis
function buildChecks(id, items, defaults) {
  $(id).innerHTML = Object.entries(items).map(([k, v]) =>
    `<label><input type="checkbox" value="${k}" ${defaults.includes(k) ? "checked" : ""}> ${v}</label>`).join("");
}

$("analyses").addEventListener("change", (e) => {
  const v = e.target.value;
  const sel = new Set(checked("analyses"));
  if (e.target.checked && DEPENDS[v]) DEPENDS[v].forEach((d) => sel.add(d));
  if (!e.target.checked) for (const [k, deps] of Object.entries(DEPENDS)) if (deps.includes(v)) sel.delete(k);
  $("analyses").querySelectorAll("input").forEach((i) => (i.checked = sel.has(i.value)));
});

function syncAudit(v) {
  if ($("audit-ok")) $("audit-ok").checked = v;
  $("audit-ok2").checked = v;
  $("audit-gate").classList.toggle("ok", v);
}
$("audit-ok2").onchange = (e) => syncAudit(e.target.checked);

$("b-run").onclick = () => {
  const analyses = checked("analyses");
  if (!analyses.length) return toast("Seleccione al menos un análisis.");
  const cfg = buildCfg();
  if (state.auditBlock && !$("audit-ok2").checked) {
    const g = $("audit-gate");
    g.scrollIntoView({ behavior: "smooth", block: "center" });
    g.classList.remove("flash"); void g.offsetWidth; g.classList.add("flash");
    return toast("Marque la casilla de confirmación que está justo encima del botón «Ejecutar análisis».", 9000);
  }
  if (cfg.k_min >= cfg.k_max && !cfg.k) return toast("K mínimo debe ser menor que K máximo.");
  busy(true);
  runClockStart();
  $("run-status").hidden = false;
  $("stage").textContent = "Iniciando…";
  setBar("bar", 0);
  worker.postMessage({ cmd: "run", cfg });
};

const num = (id) => Number($(id).value);
function searchCfg() {
  return { search_scopus: $("q-scopus").value.trim(), search_wos: $("q-wos").value.trim(),
    search_date: $("q-date").value, wos_indexes: [...$("q-idx").querySelectorAll("input:checked")].map((i) => i.value) };
}
// Procesos auxiliares para LDA: uno menos que los núcleos lógicos, hasta 6 (cada uno usa ~300 MB de memoria);
// 2 como máximo si el navegador informa ≤ 4 GB de RAM. El resultado no depende del número de procesos.
function autoWorkers() {
  const hc = navigator.hardwareConcurrency || 2;
  const mem = navigator.deviceMemory || 8;
  return Math.max(1, Math.min(hc - 1, mem <= 4 ? 2 : 6));
}
function nWorkers() {
  const v = $("o-workers").value;
  return v === "auto" ? autoWorkers() : Number(v);
}
$("o-workers").options[0].textContent = `Automático (${autoWorkers()})`;

// ------------------------------------------------------------ paleta de colores de las figuras
function paletteCfg() {
  const v = $("o-palette").value;
  if (v === "auto") return { palette: "auto", palette_variant: null };
  const [key, variant] = v.split(":");
  return { palette: key, palette_variant: Number(variant) };
}
function fillPalettes(list) {
  state.palettes = list;
  const sel = $("o-palette");
  sel.innerHTML = `<option value="auto">Automática</option>` + list.map((p) =>
    `<option value="${p.key}:0">${escH(p.name)} · A</option><option value="${p.key}:1">${escH(p.name)} · B</option>`).join("");
}
function showPalette(p) {
  if (!p || !p.categorical) return;
  if (p.auto) {
    state.paletteAuto = p;
    $("o-palette").options[0].textContent = `Automática — ${p.name} · ${p.variant ? "B" : "A"}`;
  }
  const cur = $("o-palette").value === "auto" ? state.paletteAuto || p : p;
  $("pal-swatch").innerHTML = cur.categorical.map((c) => `<i style="background:${c}"></i>`).join("");
  $("pal-note").textContent = $("o-palette").value === "auto"
    ? `elegida por la temática (${cur.topic}); cámbiela si prefiere otra`
    : `paleta para ${cur.topic}`;
}
function localPalette(v) {  // vista previa inmediata al elegir una paleta fija
  const [key, variant] = v.split(":");
  const p = (state.palettes || []).find((x) => x.key === key);
  return p && { name: p.name, topic: p.topic, categorical: Number(variant) ? p.alt : p.cat, auto: false };
}
let palTimer = null;
function requestPalette() {
  clearTimeout(palTimer);
  palTimer = setTimeout(() => {
    if (state.loaded && !state.busy) worker.postMessage({ cmd: "palette_preview", cfg: { ...searchCfg(), query_terms: csv($("o-q").value), palette: "auto" } });
  }, 500);
}
for (const id of ["q-scopus", "q-wos", "o-q"]) $(id).addEventListener("input", requestPalette);
$("o-palette").addEventListener("change", (e) => {
  const v = e.target.value;
  showPalette(v === "auto" ? state.paletteAuto : localPalette(v));
  if (state.hasResults && !state.busy) {  // con resultados: se vuelven a dibujar las figuras (sin repetir el análisis)
    const pc = paletteCfg();
    busy(true);
    worker.postMessage({ cmd: "palette", key: pc.palette, variant: pc.palette_variant });
  }
});

function buildCfg() {
  const analyses = checked("analyses");
  return {
    config_version: 2, profile: $("o-profile").value, analyses,
    year_min: num("o-ymin") || null, year_max: num("o-ymax") || null,
    doc_types: checked("o-dt").length ? checked("o-dt") : null,
    languages: checked("o-la").length ? checked("o-la") : null,
    keep_unknown_language: keepUnknown("o-la"), keep_unknown_type: keepUnknown("o-dt"),
    keep_unknown_year: !$("o-unk-year").checked,
    query_terms: csv($("o-q").value), extra_stopwords: csv($("o-sw").value), synonyms: $("o-syn").value,
    citation_source: $("o-tc").value,
    k_min: num("o-kmin"), k_max: num("o-kmax"), k_step: num("o-kstep"), k: num("o-k") || null,
    k_fine_window: num("o-kfine"), k_seeds: num("o-kseeds"), lda_method: $("o-ldamethod").value,
    lda_max_iter: num("o-ldaiter"), lda_doc_prior: num("o-alpha") || null, lda_word_prior: num("o-eta") || null,
    bert_nr_topics: $("o-bnr").value === "lda" ? "lda" : null, embedding_model: $("o-emb").value,
    min_cluster_size: num("o-mcs") || null, min_samples: num("o-msamp") || null, outlier_threshold: num("o-othr"),
    n_stab_lda: num("o-slda"), n_stab_bert: num("o-sbert"),
    n_boot: Math.min(num("o-boot"), 5000), boot_method: $("o-bootm").value, n_perm: num("o-perm"),
    tau_strong: num("o-tau"), tau_partial: num("o-taup"), trend_test: $("o-trend").value,
    period_mode: $("o-pmode").value, period_width: num("o-pw"), n_periods: num("o-np"), custom_periods: $("o-pcustom").value,
    min_df: num("o-mindf"), seed: num("o-seed"), cluster_shape: $("o-shape").value, n_workers: nWorkers(),
    ...paletteCfg(),
    composite: $("m-comp").checked, export_tiff: $("m-tiff").checked, ...searchCfg(),
  };
}

// valores de cada perfil (los mismos que PROFILES en bibliotopic/pipeline.py)
const PROFILES = {
  publication: { "o-boot": 1000, "o-bootm": "percentile", "o-perm": 1000, "o-slda": 10, "o-sbert": 10,
    "o-kmin": 5, "o-kmax": 40, "o-kstep": 5, "o-kfine": 5, "o-kseeds": 3, "o-ldamethod": "batch", "o-ldaiter": 100 },
  quick: { "o-boot": 200, "o-bootm": "percentile", "o-perm": 200, "o-slda": 3, "o-sbert": 3,
    "o-kmin": 5, "o-kmax": 30, "o-kstep": 5, "o-kfine": 0, "o-kseeds": 1, "o-ldamethod": "batch", "o-ldaiter": 30 },
};
function applyProfile(name) {
  for (const [id, v] of Object.entries(PROFILES[name])) $(id).value = v;
  requestEstimate();
}
$("o-profile").addEventListener("change", (e) => applyProfile(e.target.value));
let estTimer = null;
function requestEstimate() {
  clearTimeout(estTimer);
  estTimer = setTimeout(() => { if (!state.busy && state.loaded) worker.postMessage({ cmd: "estimate", cfg: buildCfg() }); }, 400);
}
for (const id of ["o-kmin", "o-kmax", "o-kstep", "o-kfine", "o-kseeds", "o-ldamethod", "o-ldaiter", "o-slda", "o-sbert", "o-workers",
  "o-boot", "o-k", "o-emb", "analyses"]) $(id).addEventListener("change", requestEstimate);
function showEstimate(e) {
  const fmt = (m) => (m >= 90 ? `${(m / 60).toFixed(1)} h` : `${Math.max(1, Math.round(m))} min`);
  const par = e.n_workers > 1 ? `, ${e.n_workers} procesos en paralelo` : "";
  $("estimate").textContent = `Tiempo estimado en el navegador: ~${fmt(e.minutes)} (LDA ${fmt(e.lda_minutes)}${par}; BERTopic ${fmt(e.bertopic_minutes)}) · ${e.n_docs.toLocaleString()} documentos con resumen.`;
  const local = e.local_minutes_1core;
  $("estimate-warn").innerHTML = e.minutes > 60
    ? `<div class="warn">Con este perfil y este corpus el análisis tardaría unas ${fmt(e.minutes)} en el navegador
       (la pestaña debe quedar abierta). Alternativas sin rebajar los valores: (1) descargue el script con esta
       configuración y ejecútelo en su computadora: ~${fmt(local)} con un núcleo, y aproximadamente ese tiempo
       dividido por el número de núcleos con <code>--nucleos</code>; (2) para explorar, use el perfil
       «Exploración rápida» y luego repita con «Publicación». Puede ejecutarlo igualmente aquí.</div>` : "";
}
$("b-script-pre").onclick = () => { busy(true); worker.postMessage({ cmd: "script", meta: docMeta(), cfg: buildCfg() }); };
$("f-syn").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (f) $("o-syn").value = (await f.text()).replace(/^\ufeff/, "");
  e.target.value = "";
});
$("b-syn-template").onclick = () => download("sinonimos.csv",
  new TextEncoder().encode("term,canonical\narabica coffee,coffea arabica\nrobusta coffee,coffea canephora\n"), "text/csv");
// términos de la búsqueda a partir de las ecuaciones (misma regla que io.extract_query_terms)
function extractQueryTerms(...eqs) {
  const out = [];
  for (let q of eqs) {
    if (!q) continue;
    q = q.replace(/\b(AND\s+(NOT\s+)?)?(LIMIT-TO|EXCLUDE|PUBYEAR|DOCTYPE|LANGUAGE|SUBJAREA|SRCTYPE|PUBSTAGE|OA|AFFILCOUNTRY|EXACTKEYWORD|PY|DT|LA|WC|SU|CU|OG|SO|DOP)\b\s*(=|>|<|>=|<=)?\s*(\((?:[^()]|\([^()]*\))*\)|\d{4}|"[^"]*")/gi, " ");
    q = q.replace(/\b(TITLE-ABS-KEY|TITLE-ABS|TITLE|ABS|KEY|AUTHKEY|INDEXTERMS|ALL|TS|TI|AB|AK|KP)\s*=?\s*/gi, " ");
    for (;;) {
      const m = /\bNOT\s*\(/i.exec(q);
      if (!m) break;
      let d = 0, i = m.index + m[0].length - 1;
      for (; i < q.length; i++) { if (q[i] === "(") d++; if (q[i] === ")") d--; if (d === 0) break; }
      q = q.slice(0, m.index) + " " + q.slice(i + 1);
    }
    q = q.replace(/\bNOT\s+("[^"]*"|\S+)/gi, " ");
    for (const m of q.matchAll(/["{]([^"}]+)["}]/g)) out.push(m[1].replace(/[*?$]/g, "").trim().toLowerCase().replace(/\s+/g, " "));
    q = q.replace(/["{][^"}]*["}]/g, " ");
    for (let t of q.split(/[\s()]+/)) {
      t = t.replace(/[*?$]/g, "").trim().toLowerCase().replace(/^(w|pre|near)\/\d+$/, "");
      if (t && !["and", "or", "not", "near", "same", "pre", "w"].includes(t) && !/^[\d/=<>-]+$/.test(t) && t.length > 1) out.push(t);
    }
  }
  return [...new Set(out.filter(Boolean))];
}
let qAuto = true;
$("o-q").addEventListener("input", () => { qAuto = false; });
for (const id of ["q-scopus", "q-wos"]) $(id).addEventListener("input", () => {
  if (qAuto) $("o-q").value = extractQueryTerms($("q-scopus").value, $("q-wos").value).join(", ");
});

// ------------------------------------------------------------------ resultados
let urls = [];
function showQuality(q) {
  if (!q) return;
  const icon = { red: "🔴", amber: "🟡", green: "🟢" };
  $("quality").innerHTML = `<h3>Controles de calidad</h3><ul class="quality">${q.map((c) =>
    `<li class="${c.level}">${icon[c.level]} <b>${escH(c.check)}</b> — ${escH(c.detail)}</li>`).join("")}</ul>`;
}
function showResults({ figs, tables, labels, quality, palette, palettes }) {
  showQuality(quality);
  state.hasResults = true;
  if (!state.palettes && palettes) fillPalettes(palettes);
  if (palette && palette.categorical) {
    if (palette.auto) showPalette(palette);
    else { $("o-palette").value = `${palette.key}:${palette.variant}`; showPalette(palette); }
  }
  $("s-results").hidden = false;
  $("s-write").hidden = false;
  urls.forEach(URL.revokeObjectURL);
  urls = [];
  $("gallery").innerHTML = "";
  for (const f of figs) {
    const url = URL.createObjectURL(new Blob([f.data], { type: "image/png" }));
    urls.push(url);
    const fig = document.createElement("figure");
    fig.innerHTML = `<a href="${url}" target="_blank"><img src="${url}" alt="${FIG_TITLES[f.key] || f.key}"></a>
      <figcaption><span>${FIG_TITLES[f.key] || f.key}</span><a href="${url}" download="${f.name}">PNG</a></figcaption>`;
    $("gallery").appendChild(fig);
  }
  $("tables").innerHTML = Object.entries(TABLE_TITLES).filter(([k]) => tables[k]).map(([k, title]) => {
    const t = tables[k];
    return `<h3>${title}</h3><div class="tbl"><table><thead><tr>${t.columns.map((c) => `<th>${c}</th>`).join("")}</tr></thead>
      <tbody>${t.data.slice(0, 25).map((r) => `<tr>${r.map((v) => `<td>${v ?? ""}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }).join("");
  if (!state.labels || JSON.stringify(Object.keys(labels)) !== JSON.stringify(Object.keys(state.labels))) {
    state.labels = labels;
    renderLabels(labels);
  }
}

// Nombre corto para las figuras (misma regla que short_label en bibliotopic/pipeline.py)
const SMALL = new Set(["and", "of", "the", "in", "for", "with", "on", "to", "a", "an", "by", "&", "from", "at", "via"]);
function shortLabel(name, maxlen = 28) {
  const s = String(name || "").split(/\s+/).filter(Boolean).join(" ");
  if (s.length <= maxlen) return s;
  const parts = s.split(/\s+(?:and|&)\s+|\s*[:;,(]\s*|\s+[-\u2013\u2014]\s+/).map((p) => p.trim()).filter(Boolean);
  const cand = parts.length && (parts[0].split(" ").length >= 2 || parts.length === 1) ? parts[0] : s;
  const out = [];
  for (const w of cand.split(" ")) {
    if (out.length && [...out, w].join(" ").length > maxlen) break;
    out.push(w);
  }
  while (out.length > 1 && SMALL.has(out[out.length - 1].toLowerCase().replace(/[,;:]+$/, ""))) out.pop();
  return out.join(" ").replace(/[,;:\-\u2013\s]+$/, "");
}
// Cambia el nombre completo de una fila; el corto se recalcula si no lo editó el usuario
function setFull(row, value) {
  const full = row.querySelector("input.full"), short = row.querySelector("input.short");
  full.value = value;
  if (short.dataset.auto === "1") short.value = shortLabel(value);
}
const escH = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function renderLabels(labels) {
  const sug = state.valSug || {};
  const parts = Object.entries(labels).map(([model, rows]) => `
    <div data-model="${model}"><p><strong>${model === "lda" ? "LDA" : "BERTopic"}</strong></p>
      ${rows.map((r, i) => {
        const sg = (sug[model] || [])[i] || {};
        const value = state.valPending && sg.best ? sg.best : r.label;
        const alts = (sg.all || []).map((a) => `<button type="button" class="chip" data-v="${escH(a)}">${escH(a)}</button>`).join("");
        const short = value === r.label && r.short ? r.short : shortLabel(value);
        const auto = short === shortLabel(value) ? "1" : "0";
        return `<div class="lrow ${(state.valPending || state.valFlag) && sg.spanish ? "es" : ""}"><b>${r.id}</b><input type="text" class="full" value="${escH(value)}">
        <label class="shortl">Nombre corto (figuras) <input type="text" class="short" maxlength="40" data-auto="${auto}" value="${escH(short)}"></label>
        <span class="terms">${escH(r.terms)}</span>
        ${alts ? `<span class="sugg">Expertos (${sg.agree}/${sg.n} coinciden): ${alts}</span>` : ""}</div>`;
      }).join("")}</div>`);
  $("labels").innerHTML = parts.join("");
  $("labels-box").hidden = !parts.length;
  $("labels").querySelectorAll(".chip").forEach((b) => {
    b.onclick = () => {
      const row = b.closest(".lrow");
      setFull(row, b.dataset.v);
      row.classList.remove("es");
    };
  });
  $("labels").querySelectorAll(".lrow input.full").forEach((inp) => {
    inp.addEventListener("input", () => {
      const row = inp.closest(".lrow");
      row.classList.remove("es");
      setFull(row, inp.value);
    });
  });
  $("labels").querySelectorAll(".lrow input.short").forEach((inp) => {
    inp.addEventListener("input", () => { inp.dataset.auto = inp.value.trim() ? "0" : "1"; });
  });
}

$("b-labels").onclick = () => {
  const es = $("labels").querySelectorAll(".lrow.es").length;
  if (es && !confirm(`${es} nombre(s) parecen estar en español y las figuras deben ir en inglés. ¿Aplicarlos de todos modos?`)) return;
  state.valPending = false;
  state.valFlag = false;
  const labels = {};
  $("labels").querySelectorAll("[data-model]").forEach((d) => {
    labels[d.dataset.model] = [...d.querySelectorAll("input.full")].map((i) => i.value.trim() || "Unnamed topic");
    labels[d.dataset.model + "_short"] = [...d.querySelectorAll("input.short")].map((i) => i.value.trim());
  });
  busy(true);
  toast("Actualizando figuras con los nuevos nombres…");
  worker.postMessage({ cmd: "labels", labels, aiUsed: state.aiLabels });
};

// ------------------------------------------------------------------ proyecto y validación
$("f-project").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  busy(true);
  const data = new Uint8Array(await f.arrayBuffer());
  worker.postMessage({ cmd: "open_project", data }, [data.buffer]);
  e.target.value = "";
});
$("b-save").onclick = () => { busy(true); worker.postMessage({ cmd: "save_project" }); };
$("b-val-sheet").onclick = () => { busy(true); worker.postMessage({ cmd: "val_sheet" }); };
$("f-val").addEventListener("change", async (e) => {
  const fs = [...e.target.files];
  if (!fs.length) return;
  busy(true);
  const payload = await Promise.all(fs.map(async (f) => ({ name: f.name, data: new Uint8Array(await f.arrayBuffer()) })));
  worker.postMessage({ cmd: "val_load", files: payload }, payload.map((p) => p.data.buffer));
  e.target.value = "";
});

// ------------------------------------------------------------------ IA (opcional)
let ctxResolve = null;
const getContext = () => new Promise((res) => { ctxResolve = res; worker.postMessage({ cmd: "ai_context" }); });
function needKey() {
  const k = $("ai-key").value.trim();
  if (!k) {
    $("ai-box").open = true;
    $("ai-key").focus();
    toast("Escriba su clave de API de Anthropic en «Asistencia con IA» (sección 4).");
  }
  return k;
}
$("b-ai-translate").onclick = async () => {
  const key = needKey();
  if (!key) return;
  const inputs = [...$("labels").querySelectorAll(".lrow input.full")];
  busy(true);
  toast("Traduciendo los nombres al inglés…", 15000);
  try {
    const ai = await import("./ai.js?v=20261124");
    const out = await ai.translateLabels(key, inputs.map((i) => i.value), $("m-topic").value.trim());
    out.forEach((t, i) => { if (inputs[i] && t) { setFull(inputs[i].closest(".lrow"), t); inputs[i].closest(".lrow").classList.remove("es"); } });
    state.aiLabels = true;
    state.aiModel = ai.MODEL;
    toast("Nombres traducidos. Revíselos y pulse «Aplicar nombres».", 9000);
  } catch (err) {
    toast("IA: " + (err.message || err), 12000);
  } finally {
    busy(false);
  }
};
$("b-ai-short").onclick = async () => {
  const key = needKey();
  if (!key) return;
  const rows = [...$("labels").querySelectorAll(".lrow")];
  busy(true);
  toast("Pidiendo nombres cortos a Claude…", 15000);
  try {
    const ai = await import("./ai.js?v=20261124");
    const out = await ai.shortenLabels(key, rows.map((r) => r.querySelector("input.full").value), $("m-topic").value.trim());
    out.forEach((t, i) => {
      const sh = rows[i] && rows[i].querySelector("input.short");
      if (sh && t) { sh.value = t; sh.dataset.auto = "0"; }
    });
    state.aiLabels = true;
    state.aiModel = ai.MODEL;
    toast("Nombres cortos propuestos. Revíselos y pulse «Aplicar nombres».", 9000);
  } catch (err) {
    toast("IA: " + (err.message || err), 12000);
  } finally {
    busy(false);
  }
};
$("b-ai-labels").onclick = async () => {
  const key = needKey();
  if (!key) return;
  busy(true);
  toast("Pidiendo propuestas de nombres a Claude…", 15000);
  try {
    const ai = await import("./ai.js?v=20261124");
    const ctx = await getContext();
    const out = await ai.suggestLabels(key, ctx, $("m-topic").value.trim());
    const byId = Object.fromEntries(out.map((t) => [t.id, t]));
    $("labels").querySelectorAll(".lrow").forEach((row) => {
      const id = row.querySelector("b").textContent;
      if (byId[id]) {
        setFull(row, byId[id].label);
        row.querySelector("input.full").title = byId[id].rationale;
      }
    });
    state.aiLabels = true;
    toast("Nombres propuestos. Revíselos y pulse «Aplicar nombres».", 9000);
  } catch (err) {
    toast("IA: " + (err.message || err), 12000);
  } finally {
    busy(false);
  }
};
$("b-ai-disc").onclick = async () => {
  const key = needKey();
  if (!key) return;
  busy(true);
  const out = $("ai-out");
  out.hidden = false;
  out.textContent = "Redactando la Discusión (puede tardar 1–3 minutos)…";
  try {
    const ai = await import("./ai.js?v=20261124");
    const ctx = await getContext();
    const meta = { topic_name: $("m-topic").value.trim() };
    const text = await ai.draftDiscussion(key, ctx, meta, (snap) => { out.textContent = snap; });
    out.textContent = text;
    worker.postMessage({ cmd: "ai_set", texts: { discussion: text } });
    const cb = $("sections").querySelector('input[value="discussion"]');
    if (cb) cb.checked = true;
    state.aiModel = ai.MODEL;
    toast("Discusión redactada. Se incluirá en el documento Word.", 8000);
  } catch (err) {
    out.textContent = "";
    out.hidden = true;
    toast("IA: " + (err.message || err), 12000);
  } finally {
    busy(false);
  }
};

// ------------------------------------------------------------------ título
function titleMeta() {
  return { topic_name: $("m-topic").value.trim(), query: queryText() };
}
function showTitles(list, source) {
  const fromAI = !!source;
  const box = $("title-opts");
  box.innerHTML = "";
  list.forEach((t) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = typeof t === "string" ? t : t.title;
    if (t.rationale) {
      const sm = document.createElement("small");
      sm.textContent = t.rationale;
      b.appendChild(sm);
    }
    const title = typeof t === "string" ? t : t.title;
    b.onclick = () => {
      $("m-title").value = title;
      state.titleAI = fromAI;
      box.querySelectorAll("button").forEach((x) => x.classList.toggle("sel", x === b));
    };
    box.appendChild(b);
  });
  if (list.length && (!$("m-title").value.trim() || state.autoTitle)) {
    $("m-title").value = typeof list[0] === "string" ? list[0] : list[0].title;
    state.autoTitle = true;
    state.titleAI = fromAI;
    box.firstChild.classList.add("sel");
  }
  if (source) toast(source, 6000);
}
$("m-title").addEventListener("input", () => { state.autoTitle = false; });
$("b-titles").onclick = () => worker.postMessage({ cmd: "titles", meta: titleMeta() });
$("b-ai-titles").onclick = async () => {
  const key = needKey();
  if (!key) return;
  busy(true);
  toast("Pidiendo propuestas de título a Claude…", 15000);
  try {
    const ai = await import("./ai.js?v=20261124");
    const ctx = await getContext();
    const out = await ai.suggestTitles(key, ctx, titleMeta());
    state.aiModel = ai.MODEL;
    showTitles(out, "Títulos propuestos por la IA. Elija uno o edítelo.");
  } catch (err) {
    toast("IA: " + (err.message || err), 12000);
  } finally {
    busy(false);
  }
};

// ------------------------------------------------------------------ redacción
$("q-date").value = new Date().toISOString().slice(0, 10);
function queryText() {
  const q = searchCfg();
  return [q.search_scopus && `Scopus: ${q.search_scopus}`, q.search_wos && `Web of Science: ${q.search_wos}`]
    .filter(Boolean).join(" | ");
}
$("b-doc").onclick = () => {
  const sections = checked("sections");
  if (!sections.length) return toast("Seleccione al menos un apartado.");
  busy(true);
  toast("Redactando el documento…");
  worker.postMessage({ cmd: "report", sections, meta: docMeta(), opts: figOpts() });
};
function docMeta() {
  return { topic_name: $("m-topic").value.trim(), title: $("m-title").value.trim(),
    query: queryText(), search_date: $("q-date").value, ...searchCfg(),
    composite: $("m-comp").checked, ai_model: state.aiModel ? "Claude Opus 5.5" : "",
    ai_title: !!state.titleAI, sections: checked("sections") };
}
const figOpts = () => ({ composite: $("m-comp").checked, export_tiff: $("m-tiff").checked });
$("b-zip").onclick = () => { busy(true); worker.postMessage({ cmd: "zip", opts: figOpts(), meta: docMeta() }); };
$("b-script").onclick = () => { busy(true); worker.postMessage({ cmd: "script", meta: docMeta() }); };

function download(name, data, mime) {
  const url = URL.createObjectURL(new Blob([data], { type: mime }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ------------------------------------------------------------------ mensajes
worker.onmessage = ({ data: m }) => {
  switch (m.type) {
    case "progress":
      if (!$("env").classList.contains("ready")) {
        $("env-msg").textContent = m.msg;
        setBar("env-bar", m.p);
      } else {
        setBar("bar", m.p);
        $("pmsg").textContent = m.msg;
        runTotal(m.total);
      }
      break;
    case "ready":
      $("env").classList.add("ready");
      $("env-msg").textContent = "Entorno listo";
      state.analyses = m.analyses;
      state.sections = m.sections;
      buildChecks("analyses", m.analyses, Object.keys(m.analyses));
      buildChecks("sections", m.sections, Object.keys(m.sections).filter((k) => !DEFAULT_OFF_SECTIONS.includes(k)));
      $("f-scopus").disabled = false;
      $("f-wos").disabled = false;
      $("f-project").disabled = false;
      showRecover();
      break;
    case "loaded":
      busy(false);
      state.summary = m.summary;
      showSummary(m.summary);
      break;
    case "stage":
      $("stage").textContent = m.stage;
      $("pmsg").textContent = "";
      RUN.last = Date.now();
      setBar("bar", 0);
      break;
    case "estimate":
      showEstimate(m.estimate);
      break;
    case "checkpoint":
      idbPut("checkpoint", { when: Date.now(), data: m.data.buffer.slice(m.data.byteOffset, m.data.byteOffset + m.data.byteLength) });
      break;
    case "palette":
      showPalette(m.palette);
      break;
    case "prepared":
      $("pmsg").textContent = `${(m.prep.bibliometric_documents ?? m.prep.documents).toLocaleString()} documentos en el corpus bibliométrico · `
        + `${m.prep.documents.toLocaleString()} con resumen para el modelado de temas · vocabulario de ${m.prep.vocabulary.toLocaleString()} términos`;
      break;
    case "results":
      busy(false);
      showResults(m);
      worker.postMessage({ cmd: "titles", meta: titleMeta() });
      break;
    case "done":
      $("stage").textContent = m.elapsed ? `Análisis terminado en ${fmtDur(m.elapsed)}` : "Análisis terminado";
      setBar("bar", 1);
      runClockStop();
      if (m.elapsed) { setBar("bar-total", 1); $("tot-pct").textContent = "100 %"; $("tot-eta").textContent = ""; }
      $("watchdog").hidden = true;
      showTimings(m.timings);
      $("pmsg").textContent = "";
      $("s-results").scrollIntoView({ behavior: "smooth" });
      break;
    case "file":
      busy(false);
      download(m.name, m.data, m.mime);
      break;
    case "warn":
      toast(m.msg, 9000);
      break;
    case "project":
      $("s-analyses").hidden = false;
      $("run-status").hidden = false;
      toast("Proyecto abierto.");
      break;
    case "validation": {
      const r = m.result;
      const al = Object.entries(r.alpha).map(([k, v]) => `${k}: ${v ?? "n.d."}`).join(" · ");
      $("val-result").innerHTML = `<p class="hint">${r.raters.length} evaluadores (${r.raters.join(", ")}). Alfa de Krippendorff — ${al}</p>`;
      state.valSug = r.suggestions;
      state.labels = null; // volver a dibujar el editor de nombres con las propuestas de los expertos
      const nSug = Object.values(r.suggestions || {}).flat().filter((x) => x.best).length;
      state.valPending = false;
      state.valFlag = r.spanish.length > 0;
      if (!nSug) {
        toast("Validación cargada; se incluirá en el documento. Los evaluadores no propusieron nombres.", 9000);
      } else {
        toast(`Se aplicaron a las figuras los nombres propuestos por los expertos (${nSug} temas). Puede editarlos en «Nombre de los temas».`, 10000);
        if (r.spanish.length) {
          $("val-result").innerHTML += `<p class="warn">Se aplicaron los nombres de los expertos, pero algunos parecen
            estar en español (${r.spanish.join(", ")}) y las figuras deben ir en inglés. Tradúzcalos en «Nombre de los
            temas» (marcados en naranja) o use «Traducir al inglés con IA», y pulse «Aplicar nombres».</p>`;
          $("b-ai-translate").hidden = false;
        }
      }
      break;
    }
    case "ai_context":
      if (ctxResolve) ctxResolve(m.ctx);
      ctxResolve = null;
      break;
    case "ai_saved":
      break;
    case "titles":
      showTitles(m.titles);
      break;
    case "error":
      busy(false);
      toast("Error: " + m.msg, 12000);
      console.error(m.msg);
      if (RUN.timer) {  // durante un análisis, el error queda visible en el panel de progreso
        runClockStop();
        $("watchdog").hidden = false;
        $("watchdog").innerHTML = `<b>El análisis se detuvo por un error:</b> ${escH(m.msg).slice(0, 600)}<br>
          Si ya se habían ajustado los modelos, recargue la página y use «Recuperar el último análisis».`;
      }
      break;
  }
};
worker.onerror = (e) => {
  if (RUN.timer) {
    runClockStop();
    $("watchdog").hidden = false;
    $("watchdog").innerHTML = `<b>El proceso del análisis se detuvo</b> (${escH(e.message || "error del navegador")}).
      Recargue la página; si los modelos ya estaban ajustados, use «Recuperar el último análisis».`;
  }
  $("env-msg").textContent = "No se pudo iniciar el entorno: " + (e.message || "error desconocido");
  console.error(e);
};
worker.postMessage({ cmd: "init" });
