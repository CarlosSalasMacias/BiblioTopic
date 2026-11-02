// BiblioTopic — interfaz (hilo principal). El trabajo pesado ocurre en js/worker.js.
const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
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
const csv = (s) => s.split(",").map((x) => x.trim()).filter(Boolean);

function busy(on) {
  state.busy = on;
  for (const id of ["b-load", "b-run", "b-labels", "b-doc", "b-zip", "b-save", "b-val-sheet", "b-ai-labels",
    "b-ai-disc"]) $(id).disabled = on;
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
  box.innerHTML = `
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
    ${warns.map((w) => `<div class="warn">${w}</div>`).join("")}`;
  // opciones dependientes de los datos
  $("o-ymin").value = s.year_min;
  // por defecto, hasta el último año completo
  $("o-ymax").value = Math.min(s.year_max, new Date().getFullYear() - 1);
  chips("o-dt", s.doc_types, ["Article", "Review"]);
  chips("o-la", s.languages, ["English"]);
  $("s-analyses").hidden = false;
}

function chips(id, counts, defaults) {
  const box = $(id);
  const keys = Object.keys(counts);
  if (!keys.length) {
    box.innerHTML = '<span class="hint">(no disponible en los archivos)</span>';
    return;
  }
  const useDefault = keys.some((k) => defaults.includes(k));
  box.innerHTML = keys.map((k) => `<label><input type="checkbox" value="${k}"
      ${!useDefault || defaults.includes(k) ? "checked" : ""}> ${k} (${counts[k]})</label>`).join("");
}
const checked = (id) => [...$(id).querySelectorAll("input:checked")].map((i) => i.value);

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

$("b-run").onclick = () => {
  const analyses = checked("analyses");
  if (!analyses.length) return toast("Seleccione al menos un análisis.");
  const num = (id) => Number($(id).value);
  const cfg = {
    analyses,
    year_min: num("o-ymin") || null, year_max: num("o-ymax") || null,
    doc_types: checked("o-dt").length ? checked("o-dt") : null,
    languages: checked("o-la").length ? checked("o-la") : null,
    query_terms: csv($("o-q").value), extra_stopwords: csv($("o-sw").value), synonyms: $("o-syn").value,
    k_min: num("o-kmin"), k_max: num("o-kmax"), k_step: num("o-kstep"), k: num("o-k") || null,
    bert_nr_topics: $("o-bnr").value === "lda" ? "lda" : null,
    min_cluster_size: num("o-mcs") || null, n_stab_lda: num("o-slda"), n_stab_bert: num("o-sbert"),
    period_width: num("o-pw"), min_df: num("o-mindf"), tau_strong: num("o-tau"), seed: num("o-seed"),
    n_boot: num("o-boot"), cluster_shape: $("o-shape").value, composite: $("m-comp").checked,
    export_tiff: $("m-tiff").checked,
  };
  if (cfg.k_min >= cfg.k_max && !cfg.k) return toast("K mínimo debe ser menor que K máximo.");
  busy(true);
  $("run-status").hidden = false;
  $("stage").textContent = "Iniciando…";
  setBar("bar", 0);
  worker.postMessage({ cmd: "run", cfg });
};

// ------------------------------------------------------------------ resultados
let urls = [];
function showResults({ figs, tables, labels }) {
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

function renderLabels(labels) {
  const parts = Object.entries(labels).map(([model, rows]) => `
    <div data-model="${model}"><p><strong>${model === "lda" ? "LDA" : "BERTopic"}</strong></p>
      ${rows.map((r) => `<div class="lrow"><b>${r.id}</b><input type="text" value="${r.label.replace(/"/g, "&quot;")}">
        <span class="terms">${r.terms}</span></div>`).join("")}</div>`);
  $("labels").innerHTML = parts.join("");
  $("labels-box").hidden = !parts.length;
}

$("b-labels").onclick = () => {
  const labels = {};
  $("labels").querySelectorAll("[data-model]").forEach((d) => {
    labels[d.dataset.model] = [...d.querySelectorAll("input")].map((i) => i.value.trim() || "Unnamed topic");
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
$("b-ai-labels").onclick = async () => {
  const key = needKey();
  if (!key) return;
  busy(true);
  toast("Pidiendo propuestas de nombres a Claude…", 15000);
  try {
    const ai = await import("./ai.js");
    const ctx = await getContext();
    const out = await ai.suggestLabels(key, ctx, $("m-topic").value.trim());
    const byId = Object.fromEntries(out.map((t) => [t.id, t]));
    $("labels").querySelectorAll(".lrow").forEach((row) => {
      const id = row.querySelector("b").textContent;
      if (byId[id]) {
        row.querySelector("input").value = byId[id].label;
        row.querySelector("input").title = byId[id].rationale;
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
    const ai = await import("./ai.js");
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
  return { topic_name: $("m-topic").value.trim(), query: $("m-query").value.trim() };
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
    const ai = await import("./ai.js");
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
$("m-date").value = new Date().toLocaleDateString("es-ES");
$("b-doc").onclick = () => {
  const sections = checked("sections");
  if (!sections.length) return toast("Seleccione al menos un apartado.");
  busy(true);
  toast("Redactando el documento…");
  worker.postMessage({
    cmd: "report", sections,
    meta: { topic_name: $("m-topic").value.trim(), title: $("m-title").value.trim(),
            query: $("m-query").value.trim(), search_date: $("m-date").value.trim(),
            composite: $("m-comp").checked, ai_model: state.aiModel ? "Claude Opus 5.5" : "",
            ai_title: !!state.titleAI },
    opts: figOpts(),
  });
};
const figOpts = () => ({ composite: $("m-comp").checked, export_tiff: $("m-tiff").checked });
$("b-zip").onclick = () => { busy(true); worker.postMessage({ cmd: "zip", opts: figOpts() }); };

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
      break;
    case "loaded":
      busy(false);
      state.summary = m.summary;
      showSummary(m.summary);
      break;
    case "stage":
      $("stage").textContent = `${m.stage} · ${fmtTime(m.elapsed)}`;
      setBar("bar", 0);
      break;
    case "prepared":
      $("pmsg").textContent = `${m.prep.documents.toLocaleString()} documentos · vocabulario de ${m.prep.vocabulary.toLocaleString()} términos`;
      break;
    case "results":
      busy(false);
      showResults(m);
      worker.postMessage({ cmd: "titles", meta: titleMeta() });
      break;
    case "done":
      $("stage").textContent = `Análisis terminado en ${fmtTime(m.elapsed)}`;
      setBar("bar", 1);
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
      toast("Validación cargada; se incluirá en el documento.");
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
      break;
  }
};
worker.onerror = (e) => {
  $("env-msg").textContent = "No se pudo iniciar el entorno: " + (e.message || "error desconocido");
  console.error(e);
};
worker.postMessage({ cmd: "init" });
