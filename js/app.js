// BiblioTopic — interfaz (hilo principal). El trabajo pesado ocurre en js/worker.js.
const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL("./worker.js?v=20261113", import.meta.url), { type: "module" });
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
  const unk = (f) => Object.values((s.unknown || {})[f] || {}).reduce((a, b) => a + b, 0);
  const unkBy = (f) => Object.entries((s.unknown || {})[f] || {}).map(([k, v]) => `${k}: ${v.toLocaleString()}`).join(" · ");
  chips("o-dt", s.doc_types, ["Article", "Review"], unk("DT"), unkBy("DT"));
  chips("o-la", s.languages, ["English"], unk("LA"), unkBy("LA"));
  $("s-analyses").hidden = false;
}

// Valores del campo + «Sin dato» (registros sin el campo: p. ej., Scopus sin columna de idioma). Los
// registros sin dato no son «otro idioma»: se conservan por defecto y la decisión queda explícita.
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

$("b-run").onclick = () => {
  const analyses = checked("analyses");
  if (!analyses.length) return toast("Seleccione al menos un análisis.");
  const num = (id) => Number($(id).value);
  const cfg = {
    analyses,
    year_min: num("o-ymin") || null, year_max: num("o-ymax") || null,
    doc_types: checked("o-dt").length ? checked("o-dt") : null,
    languages: checked("o-la").length ? checked("o-la") : null,
    keep_unknown_language: keepUnknown("o-la"), keep_unknown_type: keepUnknown("o-dt"),
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
    const ai = await import("./ai.js?v=20261113");
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
    const ai = await import("./ai.js?v=20261113");
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
    const ai = await import("./ai.js?v=20261113");
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
    const ai = await import("./ai.js?v=20261113");
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
    const ai = await import("./ai.js?v=20261113");
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
  worker.postMessage({ cmd: "report", sections, meta: docMeta(), opts: figOpts() });
};
function docMeta() {
  return { topic_name: $("m-topic").value.trim(), title: $("m-title").value.trim(),
    query: $("m-query").value.trim(), search_date: $("m-date").value.trim(),
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
      break;
  }
};
worker.onerror = (e) => {
  $("env-msg").textContent = "No se pudo iniciar el entorno: " + (e.message || "error desconocido");
  console.error(e);
};
worker.postMessage({ cmd: "init" });
