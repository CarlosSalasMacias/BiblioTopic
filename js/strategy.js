// BiblioTopic — etapa 1: tema de investigación y estrategia de búsqueda para Scopus y Web of Science.
// La propuesta se genera con IA (clave del usuario) o, sin clave, con reglas simples que el usuario completa.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export const SCOPUS_AREAS = {
  AGRI: "Agricultural and Biological Sciences", ARTS: "Arts and Humanities",
  BIOC: "Biochemistry, Genetics and Molecular Biology", BUSI: "Business, Management and Accounting",
  CENG: "Chemical Engineering", CHEM: "Chemistry", COMP: "Computer Science", DECI: "Decision Sciences",
  DENT: "Dentistry", EART: "Earth and Planetary Sciences", ECON: "Economics, Econometrics and Finance",
  ENER: "Energy", ENGI: "Engineering", ENVI: "Environmental Science", HEAL: "Health Professions",
  IMMU: "Immunology and Microbiology", MATE: "Materials Science", MATH: "Mathematics", MEDI: "Medicine",
  NEUR: "Neuroscience", NURS: "Nursing", PHAR: "Pharmacology, Toxicology and Pharmaceutics",
  PHYS: "Physics and Astronomy", PSYC: "Psychology", SOCI: "Social Sciences", VETE: "Veterinary",
  MULT: "Multidisciplinary",
};

// st: estado de la estrategia (se guarda también en el proyecto)
let st = null;
let ctx = null; // { toast, getKey, onApproved }

// ------------------------------------------------------------------ propuesta sin IA
const STOP = new Set(["the", "a", "an", "of", "and", "in", "on", "for", "to", "with", "by", "its", "their",
  "el", "la", "los", "las", "de", "del", "y", "e", "en", "para", "por", "con", "sobre", "un", "una"]);
const SPLIT = /\s*(?:,|;|\band\b|\by\b|\bin\b|\ben\b|\bfor\b|\bpara\b|\bon\b|\bsobre\b|\bwith\b|\bcon\b|\bamong\b|\bentre\b)\s*/i;

function truncate(word) {
  const w = word.toLowerCase();
  if (w.length < 5 || /\*$/.test(w)) return null;
  const stem = w.replace(/(ations?|izations?|isations?|ies|es|s|ing|ed|al|ic)$/, "");
  return stem.length >= 4 && stem !== w ? stem + "*" : (w.length >= 6 ? w + "*" : null);
}

function spelling(term) {
  const out = new Set();
  const swaps = [[/ization/g, "isation"], [/isation/g, "ization"], [/yze/g, "yse"], [/yse\b/g, "yze"],
    [/ize/g, "ise"], [/ise\b/g, "ize"], [/our\b/g, "or"], [/(?<=[a-z]{3})or\b/g, "our"]];
  for (const [re, rep] of swaps) {
    const v = term.replace(re, rep);
    if (v !== term) out.add(v);
  }
  return [...out].slice(0, 2);
}

export function heuristicStrategy(topic) {
  const parts = topic.split(SPLIT).map((p) => p.trim().replace(/^(the|el|la|los|las)\s+/i, ""))
    .filter((p) => p && !STOP.has(p.toLowerCase()));
  const concepts = parts.slice(0, 4).map((p) => {
    const words = p.toLowerCase().split(/\s+/).filter((w) => !STOP.has(w));
    const phrase = words.join(" ");
    const terms = new Set([phrase]);
    const last = truncate(words[words.length - 1]);
    if (last) terms.add([...words.slice(0, -1), last].join(" "));
    if (words.length === 2) terms.add(words.join("-"));
    spelling(phrase).forEach((v) => terms.add(v));
    return { name: p, rationale: "", terms: [...terms].map((t) => ({ term: t, on: true })) };
  });
  return {
    topic, topic_en: topic, ai: false, concepts,
    exclusions: [],
    scopus: [], wos: [],
    notes: "Propuesta básica generada sin IA a partir de las palabras del tema: revise y añada sinónimos, " +
      "variantes y términos de exclusión. Los términos deben estar en inglés. Con una clave de API de Anthropic, " +
      "la aplicación propone conceptos, sinónimos, exclusiones razonadas y categorías temáticas.",
  };
}

function fromAI(topic, r) {
  return {
    topic, topic_en: r.topic_en || topic, ai: true,
    concepts: r.concepts.map((c) => ({ name: c.name, rationale: c.rationale,
      terms: c.terms.map((t) => ({ term: t.replace(/^"|"$/g, ""), on: true })) })),
    exclusions: r.exclusions.filter((g) => g.terms.length).map((g) => ({
      group: g.group, rationale: g.rationale,
      // los términos con riesgo de excluir estudios pertinentes empiezan desmarcados
      terms: g.terms.map((t) => ({ term: t.term, reason: t.reason, caution: t.caution,
        note: t.caution_note, on: !t.caution })),
    })),
    scopus: r.scopus_subject_areas.filter((a) => SCOPUS_AREAS[a.code]).map((a) => ({ ...a, on: false })),
    wos: r.wos_categories.map((c) => ({ ...c, on: false })),
    notes: r.notes,
  };
}

// ------------------------------------------------------------------ ecuaciones
const fmtTerm = (t) => {
  t = t.trim().replace(/^"+|"+$/g, "");
  return /[\s\-\/]/.test(t) ? `"${t}"` : t;
};
const orBlock = (terms) => terms.map(fmtTerm).join(" OR ");
const activeTerms = (c) => c.terms.filter((t) => t.on && t.term.trim()).map((t) => t.term);
export const acceptedExclusions = (s = st) =>
  s.exclusions.flatMap((g) => g.terms.filter((t) => t.on && t.term.trim()).map((t) => ({ ...t, group: g.group })));

export function buildQueries(s = st) {
  const blocks = s.concepts.map(activeTerms).filter((ts) => ts.length);
  const core = blocks.map((ts) => `( ${orBlock(ts)} )`).join(" AND ");
  const ex = acceptedExclusions(s).map((t) => t.term);
  const title = s.exclField === "title";
  let scopus = `TITLE-ABS-KEY ( ${core} )`;
  let wos = `TS=( ${core} )`;
  if (ex.length) wos = `( ${wos}\nNOT ${title ? "TI" : "TS"}=( ${orBlock(ex)} ) )`;
  const inc = s.scopus.filter((a) => a.on && a.mode === "include");
  const exc = s.scopus.filter((a) => a.on && a.mode === "exclude");
  if (inc.length) scopus += `\nAND ( ${inc.map((a) => `LIMIT-TO ( SUBJAREA , "${a.code}" )`).join(" OR ")} )`;
  if (exc.length) scopus += `\nAND ( ${exc.map((a) => `EXCLUDE ( SUBJAREA , "${a.code}" )`).join(" OR ")} )`;
  // En Scopus AND NOT se evalúa al final: debe ir después de los filtros
  if (ex.length) scopus += `\nAND NOT ${title ? "TITLE" : "TITLE-ABS-KEY"} ( ${orBlock(ex)} )`;
  const winc = s.wos.filter((c) => c.on && c.mode === "include" && c.name.trim());
  const wexc = s.wos.filter((c) => c.on && c.mode === "exclude" && c.name.trim());
  if (winc.length) wos += `\nAND WC=( ${winc.map((c) => `"${c.name.trim()}"`).join(" OR ")} )`;
  if (wexc.length) wos += `\nNOT WC=( ${wexc.map((c) => `"${c.name.trim()}"`).join(" OR ")} )`;
  return { scopus, wos };
}

export function checkQuery(q, db) {
  const issues = [];
  let flat = q;
  for (let i = 0; i < 20 && /\([^()]*\)/.test(flat); i++) flat = flat.replace(/\([^()]*\)/g, " ");
  if (db === "scopus" && /AND\s+NOT[\s\S]*\b(AND|OR)\b(?!\s+NOT)/.test(flat))
    issues.push("en Scopus «AND NOT» se procesa al final; colóquelo como último elemento de la ecuación");
  if (db === "scopus" && /\b(TS|TI|AB|WC)=/.test(q)) issues.push("contiene etiquetas de Web of Science");
  if (db === "wos" && /TITLE-ABS-KEY|LIMIT-TO|SUBJAREA/.test(q)) issues.push("contiene sintaxis de Scopus");
  let depth = 0;
  for (const ch of q) {
    if (ch === "(") depth++;
    if (ch === ")") depth--;
    if (depth < 0) break;
  }
  if (depth !== 0) issues.push("paréntesis desbalanceados");
  if ((q.match(/"/g) || []).length % 2) issues.push("comillas sin cerrar");
  if (/(^|[\s(])\*/.test(q)) issues.push("truncamiento al inicio de una palabra (no admitido)");
  const short = [...q.matchAll(/\b([A-Za-z]{1,3})\*/g)].map((m) => m[1] + "*");
  if (short.length) issues.push(`truncamientos muy cortos que recuperan ruido: ${[...new Set(short)].join(", ")}`);
  if (/\b(AND|OR|NOT)\s*\)/.test(q) || /\(\s*(AND|OR)\b/.test(q) || /\b(AND|OR|NOT)\s*$/.test(q.trim()))
    issues.push("operador sin término");
  return issues;
}

// ------------------------------------------------------------------ interfaz
function render() {
  $("st-topic-en").value = st.topic_en;
  $("st-notes").textContent = st.notes || "";
  $("st-notes").hidden = !st.notes;
  renderConcepts();
  renderExclusions();
  renderFilters();
  renderReview();
}

function renderConcepts() {
  const box = $("st-concepts");
  box.innerHTML = st.concepts.map((c, i) => `
    <div class="concept" data-i="${i}">
      <div class="concept-head">
        <span class="blk">${i ? "AND" : ""}</span>
        <input type="text" class="c-name" value="${esc(c.name)}" aria-label="Nombre del concepto">
        <button type="button" class="mini c-del" title="Eliminar concepto">×</button>
      </div>
      ${c.rationale ? `<p class="hint">${esc(c.rationale)}</p>` : ""}
      <div class="terms">${c.terms.map((t, j) => `
        <span class="term ${t.on ? "" : "off"}" data-j="${j}">
          <input type="checkbox" ${t.on ? "checked" : ""} title="Incluir">
          <input type="text" value="${esc(t.term)}" size="${Math.max(6, t.term.length + 1)}">
          <button type="button" class="mini t-del" title="Eliminar">×</button>
        </span>`).join(" <i>OR</i> ")}
        <button type="button" class="mini t-add">+ término</button>
      </div>
    </div>`).join("");
  box.querySelectorAll(".concept").forEach((el) => {
    const c = st.concepts[+el.dataset.i];
    el.querySelector(".c-name").oninput = (e) => { c.name = e.target.value; };
    el.querySelector(".c-del").onclick = () => { st.concepts.splice(+el.dataset.i, 1); changed(); renderConcepts(); };
    el.querySelector(".t-add").onclick = () => {
      c.terms.push({ term: "", on: true });
      renderConcepts();
      const ins = $("st-concepts").querySelectorAll(`.concept[data-i="${el.dataset.i}"] .term input[type=text]`);
      ins[ins.length - 1]?.focus();
    };
    el.querySelectorAll(".term").forEach((tEl) => {
      const t = c.terms[+tEl.dataset.j];
      tEl.querySelector("input[type=checkbox]").onchange = (e) => { t.on = e.target.checked; tEl.classList.toggle("off", !t.on); changed(); };
      tEl.querySelector("input[type=text]").oninput = (e) => { t.term = e.target.value; e.target.size = Math.max(6, t.term.length + 1); changed(); };
      tEl.querySelector(".t-del").onclick = () => { c.terms.splice(+tEl.dataset.j, 1); changed(); renderConcepts(); };
    });
  });
}

function renderExclusions() {
  const box = $("st-excl");
  if (!st.exclusions.length) {
    box.innerHTML = `<p class="hint">No hay términos de exclusión propuestos${st.ai ? " (la IA no encontró exclusiones claramente seguras)" : ""}. Puede añadir los suyos.</p>`;
    return;
  }
  box.innerHTML = st.exclusions.map((g, i) => {
    const n = g.terms.filter((t) => t.on).length;
    return `
    <fieldset class="xgroup" data-i="${i}">
      <legend><label><input type="checkbox" class="g-all" ${n === g.terms.length ? "checked" : ""}>
        <b>${esc(g.group)}</b></label> <span class="hint">(${n}/${g.terms.length})</span></legend>
      ${g.rationale ? `<p class="hint">${esc(g.rationale)}</p>` : ""}
      ${g.terms.map((t, j) => `
      <div class="xterm ${t.on ? "" : "off"}" data-j="${j}">
        <input type="checkbox" ${t.on ? "checked" : ""} title="Aceptar">
        <input type="text" value="${esc(t.term)}">
        <span class="xreason">${esc(t.reason || "")}${t.caution ? `<span class="caution" title="${esc(t.note)}">⚠ precaución: ${esc(t.note)}</span>` : ""}</span>
        <button type="button" class="mini x-del" title="Eliminar">×</button>
      </div>`).join("")}
    </fieldset>`;
  }).join("");
  box.querySelectorAll(".xgroup").forEach((gEl) => {
    const g = st.exclusions[+gEl.dataset.i];
    gEl.querySelector(".g-all").onchange = (e) => {
      if (e.target.checked && g.terms.some((t) => t.caution && !t.on) &&
        !confirm("Este grupo contiene términos marcados con precaución porque podrían eliminar estudios pertinentes. ¿Marcar también esos términos?")) {
        g.terms.forEach((t) => { if (!t.caution) t.on = true; });
      } else {
        g.terms.forEach((t) => { t.on = e.target.checked; });
      }
      changed(); renderExclusions();
    };
    gEl.querySelectorAll(".xterm").forEach((tEl) => {
      const t = g.terms[+tEl.dataset.j];
      tEl.querySelector("input[type=checkbox]").onchange = (e) => { t.on = e.target.checked; changed(); renderExclusions(); };
      tEl.querySelector("input[type=text]").oninput = (e) => { t.term = e.target.value; changed(); };
      tEl.querySelector(".x-del").onclick = () => {
        g.terms.splice(+tEl.dataset.j, 1);
        if (!g.terms.length) st.exclusions.splice(+gEl.dataset.i, 1);
        changed(); renderExclusions();
      };
    });
  });
}

function renderFilters() {
  const row = (f, i, kind, label) => `
    <div class="frow ${f.on ? "" : "off"}" data-k="${kind}" data-i="${i}">
      <input type="checkbox" ${f.on ? "checked" : ""}>
      <select><option value="include" ${f.mode === "include" ? "selected" : ""}>Limitar a</option>
        <option value="exclude" ${f.mode === "exclude" ? "selected" : ""}>Excluir</option></select>
      ${label}
      <button type="button" class="mini f-del" title="Eliminar">×</button>
      ${f.reason ? `<span class="xreason">${esc(f.reason)}</span>` : ""}
    </div>`;
  $("st-scopus-f").innerHTML = st.scopus.map((a, i) => row(a, i, "scopus",
    `<b>${esc(a.code)}</b> <span>${esc(SCOPUS_AREAS[a.code] || "")}</span>`)).join("") ||
    `<p class="hint">Sin filtros. Añada un área si quiere limitar o excluir resultados.</p>`;
  $("st-wos-f").innerHTML = st.wos.map((c, i) => row(c, i, "wos",
    `<input type="text" value="${esc(c.name)}" class="wc-name">`)).join("") ||
    `<p class="hint">Sin filtros. Añada una categoría si quiere limitar o excluir resultados.</p>`;
  document.querySelectorAll("#st-filters .frow").forEach((el) => {
    const list = el.dataset.k === "scopus" ? st.scopus : st.wos;
    const f = list[+el.dataset.i];
    el.querySelector("input[type=checkbox]").onchange = (e) => { f.on = e.target.checked; el.classList.toggle("off", !f.on); changed(); };
    el.querySelector("select").onchange = (e) => { f.mode = e.target.value; changed(); };
    el.querySelector(".wc-name")?.addEventListener("input", (e) => { f.name = e.target.value; changed(); });
    el.querySelector(".f-del").onclick = () => { list.splice(+el.dataset.i, 1); changed(); renderFilters(); };
  });
}

function renderReview() {
  const ex = acceptedExclusions();
  const nTerms = st.concepts.reduce((a, c) => a + activeTerms(c).length, 0);
  $("st-review").innerHTML = `
    <p><b>${st.concepts.filter((c) => activeTerms(c).length).length}</b> conceptos con <b>${nTerms}</b> términos de búsqueda.
      <b>${ex.length}</b> términos de exclusión aceptados${ex.length ? `, aplicados ${st.exclField === "title" ? "solo al título" : "a título, resumen y palabras clave"}` : ""}.</p>
    ${ex.length ? `<ul class="xlist">${ex.map((t) => `<li><code>${esc(t.term)}</code> <span class="hint">· ${esc(t.group)}${t.caution ? " · ⚠ con precaución" : ""}</span></li>`).join("")}</ul>` : ""}`;
}

// Cualquier cambio en los bloques invalida las ecuaciones ya generadas
function changed() {
  renderReview();
  if (!$("st-queries").hidden) {
    $("st-queries").hidden = true;
    $("st-approve-box").hidden = true;
    $("st-gen").textContent = "Generar ecuaciones";
  }
}

function showQueries() {
  const bad = st.concepts.filter((c) => activeTerms(c).length).length;
  if (!bad) return ctx.toast("Añada al menos un concepto con un término de búsqueda.");
  const q = buildQueries();
  $("st-q-scopus").value = q.scopus;
  $("st-q-wos").value = q.wos;
  $("st-queries").hidden = false;
  $("st-approve-box").hidden = false;
  ["st-ok-scopus", "st-ok-wos"].forEach((id) => { $(id).checked = false; });
  validateQueries();
  $("st-gen").textContent = "Regenerar ecuaciones";
  $("st-queries").scrollIntoView({ behavior: "smooth", block: "start" });
}

function validateQueries() {
  for (const k of ["scopus", "wos"]) {
    const issues = checkQuery($(`st-q-${k}`).value, k);
    $(`st-chk-${k}`).innerHTML = issues.length
      ? `<span class="bad">Revise: ${issues.map(esc).join("; ")}.</span>`
      : `<span class="good">Sintaxis correcta.</span>`;
  }
  $("st-continue").disabled = !($("st-ok-scopus").checked && $("st-ok-wos").checked);
}

function exportText() {
  const ex = acceptedExclusions();
  const lines = [
    `Search strategy — ${st.topic_en}`, `Date: ${new Date().toLocaleDateString("en-CA")}`, "",
    "SCOPUS (Advanced search)", $("st-q-scopus").value, "",
    "WEB OF SCIENCE CORE COLLECTION (Advanced search)", $("st-q-wos").value, "",
    "Concepts (OR within, AND between):",
    ...st.concepts.map((c, i) => `  ${i + 1}. ${c.name}: ${activeTerms(c).join("; ")}`), "",
    `Exclusion terms (${st.exclField === "title" ? "title only" : "title, abstract and keywords"}):`,
    ...(ex.length ? ex.map((t) => `  - ${t.term} [${t.group}] ${t.reason || ""}`) : ["  none"]), "",
    "Subject filters:",
    ...st.scopus.filter((a) => a.on).map((a) => `  Scopus ${a.mode} ${a.code} (${SCOPUS_AREAS[a.code]})`),
    ...st.wos.filter((c) => c.on).map((c) => `  WoS ${c.mode} WC="${c.name}"`),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/plain" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "BiblioTopic_search_strategy.txt" });
  document.body.appendChild(a); a.click(); a.remove();
}

async function propose() {
  const topic = $("st-topic").value.trim();
  if (topic.length < 4) return ctx.toast("Describa el tema de investigación.");
  const key = ctx.getKey();
  $("st-next").disabled = true;
  try {
    if (key) {
      $("st-next").textContent = "Analizando el tema con IA…";
      const ai = await import("./ai.js");
      st = fromAI(topic, await ai.suggestStrategy(key, topic));
      ctx.onAIUsed?.(ai.MODEL);
    } else {
      st = heuristicStrategy(topic);
    }
  } catch (err) {
    ctx.toast("IA: " + (err.message || err) + " Se usa la propuesta básica.", 12000);
    st = heuristicStrategy(topic);
  } finally {
    $("st-next").disabled = false;
    $("st-next").textContent = "Siguiente";
  }
  st.exclField = $("st-exfield").value;
  $("st-step2").hidden = false;
  $("st-queries").hidden = true;
  $("st-approve-box").hidden = true;
  render();
  $("st-step2").scrollIntoView({ behavior: "smooth", block: "start" });
}

export function getStrategy() { return st; }

export function restore(saved) {
  if (!saved) return;
  st = saved;
  $("st-topic").value = st.topic || "";
  $("st-step2").hidden = false;
  render();
  $("st-q-scopus").value = st.query_scopus || "";
  $("st-q-wos").value = st.query_wos || "";
  $("st-queries").hidden = false;
  $("st-approve-box").hidden = false;
  $("st-ok-scopus").checked = $("st-ok-wos").checked = !!st.approved;
  validateQueries();
}

export function init(c) {
  ctx = c;
  $("st-next").onclick = propose;
  $("st-topic").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) propose(); });
  $("st-topic-en").oninput = (e) => { if (st) st.topic_en = e.target.value; };
  $("st-add-concept").onclick = () => { st.concepts.push({ name: "Nuevo concepto", rationale: "", terms: [{ term: "", on: true }] }); changed(); renderConcepts(); };
  $("st-add-excl").onclick = () => {
    const v = $("st-new-excl").value.trim();
    if (!v) return;
    let g = st.exclusions.find((x) => x.group === "Términos personalizados");
    if (!g) st.exclusions.push(g = { group: "Términos personalizados", rationale: "Añadidos por los autores.", terms: [] });
    v.split(/[;,]/).map((x) => x.trim()).filter(Boolean).forEach((term) => g.terms.push({ term, reason: "", caution: false, note: "", on: true }));
    $("st-new-excl").value = "";
    changed(); renderExclusions();
  };
  $("st-new-excl").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("st-add-excl").click(); } });
  $("st-exfield").onchange = (e) => { if (st) { st.exclField = e.target.value; changed(); } };
  const sel = $("st-add-scopus");
  sel.innerHTML = `<option value="">Añadir área de Scopus…</option>` +
    Object.entries(SCOPUS_AREAS).map(([k, v]) => `<option value="${k}">${k} · ${v}</option>`).join("");
  sel.onchange = () => {
    if (sel.value && !st.scopus.some((a) => a.code === sel.value)) {
      st.scopus.push({ code: sel.value, mode: "include", reason: "", on: true });
      changed(); renderFilters();
    }
    sel.value = "";
  };
  $("st-add-wos").onclick = () => { st.wos.push({ name: "", mode: "include", reason: "", on: true }); changed(); renderFilters(); };
  $("st-gen").onclick = showQueries;
  for (const k of ["scopus", "wos"]) {
    $(`st-q-${k}`).oninput = () => { $(`st-ok-${k}`).checked = false; validateQueries(); };
    $(`st-ok-${k}`).onchange = validateQueries;
    $(`st-copy-${k}`).onclick = async () => {
      try { await navigator.clipboard.writeText($(`st-q-${k}`).value); ctx.toast("Ecuación copiada."); }
      catch { $(`st-q-${k}`).select(); document.execCommand("copy"); ctx.toast("Ecuación copiada."); }
    };
  }
  $("st-export").onclick = exportText;
  $("st-continue").onclick = () => {
    st.query_scopus = $("st-q-scopus").value.trim();
    st.query_wos = $("st-q-wos").value.trim();
    st.approved = true;
    st.approved_at = new Date().toISOString();
    ctx.onApproved(st);
  };
}
