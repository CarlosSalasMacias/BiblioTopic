// BiblioTopic — asistencia opcional con IA (Claude, API de Anthropic) desde el navegador.
// La clave de API la escribe el usuario y solo se usa desde esta pestaña (no se guarda en ningún servidor).
import Anthropic from "https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@0.135.0/+esm";

export const MODEL = "claude-opus-5-5";
const BETAS = ["server-side-fallback-2026-07-01"];

function client(apiKey) {
  // dangerouslyAllowBrowser: la página es estática (GitHub Pages) y la clave es del propio usuario.
  return new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
}

function textOf(msg) {
  if (msg.stop_reason === "refusal") {
    throw new Error("El modelo declinó la solicitud" + (msg.stop_details?.category ? ` (${msg.stop_details.category})` : "") + ".");
  }
  return msg.content.filter((b) => b.type === "text").map((b) => b.text).join("");
}

async function run(apiKey, params, onText) {
  const stream = client(apiKey).beta.messages.stream({
    model: MODEL,
    betas: BETAS,
    fallbacks: "default", // si el modelo declina, el servidor reintenta con el modelo de respaldo recomendado
    ...params,
  });
  if (onText) stream.on("text", (_, snapshot) => onText(snapshot));
  const msg = await stream.finalMessage();
  if (msg.stop_reason === "max_tokens") throw new Error("La respuesta se cortó por longitud; inténtelo de nuevo.");
  return textOf(msg);
}

const LABEL_SCHEMA = {
  type: "object",
  properties: {
    topics: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          label: { type: "string" },
          rationale: { type: "string" },
        },
        required: ["id", "label", "rationale"],
        additionalProperties: false,
      },
    },
  },
  required: ["topics"],
  additionalProperties: false,
};

export async function suggestLabels(apiKey, ctx, field) {
  const topics = [...(ctx.topics.lda || []), ...(ctx.topics.bert || [])].map((t) => ({
    id: t.id, top_terms: t.terms, prevalence_pct: t.prevalence_pct,
    representative_titles: t.docs.map((d) => d.title),
  }));
  const prompt = `You are an expert in ${field || "the research field of this corpus"} and in bibliometric topic modelling.
Below are topics obtained with LDA (ids L1, L2, ...) and BERTopic (ids B1, B2, ...) from titles and abstracts of a scientific corpus.
For each topic, propose a concise, specific English label (2-6 words, title case, no abbreviations unless standard in the field) that a domain expert would accept, and a one-sentence rationale in Spanish.
Base each label only on the terms and titles given. If a topic is incoherent, label it "Mixed: <dominant idea>".

${JSON.stringify(topics, null, 1)}`;
  const text = await run(apiKey, {
    max_tokens: 8000,
    output_config: { effort: "medium", format: { type: "json_schema", schema: LABEL_SCHEMA } },
    messages: [{ role: "user", content: prompt }],
  });
  return JSON.parse(text).topics;
}

export async function draftDiscussion(apiKey, ctx, meta, onText) {
  const refs = ctx.references.map((r) => `[${r.id}] ${r.cite}. ${r.title}${r.citations != null ? ` (${r.citations} citations)` : ""}`).join("\n");
  const prompt = `Eres un investigador senior que redacta la sección de Discusión de un artículo científico para una revista Q1.
Tema del estudio: ${meta.topic_name || "(no indicado)"}.

A continuación están los resultados del análisis bibliométrico y de modelado de temas (LDA y BERTopic) del corpus, y una lista de documentos del propio corpus que puedes citar.

RESULTADOS (JSON):
${JSON.stringify({ topics: ctx.topics, summary: ctx.summary }, null, 1)}

DOCUMENTOS DEL CORPUS QUE PUEDES CITAR:
${refs}

Escribe en español académico una Discusión de 1000 a 1500 palabras, organizada en 4 a 6 subsecciones con encabezados que empiecen por "## ". Debe:
- Interpretar el significado de los principales temas para el campo y explicar por qué son relevantes.
- Discutir la dinámica temporal (temas emergentes y en declive) y, si hay datos, la distribución geográfica y la colaboración.
- Comparar lo que aportan LDA y BERTopic y qué temas son robustos entre ambos métodos.
- Señalar vacíos de investigación y líneas futuras concretas.
- Incluir un párrafo de limitaciones (cobertura de las bases de datos, idioma, tipos de documento, sensibilidad a los parámetros de los modelos).

Reglas estrictas sobre citas:
- Cita SOLO los documentos de la lista, con su identificador entre corchetes, por ejemplo [R3] o [R3, R7]. No inventes referencias, autores, años ni DOI.
- Si una afirmación requiere literatura externa al corpus, escribe exactamente [cita externa] en su lugar para que los autores la completen.
- No repitas cifras de los resultados salvo las imprescindibles; interpreta en lugar de describir.
Devuelve solo el texto de la Discusión (sin título principal, sin lista de referencias, sin Markdown aparte de los encabezados "## ").`;
  return run(apiKey, {
    max_tokens: 32000,
    output_config: { effort: "high" },
    messages: [{ role: "user", content: prompt }],
  }, onText);
}

const TITLE_SCHEMA = {
  type: "object",
  properties: {
    titles: {
      type: "array",
      items: {
        type: "object",
        properties: { title: { type: "string" }, rationale: { type: "string" } },
        required: ["title", "rationale"],
        additionalProperties: false,
      },
    },
  },
  required: ["titles"],
  additionalProperties: false,
};

export async function suggestTitles(apiKey, ctx, meta) {
  const topics = Object.fromEntries(Object.entries(ctx.topics).map(([m, ts]) => [m, ts.map((t) => ({
    id: t.id, label: t.label, terms: t.terms.slice(0, 8), prevalence_pct: t.prevalence_pct,
  }))]));
  const prompt = `You are a senior editor of a Q1 journal. Propose 5 English titles for a research article based on the analysis below.
Study topic: ${meta.topic_name || "(not given; infer it from the topics)"}.
Search query: ${meta.query || "(not given)"}.
The article is a bibliometric analysis of a Web of Science/Scopus corpus with topic modelling (the models used appear in the results).

Requirements:
- Specific to the subject of the corpus and to its main findings (dominant, emerging or declining themes), not generic.
- Mention the methods only when they add value; vary the style (descriptive, finding-oriented, question).
- 10-20 words each, no abbreviations except LDA, BERTopic or HJ-Biplot, sentence case.
- Do not claim findings that are not in the results.
- For each title, give a one-sentence rationale in Spanish.

RESULTS (JSON):
${JSON.stringify({ topics, summary: ctx.summary }, null, 1)}`;
  const text = await run(apiKey, {
    max_tokens: 4000,
    output_config: { effort: "medium", format: { type: "json_schema", schema: TITLE_SCHEMA } },
    messages: [{ role: "user", content: prompt }],
  });
  return JSON.parse(text).titles;
}

const STRATEGY_SCHEMA = {
  type: "object",
  properties: {
    topic_en: { type: "string" },
    concepts: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: { type: "string" },
          rationale: { type: "string" },
          terms: { type: "array", items: { type: "string" } },
        },
        required: ["name", "rationale", "terms"],
        additionalProperties: false,
      },
    },
    exclusions: {
      type: "array",
      items: {
        type: "object",
        properties: {
          group: { type: "string" },
          rationale: { type: "string" },
          terms: {
            type: "array",
            items: {
              type: "object",
              properties: {
                term: { type: "string" },
                reason: { type: "string" },
                caution: { type: "boolean" },
                caution_note: { type: "string" },
              },
              required: ["term", "reason", "caution", "caution_note"],
              additionalProperties: false,
            },
          },
        },
        required: ["group", "rationale", "terms"],
        additionalProperties: false,
      },
    },
    scopus_subject_areas: {
      type: "array",
      items: {
        type: "object",
        properties: {
          code: { type: "string" },
          mode: { type: "string", enum: ["include", "exclude"] },
          reason: { type: "string" },
        },
        required: ["code", "mode", "reason"],
        additionalProperties: false,
      },
    },
    wos_categories: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: { type: "string" },
          mode: { type: "string", enum: ["include", "exclude"] },
          reason: { type: "string" },
        },
        required: ["name", "mode", "reason"],
        additionalProperties: false,
      },
    },
    notes: { type: "string" },
  },
  required: ["topic_en", "concepts", "exclusions", "scopus_subject_areas", "wos_categories", "notes"],
  additionalProperties: false,
};

export async function suggestStrategy(apiKey, topic) {
  const prompt = `You are an information specialist (academic librarian) who designs search strategies for bibliometric studies in Scopus and Web of Science Core Collection.

Research topic given by the researcher (it may be in Spanish or another language):
"""${topic}"""

Design the building blocks of a sensitive but precise search strategy. Return:

1. topic_en: the topic as a short English phrase suitable for an article title.
2. concepts: the 1-4 main concepts of the topic (each concept becomes one OR-block; blocks are combined with AND). For each concept give a short English name, a one-sentence rationale in Spanish, and 3-15 English search terms: synonyms, spelling variants (British/American), singular/plural forms, acronyms in common use, scientific and common names when relevant. Use the truncation asterisk (*) where it captures variants without adding noise (never at the start of a word, and not on stems shorter than 4 letters). Write multi-word phrases without quotes (quotes are added later). Do not add a concept for generic words such as "research", "study" or "analysis".
3. exclusions: candidate NOT terms that would retrieve documents outside the object of study, grouped by thematic category (group name and a one-sentence rationale in Spanish for the group). For each term: the English term (truncation allowed), a short reason in Spanish, and caution = true when the term could also appear in relevant documents (for example, a different context in which the topic is also studied, a method or organism that relevant studies may mention). When caution is true, explain the risk in caution_note (Spanish); otherwise caution_note is "". Be conservative: propose only exclusions that clearly reduce noise, never terms that are part of the topic itself. Propose an empty list if no exclusion is advisable.
4. scopus_subject_areas: Scopus subject-area codes (for example AGRI, BIOC, ENVI, MEDI, SOCI) that could be used to limit (include) or exclude results, with a reason in Spanish. Only codes from the official Scopus list of 27 subject areas plus MULT.
5. wos_categories: Web of Science categories (exact official names, for example "Agronomy", "Plant Sciences") to limit (include) or exclude, with a reason in Spanish.
6. notes: 2-4 sentences in Spanish with advice on the strategy (recall-precision trade-off, terms to verify, ambiguous terms).`;
  const text = await run(apiKey, {
    max_tokens: 16000,
    output_config: { effort: "high", format: { type: "json_schema", schema: STRATEGY_SCHEMA } },
    messages: [{ role: "user", content: prompt }],
  });
  return JSON.parse(text);
}
