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
