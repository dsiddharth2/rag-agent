import { extractJsonObject, toText } from './llm-json.mjs';

export const CONTEXT_K = 5;

export const GENERATOR = Object.freeze({ member_name: 'doer', model: 'standard' });

export function formatContext(chunks) {
  return chunks
    .map((c, i) => `[${i + 1}] doc_id=${c.doc_id}\n${c.text}`)
    .join('\n\n---\n\n');
}

export function buildAnswerPrompt(question, chunks) {
  const context = formatContext(chunks);
  return `You answer IBM product support questions using only the technote excerpts below.

Question:
${question}

Context:
${context}

Rules:
- Answer only from the context above.
- If the context does not contain enough information, set abstained to true and give a one-line refusal as answer.
- Cite the doc_id values you relied on in cited_doc_ids.

Reply with only JSON in this shape:
{"answer": "...", "abstained": true|false, "cited_doc_ids": ["..."]}`;
}

export function parseAnswer(text) {
  const obj = extractJsonObject(text);
  if (obj == null) return { ok: false, error: 'no JSON object' };
  if (typeof obj.answer !== 'string') return { ok: false, error: 'missing answer' };
  if (typeof obj.abstained !== 'boolean') return { ok: false, error: 'invalid abstained' };
  if (!Array.isArray(obj.cited_doc_ids)) return { ok: false, error: 'invalid cited_doc_ids' };
  if (!obj.cited_doc_ids.every((id) => typeof id === 'string')) {
    return { ok: false, error: 'invalid cited_doc_ids entries' };
  }
  return {
    ok: true,
    value: {
      answer: obj.answer,
      abstained: obj.abstained,
      cited_doc_ids: obj.cited_doc_ids,
    },
  };
}

export async function askForJson({ agent, prompt, options, parse, failure }) {
  let currentPrompt = prompt;
  const rawReplies = [];

  for (let attempt = 0; attempt < 2; attempt += 1) {
    const reply = await agent(currentPrompt, options);
    const text = toText(reply);
    rawReplies.push(text);
    const parsed = parse(text);
    if (parsed.ok) {
      return { ...parsed.value, calls: attempt + 1 };
    }
    if (attempt === 0) {
      currentPrompt = `${prompt}\n\nYour previous reply could not be parsed: ${parsed.error}. Reply with only valid JSON.`;
    }
  }

  return {
    error: failure,
    raw: rawReplies[rawReplies.length - 1] ?? '',
    calls: 2,
  };
}

export async function generateAnswer({ agent, question, chunks }) {
  const prompt = buildAnswerPrompt(question, chunks);
  return askForJson({
    agent,
    prompt,
    options: GENERATOR,
    parse: parseAnswer,
    failure: 'generation_parse_failed',
  });
}
