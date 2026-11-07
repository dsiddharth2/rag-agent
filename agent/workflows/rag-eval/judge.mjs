import { extractJsonObject } from '../rag-shared/llm-json.mjs';
import { askForJson, formatContext } from '../rag-shared/generation.mjs';

export const JUDGE = Object.freeze({ member_name: 'reviewer', model: 'premium' });

export const CORRECTNESS_LEVELS = ['full', 'partial', 'none'];

export function buildJudgePrompt({ question, chunks, answer, goldAnswer }) {
  const context = formatContext(chunks);
  return `You evaluate a support answer against retrieved technote excerpts and a reference answer.

Question:
${question}

Retrieved context:
${context}

Generated answer:
${answer}

Reference answer (gold):
${goldAnswer}

Tasks:
1. Split the generated answer into factual claims. For each claim, mark whether it is supported using only the retrieved context above (not the reference).
2. Rate whether the generated answer contains the information in the reference (full / partial / none). Judge containment, not wording; extra correct detail is allowed.
3. Reply with only JSON in this shape:
{"claims":[{"text":"...","supported":true|false}],"correctness":"full"|"partial"|"none","reason":"..."}`;
}

export function parseJudge(text) {
  const obj = extractJsonObject(text);
  if (obj == null) return { ok: false, error: 'no JSON object' };
  if (!Array.isArray(obj.claims)) return { ok: false, error: 'invalid claims' };
  for (const c of obj.claims) {
    if (typeof c?.text !== 'string') return { ok: false, error: 'invalid claim text' };
    if (typeof c.supported !== 'boolean') return { ok: false, error: 'invalid supported' };
  }
  if (!CORRECTNESS_LEVELS.includes(obj.correctness)) {
    return { ok: false, error: 'invalid correctness' };
  }
  const reason = typeof obj.reason === 'string' ? obj.reason : '';
  return {
    ok: true,
    value: {
      claims: obj.claims.map((c) => ({ text: c.text, supported: c.supported })),
      correctness: obj.correctness,
      reason,
    },
  };
}

export async function judgeAnswer({ agent, question, chunks, answer, goldAnswer }) {
  const prompt = buildJudgePrompt({ question, chunks, answer, goldAnswer });
  return askForJson({
    agent,
    prompt,
    options: JUDGE,
    parse: parseJudge,
    failure: 'judge_error',
  });
}
