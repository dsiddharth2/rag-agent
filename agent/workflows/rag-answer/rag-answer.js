import { CONTEXT_K, generateAnswer } from '../rag-shared/generation.mjs';
import { RAG_METHODS, createRetrievalClient } from '../rag-shared/retrieval-client.mjs';

export const meta = { name: 'rag-answer' };

export async function main(context) {
  const { command, agent, log, args } = context;
  const method = args?.method ?? 'hybrid';
  const question = args?.question;

  if (!RAG_METHODS.includes(method)) {
    return { error: 'unknown_method' };
  }
  if (typeof question !== 'string' || !question.trim()) {
    return { error: 'missing_question' };
  }

  log(`rag-answer method=${method}`);

  const retrieval = createRetrievalClient(command);
  let chunks;
  try {
    chunks = await retrieval.single({ method, question });
  } catch (err) {
    return {
      method,
      question,
      error: 'retrieval_failed',
      detail: err?.message ?? String(err),
    };
  }

  const contextChunks = chunks.slice(0, CONTEXT_K);
  const context_doc_ids = [];
  const seenDocs = new Set();
  for (const c of contextChunks) {
    if (seenDocs.has(c.doc_id)) continue;
    seenDocs.add(c.doc_id);
    context_doc_ids.push(c.doc_id);
  }
  const context_chunk_ids = contextChunks.map((c) => c.chunk_id);

  const generated = await generateAnswer({ agent, question, chunks: contextChunks });

  return {
    method,
    question,
    context_doc_ids,
    context_chunk_ids,
    ...generated,
  };
}
