// Your tool catalog. To expose a new workflow, append an entry here — no
// changes to server.mjs or http.mjs are needed. `description` is read by the
// connected model when it decides which tool to call, so write it for that
// reader.
import * as z from 'zod/v4';
import { runRagAnswer } from '../workflows/rag-answer/main.mjs';

// Kit template tools (hello, weather, textstats) are omitted here so the TechQA
// host does not advertise demo tools; hello workflow tests import hello.js directly.

export const defaultRegistry = [
  {
    name: 'rag_answer',
    description:
      'Answer an IBM product support question using the indexed TechQA technotes. Runs retrieval ' +
      '(hybrid = dense + BM25 + rerank, or naive = dense only) then generates an answer from the ' +
      'top context chunks. Spends LLM tokens.',
    inputSchema: z.object({
      question: z.string().describe('The support question, including error messages and product names.'),
      method: z
        .enum(['naive', 'hybrid'])
        .optional()
        .describe('Retrieval strategy. Defaults to hybrid.'),
    }),
    annotations: { readOnlyHint: true },
    routing: {
      description: 'Answer an IBM product support question from the technote collection.',
      args: {
        question: {
          extract: 'the full support question including error messages, product names, and symptoms',
        },
        method: {
          extract: 'retrieval method naive or hybrid when the user specifies one; otherwise omit for hybrid',
        },
      },
    },
    async run({ fleetApi, args, signal, reportPhase, workspace }) {
      return await runRagAnswer({
        fleetApi,
        workspace,
        signal,
        reportPhase,
        method: args.method,
        question: args.question,
      });
    },
  },
];
