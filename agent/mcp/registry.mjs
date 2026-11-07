// Your tool catalog. To expose a new workflow, append an entry here — no
// changes to server.mjs or http.mjs are needed. `description` is read by the
// connected model when it decides which tool to call, so write it for that
// reader.
import path from 'node:path';
import * as z from 'zod/v4';
import { toolsDir, shellEscape, parseToolOutput } from './registry-helpers.mjs';
import { runHello } from '../workflows/hello/main.mjs';
import { runRagAnswer } from '../workflows/rag-answer/main.mjs';

export const defaultRegistry = [
  {
    name: 'hello',
    description:
      'Greets someone by name using an agent. A starting point to copy — replace ' +
      'the body in workflows/hello/hello.js with your own work. Spends LLM tokens.',
    inputSchema: z.object({
      name: z.string().optional().describe('Who to greet. Defaults to "world".'),
    }),
    routing: {
      description: 'Greet someone by name using an agent.',
      args: {
        name: { extract: 'the name of the person to greet' },
      },
    },
    async run({ fleetApi, args, signal, reportPhase, workspace }) {
      return await runHello({ fleetApi, workspace, name: args.name, signal, reportPhase });
    },
  },
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
  {
    name: 'weather',
    description:
      'Fetches current weather for a city using the wttr.in API. Returns temperature, ' +
      'humidity, wind, UV index, and a text description. Read-only, no LLM tokens.',
    inputSchema: z.object({
      city: z.string().optional().describe('City name to look up. Defaults to London.'),
    }),
    annotations: { readOnlyHint: true, idempotentHint: true },
    async run({ fleetApi, args }) {
      const city = shellEscape(args.city || 'London');
      const script = path.join(toolsDir, 'weather', 'weather.py');
      const raw = await fleetApi.executeCommand({
        member_name: 'doer',
        command: `python3 "${script}" "${city}"`,
      });
      return parseToolOutput(raw);
    },
  },
  {
    name: 'textstats',
    description:
      'Analyzes a text string and returns character count, word count, sentence count, ' +
      'unique words, and average word length. Read-only, no LLM tokens.',
    inputSchema: z.object({
      text: z.string().describe('The text to analyze.'),
    }),
    annotations: { readOnlyHint: true, idempotentHint: true },
    async run({ fleetApi, args }) {
      const escaped = shellEscape(args.text);
      const script = path.join(toolsDir, 'textstats', 'textstats.py');
      const raw = await fleetApi.executeCommand({
        member_name: 'doer',
        command: `python3 "${script}" "${escaped}"`,
      });
      return parseToolOutput(raw);
    },
  },
];
