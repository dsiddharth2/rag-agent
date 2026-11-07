import fs from 'node:fs';
import fsp from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { pythonModuleCommand, quotePath } from './paths.mjs';

export const RAG_METHODS = Object.freeze(['naive', 'hybrid']);
export const RETRIEVE_DEPTH = 50;

const COMMAND_OPTS = Object.freeze({
  member_name: 'doer',
  failSoft: true,
  timeout_s: 3600,
  timeoutMs: 3_700_000,
});

export function readJsonl(file) {
  const text = fs.readFileSync(file, 'utf8');
  return text
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

export function writeJsonl(file, rows) {
  const body = rows.map((row) => `${JSON.stringify(row)}\n`).join('');
  fs.writeFileSync(file, body, 'utf8');
}

function failOrParse(label, result) {
  if (!result.ok) {
    throw new Error(`${label} failed: ${result.error}`);
  }
  return result.output;
}

export function createRetrievalClient(command) {
  async function runFleetCommand(label, cmd) {
    const result = await command(cmd, COMMAND_OPTS);
    return failOrParse(label, result);
  }

  return {
    async single({ method, question, depth = RETRIEVE_DEPTH }) {
      const dir = await fsp.mkdtemp(path.join(os.tmpdir(), 'rag-query-'));
      try {
        const queryFile = path.join(dir, 'query.txt');
        await fsp.writeFile(queryFile, question, 'utf8');
        const cmd = pythonModuleCommand('rag.retrieve', [
          '--method',
          method,
          '--query-file',
          quotePath(queryFile),
          '--depth',
          String(depth),
        ]);
        const output = await runFleetCommand('retrieve', cmd);
        const parsed = JSON.parse(output);
        return parsed.chunks ?? [];
      } finally {
        await fsp.rm(dir, { recursive: true, force: true });
      }
    },

    async batch({ method, rows, outFile, depth = RETRIEVE_DEPTH }) {
      const queriesFile = outFile.replace(/\.jsonl$/i, '.queries.jsonl');
      writeJsonl(queriesFile, rows);
      const cmd = pythonModuleCommand('rag.retrieve', [
        '--method',
        method,
        '--queries-file',
        quotePath(queriesFile),
        '--out',
        quotePath(outFile),
        '--depth',
        String(depth),
      ]);
      const output = await runFleetCommand('retrieve batch', cmd);
      const summary = JSON.parse(output.trim());
      const resultRows = readJsonl(outFile);
      const chunksById = new Map();
      for (const row of resultRows) {
        chunksById.set(row.query_id, row.chunks ?? []);
      }
      return { summary, chunksById };
    },

    async metrics({ resultsFile, qrelsFile }) {
      const cmd = pythonModuleCommand('rag.retrieval_metrics', [
        '--results',
        quotePath(resultsFile),
        '--qrels',
        quotePath(qrelsFile),
      ]);
      const output = await runFleetCommand('retrieval metrics', cmd);
      return JSON.parse(output.trim());
    },
  };
}
