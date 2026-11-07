import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  INDEX_MANIFEST_PATH,
  QRELS_PATH,
  REPO_ROOT,
  REPORTS_DIR,
  TEST_QUERIES_PATH,
} from '../rag-shared/paths.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const engineScript = path.join(here, 'rag-eval.js');

export const selfExecuting = true;

export function parseCli(argv) {
  let methods = ['naive', 'hybrid'];
  let sampleSize = 40;
  let seed = null;
  let retrievalOnly = false;
  let reportDir;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--methods') {
      const value = argv[++i];
      methods = value.split(',').map((m) => m.trim()).filter(Boolean);
    } else if (arg === '--sample') {
      const value = argv[++i];
      const n = Number(value);
      if (!Number.isInteger(n) || n < 1) {
        throw new Error('sample must be a positive integer');
      }
      sampleSize = n;
    } else if (arg === '--seed') {
      const value = argv[++i];
      const n = Number(value);
      if (!Number.isInteger(n)) {
        throw new Error('seed must be an integer');
      }
      seed = n;
    } else if (arg === '--retrieval-only') {
      retrievalOnly = true;
    } else if (arg === '--report-dir') {
      reportDir = argv[++i];
    } else {
      throw new Error(`unknown argument: ${arg}`);
    }
  }

  return { methods, sampleSize, seed, retrievalOnly, reportDir };
}

export function localCommand(cmd) {
  const isWin = process.platform === 'win32';
  const result = isWin
    ? spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', cmd], {
        encoding: 'utf8',
        maxBuffer: 50 * 1024 * 1024,
      })
    : spawnSync(cmd, {
        encoding: 'utf8',
        shell: true,
        maxBuffer: 50 * 1024 * 1024,
      });

  const output = (result.stdout ?? '').trim();
  const stderr = (result.stderr ?? '').trim();
  const spawnErr = result.error?.message;
  const exitErr = result.status !== 0 ? stderr || `exit ${result.status}` : '';
  return {
    ok: result.status === 0 && !result.error,
    output,
    error: spawnErr ?? exitErr,
  };
}

function defaultReportDir(now = Date.now()) {
  const stamp = new Date(now).toISOString().replace(/[:.]/g, '-');
  return path.join(REPORTS_DIR, stamp);
}

function readGitCommit() {
  const result = spawnSync('git', ['rev-parse', 'HEAD'], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  });
  const out = result.stdout?.trim();
  return out || null;
}

function readIndexManifest() {
  try {
    return JSON.parse(fs.readFileSync(INDEX_MANIFEST_PATH, 'utf8'));
  } catch {
    return null;
  }
}

export async function runRagEval({ fleetApi, workspace, signal, reportPhase, options }) {
  if (options?.retrievalOnly) {
    const { main } = await import('./rag-eval.js');
    const command = async (cmd) => localCommand(cmd);
    return main({
      command,
      agent: async () => {
        throw new Error('agent should not be called in retrieval-only mode');
      },
      log: () => {},
      args: { options },
    });
  }

  if (!process.env.CLAUDE_CODE_OAUTH_TOKEN) {
    throw new Error('set CLAUDE_CODE_OAUTH_TOKEN or use --retrieval-only');
  }

  if (!fleetApi) {
    const { withStandaloneLease } = await import('../standalone.mjs');
    return withStandaloneLease((ctx) => runRagEval({ ...ctx, signal, reportPhase, options }));
  }

  const { FleetWorkflow } = await import(
    '@apralabs/apra-fleet/packages/apra-fleet-workflow/src/workflow/index.mjs'
  );
  const { WorkflowEngine } = await import(
    '@apralabs/apra-fleet/packages/apra-fleet-workflow/src/workflow/engine.mjs'
  );

  const workflow = new FleetWorkflow(fleetApi);
  const engine = new WorkflowEngine(workflow);
  return engine.executeFile(engineScript, {
    fleetApi,
    workspace,
    signal,
    reportPhase,
    args: { options },
  });
}

function isMainModule() {
  const invoked = process.argv[1];
  if (!invoked) return false;
  try {
    return fs.realpathSync(invoked) === fs.realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return path.resolve(invoked) === fileURLToPath(import.meta.url);
  }
}

if (isMainModule()) {
  try {
    const cli = parseCli(process.argv.slice(2));
    const reportDir = cli.reportDir ?? defaultReportDir();
    const options = {
      methods: cli.methods,
      sampleSize: cli.sampleSize,
      seed: cli.seed,
      retrievalOnly: cli.retrievalOnly,
      reportDir,
      testQueriesPath: TEST_QUERIES_PATH,
      qrelsPath: QRELS_PATH,
      gitCommit: readGitCommit(),
      indexManifest: readIndexManifest(),
    };
    const result = await runRagEval({ options });
    const summaryMd = result?.summaryMd ?? path.join(reportDir, 'summary.md');
    console.log(summaryMd);
    process.exit(0);
  } catch (err) {
    console.error(err?.message ?? err);
    process.exit(1);
  }
}
