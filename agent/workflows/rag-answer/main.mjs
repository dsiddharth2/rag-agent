import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const engineScript = path.join(here, 'rag-answer.js');

export const selfExecuting = true;

export async function runRagAnswer({ fleetApi, workspace, signal, reportPhase, method, question } = {}) {
  if (!fleetApi) {
    const { withStandaloneLease } = await import('../standalone.mjs');
    return withStandaloneLease((ctx) =>
      runRagAnswer({ ...ctx, signal, reportPhase, method, question }),
    );
  }
  const { FleetWorkflow } = await import(
    '@apralabs/apra-fleet/packages/apra-fleet-workflow/src/workflow/index.mjs'
  );
  const { WorkflowEngine } = await import(
    '@apralabs/apra-fleet/packages/apra-fleet-workflow/src/workflow/engine.mjs'
  );

  const workflow = new FleetWorkflow(fleetApi);
  const engine = new WorkflowEngine(workflow);
  return await engine.executeFile(engineScript, {
    fleetApi,
    workspace,
    signal,
    reportPhase,
    method,
    question,
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
    const method = process.argv[2] ?? 'hybrid';
    const question = process.argv[3];
    const result = await runRagAnswer({ method, question });
    console.log(JSON.stringify(result, null, 2));
    process.exit(result?.error ? 1 : 0);
  } catch (err) {
    console.error(err?.message ?? err);
    process.exit(1);
  }
}
