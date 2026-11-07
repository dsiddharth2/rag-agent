import { createRetrievalClient } from '../rag-shared/retrieval-client.mjs';
import { runEval } from './eval-runner.mjs';

export const meta = { name: 'rag-eval' };

export async function main(context) {
  const { command, agent, log, args } = context;
  const options = args?.options ?? args;
  return runEval({
    agent,
    log: log ?? (() => {}),
    retrieval: createRetrievalClient(command),
    options,
  });
}
