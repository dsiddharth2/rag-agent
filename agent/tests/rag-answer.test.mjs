import { test } from 'node:test';
import assert from 'node:assert/strict';
import { main } from '../workflows/rag-answer/rag-answer.js';
import { fakeAgent, fakeCommand, chunk } from './rag-fakes.mjs';

const validReply = JSON.stringify({
  answer: 'Apply patch 123.',
  abstained: false,
  cited_doc_ids: ['d1'],
});

function sixChunks() {
  return [1, 2, 3, 4, 5, 6].map((n) => chunk(`d${n}`, n));
}

function makeContext({ method = 'hybrid', question = 'Why does error XYZ occur?', commandResult, agentReplies }) {
  const { command, calls: commandCalls } = fakeCommand(commandResult);
  const { agent, calls: agentCalls } = fakeAgent(agentReplies ?? [validReply]);
  const context = {
    log: () => {},
    args: { method, question },
    command,
    agent,
  };
  return { context, commandCalls, agentCalls };
}

test('rag-answer happy path uses hybrid retrieval and top-5 context', async () => {
  const chunks = sixChunks();
  const { context, commandCalls, agentCalls } = makeContext({
    commandResult: {
      ok: true,
      output: JSON.stringify({ method: 'hybrid', query: 'q', chunks }),
    },
  });

  const result = await main(context);

  assert.equal(result.answer, 'Apply patch 123.');
  assert.equal(commandCalls.length, 1);
  assert.equal(commandCalls[0].member_name, 'doer');
  assert.match(commandCalls[0].cmd, /-m rag\.retrieve/);
  assert.match(commandCalls[0].cmd, /--method hybrid/);

  assert.equal(agentCalls.length, 1);
  assert.equal(agentCalls[0].member_name, 'doer');
  assert.equal(agentCalls[0].model, 'standard');
  assert.match(agentCalls[0].prompt, /doc_id=d5/);
  assert.doesNotMatch(agentCalls[0].prompt, /doc_id=d6/);

  assert.equal(result.context_doc_ids.length, 5);
  assert.deepEqual(result.context_doc_ids, ['d1', 'd2', 'd3', 'd4', 'd5']);
});

test('rag-answer retries generation once when first reply is not JSON', async () => {
  const { context } = makeContext({
    commandResult: {
      ok: true,
      output: JSON.stringify({ method: 'hybrid', query: 'q', chunks: sixChunks() }),
    },
    agentReplies: ['not json', validReply],
  });

  const result = await main(context);
  assert.equal(result.answer, 'Apply patch 123.');
  assert.equal(result.calls, 2);
});

test('rag-answer returns generation_parse_failed after two bad replies', async () => {
  const { context, agentCalls } = makeContext({
    commandResult: {
      ok: true,
      output: JSON.stringify({ method: 'hybrid', query: 'q', chunks: sixChunks() }),
    },
    agentReplies: ['bad', 'worse'],
  });

  const result = await main(context);
  assert.equal(result.error, 'generation_parse_failed');
  assert.equal(agentCalls.length, 2);
});

test('rag-answer retrieval failure skips the LLM', async () => {
  const { context, agentCalls } = makeContext({
    commandResult: { ok: false, output: '', error: 'exit 2: no index' },
  });

  const result = await main(context);
  assert.equal(result.error, 'retrieval_failed');
  assert.equal(agentCalls.length, 0);
});

test('rag-answer rejects unknown method', async () => {
  const { context, commandCalls, agentCalls } = makeContext({
    method: 'graph',
    commandResult: { ok: true, output: '{}' },
  });

  const result = await main(context);
  assert.equal(result.error, 'unknown_method');
  assert.equal(commandCalls.length, 0);
  assert.equal(agentCalls.length, 0);
});

test('rag-answer rejects missing question', async () => {
  const { command, calls: commandCalls } = fakeCommand({ ok: true, output: '{}' });
  const { agent, calls: agentCalls } = fakeAgent([validReply]);
  const result = await main({
    log: () => {},
    args: { method: 'hybrid', question: '' },
    command,
    agent,
  });
  assert.equal(result.error, 'missing_question');
  assert.equal(commandCalls.length, 0);
  assert.equal(agentCalls.length, 0);
});
