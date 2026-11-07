import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildJudgePrompt,
  parseJudge,
  judgeAnswer,
  JUDGE,
} from '../workflows/rag-eval/judge.mjs';
import { fakeAgent } from './rag-fakes.mjs';

const validJudge =
  '{"claims":[{"text":"Claim A","supported":true},{"text":"Claim B","supported":false}],"correctness":"full","reason":"Matches reference."}';

test('parseJudge accepts valid JSON', () => {
  const r = parseJudge(validJudge);
  assert.equal(r.ok, true);
  assert.equal(r.value.correctness, 'full');
  assert.equal(r.value.reason, 'Matches reference.');
  assert.equal(r.value.claims.length, 2);
  assert.equal(r.value.claims[0].supported, true);
});

test('parseJudge defaults reason to empty string', () => {
  const r = parseJudge(
    '{"claims":[{"text":"x","supported":true}],"correctness":"partial"}',
  );
  assert.equal(r.ok, true);
  assert.equal(r.value.reason, '');
});

test('parseJudge rejects bad correctness value', () => {
  const r = parseJudge(
    '{"claims":[{"text":"x","supported":true}],"correctness":"mostly","reason":""}',
  );
  assert.equal(r.ok, false);
});

test('parseJudge rejects supported not a boolean', () => {
  const r = parseJudge(
    '{"claims":[{"text":"x","supported":"yes"}],"correctness":"full","reason":""}',
  );
  assert.equal(r.ok, false);
});

test('parseJudge rejects no JSON', () => {
  const r = parseJudge('no json here');
  assert.equal(r.ok, false);
});

test('buildJudgePrompt includes question, context, answer, and gold', () => {
  const chunks = [{ doc_id: 'd1', text: 'context one' }];
  const prompt = buildJudgePrompt({
    question: 'What is X?',
    chunks,
    answer: 'X is Y.',
    goldAnswer: 'Y defines X.',
  });
  assert.match(prompt, /What is X\?/);
  assert.match(prompt, /doc_id=d1/);
  assert.match(prompt, /X is Y\./);
  assert.match(prompt, /Y defines X\./);
  assert.match(prompt, /only the retrieved context/i);
  assert.match(prompt, /"claims"/);
});

test('judgeAnswer uses reviewer premium', async () => {
  const { agent, calls } = fakeAgent([validJudge]);
  await judgeAnswer({
    agent,
    question: 'Q?',
    chunks: [{ doc_id: 'a', text: 't' }],
    answer: 'A.',
    goldAnswer: 'G.',
  });
  assert.equal(calls.length, 1);
  assert.deepEqual(
    { member_name: calls[0].member_name, model: calls[0].model },
    JUDGE,
  );
});

test('judgeAnswer retries then succeeds', async () => {
  const { agent } = fakeAgent(['garbage', validJudge]);
  const result = await judgeAnswer({
    agent,
    question: 'Q?',
    chunks: [],
    answer: 'A.',
    goldAnswer: 'G.',
  });
  assert.equal(result.correctness, 'full');
  assert.equal(result.calls, 2);
});

test('judgeAnswer fails after two garbage replies', async () => {
  const { agent } = fakeAgent(['bad', 'still bad']);
  const result = await judgeAnswer({
    agent,
    question: 'Q?',
    chunks: [],
    answer: 'A.',
    goldAnswer: 'G.',
  });
  assert.equal(result.error, 'judge_error');
  assert.equal(result.raw, 'still bad');
  assert.equal(result.calls, 2);
});
