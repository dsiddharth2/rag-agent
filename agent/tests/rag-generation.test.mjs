import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildAnswerPrompt,
  parseAnswer,
  askForJson,
  formatContext,
  CONTEXT_K,
} from '../workflows/rag-shared/generation.mjs';
import { fakeAgent } from './rag-fakes.mjs';

const validJson =
  '{"answer":"Reinstall the component.","abstained":false,"cited_doc_ids":["swg21996508"]}';

test('parseAnswer accepts valid JSON shape', () => {
  const r = parseAnswer(validJson);
  assert.equal(r.ok, true);
  assert.equal(r.value.answer, 'Reinstall the component.');
  assert.equal(r.value.abstained, false);
  assert.deepEqual(r.value.cited_doc_ids, ['swg21996508']);
});

test('parseAnswer rejects missing answer', () => {
  const r = parseAnswer('{"abstained":false,"cited_doc_ids":[]}');
  assert.equal(r.ok, false);
});

test('parseAnswer rejects non-boolean abstained', () => {
  const r = parseAnswer('{"answer":"x","abstained":"no","cited_doc_ids":[]}');
  assert.equal(r.ok, false);
});

test('parseAnswer rejects non-array cited_doc_ids', () => {
  const r = parseAnswer('{"answer":"x","abstained":false,"cited_doc_ids":"one"}');
  assert.equal(r.ok, false);
});

test('buildAnswerPrompt contains question and every doc_id', () => {
  const chunks = [
    { doc_id: 'd1', text: 'first' },
    { doc_id: 'd2', text: 'second' },
  ];
  const prompt = buildAnswerPrompt('How do I fix error ABC?', chunks);
  assert.match(prompt, /How do I fix error ABC\?/);
  assert.match(prompt, /doc_id=d1/);
  assert.match(prompt, /doc_id=d2/);
});

test('formatContext labels chunks with index and doc_id', () => {
  const ctx = formatContext([
    { doc_id: 'a', text: 'one' },
    { doc_id: 'b', text: 'two' },
  ]);
  assert.match(ctx, /\[1\] doc_id=a\none/);
  assert.match(ctx, /\[2\] doc_id=b\ntwo/);
  assert.match(ctx, /\n\n---\n\n/);
});

test('askForJson succeeds on first valid reply', async () => {
  const { agent } = fakeAgent([validJson]);
  const result = await askForJson({
    agent,
    prompt: 'go',
    options: { member_name: 'doer', model: 'standard' },
    parse: parseAnswer,
    failure: 'generation_parse_failed',
  });
  assert.equal(result.answer, 'Reinstall the component.');
  assert.equal(result.calls, 1);
});

test('askForJson retries once after invalid JSON', async () => {
  const { agent, calls } = fakeAgent(['not json', validJson]);
  const result = await askForJson({
    agent,
    prompt: 'go',
    options: { member_name: 'doer', model: 'standard' },
    parse: parseAnswer,
    failure: 'generation_parse_failed',
  });
  assert.equal(result.answer, 'Reinstall the component.');
  assert.equal(result.calls, 2);
  assert.match(calls[1].prompt, /could not be parsed/i);
});

test('askForJson returns failure after two invalid replies', async () => {
  const { agent } = fakeAgent(['nope', 'still bad']);
  const result = await askForJson({
    agent,
    prompt: 'go',
    options: { member_name: 'doer', model: 'standard' },
    parse: parseAnswer,
    failure: 'generation_parse_failed',
  });
  assert.equal(result.error, 'generation_parse_failed');
  assert.equal(result.raw, 'still bad');
  assert.equal(result.calls, 2);
});

test('CONTEXT_K is 5', () => {
  assert.equal(CONTEXT_K, 5);
});
