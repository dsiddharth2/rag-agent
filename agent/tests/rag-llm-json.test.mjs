import { test } from 'node:test';
import assert from 'node:assert/strict';
import { toText, extractJsonObject } from '../workflows/rag-shared/llm-json.mjs';

test('toText accepts a plain string', () => {
  assert.equal(toText('hello'), 'hello');
});

test('toText accepts { text }', () => {
  assert.equal(toText({ text: 'from text' }), 'from text');
});

test('toText accepts { content: [{ text }] }', () => {
  assert.equal(
    toText({ content: [{ text: 'block one' }, { text: 'block two' }] }),
    'block one\nblock two',
  );
});

test('toText accepts { output }', () => {
  assert.equal(toText({ output: 'model output' }), 'model output');
});

test('extractJsonObject parses plain JSON', () => {
  assert.deepEqual(extractJsonObject('{"a":1}'), { a: 1 });
});

test('extractJsonObject parses fenced JSON block', () => {
  const text = 'Here is the result:\n```json\n{"answer":"yes"}\n```\nDone.';
  assert.deepEqual(extractJsonObject(text), { answer: 'yes' });
});

test('extractJsonObject handles prose before and after JSON', () => {
  const text = 'Preamble.\n{"x":42}\nPostamble.';
  assert.deepEqual(extractJsonObject(text), { x: 42 });
});

test('extractJsonObject respects braces inside JSON strings', () => {
  const text = '{"answer":"use {x}"}';
  assert.deepEqual(extractJsonObject(text), { answer: 'use {x}' });
});

test('extractJsonObject returns the last parseable object', () => {
  const text = '{"first":1} trailing {"second":2}';
  assert.deepEqual(extractJsonObject(text), { second: 2 });
});

test('extractJsonObject returns null when no object', () => {
  assert.equal(extractJsonObject('no json here'), null);
});
