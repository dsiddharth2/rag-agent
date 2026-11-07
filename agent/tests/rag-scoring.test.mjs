import { test } from 'node:test';
import assert from 'node:assert/strict';
import { scoreRow, aggregate } from '../workflows/rag-eval/scoring.mjs';

const answerable = { is_impossible: false, quality_ok: true };
const unanswerable = { is_impossible: true };

test('scoreRow: answered full, faithfulness 2/3, wrong_refusal false', () => {
  const s = scoreRow({
    question: answerable,
    goldDocIds: ['g1'],
    contextDocIds: ['g1', 'other'],
    generation: { abstained: false, answer: 'A.' },
    judgment: {
      correctness: 'full',
      claims: [
        { text: 'a', supported: true },
        { text: 'b', supported: true },
        { text: 'c', supported: false },
      ],
    },
  });
  assert.equal(s.stratum, 'answerable_ok');
  assert.equal(s.correctness, 1);
  assert.ok(Math.abs(s.faithfulness - 2 / 3) < 1e-9);
  assert.equal(s.wrong_refusal, false);
  assert.equal(s.context_hit, true);
  assert.equal(s.zero_claims, false);
  assert.deepEqual(s.errors, []);
});

test('scoreRow: abstained answerable', () => {
  const s = scoreRow({
    question: answerable,
    goldDocIds: ['g1'],
    contextDocIds: ['x'],
    generation: { abstained: true, answer: 'Cannot answer.' },
    judgment: null,
  });
  assert.equal(s.correctness, 0);
  assert.equal(s.wrong_refusal, true);
  assert.equal(s.faithfulness, null);
  assert.equal(s.context_hit, false);
});

test('scoreRow: judge error', () => {
  const s = scoreRow({
    question: answerable,
    goldDocIds: ['g1'],
    contextDocIds: ['g1'],
    generation: { abstained: false, answer: 'A.' },
    judgment: { error: 'judge_error' },
  });
  assert.deepEqual(s.errors, ['judge_error']);
  assert.equal(s.correctness, null);
  assert.equal(s.wrong_refusal, false);
});

test('scoreRow: zero claims', () => {
  const s = scoreRow({
    question: answerable,
    goldDocIds: ['g1'],
    contextDocIds: ['g1'],
    generation: { abstained: false, answer: 'OK.' },
    judgment: { correctness: 'partial', claims: [] },
  });
  assert.equal(s.zero_claims, true);
  assert.equal(s.faithfulness, null);
  assert.equal(s.correctness, 0.5);
});

test('scoreRow: unanswerable abstained', () => {
  const s = scoreRow({
    question: unanswerable,
    goldDocIds: [],
    contextDocIds: ['x'],
    generation: { abstained: true, answer: 'Refuse.' },
    judgment: null,
  });
  assert.equal(s.abstention_correct, true);
  assert.equal(s.context_hit, null);
  assert.equal(s.wrong_refusal, null);
});

test('scoreRow: unanswerable answered', () => {
  const s = scoreRow({
    question: unanswerable,
    goldDocIds: [],
    contextDocIds: ['x'],
    generation: { abstained: false, answer: 'Hallucinated.' },
    judgment: null,
  });
  assert.equal(s.abstention_correct, false);
});

test('scoreRow: generation error', () => {
  const s = scoreRow({
    question: answerable,
    goldDocIds: ['g1'],
    contextDocIds: ['g1'],
    generation: { error: 'generation_parse_failed' },
    judgment: null,
  });
  assert.deepEqual(s.errors, ['generation_parse_failed']);
  assert.equal(s.correctness, null);
  assert.equal(s.wrong_refusal, null);
});

test('aggregate over four rows for one method', () => {
  const method = 'hybrid';
  const rows = [
    {
      method,
      scores: scoreRow({
        question: answerable,
        goldDocIds: ['g1'],
        contextDocIds: ['g1'],
        generation: { abstained: false, answer: 'A.' },
        judgment: {
          correctness: 'full',
          claims: [
            { text: 'a', supported: true },
            { text: 'b', supported: false },
          ],
        },
      }),
    },
    {
      method,
      scores: scoreRow({
        question: answerable,
        goldDocIds: ['g1'],
        contextDocIds: ['g1'],
        generation: { abstained: true, answer: 'No.' },
        judgment: null,
      }),
    },
    {
      method,
      scores: scoreRow({
        question: answerable,
        goldDocIds: ['g1'],
        contextDocIds: ['g1'],
        generation: { abstained: false, answer: 'A.' },
        judgment: { error: 'judge_error' },
      }),
    },
    {
      method,
      scores: scoreRow({
        question: unanswerable,
        goldDocIds: [],
        contextDocIds: [],
        generation: { abstained: false, answer: 'Wrong.' },
        judgment: null,
      }),
    },
  ];

  const agg = aggregate(rows);
  const ok = agg[method].answerable_ok;
  assert.equal(ok.n, 3);
  assert.equal(ok.correctness, 0.5);
  assert.equal(ok.faithfulness, 0.5);
  assert.equal(ok.faithfulness_n, 1);
  assert.equal(ok.wrong_refusal_rate, 1 / 3);
  assert.deepEqual(ok.errors, { judge_error: 1 });
  assert.equal(ok.hallucination_rate, null);

  const un = agg[method].unanswerable;
  assert.equal(un.abstention_accuracy, 0);
  assert.equal(un.hallucination_rate, 1);

  const flagged = agg[method].answerable_flagged;
  assert.equal(flagged.n, 0);
  assert.equal(flagged.correctness, null);
});
