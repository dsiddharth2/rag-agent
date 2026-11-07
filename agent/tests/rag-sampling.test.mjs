import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  STRATA,
  stratumOf,
  mulberry32,
  randomSeed,
  allocate,
  stratifiedSample,
} from '../workflows/rag-eval/sampling.mjs';

test('stratumOf covers all three cases', () => {
  assert.equal(stratumOf({ is_impossible: true }), 'unanswerable');
  assert.equal(stratumOf({ is_impossible: true, quality_ok: false }), 'unanswerable');
  assert.equal(stratumOf({ is_impossible: false, quality_ok: false }), 'answerable_flagged');
  assert.equal(stratumOf({ is_impossible: false, quality_ok: true }), 'answerable_ok');
  assert.equal(stratumOf({ quality_ok: true }), 'answerable_ok');
});

test('allocate largest-remainder rounding', () => {
  const sizes = { answerable_ok: 173, answerable_flagged: 10, unanswerable: 90 };
  assert.deepEqual(allocate(sizes, 40), {
    answerable_ok: 25,
    answerable_flagged: 2,
    unanswerable: 13,
  });
  assert.deepEqual(allocate(sizes, 10), {
    answerable_ok: 6,
    answerable_flagged: 1,
    unanswerable: 3,
  });
  assert.deepEqual(allocate(sizes, 273), {
    answerable_ok: 173,
    answerable_flagged: 10,
    unanswerable: 90,
  });
});

test('allocate throws on invalid total', () => {
  const sizes = { answerable_ok: 173, answerable_flagged: 10, unanswerable: 90 };
  assert.throws(() => allocate(sizes, 0), /total/i);
  assert.throws(() => allocate(sizes, 274), /total|population|exceed/i);
  assert.throws(() => allocate(sizes, 2.5), /integer/i);
});

function makeSynthetic273() {
  const questions = [];
  let n = 0;
  for (let i = 0; i < 173; i++) {
    questions.push({
      query_id: `ok-${String(i).padStart(4, '0')}`,
      is_impossible: false,
      quality_ok: true,
    });
  }
  for (let i = 0; i < 10; i++) {
    questions.push({
      query_id: `flag-${String(i).padStart(3, '0')}`,
      is_impossible: false,
      quality_ok: false,
    });
  }
  for (let i = 0; i < 90; i++) {
    questions.push({
      query_id: `un-${String(i).padStart(4, '0')}`,
      is_impossible: true,
    });
  }
  assert.equal(questions.length, 273);
  return questions;
}

test('stratifiedSample determinism and counts at size 40', () => {
  const questions = makeSynthetic273();
  const sizes = { answerable_ok: 173, answerable_flagged: 10, unanswerable: 90 };
  const expectedCounts = allocate(sizes, 40);

  const a = stratifiedSample(questions, 40, 7);
  const b = stratifiedSample(questions, 40, 7);
  assert.equal(a.seed, 7);
  assert.equal(b.seed, 7);
  assert.deepEqual(a.counts, expectedCounts);
  assert.deepEqual(b.counts, expectedCounts);
  assert.deepEqual(
    a.questions.map((q) => q.query_id),
    b.questions.map((q) => q.query_id),
  );

  const c = stratifiedSample(questions, 40, 8);
  assert.notDeepEqual(
    a.questions.map((q) => q.query_id),
    c.questions.map((q) => q.query_id),
  );

  assert.equal(a.questions.length, 40);
  const ids = new Set(a.questions.map((q) => q.query_id));
  assert.equal(ids.size, 40);
});

test('stratifiedSample at size 273 returns full population for any seed', () => {
  const questions = makeSynthetic273();
  const a = stratifiedSample(questions, 273, 7);
  const b = stratifiedSample(questions, 273, 8);
  assert.deepEqual(
    a.questions.map((q) => q.query_id).sort(),
    b.questions.map((q) => q.query_id).sort(),
  );
  assert.equal(a.questions.length, 273);
});

test('mulberry32 is deterministic and in [0, 1)', () => {
  const rng1 = mulberry32(42);
  const rng2 = mulberry32(42);
  const v1 = rng1();
  const v2 = rng2();
  assert.equal(v1, v2);
  assert.ok(v1 >= 0 && v1 < 1);
});

test('randomSeed returns integer in range', () => {
  for (let i = 0; i < 20; i++) {
    const s = randomSeed();
    assert.equal(s, Math.floor(s));
    assert.ok(s >= 1 && s < 2 ** 31 - 1);
  }
});

test('STRATA order', () => {
  assert.deepEqual(STRATA, ['answerable_ok', 'answerable_flagged', 'unanswerable']);
});
