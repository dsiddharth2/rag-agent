import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { renderSummary, writeReport } from '../workflows/rag-eval/report.mjs';

const baseRun = {
  started_at: '2026-11-07T12:00:00.000Z',
  finished_at: '2026-11-07T12:05:00.000Z',
  methods: ['naive', 'hybrid'],
  retrieval_only: false,
  git_commit: 'abc1234',
  sample: {
    size: 40,
    seed: 7,
    counts: { answerable_ok: 25, answerable_flagged: 2, unanswerable: 13 },
    query_ids: ['q1'],
  },
  settings: {
    retrieve_depth: 50,
    context_k: 5,
    generator: { member_name: 'doer', model: 'standard' },
    judge: { member_name: 'reviewer', model: 'premium' },
    index: null,
    retrieval: {},
  },
};

const retrieval = {
  naive: {
    method: 'naive',
    n: 183,
    'hit@1': 0.792,
    'hit@3': 0.891,
    'hit@5': 0.923,
    'hit@10': 0.945,
    mrr: 0.849,
  },
  hybrid: {
    method: 'hybrid',
    n: 183,
    'hit@1': 0.792,
    'hit@3': 0.885,
    'hit@5': 0.918,
    'hit@10': 0.973,
    mrr: 0.85,
  },
};

test('retrieval-only render: retrieval table, retrieval-only line, no Answer quality', () => {
  const md = renderSummary({
    run: { ...baseRun, retrieval_only: true, sample: undefined },
    retrieval,
    answers: null,
    calls: null,
  });
  assert.match(md, /^# RAG eval — 2026-11-07T12:00:00\.000Z/m);
  assert.match(md, /abc1234/);
  assert.match(md, /## Retrieval/);
  assert.match(md, /\| naive \| 183 \|/);
  assert.match(md, /\| hybrid \| 183 \|/);
  assert.match(md, /_Retrieval-only run: answer quality not measured\._/);
  assert.doesNotMatch(md, /Answer quality/);
});

test('full render: methods, seed, strata, not-a-headline, null as en-dash', () => {
  const answers = {
    naive: {
      answerable_ok: {
        n: 2,
        correctness: 1,
        faithfulness: 0.5,
        faithfulness_n: 1,
        zero_claims: 0,
        wrong_refusal_rate: 0,
        context_hit_rate: 0.5,
        abstention_accuracy: null,
        hallucination_rate: null,
        errors: { judge_error: 1 },
      },
      answerable_flagged: {
        n: 0,
        correctness: null,
        faithfulness: null,
        faithfulness_n: 0,
        zero_claims: 0,
        wrong_refusal_rate: null,
        context_hit_rate: null,
        abstention_accuracy: null,
        hallucination_rate: null,
        errors: {},
      },
      unanswerable: {
        n: 1,
        correctness: null,
        faithfulness: null,
        faithfulness_n: 0,
        zero_claims: 0,
        wrong_refusal_rate: null,
        context_hit_rate: null,
        abstention_accuracy: 1,
        hallucination_rate: 0,
        errors: {},
      },
    },
    hybrid: {
      answerable_ok: {
        n: 1,
        correctness: null,
        faithfulness: null,
        faithfulness_n: 0,
        zero_claims: 1,
        wrong_refusal_rate: null,
        context_hit_rate: null,
        abstention_accuracy: null,
        hallucination_rate: null,
        errors: {},
      },
      answerable_flagged: {
        n: 0,
        correctness: null,
        faithfulness: null,
        faithfulness_n: 0,
        zero_claims: 0,
        wrong_refusal_rate: null,
        context_hit_rate: null,
        abstention_accuracy: null,
        hallucination_rate: null,
        errors: {},
      },
      unanswerable: {
        n: 0,
        correctness: null,
        faithfulness: null,
        faithfulness_n: 0,
        zero_claims: 0,
        wrong_refusal_rate: null,
        context_hit_rate: null,
        abstention_accuracy: null,
        hallucination_rate: null,
        errors: {},
      },
    },
  };

  const calls = {
    naive: {
      generation_calls: 3,
      judge_calls: 2,
      total_ms: 4500,
      median_generation_ms: 1200,
    },
    hybrid: {
      generation_calls: 3,
      judge_calls: 1,
      total_ms: 3000,
      median_generation_ms: 900,
    },
  };

  const md = renderSummary({
    run: baseRun,
    retrieval,
    answers,
    calls,
  });

  assert.match(md, /## Answer quality \(sample of 40, seed 7: 25 \/ 2 \/ 13\)/);
  assert.match(md, /not a headline/i);
  assert.match(md, /answerable_ok|Answerable \(OK\)/i);
  assert.match(md, /answerable_flagged|flagged/i);
  assert.match(md, /unanswerable/i);
  assert.match(md, /\| naive \|/);
  assert.match(md, /\| hybrid \|/);
  assert.match(md, /judge_error: 1/);
  assert.match(md, /\| 0 \|/);
  assert.match(md, /–/);
  assert.match(md, /## Calls/);
  assert.match(md, /## Notes/);
  assert.match(md, /doer/);
  assert.match(md, /reviewer/);
  assert.match(md, /standard/);
  assert.match(md, /premium/);
  assert.match(md, /--seed/);
});

test('writeReport writes run.json, results.jsonl, summary.md', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rag-report-'));
  const run = { ...baseRun, retrieval_only: true };
  const rows = [
    { query_id: 'q1', method: 'naive', stratum: 'answerable_ok', scores: {} },
    { query_id: 'q2', method: 'hybrid', stratum: 'unanswerable', scores: {} },
  ];

  const paths = writeReport(dir, {
    run,
    retrieval,
    answers: null,
    calls: null,
    rows,
  });

  assert.equal(paths.runJson, path.join(dir, 'run.json'));
  assert.equal(paths.resultsJsonl, path.join(dir, 'results.jsonl'));
  assert.equal(paths.summaryMd, path.join(dir, 'summary.md'));

  assert.ok(fs.existsSync(paths.runJson));
  assert.ok(fs.existsSync(paths.resultsJsonl));
  assert.ok(fs.existsSync(paths.summaryMd));

  const lines = fs.readFileSync(paths.resultsJsonl, 'utf8').trim().split('\n');
  assert.equal(lines.length, 2);
  assert.deepEqual(JSON.parse(lines[0]), rows[0]);
  assert.deepEqual(JSON.parse(lines[1]), rows[1]);

  const parsed = JSON.parse(fs.readFileSync(paths.runJson, 'utf8'));
  assert.deepEqual(parsed, run);
  assert.match(fs.readFileSync(paths.summaryMd, 'utf8'), /Retrieval-only/);
});
