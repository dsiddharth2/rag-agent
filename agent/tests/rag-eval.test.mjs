import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { parseCli } from '../workflows/rag-eval/main.mjs';
import { main as ragEvalMain } from '../workflows/rag-eval/rag-eval.js';
import { loadQrels, runEval } from '../workflows/rag-eval/eval-runner.mjs';
import { chunk, fakeAgent } from './rag-fakes.mjs';

const genOk = JSON.stringify({
  answer: 'Short answer.',
  abstained: false,
  cited_doc_ids: ['g1'],
});

const genAbstain = JSON.stringify({
  answer: 'Cannot answer from context.',
  abstained: true,
  cited_doc_ids: [],
});

const judgeQ1 = JSON.stringify({
  claims: [
    { text: 'claim a', supported: true },
    { text: 'claim b', supported: false },
  ],
  correctness: 'full',
  reason: 'ok',
});

function writeJsonl(file, rows) {
  fs.writeFileSync(file, `${rows.map((r) => JSON.stringify(r)).join('\n')}\n`);
}

function makeFixtureDir() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rag-eval-'));
  const queries = [
    {
      query_id: 'q1',
      text: 'Question one?',
      ground_truth_answer: 'Gold answer one.',
      is_impossible: false,
      quality_ok: true,
    },
    {
      query_id: 'q2',
      text: 'Question two?',
      ground_truth_answer: 'Gold answer two.',
      is_impossible: false,
      quality_ok: true,
    },
    {
      query_id: 'q3',
      text: 'Question three?',
      ground_truth_answer: '',
      is_impossible: true,
      quality_ok: true,
    },
  ];
  const testQueriesPath = path.join(dir, 'queries.jsonl');
  const qrelsPath = path.join(dir, 'qrels.jsonl');
  writeJsonl(testQueriesPath, queries);
  writeJsonl(qrelsPath, [
    { query_id: 'q1', doc_id: 'g1', relevance: 1 },
    { query_id: 'q2', doc_id: 'g2', relevance: 1 },
  ]);
  return { dir, testQueriesPath, qrelsPath, queries };
}

function makeFakeRetrieval(reportDir) {
  const batchCalls = [];

  async function batch({ method, rows, outFile }) {
    batchCalls.push({ method, queryIds: rows.map((r) => r.query_id), outFile });
    fs.mkdirSync(path.dirname(outFile), { recursive: true });
    const outRows = rows.map((row) => {
      const chunks =
        row.query_id === 'q1'
          ? [chunk('g1', 1), chunk('other', 2)]
          : [chunk('miss', 1), chunk('miss2', 2)];
      return { query_id: row.query_id, chunks };
    });
    writeJsonl(outFile, outRows);
    return {
      summary: {
        method,
        depth: 50,
        embedding_model: 'mini',
        ...(method === 'hybrid' ? { rerank_model: 'cross' } : {}),
      },
      chunksById: new Map(outRows.map((r) => [r.query_id, r.chunks])),
    };
  }

  async function metrics() {
    return {
      method: 'metrics',
      n: 2,
      'hit@1': 0.5,
      'hit@3': 0.5,
      'hit@5': 0.5,
      'hit@10': 0.5,
      mrr: 0.5,
    };
  }

  return { batch, metrics, batchCalls };
}

function makeDoerReviewerAgent({ fleetErrorOnFirst = false } = {}) {
  let fleetErrorThrown = false;
  return fakeAgent(({ prompt, member_name }) => {
    if (member_name === 'doer') {
      if (fleetErrorOnFirst && !fleetErrorThrown) {
        fleetErrorThrown = true;
        throw new Error('fleet down');
      }
      if (prompt.includes('Question three?')) return genAbstain;
      return genOk;
    }
    if (member_name === 'reviewer') {
      if (prompt.includes('Question one?')) return judgeQ1;
      return 'not json';
    }
    throw new Error('unexpected member');
  });
}

test('rag-eval body main accepts engine-shaped context.args', async () => {
  const { dir, testQueriesPath, qrelsPath } = makeFixtureDir();
  const reportDir = path.join(dir, 'report-engine-args');

  const command = async (cmd) => {
    if (cmd.includes('rag.retrieve') && cmd.includes('--out')) {
      const outMatch = cmd.match(/--out\s+'([^']+)'/) ?? cmd.match(/--out\s+"([^"]+)"/);
      const outFile = outMatch?.[1];
      if (outFile) {
        fs.mkdirSync(path.dirname(outFile), { recursive: true });
        writeJsonl(outFile, [
          { query_id: 'q1', chunks: [chunk('g1', 1)] },
          { query_id: 'q2', chunks: [] },
          { query_id: 'q3', chunks: [] },
        ]);
      }
      return {
        ok: true,
        output: JSON.stringify({ method: 'naive', depth: 50, embedding_model: 'mini' }),
      };
    }
    if (cmd.includes('retrieval_metrics')) {
      return { ok: true, output: JSON.stringify({ method: 'naive', n: 2, mrr: 0 }) };
    }
    return { ok: false, error: `unexpected command: ${cmd}` };
  };

  const engineArgs = {
    fleetApi: {},
    workspace: '/tmp/ws',
    signal: undefined,
    reportPhase: () => {},
    options: {
      methods: ['naive'],
      sampleSize: 3,
      seed: 7,
      retrievalOnly: true,
      reportDir,
      testQueriesPath,
      qrelsPath,
      gitCommit: null,
      indexManifest: null,
    },
  };

  await assert.doesNotReject(async () =>
    ragEvalMain({
      command,
      agent: async () => {
        throw new Error('agent should not run');
      },
      log: () => {},
      args: engineArgs,
    }),
  );

  assert.ok(fs.existsSync(path.join(reportDir, 'summary.md')));
});

test('parseCli defaults and validation', () => {
  assert.deepEqual(parseCli([]), {
    methods: ['naive', 'hybrid'],
    sampleSize: 40,
    seed: null,
    retrievalOnly: false,
    reportDir: undefined,
  });

  assert.deepEqual(parseCli(['--methods', 'naive', '--sample', '10', '--seed', '7', '--retrieval-only']), {
    methods: ['naive'],
    sampleSize: 10,
    seed: 7,
    retrievalOnly: true,
    reportDir: undefined,
  });

  assert.throws(() => parseCli(['--sample', '0']), /sample/i);
  assert.throws(() => parseCli(['--seed', 'x']), /seed/i);
});

test('loadQrels groups doc ids', () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'qrels-')), 'q.jsonl');
  writeJsonl(file, [
    { query_id: 'q1', doc_id: 'a' },
    { query_id: 'q1', doc_id: 'b' },
    { query_id: 'q2', doc_id: 'c' },
  ]);
  const map = loadQrels(file);
  assert.deepEqual(map.get('q1'), ['a', 'b']);
  assert.deepEqual(map.get('q2'), ['c']);
});

test('runEval end to end with fake retrieval and agent', async () => {
  const { dir, testQueriesPath, qrelsPath } = makeFixtureDir();
  const reportDir = path.join(dir, 'report');
  const retrieval = makeFakeRetrieval(reportDir);
  const { agent, calls: agentCalls } = makeDoerReviewerAgent();
  const indexManifest = { embedding_model: 'mini', n_chunks: 10 };

  const result = await runEval({
    agent,
    log: () => {},
    retrieval,
    options: {
      methods: ['naive', 'hybrid'],
      sampleSize: 3,
      seed: 7,
      retrievalOnly: false,
      reportDir,
      testQueriesPath,
      qrelsPath,
      gitCommit: 'deadbeef',
      indexManifest,
    },
    now: Date.parse('2026-11-07T12:00:00.000Z'),
  });

  assert.equal(retrieval.batchCalls.length, 2);
  for (const call of retrieval.batchCalls) {
    assert.deepEqual(call.queryIds, ['q1', 'q2', 'q3']);
  }

  for (const method of ['naive', 'hybrid']) {
    const ok = result.answers[method].answerable_ok;
    assert.equal(ok.n, 2);
    assert.equal(ok.correctness, 1);
    assert.equal(ok.faithfulness, 0.5);
    assert.equal(ok.errors.judge_error, 1);
    assert.equal(ok.context_hit_rate, 0.5);
    assert.equal(result.answers[method].unanswerable.abstention_accuracy, 1);
  }

  const reviewerCalls = agentCalls.filter((c) => c.member_name === 'reviewer');
  assert.equal(reviewerCalls.length, 6);
  assert.equal(result.calls.naive.judge_calls, 3);
  assert.equal(result.calls.hybrid.judge_calls, 3);

  assert.ok(fs.existsSync(path.join(reportDir, 'summary.md')));
  assert.ok(fs.existsSync(path.join(reportDir, 'run.json')));
  assert.ok(fs.existsSync(path.join(reportDir, 'results.jsonl')));
  assert.ok(fs.existsSync(path.join(reportDir, 'retrieval', 'naive.jsonl')));

  const lines = fs.readFileSync(path.join(reportDir, 'results.jsonl'), 'utf8').trim().split('\n');
  assert.equal(lines.length, 6);

  const run = JSON.parse(fs.readFileSync(path.join(reportDir, 'run.json'), 'utf8'));
  assert.equal(run.sample.seed, 7);
  assert.equal(run.sample.query_ids.length, 3);
  assert.deepEqual(run.settings.index, indexManifest);
  assert.equal(run.settings.retrieval.hybrid.rerank_model, 'cross');
});

test('runEval retrieval-only skips agent', async () => {
  const { dir, testQueriesPath, qrelsPath } = makeFixtureDir();
  const reportDir = path.join(dir, 'report-ro');
  const retrieval = makeFakeRetrieval(reportDir);
  let agentCalled = false;

  const result = await runEval({
    agent: () => {
      agentCalled = true;
      throw new Error('should not run');
    },
    log: () => {},
    retrieval,
    options: {
      methods: ['naive'],
      sampleSize: 3,
      seed: 7,
      retrievalOnly: true,
      reportDir,
      testQueriesPath,
      qrelsPath,
      gitCommit: null,
      indexManifest: null,
    },
  });

  assert.equal(agentCalled, false);
  assert.equal(result.answers, undefined);
  const md = fs.readFileSync(path.join(reportDir, 'summary.md'), 'utf8');
  assert.match(md, /Retrieval-only/);
});

test('runEval fleet_error on generation continues', async () => {
  const { dir, testQueriesPath, qrelsPath } = makeFixtureDir();
  const reportDir = path.join(dir, 'report-fe');
  const retrieval = makeFakeRetrieval(reportDir);
  const { agent } = makeDoerReviewerAgent({ fleetErrorOnFirst: true });

  const result = await runEval({
    agent,
    log: () => {},
    retrieval,
    options: {
      methods: ['naive'],
      sampleSize: 3,
      seed: 7,
      retrievalOnly: false,
      reportDir,
      testQueriesPath,
      qrelsPath,
      gitCommit: null,
      indexManifest: null,
    },
  });

  assert.ok(result.answers);
  const rows = fs
    .readFileSync(path.join(reportDir, 'results.jsonl'), 'utf8')
    .trim()
    .split('\n')
    .map((l) => JSON.parse(l));
  const errRow = rows.find((r) => r.generation?.error === 'fleet_error');
  assert.ok(errRow);
});

test('runEval rethrows CANCELLED', async () => {
  const { dir, testQueriesPath, qrelsPath } = makeFixtureDir();
  const reportDir = path.join(dir, 'report-cancel');
  const retrieval = makeFakeRetrieval(reportDir);
  const err = new Error('stopped');
  err.code = 'CANCELLED';

  await assert.rejects(
    () =>
      runEval({
        agent: () => {
          throw err;
        },
        log: () => {},
        retrieval,
        options: {
          methods: ['naive'],
          sampleSize: 1,
          seed: 7,
          retrievalOnly: false,
          reportDir,
          testQueriesPath,
          qrelsPath,
          gitCommit: null,
          indexManifest: null,
        },
      }),
    /stopped/,
  );
});
