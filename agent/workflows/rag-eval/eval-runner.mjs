import fs from 'node:fs';
import path from 'node:path';

import { CONTEXT_K, GENERATOR, generateAnswer } from '../rag-shared/generation.mjs';
import {
  RAG_METHODS,
  RETRIEVE_DEPTH,
  readJsonl,
} from '../rag-shared/retrieval-client.mjs';
import { JUDGE, judgeAnswer } from './judge.mjs';
import { writeReport } from './report.mjs';
import { randomSeed, stratumOf, stratifiedSample } from './sampling.mjs';
import { aggregate, scoreRow } from './scoring.mjs';

export function loadQrels(file) {
  const map = new Map();
  for (const row of readJsonl(file)) {
    const queryId = row.query_id;
    const docId = row.doc_id;
    if (!queryId || !docId) continue;
    const list = map.get(queryId) ?? [];
    list.push(docId);
    map.set(queryId, list);
  }
  return map;
}

function validateMethods(methods) {
  for (const method of methods) {
    if (!RAG_METHODS.includes(method)) {
      throw new Error(`unknown method: ${method}`);
    }
  }
}

function isFleetStop(err) {
  return err?.code === 'CANCELLED' || err?.code === 'BUDGET_EXCEEDED';
}

async function safeGenerateAnswer(params) {
  const t0 = Date.now();
  try {
    const generation = await generateAnswer(params);
    return { generation, generation_ms: Date.now() - t0 };
  } catch (err) {
    if (isFleetStop(err)) throw err;
    return {
      generation: {
        error: 'fleet_error',
        detail: err?.message ?? String(err),
        calls: 1,
      },
      generation_ms: Date.now() - t0,
    };
  }
}

async function safeJudgeAnswer(params) {
  const t0 = Date.now();
  try {
    const judgment = await judgeAnswer(params);
    return { judgment, judge_ms: Date.now() - t0 };
  } catch (err) {
    if (isFleetStop(err)) throw err;
    return {
      judgment: {
        error: 'fleet_error',
        detail: err?.message ?? String(err),
        calls: 1,
      },
      judge_ms: Date.now() - t0,
    };
  }
}

function contextFromChunks(chunks) {
  const contextChunks = (chunks ?? []).slice(0, CONTEXT_K);
  const context_chunk_ids = contextChunks.map((c) => c.chunk_id);
  const context_doc_ids = [];
  const seen = new Set();
  for (const c of contextChunks) {
    if (seen.has(c.doc_id)) continue;
    seen.add(c.doc_id);
    context_doc_ids.push(c.doc_id);
  }
  return { contextChunks, context_chunk_ids, context_doc_ids };
}

function median(values) {
  const nums = values.filter((v) => typeof v === 'number' && !Number.isNaN(v));
  if (nums.length === 0) return null;
  const sorted = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 1) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function summarizeCalls(rows) {
  let generation_calls = 0;
  let judge_calls = 0;
  let total_ms = 0;
  const genTimes = [];

  for (const row of rows) {
    total_ms += (row.generation_ms ?? 0) + (row.judge_ms ?? 0);
    if (row.generation?.calls != null) {
      generation_calls += row.generation.calls;
    }
    if (row.judgment?.calls != null) {
      judge_calls += row.judgment.calls;
    }
    if (row.generation_ms != null) genTimes.push(row.generation_ms);
  }

  return {
    generation_calls,
    judge_calls,
    total_ms,
    median_generation_ms: median(genTimes),
  };
}

export async function runEval({ agent, log, retrieval, options, now = Date.now() }) {
  const {
    methods,
    sampleSize,
    seed: seedOpt,
    retrievalOnly,
    reportDir,
    testQueriesPath,
    qrelsPath,
    gitCommit,
    indexManifest,
  } = options;

  validateMethods(methods);
  const questions = readJsonl(testQueriesPath);
  const qrels = loadQrels(qrelsPath);
  const started_at = new Date(now).toISOString();

  fs.mkdirSync(reportDir, { recursive: true });
  fs.mkdirSync(path.join(reportDir, 'retrieval'), { recursive: true });

  const retrievalSettings = {};
  const retrievalMetrics = {};
  const chunksByMethod = {};

  for (const method of methods) {
    log(`retrieval batch method=${method}`);
    const outFile = path.join(reportDir, 'retrieval', `${method}.jsonl`);
    const rows = questions.map((q) => ({ query_id: q.query_id, text: q.text }));
    const { summary, chunksById } = await retrieval.batch({ method, rows, outFile });
    retrievalSettings[method] = summary;
    retrievalMetrics[method] = await retrieval.metrics({
      resultsFile: outFile,
      qrelsFile: qrelsPath,
    });
    chunksByMethod[method] = chunksById;
  }

  if (retrievalOnly) {
    const run = {
      started_at,
      finished_at: new Date().toISOString(),
      methods,
      retrieval_only: true,
      git_commit: gitCommit ?? null,
      settings: {
        retrieve_depth: RETRIEVE_DEPTH,
        context_k: CONTEXT_K,
        generator: GENERATOR,
        judge: JUDGE,
        index: indexManifest ?? null,
        retrieval: retrievalSettings,
      },
    };
    writeReport(reportDir, {
      run,
      retrieval: retrievalMetrics,
      answers: null,
      calls: null,
      rows: [],
    });
    return { reportDir, retrieval: retrievalMetrics };
  }

  const seed = seedOpt ?? randomSeed();
  const sample = stratifiedSample(questions, sampleSize, seed);
  const rows = [];
  const calls = {};

  for (const method of methods) {
    const chunksById = chunksByMethod[method];
    const methodRows = [];

    for (const question of sample.questions) {
      const query_id = question.query_id;
      const goldDocIds = qrels.get(query_id) ?? [];
      const { contextChunks, context_chunk_ids, context_doc_ids } = contextFromChunks(
        chunksById.get(query_id),
      );

      const { generation, generation_ms } = await safeGenerateAnswer({
        agent,
        question: question.text,
        chunks: contextChunks,
      });

      let judgment = null;
      let judge_ms = null;
      const answerable = !question.is_impossible;
      const genOk = !generation?.error && generation?.abstained !== true;
      if (answerable && genOk) {
        const judged = await safeJudgeAnswer({
          agent,
          question: question.text,
          chunks: contextChunks,
          answer: generation.answer,
          goldAnswer: question.ground_truth_answer ?? '',
        });
        judgment = judged.judgment;
        judge_ms = judged.judge_ms;
      }

      const scores = scoreRow({
        question,
        goldDocIds,
        contextDocIds: context_doc_ids,
        generation,
        judgment,
      });

      const row = {
        query_id,
        method,
        stratum: scores.stratum,
        context_chunk_ids,
        context_doc_ids,
        gold_doc_ids: goldDocIds,
        generation,
        generation_ms,
        judgment,
        judge_ms,
        scores,
      };
      methodRows.push(row);
      rows.push(row);
    }

    calls[method] = summarizeCalls(methodRows);
  }

  const answers = aggregate(rows);
  const finished_at = new Date().toISOString();
  const run = {
    started_at,
    finished_at,
    methods,
    retrieval_only: false,
    git_commit: gitCommit ?? null,
    sample: {
      size: sampleSize,
      seed,
      counts: sample.counts,
      query_ids: sample.questions.map((q) => q.query_id),
    },
    settings: {
      retrieve_depth: RETRIEVE_DEPTH,
      context_k: CONTEXT_K,
      generator: GENERATOR,
      judge: JUDGE,
      index: indexManifest ?? null,
      retrieval: retrievalSettings,
    },
  };

  const paths = writeReport(reportDir, {
    run,
    retrieval: retrievalMetrics,
    answers,
    calls,
    rows,
  });

  return { reportDir, retrieval: retrievalMetrics, answers, calls, summaryMd: paths.summaryMd };
}
