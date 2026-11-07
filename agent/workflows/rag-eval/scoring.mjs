import { STRATA, stratumOf } from './sampling.mjs';

export const CORRECTNESS_SCORE = { full: 1, partial: 0.5, none: 0 };

function meanOf(values) {
  const nums = values.filter((v) => v != null);
  if (nums.length === 0) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function boolMean(values) {
  return meanOf(values.map((v) => (v == null ? null : v ? 1 : 0)));
}

export function scoreRow({ question, goldDocIds, contextDocIds, generation, judgment }) {
  const stratum = stratumOf(question);
  const out = {
    stratum,
    errors: [],
    abstained: null,
    context_hit: null,
    correctness: null,
    faithfulness: null,
    zero_claims: false,
    wrong_refusal: null,
    abstention_correct: null,
  };

  if (stratum !== 'unanswerable') {
    const gold = new Set(goldDocIds ?? []);
    out.context_hit = (contextDocIds ?? []).some((id) => gold.has(id));
  }

  if (generation?.error) {
    out.errors.push(generation.error);
    return out;
  }

  out.abstained = generation?.abstained ?? null;

  if (stratum === 'unanswerable') {
    out.abstention_correct = out.abstained === true;
    return out;
  }

  out.wrong_refusal = out.abstained === true;

  if (out.abstained) {
    out.correctness = 0;
    return out;
  }

  if (judgment?.error) {
    out.errors.push(judgment.error);
    return out;
  }

  const level = judgment?.correctness;
  out.correctness = CORRECTNESS_SCORE[level] ?? null;

  const claims = judgment?.claims ?? [];
  if (claims.length === 0) {
    out.zero_claims = true;
    out.faithfulness = null;
  } else {
    const supported = claims.filter((c) => c.supported).length;
    out.faithfulness = supported / claims.length;
  }

  return out;
}

function emptySummary() {
  return {
    n: 0,
    errors: {},
    correctness: null,
    faithfulness: null,
    faithfulness_n: 0,
    zero_claims: 0,
    wrong_refusal_rate: null,
    context_hit_rate: null,
    abstention_accuracy: null,
    hallucination_rate: null,
  };
}

function tallyErrors(errorLists) {
  const counts = {};
  for (const list of errorLists) {
    for (const name of list) {
      counts[name] = (counts[name] ?? 0) + 1;
    }
  }
  return counts;
}

function summarizeScores(scoreList) {
  const summary = emptySummary();
  summary.n = scoreList.length;
  summary.errors = tallyErrors(scoreList.map((s) => s.errors));
  summary.correctness = meanOf(scoreList.map((s) => s.correctness));
  const faithVals = scoreList.map((s) => s.faithfulness);
  summary.faithfulness = meanOf(faithVals);
  summary.faithfulness_n = faithVals.filter((v) => v != null).length;
  summary.zero_claims = scoreList.filter((s) => s.zero_claims === true).length;
  summary.wrong_refusal_rate = boolMean(scoreList.map((s) => s.wrong_refusal));
  summary.context_hit_rate = boolMean(scoreList.map((s) => s.context_hit));
  summary.abstention_accuracy = boolMean(scoreList.map((s) => s.abstention_correct));
  summary.hallucination_rate =
    summary.abstention_accuracy == null ? null : 1 - summary.abstention_accuracy;
  return summary;
}

export function aggregate(rows) {
  const byMethod = {};

  for (const row of rows) {
    const { method, scores } = row;
    if (!byMethod[method]) {
      byMethod[method] = Object.fromEntries(STRATA.map((k) => [k, []]));
    }
    const key = scores.stratum;
    if (byMethod[method][key]) {
      byMethod[method][key].push(scores);
    }
  }

  const result = {};
  for (const [method, strata] of Object.entries(byMethod)) {
    result[method] = {};
    for (const key of STRATA) {
      const list = strata[key] ?? [];
      result[method][key] = list.length === 0 ? emptySummary() : summarizeScores(list);
    }
  }

  return result;
}
