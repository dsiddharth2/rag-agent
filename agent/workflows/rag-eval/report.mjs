import fs from 'node:fs';
import path from 'node:path';
import { STRATA } from './sampling.mjs';

export function formatNum(value) {
  if (value == null) return '–';
  return Number(value).toFixed(3);
}

export function formatErrors(errors) {
  const entries = Object.entries(errors ?? {})
    .filter(([, count]) => count > 0)
    .sort(([a], [b]) => a.localeCompare(b));
  if (entries.length === 0) return '0';
  return entries.map(([name, count]) => `${name}: ${count}`).join(', ');
}

function msToSeconds(ms) {
  if (ms == null) return '–';
  return (ms / 1000).toFixed(3);
}

function faithfulnessCell(summary) {
  return `${formatNum(summary.faithfulness)} (${summary.faithfulness_n ?? 0})`;
}

const STRATUM_HEADING = {
  answerable_ok: '### answerable_ok',
  answerable_flagged: '### answerable_flagged (not a headline number)',
  unanswerable: '### unanswerable',
};

function retrievalTable(methods, retrieval) {
  const lines = [
    '## Retrieval (all answerable test questions, no LLM)',
    '',
    '| Method | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
  ];
  for (const method of methods) {
    const m = retrieval[method];
    if (!m) continue;
    lines.push(
      `| ${method} | ${m.n} | ${formatNum(m['hit@1'])} | ${formatNum(m['hit@3'])} | ${formatNum(m['hit@5'])} | ${formatNum(m['hit@10'])} | ${formatNum(m.mrr)} |`,
    );
  }
  return lines.join('\n');
}

function answerableRow(method, summary) {
  return `| ${method} | ${summary.n} | ${formatNum(summary.correctness)} | ${faithfulnessCell(summary)} | ${summary.zero_claims ?? 0} | ${formatNum(summary.wrong_refusal_rate)} | ${formatNum(summary.context_hit_rate)} | ${formatErrors(summary.errors)} |`;
}

function unanswerableRow(method, summary) {
  return `| ${method} | ${summary.n} | ${formatNum(summary.abstention_accuracy)} | ${formatNum(summary.hallucination_rate)} | ${formatErrors(summary.errors)} |`;
}

function answerQualitySection(run, methods, answers) {
  const sample = run.sample ?? {};
  const counts = sample.counts ?? {};
  const parts = STRATA.map((k) => counts[k] ?? 0);
  const lines = [
    `## Answer quality (sample of ${sample.size ?? 0}, seed ${sample.seed ?? '–'}: ${parts.join(' / ')})`,
    '',
  ];

  for (const stratum of STRATA) {
    lines.push(STRATUM_HEADING[stratum], '');
    if (stratum === 'unanswerable') {
      lines.push(
        '| Method | n | Abstention accuracy | Hallucination rate | Errors |',
        '| --- | ---: | ---: | ---: | --- |',
      );
      for (const method of methods) {
        const summary = answers?.[method]?.[stratum];
        if (summary) lines.push(unanswerableRow(method, summary));
      }
    } else {
      lines.push(
        '| Method | n | Correctness | Faithfulness (n) | Zero-claim answers | Wrong refusals | Context hit | Errors |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
      );
      for (const method of methods) {
        const summary = answers?.[method]?.[stratum];
        if (summary) lines.push(answerableRow(method, summary));
      }
    }
    lines.push('');
  }

  return lines.join('\n').trimEnd();
}

function callsSection(methods, calls) {
  const lines = [
    '## Calls',
    '',
    '| Method | Generation calls | Judge calls | Total time (s) | Median generation (s) |',
    '| --- | ---: | ---: | ---: | ---: |',
  ];
  for (const method of methods) {
    const c = calls[method];
    if (!c) continue;
    lines.push(
      `| ${method} | ${c.generation_calls ?? 0} | ${c.judge_calls ?? 0} | ${msToSeconds(c.total_ms)} | ${msToSeconds(c.median_generation_ms)} |`,
    );
  }
  return lines.join('\n');
}

function notesSection(run) {
  const gen = run.settings?.generator ?? {};
  const judge = run.settings?.judge ?? {};
  return [
    '## Notes',
    '',
    `- Generator: \`${gen.member_name ?? 'doer'}\` (${gen.model ?? 'standard'} tier). Judge: \`${judge.member_name ?? 'reviewer'}\` (${judge.model ?? 'premium'} tier).`,
    '- Generator and judge use the same provider family (Claude); scores may be optimistically biased.',
    '- Re-run with the same `--seed` for like-for-like comparisons across methods or retrieval settings.',
  ].join('\n');
}

export function renderSummary({ run, retrieval, answers, calls }) {
  const methods = run.methods ?? [];
  const parts = [
    `# RAG eval — ${run.started_at}`,
    '',
    `Methods: ${methods.join(', ')}. Git commit: ${run.git_commit ?? '–'}.`,
    '',
    retrievalTable(methods, retrieval),
    '',
  ];

  if (answers == null) {
    parts.push('_Retrieval-only run: answer quality not measured._', '');
  } else {
    parts.push(answerQualitySection(run, methods, answers), '');
  }

  if (calls != null) {
    parts.push(callsSection(methods, calls), '');
  }

  parts.push(notesSection(run), '');
  return parts.join('\n');
}

export function writeReport(dir, { run, retrieval, answers, calls, rows }) {
  fs.mkdirSync(dir, { recursive: true });
  const runJson = path.join(dir, 'run.json');
  const resultsJsonl = path.join(dir, 'results.jsonl');
  const summaryMd = path.join(dir, 'summary.md');

  fs.writeFileSync(runJson, `${JSON.stringify(run, null, 2)}\n`);
  const body = rows.map((row) => JSON.stringify(row)).join('\n');
  fs.writeFileSync(resultsJsonl, body ? `${body}\n` : '');
  fs.writeFileSync(summaryMd, renderSummary({ run, retrieval, answers, calls }));

  return { runJson, resultsJsonl, summaryMd };
}
