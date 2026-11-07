import crypto from 'node:crypto';

export const STRATA = ['answerable_ok', 'answerable_flagged', 'unanswerable'];

export function stratumOf(q) {
  if (q.is_impossible) return 'unanswerable';
  if (q.quality_ok === false) return 'answerable_flagged';
  return 'answerable_ok';
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomSeed() {
  return crypto.randomInt(1, 2 ** 31 - 1);
}

export function allocate(sizes, total) {
  if (!Number.isInteger(total)) {
    throw new Error('total must be an integer');
  }
  if (total < 1) {
    throw new Error('total must be at least 1');
  }

  const population = STRATA.reduce((sum, key) => sum + (sizes[key] ?? 0), 0);
  if (total > population) {
    throw new Error('total exceeds population');
  }

  const parts = STRATA.map((key, index) => {
    const size = sizes[key] ?? 0;
    const exact = population === 0 ? 0 : (size / population) * total;
    const floor = Math.floor(exact);
    return { key, index, floor, remainder: exact - floor, cap: size };
  });

  let assigned = parts.reduce((sum, p) => sum + p.floor, 0);
  let leftover = total - assigned;

  const rank = [...parts].sort((a, b) => {
    if (b.remainder !== a.remainder) return b.remainder - a.remainder;
    return a.index - b.index;
  });

  const counts = Object.fromEntries(STRATA.map((key) => [key, 0]));
  for (const p of parts) {
    counts[p.key] = p.floor;
  }

  for (let i = 0; i < leftover; i++) {
    const key = rank[i].key;
    counts[key] += 1;
  }

  for (const key of STRATA) {
    counts[key] = Math.min(counts[key], sizes[key] ?? 0);
  }

  let sum = STRATA.reduce((s, key) => s + counts[key], 0);
  if (sum < total) {
    for (const key of STRATA) {
      while (sum < total && counts[key] < (sizes[key] ?? 0)) {
        counts[key] += 1;
        sum += 1;
      }
    }
  }

  return counts;
}

function fisherYates(items, rng) {
  const arr = items.slice();
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function stratifiedSample(questions, size, seed) {
  const byStratum = Object.fromEntries(STRATA.map((k) => [k, []]));
  for (const q of questions) {
    byStratum[stratumOf(q)].push(q);
  }

  const sizes = Object.fromEntries(
    STRATA.map((k) => [k, byStratum[k].length]),
  );
  const counts = allocate(sizes, size);
  const rng = mulberry32(seed);

  const picked = [];
  for (const key of STRATA) {
    const stratum = byStratum[key]
      .slice()
      .sort((a, b) => String(a.query_id).localeCompare(String(b.query_id)));
    const shuffled = fisherYates(stratum, rng);
    picked.push(...shuffled.slice(0, counts[key]));
  }

  return { seed, counts, questions: picked };
}
