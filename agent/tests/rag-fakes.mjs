export function chunk(docId, n, text = `Technote text for ${docId}`) {
  return {
    rank: n,
    chunk_id: `${docId}#${n - 1}`,
    doc_id: docId,
    score: 1 - n * 0.01,
    text,
  };
}

export function fakeAgent(repliesOrFn) {
  const calls = [];
  const queue = Array.isArray(repliesOrFn) ? [...repliesOrFn] : null;

  async function agent(prompt, options) {
    calls.push({ prompt, ...options });
    if (typeof repliesOrFn === 'function') {
      return repliesOrFn({ prompt, ...options }, calls);
    }
    if (queue.length === 0) {
      throw new Error('fakeAgent: no more queued replies');
    }
    return queue.shift();
  }

  return { agent, calls };
}

export function fakeCommand(resultOrFn) {
  const calls = [];

  async function command(cmd, options) {
    calls.push({ cmd, ...options });
    const result = typeof resultOrFn === 'function' ? resultOrFn({ cmd, ...options }, calls) : resultOrFn;
    if (options?.failSoft) return result;
    if (result && typeof result === 'object' && result.ok === false) {
      throw new Error(result.error ?? 'command failed');
    }
    if (result && typeof result === 'object' && 'output' in result) {
      return result.output;
    }
    return result;
  }

  return { command, calls };
}
