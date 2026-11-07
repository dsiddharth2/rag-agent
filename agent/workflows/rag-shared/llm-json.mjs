export function toText(reply) {
  if (typeof reply === 'string') return reply;
  if (reply && typeof reply.text === 'string') return reply.text;
  if (reply && Array.isArray(reply.content)) {
    return reply.content
      .map((block) => (block && typeof block.text === 'string' ? block.text : ''))
      .filter(Boolean)
      .join('\n');
  }
  if (reply && typeof reply.output === 'string') return reply.output;
  return '';
}

export function extractJsonObject(text) {
  const stripped = text.replace(/```(?:json)?\s*\n?/gi, '').replace(/```/g, '');

  const parseable = [];
  let i = 0;
  while (i < stripped.length) {
    if (stripped[i] !== '{') {
      i += 1;
      continue;
    }
    const start = i;
    let depth = 0;
    let inString = false;
    let escape = false;
    let closedAt = -1;

    for (let j = i; j < stripped.length; j += 1) {
      const c = stripped[j];
      if (inString) {
        if (escape) escape = false;
        else if (c === '\\') escape = true;
        else if (c === '"') inString = false;
        continue;
      }
      if (c === '"') {
        inString = true;
        continue;
      }
      if (c === '{') depth += 1;
      else if (c === '}') {
        depth -= 1;
        if (depth === 0) {
          closedAt = j;
          break;
        }
      }
    }

    if (closedAt === -1) break;

    const candidate = stripped.slice(start, closedAt + 1);
    try {
      parseable.push(JSON.parse(candidate));
    } catch {
      // skip unparseable spans
    }
    i = closedAt + 1;
  }

  if (parseable.length === 0) return null;
  return parseable[parseable.length - 1];
}
