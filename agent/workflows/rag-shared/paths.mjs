import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const REPO_ROOT = path.resolve(__dirname, '../../..');

export const TEST_QUERIES_PATH = path.join(
  REPO_ROOT,
  'RAG_Dataset/data/processed/queries_test.jsonl',
);
export const QRELS_PATH = path.join(
  REPO_ROOT,
  'RAG_Dataset/data/processed/qrels.jsonl',
);
export const INDEX_MANIFEST_PATH = path.join(
  REPO_ROOT,
  'rag/indexes/manifest.json',
);
export const REPORTS_DIR = path.join(REPO_ROOT, 'agent/evals/reports/rag');

export function pythonBin(platform = process.platform, root = REPO_ROOT) {
  if (platform === 'win32') {
    return path.win32.join(root, '.venv', 'Scripts', 'python.exe');
  }
  return path.posix.join(root, '.venv', 'bin', 'python');
}

export function quotePath(p) {
  if (p.includes("'") || p.includes('\n')) {
    throw new Error('path contains unsafe characters');
  }
  return `'${p}'`;
}

export function pythonModuleCommand(
  moduleName,
  args = [],
  { platform = process.platform, root = REPO_ROOT } = {},
) {
  const python = pythonBin(platform, root);
  const quotedRoot = quotePath(root);
  const quotedPython = quotePath(python);
  const argStr = args.length ? ` ${args.join(' ')}` : '';
  if (platform === 'win32') {
    return `Set-Location -LiteralPath ${quotedRoot}; & ${quotedPython} -m ${moduleName}${argStr}`;
  }
  return `cd ${quotedRoot} && ${quotedPython} -m ${moduleName}${argStr}`;
}
