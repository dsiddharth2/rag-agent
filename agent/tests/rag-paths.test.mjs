import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {
  REPO_ROOT,
  pythonBin,
  quotePath,
  pythonModuleCommand,
} from '../workflows/rag-shared/paths.mjs';

test('REPO_ROOT contains a RAG_Dataset directory', () => {
  const ragDataset = path.join(REPO_ROOT, 'RAG_Dataset');
  assert.ok(fs.existsSync(ragDataset), `expected ${ragDataset} to exist`);
  assert.ok(fs.statSync(ragDataset).isDirectory());
});

test('pythonBin resolves venv python per platform', () => {
  assert.equal(pythonBin('linux', '/r'), '/r/.venv/bin/python');
  assert.ok(
    pythonBin('win32', 'C:\\r').endsWith('.venv\\Scripts\\python.exe'),
  );
});

test('pythonModuleCommand builds POSIX shell command', () => {
  assert.equal(
    pythonModuleCommand('rag.retrieve', ['--method', 'naive'], {
      platform: 'linux',
      root: '/r',
    }),
    "cd '/r' && '/r/.venv/bin/python' -m rag.retrieve --method naive",
  );
});

test('pythonModuleCommand builds PowerShell command on win32', () => {
  assert.equal(
    pythonModuleCommand('rag.retrieve', [], { platform: 'win32', root: 'C:\\r' }),
    "Set-Location -LiteralPath 'C:\\r'; & 'C:\\r\\.venv\\Scripts\\python.exe' -m rag.retrieve",
  );
});

test('quotePath wraps paths and rejects unsafe characters', () => {
  assert.equal(quotePath('/a b $HOME'), "'/a b $HOME'");
  assert.throws(() => quotePath("a'b"));
  assert.throws(() => quotePath('/bad\npath'));
});
