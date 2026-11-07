// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Delete specs in modules/analysis/test that do not exercise analysis-owned code (directly or via
// local test helpers): they tested forked luma.gl code that now comes from npm.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';

const require = createRequire(path.resolve('package.json'));
const ts = require('typescript');
const TEST = 'modules/analysis/test';
const SRC = path.resolve('modules/analysis/src');
const walk = d =>
  fs.readdirSync(d, {withFileTypes: true}).flatMap(e => {
    const f = path.join(d, e.name);
    return e.isDirectory() ? walk(f) : /\.(ts|tsx)$/.test(e.name) ? [f] : [];
  });
const resolve = (from, spec) => {
  const b = path.normalize(path.join(path.dirname(from), spec.replace(/\?.*$/, ''))).replace(/\.js$/, '');
  return [b + '.ts', b + '.tsx', b + '/index.ts', b].find(c => fs.existsSync(c) && fs.statSync(c).isFile());
};
const memo = new Map();
const usesAnalysis = (file, stack = new Set()) => {
  if (memo.has(file)) return memo.get(file);
  if (stack.has(file)) return false;
  stack.add(file);
  const text = fs.readFileSync(file, 'utf8');
  let hit = false;
  for (const {fileName} of ts.preProcessFile(text, true, true).importedFiles) {
    if (fileName.startsWith('@luma-contributors/')) { hit = true; break; }
    if (!fileName.startsWith('.')) continue;
    const target = resolve(file, fileName);
    if (!target) continue;
    if (path.resolve(target).startsWith(SRC)) { hit = true; break; }
    if (target.startsWith(TEST) && /\.tsx?$/.test(target) && usesAnalysis(target, stack)) { hit = true; break; }
  }
  // tests that read analysis sources or example data by path also count
  if (!hit && /modules\/analysis\/src|spatial-analysis-showcase|docs\/lens-records/.test(text)) hit = true;
  memo.set(file, hit);
  return hit;
};
// Analysis-owned test directories hold oracle, fixture and layering checks that need no source import.
const ANALYSIS_TEST_DIRECTORIES = /^modules\/analysis\/test\/(analysis-core|computation-record|gpu-network|gpu-spatial-analysis|gpu-terrain|geospatial-expansion|oracles)\//;
const specs = walk(TEST).filter(f => /\.spec\.tsx?$/.test(f) && !ANALYSIS_TEST_DIRECTORIES.test(f));
const removed = specs.filter(f => !usesAnalysis(f));
removed.forEach(f => fs.rmSync(f));
// remove helper .ts files no longer imported by any remaining test file
let removedHelpers = 0;
for (let pass = 0; pass < 5; pass++) {
  const files = walk(TEST);
  const imported = new Set();
  for (const f of files) {
    for (const {fileName} of ts.preProcessFile(fs.readFileSync(f, 'utf8'), true, true).importedFiles) {
      if (fileName.startsWith('.')) { const t = resolve(f, fileName); if (t) imported.add(path.resolve(t)); }
      if (fileName.startsWith('@luma-contributors/analysis/test/')) { const t = resolve('x', path.join(TEST, fileName.slice('@luma-contributors/analysis/test/'.length))); if (t) imported.add(path.resolve(t)); }
    }
  }
  const orphans = files.filter(f => !/\.spec\.tsx?$/.test(f) && !imported.has(path.resolve(f)) && !f.includes('/oracles/'));
  if (!orphans.length) break;
  orphans.forEach(f => fs.rmSync(f));
  removedHelpers += orphans.length;
}
const byDirectory = {};
removed.forEach(f => { const d = f.split('/')[3]; byDirectory[d] = (byDirectory[d] || 0) + 1; });
console.log(`removed ${removed.length} of ${specs.length} specs, ${removedHelpers} orphan helpers`);
console.log(JSON.stringify(byDirectory));
fs.writeFileSync('scripts/split/pruned-luma-tests.txt', removed.join('\n') + '\n');
