// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Enforce dev-docs/upstream-first-plan.md:
// - luma.gl is imported only through published public entry points
// - analysis source lives only in analysis-owned directories
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';

const require = createRequire(import.meta.url);
const ts = require('typescript');
const ALLOWED_DIRECTORIES = new Set([
  'analysis-core',
  'computation-record',
  'gpu-spatial-analysis',
  'gpu-network',
  'gpu-terrain',
  'testing',
  'dataframe-analysis',
  'raster-analysis',
  'pointcloud-analysis',
  'geospatial-analysis'
]);
const PUBLIC_EXPORTS = {};
for (const name of ['core', 'shaders', 'graph', 'engine']) {
  const packageJson = JSON.parse(fs.readFileSync(`node_modules/@luma.gl/${name}/package.json`, 'utf8'));
  PUBLIC_EXPORTS[name] = new Set(Object.keys(packageJson.exports).filter(key => !key.includes('*')));
}
const walk = directory =>
  fs.existsSync(directory)
    ? fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
        if (['node_modules', 'dist'].includes(entry.name)) return [];
        const full = path.join(directory, entry.name);
        return entry.isDirectory() ? walk(full) : /\.(ts|tsx|mts)$/.test(entry.name) ? [full] : [];
      })
    : [];

const problems = [];
const analysisSource = 'modules/analysis/src';
for (const entry of fs.readdirSync(analysisSource, {withFileTypes: true})) {
  if (entry.isDirectory() && !ALLOWED_DIRECTORIES.has(entry.name)) {
    const count = walk(path.join(analysisSource, entry.name)).length;
    problems.push(`${analysisSource}/${entry.name}: not an analysis-owned directory (${count} files)`);
  }
}
const roots = ['modules/analysis/src', 'modules/lenses/src', 'modules/kepler/src', 'modules/analysis/test', 'modules/lenses/test', 'modules/kepler/test', 'examples/deck'];
for (const file of roots.flatMap(walk)) {
  const imports = ts.preProcessFile(fs.readFileSync(file, 'utf8'), true, true).importedFiles;
  for (const {fileName} of imports) {
    const match = fileName.match(/^@luma\.gl\/([a-z0-9-]+)(\/.*)?$/);
    if (!match) continue;
    const [, name, sub = ''] = match;
    const exportsOfPackage = PUBLIC_EXPORTS[name];
    if (!exportsOfPackage) continue; // other @luma.gl packages are not dependencies; tsc reports them
    if (!exportsOfPackage.has(sub ? `.${sub}` : '.')) problems.push(`${file}: non-public luma.gl import '${fileName}'`);
  }
}
// Re-exporting luma.gl APIs from the package source is forbidden (AGENTS.md); import the symbol where it is used.
export const findLumaReExports = (sourceText, fileName) => {
  const sourceFile = ts.createSourceFile(fileName, sourceText, ts.ScriptTarget.Latest, true);
  return sourceFile.statements
    .filter(
      statement =>
        ts.isExportDeclaration(statement) &&
        statement.moduleSpecifier &&
        ts.isStringLiteral(statement.moduleSpecifier) &&
        /^@luma\.gl\//.test(statement.moduleSpecifier.text)
    )
    .map(statement => `${fileName}: re-export from luma.gl '${statement.moduleSpecifier.text}'`);
};
const sourceRoots = fs.readdirSync('modules').map(name => `modules/${name}/src`);
for (const file of sourceRoots.flatMap(walk)) {
  problems.push(...findLumaReExports(fs.readFileSync(file, 'utf8'), file));
}
// Guard self-test: UPSTREAM_FIRST_SCRATCH_FILE=<path> also checks a file outside the tree.
if (process.env.UPSTREAM_FIRST_SCRATCH_FILE) {
  const scratchFile = process.env.UPSTREAM_FIRST_SCRATCH_FILE;
  problems.push(...findLumaReExports(fs.readFileSync(scratchFile, 'utf8'), scratchFile));
}
if (problems.length) {
  process.stdout.write(`${problems.join('\n')}\n${problems.length} upstream-first violations\n`);
  process.exitCode = 1;
} else {
  process.stdout.write('upstream-first: no violations\n');
}
