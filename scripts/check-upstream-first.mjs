// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Enforce dev-docs/upstream-first-plan.md:
// - luma.gl is imported only through published public entry points
// - analysis source lives only in analysis-owned directories
import fs from 'node:fs';
import {createRequire} from 'node:module';
import path from 'node:path';

const require = createRequire(import.meta.url);
const typeScript = require('typescript');
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
  const packageJson = JSON.parse(
    fs.readFileSync(`node_modules/@luma.gl/${name}/package.json`, 'utf8')
  );
  PUBLIC_EXPORTS[name] = new Set(
    Object.keys(packageJson.exports).filter(key => !key.includes('*'))
  );
}
const walk = directory =>
  fs.existsSync(directory)
    ? fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
        if (['node_modules', 'dist'].includes(entry.name)) return [];
        const full = path.join(directory, entry.name);
        return entry.isDirectory()
          ? walk(full)
          : /\.(?:[cm]?[jt]s|[jt]sx)$/.test(entry.name)
            ? [full]
            : [];
      })
    : [];

const problems = [];
const analysisSource = 'modules/analysis/src';
for (const entry of fs.readdirSync(analysisSource, {withFileTypes: true})) {
  if (entry.isDirectory() && !ALLOWED_DIRECTORIES.has(entry.name)) {
    const count = walk(path.join(analysisSource, entry.name)).length;
    problems.push(
      `${analysisSource}/${entry.name}: not an analysis-owned directory (${count} files)`
    );
  } else if (entry.isFile() && entry.name !== 'index.ts') {
    problems.push(
      `${analysisSource}/${entry.name}: only index.ts is allowed directly in the source root`
    );
  }
}
const roots = [
  'modules/analysis/src',
  'modules/lenses/src',
  'modules/kepler/src',
  'modules/analysis/test',
  'modules/lenses/test',
  'modules/kepler/test',
  'examples/deck'
];
// A luma.gl specifier is public only when it is an explicit, wildcard-free key of the package `exports`
// and has no `_internal` path segment. `@luma.gl/graph` lists `./_internal/arrow` as an explicit key, so
// the exports map alone does not decide it: `_internal` is rejected for every `@luma.gl/*` package.
export const findNonPublicLumaImports = (sourceText, fileName) => {
  const found = [];
  for (const {fileName: specifier} of typeScript.preProcessFile(sourceText, true, true)
    .importedFiles) {
    const match = specifier.match(/^@luma\.gl\/([a-z0-9-]+)(\/.*)?$/);
    if (!match) continue;
    const [, name, sub = ''] = match;
    if (/\/_internal(\/|$)/.test(sub)) {
      found.push(`${fileName}: luma.gl internal import '${specifier}'`);
      continue;
    }
    if (!Object.hasOwn(PUBLIC_EXPORTS, name)) {
      found.push(
        `${fileName}: disallowed luma.gl package '${specifier}' (allowed: core, shaders, graph; engine in tests/examples)`
      );
      continue;
    }
    const normalizedFile = path
      .relative(process.cwd(), path.resolve(fileName))
      .split(path.sep)
      .join('/');
    const testOrExample = /^(?:examples\/|modules\/[^/]+\/test\/|test\/)/.test(normalizedFile);
    if (name === 'engine' && !testOrExample) {
      found.push(`${fileName}: engine is restricted to tests and examples: '${specifier}'`);
      continue;
    }
    const exportsOfPackage = PUBLIC_EXPORTS[name];
    if (!exportsOfPackage.has(sub ? `.${sub}` : '.'))
      found.push(`${fileName}: non-public luma.gl import '${specifier}'`);
  }
  return found;
};
for (const file of roots.flatMap(walk)) {
  problems.push(...findNonPublicLumaImports(fs.readFileSync(file, 'utf8'), file));
}
// Re-exporting luma.gl APIs from the package source is forbidden (AGENTS.md); import the symbol where it is used.
export const findLumaReExports = (sourceText, fileName) => {
  const sourceFile = typeScript.createSourceFile(
    fileName,
    sourceText,
    typeScript.ScriptTarget.Latest,
    true
  );
  return sourceFile.statements
    .filter(
      statement =>
        typeScript.isExportDeclaration(statement) &&
        statement.moduleSpecifier &&
        typeScript.isStringLiteral(statement.moduleSpecifier) &&
        /^@luma\.gl\//.test(statement.moduleSpecifier.text)
    )
    .map(statement => `${fileName}: re-export from luma.gl '${statement.moduleSpecifier.text}'`);
};
const sourceRoots = fs.readdirSync('modules').map(name => `modules/${name}/src`);
for (const file of sourceRoots.flatMap(walk)) {
  problems.push(...findLumaReExports(fs.readFileSync(file, 'utf8'), file));
}
// Guard self-test: UPSTREAM_FIRST_SCRATCH_FILE=<path> also checks a file outside the tree, with both the
// import rule and the re-export rule (modules/analysis/test/analysis-core/upstream-first-guard.node.spec.ts).
if (process.env.UPSTREAM_FIRST_SCRATCH_FILE) {
  const scratchFile = process.env.UPSTREAM_FIRST_SCRATCH_FILE;
  const scratchText = fs.readFileSync(scratchFile, 'utf8');
  problems.push(...findNonPublicLumaImports(scratchText, scratchFile));
  problems.push(...findLumaReExports(scratchText, scratchFile));
}
if (problems.length) {
  process.stdout.write(`${problems.join('\n')}\n${problems.length} upstream-first violations\n`);
  process.exitCode = 1;
} else {
  process.stdout.write('upstream-first: no violations\n');
}
