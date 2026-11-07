// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Second mechanical pass after restructure-upstream-first.mjs:
// - '@luma.gl/graph/_internal/compute/kernel' -> '@luma.gl/graph' (Kernel is public at the root)
// - named imports from '@luma.gl/graph[/sub]' of symbols that npm does not export but that an
//   analysis source file defines are redirected to that file (relative inside modules/analysis/src,
//   or a public @luma-contributors/analysis entry elsewhere when one exports the symbol).
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';

const require = createRequire(path.resolve('package.json'));
const ts = require('typescript');
const SRC = 'modules/analysis/src/';
const walk = d =>
  fs.existsSync(d)
    ? fs.readdirSync(d, {withFileTypes: true}).flatMap(e => {
        if (['node_modules', 'dist'].includes(e.name)) return [];
        const f = path.join(d, e.name);
        return e.isDirectory() ? walk(f) : /\.(ts|tsx)$/.test(e.name) ? [f] : [];
      })
    : [];
const hasExport = n => (ts.getModifiers?.(n) || []).some(m => m.kind === ts.SyntaxKind.ExportKeyword);
const parse = f => ts.createSourceFile(f, fs.readFileSync(f, 'utf8'), ts.ScriptTarget.Latest, true);
const resolve = (from, spec) => {
  const b = path.normalize(path.join(path.dirname(from), spec)).replace(/\.js$/, '');
  return [b + '.ts', b + '.tsx', b + '/index.ts', b].find(c => fs.existsSync(c) && fs.statSync(c).isFile());
};
const exportsOf = (file, seen = new Set()) => {
  const names = new Set();
  if (!file || seen.has(file)) return names;
  seen.add(file);
  for (const s of parse(file).statements) {
    if (ts.isExportDeclaration(s)) {
      if (s.exportClause && ts.isNamedExports(s.exportClause)) s.exportClause.elements.forEach(e => names.add(e.name.text));
      else if (!s.exportClause && s.moduleSpecifier?.text.startsWith('.')) exportsOf(resolve(file, s.moduleSpecifier.text), seen).forEach(n => names.add(n));
    } else if (hasExport(s)) {
      if (ts.isVariableStatement(s)) s.declarationList.declarations.forEach(d => ts.isIdentifier(d.name) && names.add(d.name.text));
      else if (s.name) names.add(s.name.text);
    }
  }
  return names;
};
// definitions: symbol -> defining analysis file (declarations only, excluding _pending)
const definedIn = new Map();
for (const f of walk(SRC)) {
  if (f.includes('/_pending/')) continue;
  for (const s of parse(f).statements) {
    if (!hasExport(s)) continue;
    const names = ts.isVariableStatement(s) ? s.declarationList.declarations.map(d => d.name.getText()) : s.name ? [s.name.text] : [];
    for (const n of names) if (!definedIn.has(n)) definedIn.set(n, f);
  }
}
const analysisPackage = JSON.parse(fs.readFileSync('modules/analysis/package.json', 'utf8'));
const entrySymbols = new Map();
for (const key of Object.keys(analysisPackage.exports)) {
  const sub = key === '.' ? '' : key.slice(2);
  const index = sub ? `${SRC}${sub}/index.ts` : `${SRC}index.ts`;
  if (fs.existsSync(index)) for (const n of exportsOf(index)) if (!entrySymbols.has(n)) entrySymbols.set(n, sub ? `@luma-contributors/analysis/${sub}` : '@luma-contributors/analysis');
}
const npmGraph = JSON.parse(fs.readFileSync('node_modules/@luma.gl/graph/package.json', 'utf8'));
const npmCache = new Map();
const npmNames = sub => {
  if (npmCache.has(sub)) return npmCache.get(sub);
  const entry = npmGraph.exports[sub ? `./${sub}` : '.'];
  const names = new Set();
  const visit = (f, seen = new Set()) => {
    if (!f || seen.has(f) || !fs.existsSync(f)) return;
    seen.add(f);
    for (const s of parse(f).statements) {
      if (ts.isExportDeclaration(s)) {
        if (s.exportClause && ts.isNamedExports(s.exportClause)) s.exportClause.elements.forEach(e => names.add(e.name.text));
        else if (!s.exportClause && s.moduleSpecifier) visit(path.join(path.dirname(f), s.moduleSpecifier.text.replace(/\.js$/, '.d.ts')), seen);
      } else if (hasExport(s)) {
        if (ts.isVariableStatement(s)) s.declarationList.declarations.forEach(d => ts.isIdentifier(d.name) && names.add(d.name.text));
        else if (s.name) names.add(s.name.text);
      }
    }
  };
  if (entry) visit(path.join('node_modules/@luma.gl/graph', entry.types));
  npmCache.set(sub, names);
  return names;
};
const stats = {files: 0, kernel: 0, relocated: 0, unresolved: new Map()};
const roots = ['modules/analysis', 'modules/lenses', 'modules/kepler', 'examples/deck/spatial-analysis-showcase', 'examples/deck/spatial-analysis-explorer', 'examples/deck/spatial-analysis-shared'];
for (const file of roots.flatMap(r => walk(r))) {
  let text = fs.readFileSync(file, 'utf8');
  const original = text;
  text = text.replace(/(['"])@luma\.gl\/graph\/_internal\/compute\/kernel\1/g, (m, q) => (stats.kernel++, `${q}@luma.gl/graph${q}`));
  const source = ts.createSourceFile(file, text, ts.ScriptTarget.Latest, true);
  const edits = [];
  for (const s of source.statements) {
    const isImport = ts.isImportDeclaration(s);
    if (!(isImport || ts.isExportDeclaration(s)) || !s.moduleSpecifier || !ts.isStringLiteral(s.moduleSpecifier)) continue;
    const spec = s.moduleSpecifier.text;
    const m = spec.match(/^@luma\.gl\/graph(?:\/(.*))?$/);
    if (!m || (m[1] || '').startsWith('_internal')) continue;
    const available = npmNames(m[1] || '');
    const elements = isImport
      ? s.importClause?.namedBindings && ts.isNamedImports(s.importClause.namedBindings) ? s.importClause.namedBindings.elements : null
      : s.exportClause && ts.isNamedExports(s.exportClause) ? s.exportClause.elements : null;
    if (!elements) continue;
    const typeOnly = isImport ? Boolean(s.importClause?.isTypeOnly) : s.isTypeOnly;
    const groups = new Map();
    for (const e of elements) {
      const name = (e.propertyName || e.name).text;
      let target = spec;
      if (!available.has(name)) {
        if (file.startsWith(SRC) && definedIn.has(name)) {
          let rel = path.relative(path.dirname(file), definedIn.get(name)).replace(/\.tsx?$/, '');
          if (!rel.startsWith('.')) rel = `./${rel}`;
          target = rel;
        } else if (entrySymbols.has(name)) target = entrySymbols.get(name);
        else if (definedIn.has(name) && file.startsWith('modules/analysis/test/')) {
          target = path.relative(path.dirname(file), definedIn.get(name)).replace(/\.tsx?$/, '');
        } else stats.unresolved.set(name, (stats.unresolved.get(name) || 0) + 1);
      }
      if (target !== spec) stats.relocated++;
      if (!groups.has(target)) groups.set(target, []);
      groups.get(target).push(e.getText(source));
    }
    if (groups.size === 1 && groups.has(spec)) continue;
    const q = text[s.moduleSpecifier.getStart(source)];
    const keyword = isImport ? 'import' : 'export';
    edits.push([s.getStart(source), s.getEnd(), [...groups].map(([t, n]) => `${keyword}${typeOnly ? ' type' : ''} {${n.join(', ')}} from ${q}${t}${q};`).join('\n')]);
  }
  for (const [a, b, v] of edits.sort((x, y) => y[0] - x[0])) text = text.slice(0, a) + v + text.slice(b);
  if (text !== original) {
    fs.writeFileSync(file, text);
    stats.files++;
  }
}
console.log(`files ${stats.files}, kernel remaps ${stats.kernel}, relocated names ${stats.relocated}`);
console.log('still missing from npm:', [...stats.unresolved].sort((a, b) => b[1] - a[1]).map(([n, c]) => `${n}(${c})`).join(' '));
