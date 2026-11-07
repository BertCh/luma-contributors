// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// One-shot restructure for dev-docs/upstream-first-plan.md:
// - rename modules/graph (@luma-contributors/graph) to modules/analysis (@luma-contributors/analysis)
// - delete every forked luma.gl graph file (scripts/split/closure.json "needed")
// - move branch-added analysis files out of luma-owned directories into analysis-owned ones
// - quarantine non-analysis luma extensions under src/_pending/ for follow-up
// - rewrite every import; named imports from luma subpaths are split per symbol between
//   npm @luma.gl/graph and the new analysis entry points
// Usage: node scripts/split/restructure-upstream-first.mjs   (from the repository root)

import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';

const require = createRequire(path.resolve('package.json'));
const ts = require('typescript');
const log = [];

const OLD_ROOT = 'modules/graph';
const NEW_ROOT = 'modules/analysis';
const OLD_SRC = `${OLD_ROOT}/src/`;
const NEW_SRC = `${NEW_ROOT}/src/`;
const OLD_PACKAGE = '@luma-contributors/graph';
const NEW_PACKAGE = '@luma-contributors/analysis';
const CONTRIBUTOR_DIRECTORIES = ['analysis-core', 'computation-record', 'gpu-network', 'gpu-spatial-analysis', 'gpu-terrain', 'testing'];
const CONTRIBUTOR_SUBPATHS = ['gpu-spatial-analysis', 'gpu-network', 'gpu-terrain', 'computation-record', 'testing'];
// Branch-added files inside luma-owned directories move here (old prefix -> new prefix).
const ANALYSIS_MOVES = [
  ['gpu-dataframe/gpu-query-describe.ts', '_pending/gpu-dataframe/gpu-query-describe.ts'],
  ['gpu-dataframe/', 'dataframe-analysis/'],
  ['gpu-raster/', 'raster-analysis/'],
  ['gpu-pointcloud/', 'pointcloud-analysis/'],
  ['gpu-project/', 'geospatial-analysis/'],
  ['gpu-geometry/', 'geospatial-analysis/'],
  ['utils/', 'analysis-core/kernels/']
];
const NEW_ENTRIES = ['dataframe-analysis', 'raster-analysis', 'pointcloud-analysis', 'geospatial-analysis'];

const closure = JSON.parse(fs.readFileSync('scripts/split/closure.json', 'utf8'));
const forkedFiles = new Set(Object.keys(closure.needed));
forkedFiles.add(`${OLD_SRC}utils/luma-core-compat.ts`);
forkedFiles.add(`${OLD_SRC}index.ts`);
const seedFiles = new Set(closure.seeds);

const walk = directory =>
  fs.existsSync(directory)
    ? fs.readdirSync(directory, {withFileTypes: true}).flatMap(entry => {
        if (['node_modules', 'dist'].includes(entry.name)) return [];
        const full = path.join(directory, entry.name);
        return entry.isDirectory() ? walk(full) : [full];
      })
    : [];

// ---------- 1. plan file destinations ----------
const oldFiles = walk(OLD_ROOT);
const oldFileSet = new Set(oldFiles);
const destination = new Map(); // old path -> new path | null (deleted)
const srcRelative = file => file.slice(OLD_SRC.length);
const mapSourcePath = relative => {
  const [top] = relative.split('/');
  if (CONTRIBUTOR_DIRECTORIES.includes(top)) return relative;
  for (const [from, to] of ANALYSIS_MOVES) if (relative.startsWith(from)) return to + relative.slice(from.length);
  return `_pending/${relative}`;
};
// Directories whose TypeScript all moves to the same place carry their non-TypeScript assets along.
const directoryMoves = new Map();
for (const file of oldFiles) {
  if (!file.startsWith(OLD_SRC) || !/\.tsx?$/.test(file)) continue;
  const directory = path.dirname(file);
  const target = forkedFiles.has(file) ? null : path.dirname(mapSourcePath(srcRelative(file)));
  const previous = directoryMoves.get(directory);
  directoryMoves.set(directory, previous === undefined ? target : previous === target ? target : false);
}
for (const file of oldFiles) {
  if (file.startsWith(OLD_SRC)) {
    if (forkedFiles.has(file)) destination.set(file, null);
    else if (/\.tsx?$/.test(file)) {
      if (!seedFiles.has(file)) log.push(`WARN unclassified source kept in place: ${file}`);
      destination.set(file, NEW_SRC + mapSourcePath(srcRelative(file)));
    } else {
      const directoryTarget = directoryMoves.get(path.dirname(file));
      destination.set(file, directoryTarget ? `${NEW_SRC}${directoryTarget}/${path.basename(file)}` : null);
    }
  } else {
    destination.set(file, NEW_ROOT + file.slice(OLD_ROOT.length));
  }
}

// ---------- 2. symbol index of new analysis entries ----------
const sourceCache = new Map();
const readSource = file => {
  if (!sourceCache.has(file)) sourceCache.set(file, ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true));
  return sourceCache.get(file);
};
const resolveOld = (fromFile, specifier) => {
  const base = path.normalize(path.join(path.dirname(fromFile), specifier.replace(/\?.*$/, '')));
  const stripped = base.replace(/\.js$/, '');
  for (const candidate of [base, `${stripped}.ts`, `${stripped}.tsx`, `${stripped}/index.ts`]) {
    if (oldFileSet.has(candidate)) return candidate;
  }
  return null;
};
const hasExportModifier = node => (ts.getModifiers?.(node) || []).some(m => m.kind === ts.SyntaxKind.ExportKeyword);
const collectExports = (file, seen = new Set()) => {
  const names = new Set();
  if (!file || seen.has(file) || !fs.existsSync(file)) return names;
  seen.add(file);
  for (const statement of readSource(file).statements) {
    if (ts.isExportDeclaration(statement)) {
      if (statement.exportClause && ts.isNamedExports(statement.exportClause)) {
        statement.exportClause.elements.forEach(element => names.add(element.name.text));
      } else if (statement.exportClause && ts.isNamespaceExport(statement.exportClause)) {
        names.add(statement.exportClause.name.text);
      } else if (statement.moduleSpecifier && statement.moduleSpecifier.text.startsWith('.')) {
        collectExports(resolveOld(file, statement.moduleSpecifier.text), seen).forEach(name => names.add(name));
      }
    } else if (hasExportModifier(statement)) {
      if (ts.isVariableStatement(statement)) {
        statement.declarationList.declarations.forEach(d => ts.isIdentifier(d.name) && names.add(d.name.text));
      } else if (statement.name) names.add(statement.name.text);
    }
  }
  return names;
};
// Generate entry barrels from the files that land directly in each new entry directory.
const entryBarrels = new Map();
const symbolEntry = new Map(); // symbol -> entry name
for (const entry of NEW_ENTRIES) {
  const prefix = `${NEW_SRC}${entry}/`;
  const landed = [...destination].filter(([, to]) => to && to.startsWith(prefix) && /\.tsx?$/.test(to));
  const lines = [];
  const exportedOldFiles = [];
  const subdirectories = new Set();
  for (const [from, to] of landed) {
    const rest = to.slice(prefix.length);
    if (!rest.includes('/')) {
      lines.push(`export * from './${rest.replace(/\.tsx?$/, '')}';`);
      exportedOldFiles.push(from);
    } else subdirectories.add(rest.split('/')[0]);
  }
  for (const subdirectory of [...subdirectories].sort()) {
    const index = landed.find(([, to]) => to === `${prefix}${subdirectory}/index.ts`);
    if (index) {
      lines.push(`export * from './${subdirectory}/index';`);
      exportedOldFiles.push(index[0]);
    } else log.push(`INFO ${entry}/${subdirectory} has no index.ts; not re-exported`);
  }
  lines.sort();
  entryBarrels.set(entry, lines);
  for (const file of exportedOldFiles) {
    for (const name of collectExports(file)) {
      if (symbolEntry.has(name) && symbolEntry.get(name) !== entry) log.push(`WARN symbol ${name} exported by ${symbolEntry.get(name)} and ${entry}`);
      symbolEntry.set(name, entry);
    }
  }
}

// ---------- 3. npm mapping for deleted luma targets ----------
const npmGraph = JSON.parse(fs.readFileSync('node_modules/@luma.gl/graph/package.json', 'utf8'));
const npmSubpaths = new Set(Object.keys(npmGraph.exports).filter(key => !key.includes('*')).map(key => key.slice(2)));
const npmExportCache = new Map();
const npmExports = subpath => {
  if (!npmExportCache.has(subpath)) {
    const entry = npmGraph.exports[subpath ? `./${subpath}` : '.'];
    const typesFile = entry && path.join('node_modules/@luma.gl/graph', entry.types);
    const names = new Set();
    const visit = (file, seen = new Set()) => {
      if (!file || seen.has(file) || !fs.existsSync(file)) return;
      seen.add(file);
      const source = ts.createSourceFile(file, fs.readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true);
      for (const statement of source.statements) {
        if (ts.isExportDeclaration(statement)) {
          if (statement.exportClause && ts.isNamedExports(statement.exportClause)) statement.exportClause.elements.forEach(e => names.add(e.name.text));
          else if (!statement.exportClause && statement.moduleSpecifier) {
            const target = path.join(path.dirname(file), statement.moduleSpecifier.text.replace(/\.js$/, '.d.ts'));
            visit(target, seen);
          }
        } else if (hasExportModifier(statement)) {
          if (ts.isVariableStatement(statement)) statement.declarationList.declarations.forEach(d => ts.isIdentifier(d.name) && names.add(d.name.text));
          else if (statement.name) names.add(statement.name.text);
        }
      }
    };
    visit(typesFile);
    npmExportCache.set(subpath, names);
  }
  return npmExportCache.get(subpath);
};
const npmSpecifierForOldFile = oldFile => {
  const relative = srcRelative(oldFile).replace(/(\/index)?\.tsx?$/, '');
  const [top, ...rest] = relative.split('/');
  if (top === 'execution') return `@luma.gl/graph/_internal/${rest.join('/')}`;
  if (npmSubpaths.has(relative)) return `@luma.gl/graph/${relative}`;
  if (npmSubpaths.has(top)) return `@luma.gl/graph/${top}`;
  return '@luma.gl/graph';
};

// ---------- 4. rewrite one TypeScript file ----------
const isContributorSubpath = sub => CONTRIBUTOR_SUBPATHS.some(c => sub === c || sub.startsWith(`${c}/`));
const relativeSpecifier = (fromNewFile, toNewFile, originalSpecifier) => {
  let relative = path.relative(path.dirname(fromNewFile), toNewFile).replace(/\\/g, '/');
  if (!relative.startsWith('.')) relative = `./${relative}`;
  const original = originalSpecifier.replace(/\?.*$/, '');
  const query = originalSpecifier.includes('?') ? originalSpecifier.slice(originalSpecifier.indexOf('?')) : '';
  if (/\.(ts|tsx)$/.test(original)) return relative + query;
  if (/\.js$/.test(original)) return relative.replace(/\.tsx?$/, '.js') + query;
  if (!/\.(ts|tsx)$/.test(relative)) return relative + query; // json, wgsl, etc.
  if (!/\/index$/.test(original) && /\/index\.tsx?$/.test(relative) && !/\.\/index/.test(original.split('/').pop() === 'index' ? './index' : '')) {
    // original pointed at a directory, keep pointing at the directory index explicitly
    return relative.replace(/\.tsx?$/, '') + query;
  }
  return relative.replace(/\.tsx?$/, '') + query;
};
const entrySpecifierFor = (newImporter, entry) =>
  newImporter.startsWith(NEW_SRC)
    ? relativeSpecifier(newImporter, `${NEW_SRC}${entry}/index.ts`, './index')
    : `${NEW_PACKAGE}/${entry}`;

const stats = {files: 0, splitDeclarations: 0, droppedReExports: 0, npmMapped: 0};
function rewriteFile(oldFile, newFile) {
  const text = fs.readFileSync(oldFile, 'utf8');
  const source = ts.createSourceFile(oldFile, text, ts.ScriptTarget.Latest, true);
  const edits = [];
  // returns {kind: 'keep'|'replace'|'npm', specifier}
  const mapSpecifier = specifier => {
    if (specifier.startsWith('.')) {
      const target = resolveOld(oldFile, specifier);
      if (!target) {
        // target outside modules/graph (examples, docs) or unresolvable: recompute against moved importer
        const absolute = path.normalize(path.join(path.dirname(oldFile), specifier));
        if (absolute.startsWith(OLD_ROOT + '/')) return {kind: 'keep', specifier};
        return {kind: 'replace', specifier: relativeSpecifier(newFile, absolute, specifier)};
      }
      const targetNew = destination.get(target);
      if (targetNew) return {kind: 'replace', specifier: relativeSpecifier(newFile, targetNew, specifier)};
      if (targetNew === null && target.startsWith(OLD_SRC)) return {kind: 'npm', specifier: npmSpecifierForOldFile(target)};
      return {kind: 'keep', specifier};
    }
    if (specifier === OLD_PACKAGE || specifier.startsWith(`${OLD_PACKAGE}/`)) {
      const sub = specifier.slice(OLD_PACKAGE.length + 1);
      if (sub.startsWith('test/')) return {kind: 'replace', specifier: `${NEW_PACKAGE}/${sub}`};
      if (!sub) return {kind: 'npm', specifier: '@luma.gl/graph'};
      if (isContributorSubpath(sub)) return {kind: 'replace', specifier: `${NEW_PACKAGE}/${sub}`};
      return {kind: 'npm', specifier: `@luma.gl/graph/${sub}`};
    }
    return {kind: 'keep', specifier};
  };
  const quote = node => text[node.getStart(source)];
  for (const statement of source.statements) {
    const isImport = ts.isImportDeclaration(statement);
    const isExport = ts.isExportDeclaration(statement);
    if ((!isImport && !isExport) || !statement.moduleSpecifier || !ts.isStringLiteral(statement.moduleSpecifier)) continue;
    const specifierNode = statement.moduleSpecifier;
    const mapped = mapSpecifier(specifierNode.text);
    if (mapped.kind === 'keep') continue;
    const q = quote(specifierNode);
    if (mapped.kind === 'replace') {
      edits.push([specifierNode.getStart(source), specifierNode.getEnd(), `${q}${mapped.specifier}${q}`]);
      continue;
    }
    // npm-mapped: split named bindings between analysis entries and npm
    stats.npmMapped++;
    const npmSubpath = mapped.specifier === '@luma.gl/graph' ? '' : mapped.specifier.slice('@luma.gl/graph/'.length);
    const npmNames = npmSubpath.startsWith('_internal') ? new Set() : npmExports(npmSubpath);
    let elements = null;
    let typeOnly = false;
    if (isImport) {
      const clause = statement.importClause;
      typeOnly = Boolean(clause?.isTypeOnly);
      if (clause && !clause.name && clause.namedBindings && ts.isNamedImports(clause.namedBindings)) elements = clause.namedBindings.elements;
    } else {
      typeOnly = statement.isTypeOnly;
      if (statement.exportClause && ts.isNamedExports(statement.exportClause)) elements = statement.exportClause.elements;
      else if (!statement.exportClause) {
        // `export * from <deleted luma file>` would republish luma API from this package: drop it
        edits.push([statement.getStart(source), statement.getEnd(), `// upstream-first: removed re-export of luma.gl ${mapped.specifier}`]);
        stats.droppedReExports++;
        log.push(`DROPPED export * of ${mapped.specifier} in ${newFile}`);
        continue;
      }
    }
    if (!elements) {
      edits.push([specifierNode.getStart(source), specifierNode.getEnd(), `${q}${mapped.specifier}${q}`]);
      continue;
    }
    const groups = new Map(); // specifier -> element texts
    for (const element of elements) {
      const importedName = (element.propertyName || element.name).text;
      let target = mapped.specifier;
      if (!npmNames.has(importedName) && symbolEntry.has(importedName)) target = entrySpecifierFor(newFile, symbolEntry.get(importedName));
      if (!groups.has(target)) groups.set(target, []);
      groups.get(target).push(element.getText(source));
    }
    if (groups.size > 1) stats.splitDeclarations++;
    const keyword = isImport ? 'import' : 'export';
    const typeKeyword = typeOnly ? ' type' : '';
    const replacement = [...groups]
      .map(([target, names]) => `${keyword}${typeKeyword} {${names.join(', ')}} from ${q}${target}${q};`)
      .join('\n');
    edits.push([statement.getStart(source), statement.getEnd(), replacement]);
  }
  // dynamic import('...') and vi.mock('...') string arguments
  const visit = node => {
    if (ts.isCallExpression(node) && node.arguments.length && ts.isStringLiteral(node.arguments[0]) &&
        (node.expression.kind === ts.SyntaxKind.ImportKeyword || /\.mock$|^require$/.test(node.expression.getText(source)))) {
      const mapped = mapSpecifier(node.arguments[0].text);
      if (mapped.kind !== 'keep') {
        const q = quote(node.arguments[0]);
        edits.push([node.arguments[0].getStart(source), node.arguments[0].getEnd(), `${q}${mapped.specifier}${q}`]);
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(source);
  let output = text;
  for (const [start, end, value] of edits.sort((a, b) => b[0] - a[0])) output = output.slice(0, start) + value + output.slice(end);
  // remaining textual mentions (comments, strings, test path literals)
  output = output.replaceAll(`${OLD_PACKAGE}/`, `${NEW_PACKAGE}/`).replaceAll(`'${OLD_PACKAGE}'`, `'${NEW_PACKAGE}'`);
  output = output.replaceAll('modules/graph/', 'modules/analysis/');
  stats.files++;
  return output;
}

// ---------- 5. apply to modules/graph, then rewrite consumers in place ----------
const writes = new Map();
for (const [from, to] of destination) {
  if (!to) continue;
  writes.set(to, /\.tsx?$/.test(from) ? rewriteFile(from, to) : fs.readFileSync(from));
}
const consumerRoots = ['modules/lenses', 'modules/kepler', 'examples/deck/spatial-analysis-showcase', 'examples/deck/spatial-analysis-explorer', 'examples/deck/spatial-analysis-shared', 'examples/deck/kepler-compute', 'test'];
for (const root of consumerRoots) {
  for (const file of walk(root)) {
    if (!/\.(tsx?|mts)$/.test(file)) continue;
    writes.set(file, rewriteFile(file, file));
  }
}
// new entry barrels and root index
const header = '// luma.gl contributors\n// SPDX-License-Identifier: MIT\n// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors\n\n';
for (const [entry, lines] of entryBarrels) writes.set(`${NEW_SRC}${entry}/index.ts`, `${header}${lines.join('\n')}\n`);
writes.set(`${NEW_SRC}index.ts`, `${header}export * from './computation-record/index';\n`);

fs.rmSync(OLD_ROOT, {recursive: true, force: true});
for (const [file, content] of writes) {
  fs.mkdirSync(path.dirname(file), {recursive: true});
  fs.writeFileSync(file, content);
}
const deleted = [...destination.values()].filter(v => v === null).length;
const pending = [...writes.keys()].filter(f => f.startsWith(`${NEW_SRC}_pending/`));
log.unshift(`files rewritten ${stats.files}; deleted ${deleted}; npm-mapped declarations ${stats.npmMapped}; split ${stats.splitDeclarations}; dropped re-exports ${stats.droppedReExports}; pending ${pending.length}`);
log.push(...pending.map(f => `PENDING ${f}`));
fs.writeFileSync('scripts/split/restructure-upstream-first.log', log.join('\n') + '\n');
console.log(log.slice(0, 1).join('\n'));
for (const [entry, lines] of entryBarrels) console.log(entry, lines.length, 'barrel lines');
