// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Import-statement census, not a semantic reachability analysis. Run from the repository root.
// Usage is attributed to the exact entry point named in each import (entry plus name); a value
// imported through a re-exporting entry is reported under `usedThroughOtherEntries`.
import {existsSync, readFileSync, readdirSync, writeFileSync} from 'node:fs';
import {join} from 'node:path';
import typescript from 'typescript';

const [exportsPath, usagePath, reportPath] = process.argv.slice(2);
if (!exportsPath || !usagePath || !reportPath)
  throw new Error('Usage: usage.mjs <exports.json> <usage.json> <report.json>');
const exportsByEntry = JSON.parse(readFileSync(exportsPath, 'utf8'));
const areas = {
  'analysis-src': ['modules/analysis/src'],
  'lenses-src': ['modules/lenses/src'],
  'kepler-src': ['modules/kepler/src'],
  tests: ['modules/analysis/test', 'modules/lenses/test', 'modules/kepler/test', 'test'],
  website: ['website/src'],
  examples: ['examples'],
  docs: ['docs']
};
function* walkDirectory(directory) {
  if (!existsSync(directory)) return;
  for (const entry of readdirSync(directory, {withFileTypes: true})) {
    if (entry.name === 'node_modules' || entry.name === 'dist' || entry.name.startsWith('.'))
      continue;
    const filePath = join(directory, entry.name);
    if (entry.isDirectory()) yield* walkDirectory(filePath);
    else if (/\.(ts|tsx|mts|js|mjs|md|mdx)$/.test(entry.name)) yield filePath;
  }
}
const usage = {};
const specifierCounts = {};
function getImportDeclarations(filePath) {
  const contents = readFileSync(filePath, 'utf8');
  const fragments = /\.mdx?$/.test(filePath)
    ? [...contents.matchAll(/^```([^\n]*)\n([\s\S]*?)^```[ \t]*$/gm)]
        .filter(match =>
          ['', 'ts', 'typescript', 'tsx', 'js', 'javascript', 'jsx', 'mts', 'mjs'].includes(
            match[1].trim().split(/\s+/)[0]
          )
        )
        .map(match => match[2])
    : [contents];
  return fragments.flatMap(fragment => {
    const source = typescript.createSourceFile(
      filePath,
      fragment,
      typescript.ScriptTarget.Latest,
      false,
      /\.[jt]sx$/.test(filePath) ? typescript.ScriptKind.TSX : typescript.ScriptKind.TS
    );
    return source.statements.filter(typescript.isImportDeclaration);
  });
}
for (const [area, directories] of Object.entries(areas)) {
  for (const directory of directories)
    for (const filePath of walkDirectory(join(process.cwd(), directory))) {
      for (const declaration of getImportDeclarations(filePath)) {
        if (!typescript.isStringLiteral(declaration.moduleSpecifier)) continue;
        const specifier = declaration.moduleSpecifier.text;
        if (!specifier.startsWith('@luma.gl/')) continue;
        const clause = declaration.importClause;
        specifierCounts[specifier] ??= {};
        specifierCounts[specifier][area] = (specifierCounts[specifier][area] || 0) + 1;
        const names = [];
        if (clause?.name) names.push(['default', clause.isTypeOnly]);
        const bindings = clause?.namedBindings;
        if (bindings && typescript.isNamedImports(bindings))
          for (const element of bindings.elements)
            names.push([
              (element.propertyName ?? element.name).text,
              clause.isTypeOnly || element.isTypeOnly
            ]);
        else if (bindings && typescript.isNamespaceImport(bindings))
          names.push(['*', clause.isTypeOnly]);
        for (const [name, isType] of names) {
          const key = `${specifier}#${name}`;
          usage[key] ??= {};
          usage[key][area] ??= {value: 0, type: 0};
          usage[key][area][isType ? 'type' : 'value']++;
        }
      }
    }
}
writeFileSync(usagePath, JSON.stringify({usage, specifierCounts}));
console.log('SPECIFIERS (import statements per area)');
for (const [specifier, counts] of Object.entries(specifierCounts).sort())
  console.log(' ', specifier.padEnd(44), JSON.stringify(counts));
// Usage is joined by (entry, name): an import of `GPUScan` from `@luma.gl/graph/gpu-core` counts for
// that entry only, even when the graph root re-exports the same declaration. A name imported only
// through another entry that exports the same declaration file is listed separately as
// `usedThroughOtherEntries` instead of being credited to this entry.
const usedKeys = new Map();
for (const [key, areasUsed] of Object.entries(usage)) {
  const total = {source: 0, other: 0};
  for (const [area, count] of Object.entries(areasUsed)) {
    total[area.endsWith('-src') ? 'source' : 'other'] += count.value + count.type;
  }
  usedKeys.set(key, total);
}
const getUsedEntries = (entry, name, declarationFile) =>
  Object.entries(exportsByEntry)
    .filter(
      ([otherEntry, otherExports]) =>
        otherEntry !== entry &&
        otherExports?.[name] &&
        (otherExports[name].file ?? '') === (declarationFile ?? '') &&
        usedKeys.has(`${otherEntry}#${name}`)
    )
    .map(([otherEntry]) => otherEntry);
const report = {};
for (const [entry, exports] of Object.entries(exportsByEntry)) {
  if (!exports) continue;
  const values = Object.entries(exports).filter(([, value]) =>
    ['class', 'function', 'const'].includes(value.kind)
  );
  const getUse = name => usedKeys.get(`${entry}#${name}`);
  const unused = values.filter(([name]) => !getUse(name));
  const usedThroughOtherEntries = Object.fromEntries(
    unused
      .map(([name, value]) => [name, getUsedEntries(entry, name, value.file)])
      .filter(([, entries]) => entries.length > 0)
  );
  report[entry] = {
    values: values.length,
    usedInSrc: values.filter(([name]) => (getUse(name)?.source ?? 0) > 0).length,
    unused: unused.map(([name]) => name),
    testOnly: values
      .filter(([name]) => getUse(name) && getUse(name).source === 0)
      .map(([name]) => name),
    usedThroughOtherEntries
  };
}
writeFileSync(reportPath, JSON.stringify(report, null, 1));
console.log(
  '\nENTRY value exports / used in src / used only outside src / unused (imported from this exact entry)'
);
for (const [entry, counts] of Object.entries(report))
  console.log(
    ' ',
    entry.padEnd(44),
    counts.values,
    counts.usedInSrc,
    counts.testOnly.length,
    counts.unused.length
  );
