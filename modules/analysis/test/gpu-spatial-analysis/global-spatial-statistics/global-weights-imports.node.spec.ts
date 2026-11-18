// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {dirname, relative, resolve} from 'node:path';
import ts from 'typescript';
import {expect, it} from 'vitest';

/** Follow runtime example imports without loading either application's UI in the node worker. */
function getExampleImportChains(entry: string): {modules: string[]; forbidden: string[]} {
  const seen = new Set<string>();
  const forbidden: string[] = [];
  function visit(file: string, chain: string[]) {
    if (seen.has(file)) return;
    seen.add(file);
    const source = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest);
    for (const statement of source.statements) {
      if (!ts.isImportDeclaration(statement) && !ts.isExportDeclaration(statement)) continue;
      if (!statement.moduleSpecifier || !ts.isStringLiteral(statement.moduleSpecifier)) continue;
      if (ts.isImportDeclaration(statement)) {
        if (statement.importClause?.isTypeOnly) continue;
        const bindings = statement.importClause?.namedBindings;
        if (
          !statement.importClause?.name &&
          bindings &&
          ts.isNamedImports(bindings) &&
          bindings.elements.every(element => element.isTypeOnly)
        )
          continue;
      } else if (statement.isTypeOnly) continue;
      const name = statement.moduleSpecifier.text;
      if (/^(@deck\.gl\/|deck\.gl$|react(?:\/|$))/.test(name)) {
        forbidden.push([...chain, relative(process.cwd(), file), name].join(' -> '));
      } else if (name.startsWith('.')) {
        const target = resolve(dirname(file), name);
        const dependency = [target, `${target}.ts`, `${target}.tsx`, `${target}/index.ts`].find(
          path => /\.[cm]?[jt]sx?$/.test(path) && existsSync(path)
        );
        if (dependency && relative(process.cwd(), dependency).startsWith('examples/'))
          visit(dependency, [...chain, relative(process.cwd(), file)]);
      }
    }
  }
  const entryPath = resolve(entry);
  // Begin with the spec, then follow only its example dependencies.
  visit(entryPath, []);
  return {modules: [...seen].map(file => relative(process.cwd(), file)), forbidden};
}

it('the global weights headless regression imports no example UI transitively', () => {
  const result = getExampleImportChains(
    'modules/analysis/test/gpu-spatial-analysis/global-spatial-statistics/global-weights-status.spec.ts'
  );
  expect(result.modules.some(file => file.includes('spatial-analysis-showcase'))).toBe(true);
  expect(result.modules.some(file => file.includes('spatial-analysis-explorer'))).toBe(true);
  expect(result.forbidden).toEqual([]);
});
