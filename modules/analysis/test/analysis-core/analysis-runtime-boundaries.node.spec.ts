// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync, readFileSync} from 'node:fs';
import {dirname, join, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import typescript from 'typescript';
import {expect, it} from 'vitest';

const SOURCE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../src');
const ANALYSIS_ENTRIES = [
  'geospatial-analysis',
  'dataframe-analysis',
  'raster-analysis',
  'pointcloud-analysis',
  'gpu-terrain',
  'gpu-spatial-analysis',
  'gpu-network'
];

function getRuntimeSpecifiers(source: string): string[] {
  const sourceFile = typescript.createSourceFile(
    'source.ts',
    source,
    typescript.ScriptTarget.Latest
  );
  const specifiers: string[] = [];
  function visit(node: typescript.Node): void {
    if (typescript.isImportDeclaration(node)) {
      const clause = node.importClause;
      const bindings = clause?.namedBindings;
      const onlyTypes =
        clause?.isTypeOnly ||
        (!clause?.name &&
          bindings &&
          typescript.isNamedImports(bindings) &&
          bindings.elements.length > 0 &&
          bindings.elements.every(element => element.isTypeOnly));
      if (!onlyTypes && typescript.isStringLiteral(node.moduleSpecifier)) {
        specifiers.push(node.moduleSpecifier.text);
      }
    } else if (typescript.isExportDeclaration(node)) {
      const clause = node.exportClause;
      const onlyTypes =
        node.isTypeOnly ||
        (clause &&
          typescript.isNamedExports(clause) &&
          clause.elements.length > 0 &&
          clause.elements.every(element => element.isTypeOnly));
      if (!onlyTypes && node.moduleSpecifier && typescript.isStringLiteral(node.moduleSpecifier)) {
        specifiers.push(node.moduleSpecifier.text);
      }
    } else if (
      typescript.isCallExpression(node) &&
      (node.expression.kind === typescript.SyntaxKind.ImportKeyword ||
        (typescript.isIdentifier(node.expression) && node.expression.text === 'require')) &&
      node.arguments[0] &&
      (typescript.isStringLiteral(node.arguments[0]) ||
        typescript.isNoSubstitutionTemplateLiteral(node.arguments[0]))
    ) {
      specifiers.push(node.arguments[0].text);
    }
    typescript.forEachChild(node, visit);
  }
  visit(sourceFile);
  return specifiers;
}

it('recognizes runtime imports and re-exports while allowing type-only dependencies', () => {
  expect(
    getRuntimeSpecifiers(`
      import type {ViewPack} from './type-import';
      import {type ViewPack} from './inline-type-import';
      export type * from './type-export';
      export {type ViewPack} from './inline-type-export';
      import './registration';
      import {type ViewPack, register} from './mixed-import';
      export * from './barrel';
      export {register} from './named-export';
      void import('./lazy-view');
      require('./required-view');
    `)
  ).toEqual([
    './registration',
    './mixed-import',
    './barrel',
    './named-export',
    './lazy-view',
    './required-view'
  ]);
});

function findRuntimeBoundaryViolations(
  entry: string,
  readSource: (file: string) => string = file => readFileSync(file, 'utf8'),
  fileExists: (file: string) => boolean = existsSync
): {visited: Set<string>; violations: string[]} {
  const visited = new Set<string>();
  const violations: string[] = [];
  function visit(file: string, chain: string[]): void {
    if (visited.has(file)) return;
    visited.add(file);
    for (const specifier of getRuntimeSpecifiers(readSource(file))) {
      if (
        /^(?:@deck\.gl(?:\/|$)|deck\.gl(?:\/|$)|@kepler\.gl(?:\/|$)|react(?:-dom)?(?:\/|$)|@luma\.gl\/engine(?:\/|$)|@luma-contributors\/(?:lenses|kepler)(?:\/|$)|@luma-contributors\/analysis\/(?:testing|test)(?:\/|$))/.test(
          specifier
        )
      ) {
        violations.push([...chain, specifier].join(' -> '));
        continue;
      }
      const target = specifier.startsWith('.')
        ? resolve(dirname(file), specifier.replace(/\.(?:js|mjs|cjs)$/, ''))
        : specifier === '@luma-contributors/analysis'
          ? resolve(SOURCE_ROOT, 'index.ts')
          : specifier.startsWith('@luma-contributors/analysis/')
            ? resolve(SOURCE_ROOT, specifier.slice('@luma-contributors/analysis/'.length))
            : undefined;
      if (!target) continue;
      const targetPath = relative(SOURCE_ROOT, target);
      const nextChain = [...chain, targetPath];
      if (
        targetPath === 'testing' ||
        targetPath.startsWith('testing/') ||
        targetPath.startsWith('..')
      ) {
        violations.push(nextChain.join(' -> '));
        continue;
      }
      const resolvedFile = [target, `${target}.ts`, `${target}.tsx`, join(target, 'index.ts')].find(
        candidate =>
          candidate.endsWith('.ts') || candidate.endsWith('.tsx') ? fileExists(candidate) : false
      );
      if (resolvedFile) visit(resolvedFile, nextChain);
    }
  }
  visit(join(SOURCE_ROOT, entry, 'index.ts'), [entry]);
  return {visited, violations};
}

it('rejects a transitive runtime visualization dependency', () => {
  const files = new Map([
    [join(SOURCE_ROOT, 'fixture/index.ts'), "export * from './helper';"],
    [join(SOURCE_ROOT, 'fixture/helper.ts'), "import {Layer} from '@deck.gl/core';"]
  ]);
  const result = findRuntimeBoundaryViolations(
    'fixture',
    file => files.get(file)!,
    file => files.has(file)
  );
  expect(result.violations).toEqual(['fixture -> fixture/helper -> @deck.gl/core']);
});

it.each([
  "import '@kepler.gl/layers';",
  "import '@luma-contributors/analysis/test/utils/require-test-device';",
  'void import(`@deck.gl/core`);'
])('rejects transitive runtime dependency %s', source => {
  const files = new Map([
    [join(SOURCE_ROOT, 'fixture/index.ts'), "export * from './helper';"],
    [join(SOURCE_ROOT, 'fixture/helper.ts'), source]
  ]);
  const result = findRuntimeBoundaryViolations(
    'fixture',
    file => files.get(file)!,
    file => files.has(file)
  );
  expect(result.visited.size).toBe(2);
  expect(result.violations).toHaveLength(1);
});

it('allows a transitive type-only host dependency', () => {
  const files = new Map([
    [join(SOURCE_ROOT, 'fixture/index.ts'), "export * from './helper';"],
    [join(SOURCE_ROOT, 'fixture/helper.ts'), "export type {Layer} from '@kepler.gl/layers';"]
  ]);
  expect(
    findRuntimeBoundaryViolations(
      'fixture',
      file => files.get(file)!,
      file => files.has(file)
    ).violations
  ).toEqual([]);
});

it('checks a nonempty inventory of runtime entry points', () => {
  expect(ANALYSIS_ENTRIES.length).toBeGreaterThan(0);
});

it.each([
  '@luma.gl/engine',
  '@luma-contributors/lenses',
  '../testing/index'
])('rejects the runtime boundary %s', specifier => {
  const result = findRuntimeBoundaryViolations(
    'fixture',
    () => `import '${specifier}';`,
    () => false
  );
  expect(result.violations).toHaveLength(1);
});

it.each(ANALYSIS_ENTRIES)('%s never reaches visualization adapters at runtime', entry => {
  const {visited, violations} = findRuntimeBoundaryViolations(entry);
  expect(visited.size, entry).toBeGreaterThan(1);
  expect(violations, entry).toEqual([]);
});
