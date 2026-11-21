// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {spawnSync} from 'node:child_process';
import {mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {pathToFileURL} from 'node:url';
import typescript from 'typescript';
import {afterEach, expect, it} from 'vitest';
import {
  createPublicSourceInventory,
  createSourceResolvers
} from '../../../../scripts/contributor-contract-source-inventory.mjs';
import {groupPublicEntries} from '../../../../scripts/public-declaration-inventory.mjs';
import {
  collectClassParameters,
  collectClassParametersFromInventory,
  generateDocPages
} from './doc-parameters';
import {getPublicDeclarations} from './public-declarations';
import * as auditInventory from './kernel-audit/analysis-contributors';
import {createVirtualProgram, getRepositoryCompilerOptions, REPOSITORY_ROOT} from './doc-program';

const directories: string[] = [];
function createReviewFixture(files: Record<string, string>, exports: Record<string, unknown>) {
  const directory = mkdtempSync(join(tmpdir(), 'public-inventory-review-'));
  directories.push(directory);
  const sourceRoot = join(directory, 'modules/analysis/src');
  for (const [file, text] of Object.entries(files)) {
    mkdirSync(dirname(join(sourceRoot, file)), {recursive: true});
    writeFileSync(join(sourceRoot, file), text);
  }
  writeFileSync(
    join(directory, 'modules/analysis/package.json'),
    JSON.stringify({name: '@luma-contributors/analysis', exports})
  );
  const sourceTrees = new Map(
    Object.entries(files).map(([file, text]) => {
      const path = join(sourceRoot, file);
      return [
        path,
        typescript.createSourceFile(path, text, typescript.ScriptTarget.Latest, true)
      ] as const;
    })
  );
  const inventory = createPublicSourceInventory({
    analysisManifest: {name: '@luma-contributors/analysis', exports},
    sourceRoot,
    sourceTrees,
    ...createSourceResolvers(sourceTrees)
  });
  return {directory, sourceRoot, sourceTrees, ...inventory};
}
afterEach(() => {
  for (const directory of directories.splice(0)) rmSync(directory, {recursive: true, force: true});
});

function getRuntimeExports(file: string): Record<string, unknown> {
  const result = spawnSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `const value = await import(${JSON.stringify(pathToFileURL(file).href)}); process.stdout.write(JSON.stringify(value));`
    ],
    {encoding: 'utf8', timeout: 10000}
  );
  expect(result.status, result.stderr).toBe(0);
  return JSON.parse(result.stdout) as Record<string, unknown>;
}

it('preserves explicit export priority and omits ambiguous stars, matching Node (review B7)', () => {
  const fixture = createReviewFixture(
    {
      'a.mjs': "export const shared = 'a'; export const first = 1;",
      'b.mjs': "export const shared = 'b'; export const second = 2;",
      'explicit.mjs': "export {shared} from './a.mjs'; export * from './b.mjs';",
      'ambiguous.mjs': "export * from './a.mjs'; export * from './b.mjs';"
    },
    {'./explicit': './dist/explicit.mjs', './ambiguous': './dist/ambiguous.mjs'}
  );
  const explicitRuntime = getRuntimeExports(join(fixture.sourceRoot, 'explicit.mjs'));
  expect(explicitRuntime['shared']).toBe('a');
  expect(fixture.getExports(join(fixture.sourceRoot, 'explicit.mjs')).get('shared')).toBe(
    join(fixture.sourceRoot, 'a.mjs') + '#shared'
  );
  const ambiguousRuntime = getRuntimeExports(join(fixture.sourceRoot, 'ambiguous.mjs'));
  expect(Object.keys(ambiguousRuntime).sort()).toEqual(['first', 'second']);
  expect([...fixture.getExports(join(fixture.sourceRoot, 'ambiguous.mjs')).keys()].sort()).toEqual(
    Object.keys(ambiguousRuntime).sort()
  );
});

it('resolves cyclic stars without caching a partial traversal, matching Node (review earlier cycles)', () => {
  const fixture = createReviewFixture(
    {
      'first.mjs': "export const first = 1; export * from './second.mjs';",
      'second.mjs': "export const second = 2; export * from './first.mjs';"
    },
    {'./first': './dist/first.mjs', './second': './dist/second.mjs'}
  );
  for (const file of ['first.mjs', 'second.mjs']) {
    expect([...fixture.getExports(join(fixture.sourceRoot, file)).keys()].sort()).toEqual(
      Object.keys(getRuntimeExports(join(fixture.sourceRoot, file))).sort()
    );
  }
});

it('does not lend a conventional packer to a distinct same-name class in another entry (review B6)', () => {
  const fixture = createReviewFixture(
    {
      'first.ts':
        'export class GPUProbe {}\nexport function getGPUProbeParameterValues(settings: {first: number}) {return new Float32Array([settings.first]);}',
      'second.ts': 'export class GPUProbe {}'
    },
    {'./first': './dist/first.js', './second': './dist/second.js'}
  );
  const program = typescript.createProgram([...fixture.sourceTrees.keys()], {
    target: typescript.ScriptTarget.ES2022
  });
  const parameters = collectClassParametersFromInventory(
    program,
    getPublicDeclarations(program, fixture.directory)
  );
  expect(parameters.map(record => record.entryPoint)).toEqual([
    '@luma-contributors/analysis/first'
  ]);
});

it('preserves both conditional source origins through grouping and documentation (review B8)', () => {
  const fixture = createReviewFixture(
    {
      'api.mts':
        'export class GPUProbe {}\nexport function getGPUProbeParameterValues(settings: {module: number}) {return new Float32Array([settings.module]);}',
      'api.cts':
        'export class GPUProbe {}\nexport function getGPUProbeParameterValues(settings: {common: number}) {return new Uint32Array([settings.common]);}'
    },
    {'./probe': {import: './dist/api.mjs', require: './dist/api.cjs'}}
  );
  const entry = groupPublicEntries(fixture.inventory).get('@luma-contributors/analysis/probe');
  expect(entry?.get('getGPUProbeParameterValues')?.size).toBe(2);
  const program = typescript.createProgram([...fixture.sourceTrees.keys()], {
    target: typescript.ScriptTarget.ES2022
  });
  const parameters = collectClassParametersFromInventory(
    program,
    getPublicDeclarations(program, fixture.directory)
  );
  expect(
    parameters
      .map(record => [record.packers[0].fields[0].name, record.packers[0].format])
      .sort(([left], [right]) => left.localeCompare(right))
  ).toEqual([
    ['common', 'uint32'],
    ['module', 'float32']
  ]);
});

it('declares the existing geometry page as the canonical line-density owner (review broken docs consumer)', () => {
  const generated = generateDocPages(collectClassParameters());
  expect(generated.undocumented).not.toContain('GPULineDensity');
  expect(generated.pages.get('gpu-spatial-analysis/operators-geometry.md')).toContain(
    '{/* parameters: GPULineDensity '
  );
  expect(generated.pages.get('gpu-spatial-analysis/operators-density.md')).not.toContain(
    '{/* parameters: GPULineDensity '
  );
});

it('expresses the generic inventory argument as a JSDoc record array (review S4)', () => {
  expect(readFileSync('scripts/public-declaration-inventory.mjs', 'utf8')).not.toMatch(
    /typeof createPublicDeclarationInventory\s*</
  );
});

it('keeps cyclic ambiguity and repeated same-origin stars consistent with Node (review B7 cycles)', () => {
  const fixture = createReviewFixture(
    {
      'first.mjs': "export * from './second.mjs'; export * from './third.mjs';",
      'second.mjs': "export * from './first.mjs'; export * from './fourth.mjs';",
      'third.mjs': "export const shared = 'third';",
      'fourth.mjs': "export const shared = 'fourth';",
      'same.mjs': "export * from './third.mjs'; export * from './alias.mjs';",
      'alias.mjs': "export {shared} from './third.mjs';"
    },
    {'./first': './dist/first.mjs', './second': './dist/second.mjs', './same': './dist/same.mjs'}
  );
  for (const file of ['first.mjs', 'second.mjs', 'same.mjs']) {
    expect([...fixture.getExports(join(fixture.sourceRoot, file)).keys()].sort()).toEqual(
      Object.keys(getRuntimeExports(join(fixture.sourceRoot, file))).sort()
    );
  }
});

it('detects manifest entries and runtime bindings omitted by the audit inventory (review unfinished parity)', () => {
  expect(typeof auditInventory.getRuntimeInventoryIssues).toBe('function');
  const records = [
    {
      entryPoint: '@luma-contributors/analysis/first',
      exportName: 'addSharedRecipe',
      declaration: 'shared.ts#addSharedRecipe',
      runtimeKind: 'function' as const
    },
    {
      entryPoint: '@luma-contributors/analysis/second',
      exportName: 'addSharedRecipe',
      declaration: 'shared.ts#addSharedRecipe',
      runtimeKind: 'function' as const
    },
    {
      entryPoint: '@luma-contributors/analysis/new-entry',
      exportName: 'newValue',
      declaration: 'new.ts#newValue',
      runtimeKind: 'value' as const
    }
  ];
  const addSharedRecipe = () => {};
  expect(
    auditInventory.getRuntimeInventoryIssues(records, {
      first: {addSharedRecipe},
      second: {addSharedRecipe}
    })
  ).toEqual(['missing runtime audit entry @luma-contributors/analysis/new-entry']);
  const barrels = {first: {addSharedRecipe}, second: {addSharedRecipe}, 'new-entry': {newValue: 1}};
  expect(auditInventory.getRuntimeInventoryIssues(records, barrels)).toEqual([]);
  expect(
    auditInventory.getRuntimeInventoryIssues(records, {
      ...barrels,
      second: {addSharedRecipe: () => {}}
    })
  ).toEqual([
    'runtime audit binding @luma-contributors/analysis/second:addSharedRecipe disagrees with shared declaration shared.ts#addSharedRecipe'
  ]);
  expect(auditInventory.getRuntimeInventoryIssues(records, {...barrels, second: {}})).toEqual([
    'missing runtime audit binding @luma-contributors/analysis/second:addSharedRecipe'
  ]);
});

it('matches admitted current manifest exports to the actual runtime audit bindings (review unfinished parity)', () => {
  const manifest = JSON.parse(
    readFileSync(join(REPOSITORY_ROOT, 'modules/analysis/package.json'), 'utf8')
  ) as {name: string; exports: Record<string, unknown>};
  const probePath = join(
    REPOSITORY_ROOT,
    'modules/analysis/test/analysis-core/__runtime_inventory_probe__.ts'
  );
  const imports = Object.keys(manifest.exports)
    .map(
      (entry, index) =>
        `export * as entry${index} from '${manifest.name}${entry === '.' ? '' : entry.slice(1)}';`
    )
    .join('\n');
  const program = createVirtualProgram(
    new Map([[probePath, imports]]),
    getRepositoryCompilerOptions()
  );
  const checker = program.getTypeChecker();
  const records = getPublicDeclarations(program).map(record => {
    const separator = record.declaration.lastIndexOf('#');
    const source = program.getSourceFile(record.declaration.slice(0, separator));
    const declarationName = record.declaration.slice(separator + 1);
    let symbol: typescript.Symbol | undefined;
    for (const statement of source?.statements ?? []) {
      const declarations = typescript.isVariableStatement(statement)
        ? statement.declarationList.declarations
        : [statement];
      for (const declaration of declarations) {
        const name = (declaration as typescript.NamedDeclaration).name;
        if (name?.getText() === declarationName) symbol = checker.getSymbolAtLocation(name);
      }
    }
    const declaration = symbol?.valueDeclaration;
    const valueType =
      symbol && declaration ? checker.getTypeOfSymbolAtLocation(symbol, declaration) : undefined;
    const runtimeKind =
      symbol && symbol.flags & typescript.SymbolFlags.Value
        ? valueType &&
          (valueType.getCallSignatures().length || valueType.getConstructSignatures().length)
          ? ('function' as const)
          : ('value' as const)
        : undefined;
    return {...record, runtimeKind};
  });
  expect(
    auditInventory.getRuntimeInventoryIssues(records, auditInventory.ANALYSIS_BARRELS)
  ).toEqual([]);
});
