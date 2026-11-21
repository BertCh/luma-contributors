// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {mkdirSync, mkdtempSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import typescript from 'typescript';
import {expect, it} from 'vitest';
import {
  collectClassParameters,
  collectClassParametersFromInventory,
  generateDocPages
} from './doc-parameters';
import {getPublicDeclarations} from './public-declarations';
import {getAnalysisContributors, getContributorNames} from './kernel-audit/analysis-contributors';

it('keeps entry, export spelling and declaration identity across source and runtime consumers (ERG-106)', () => {
  const directory = mkdtempSync(join(tmpdir(), 'public-declaration-inventory-'));
  try {
    const sourceRoot = join(directory, 'modules/analysis/src');
    const files = {
      'modules/analysis/package.json': JSON.stringify({
        name: '@luma-contributors/analysis',
        exports: {
          './first': './dist/first/index.js',
          './second': './dist/second/index.js'
        }
      }),
      'modules/analysis/src/shared.ts': 'export function addSharedRecipe() {}',
      'modules/analysis/src/first/index.ts': `export {addSharedRecipe} from '../shared';
export function addDuplicateRecipe() {}
export class GPUFixture {getCommandNodes() {}}
export function getGPUFixtureParameterValues(settings: {first: number}): Float32Array {return new Float32Array([settings.first]);}`,
      'modules/analysis/src/second/index.ts': `export {addSharedRecipe} from '../shared';
export function addDuplicateRecipe() {}
export class GPUFixture {getCommandNodes() {}}
export function getGPUFixtureParameterValues(settings: {second: number}): Uint32Array {return new Uint32Array([settings.second]);}`
    };
    for (const [file, text] of Object.entries(files)) {
      const path = join(directory, file);
      mkdirSync(dirname(path), {recursive: true});
      writeFileSync(path, text);
    }
    const program = typescript.createProgram(
      [join(sourceRoot, 'first/index.ts'), join(sourceRoot, 'second/index.ts')],
      {
        target: typescript.ScriptTarget.ES2022,
        moduleResolution: typescript.ModuleResolutionKind.Bundler,
        module: typescript.ModuleKind.ESNext
      }
    );
    const inventory = getPublicDeclarations(program, directory);
    const recipes = inventory.filter(record => record.exportName.startsWith('add'));
    expect(recipes).toHaveLength(4);
    const shared = recipes.filter(record => record.exportName === 'addSharedRecipe');
    expect(new Set(shared.map(record => record.entryPoint)).size).toBe(2);
    expect(new Set(shared.map(record => record.declaration)).size).toBe(1);
    expect(
      new Set(
        recipes
          .filter(record => record.exportName === 'addDuplicateRecipe')
          .map(record => record.declaration)
      ).size
    ).toBe(2);
    // Hand-written declarations above deliberately have different settings and return formats;
    // neither entry's public packer may disappear behind a first-symbol-wins name map.
    const parameters = collectClassParametersFromInventory(program, inventory);
    expect(parameters.map(record => record.packers[0].fields[0].name).sort()).toEqual([
      'first',
      'second'
    ]);
    expect(parameters.map(record => record.packers[0].format).sort()).toEqual([
      'float32',
      'uint32'
    ]);

    expect(() => generateDocPages(parameters, new Map([['fixture.md', '# GPUFixture\n']]))).toThrow(
      'ambiguous public class identity'
    );
    const generated = generateDocPages(
      parameters,
      new Map([
        ['first.md', '# GPUFixture\n'],
        ['second.md', '# GPUFixture\n']
      ]),
      {
        '@luma-contributors/analysis/first:GPUFixture': {page: 'first.md', heading: 'GPUFixture'},
        '@luma-contributors/analysis/second:GPUFixture': {page: 'second.md', heading: 'GPUFixture'}
      }
    );
    expect(generated.pages.get('first.md')).toContain('`first`');
    expect(generated.pages.get('first.md')).not.toContain('`second`');
    expect(generated.pages.get('second.md')).toContain('`second`');

    const addSharedRecipe = () => {};
    const addFirstRecipe = () => {};
    const addSecondRecipe = () => {};
    expect([
      ...getContributorNames({
        first: {addSharedRecipe, addDuplicateRecipe: addFirstRecipe},
        second: {addSharedRecipe, addDuplicateRecipe: addSecondRecipe}
      })
    ]).toEqual([
      ['addSharedRecipe', 'first'],
      ['first/addDuplicateRecipe', 'first'],
      ['second/addDuplicateRecipe', 'second']
    ]);
    const contributors = getAnalysisContributors();
    expect(contributors.get('gpu-spatial-analysis/addVisibilityGraphRecipe')).toBe(
      'gpu-spatial-analysis'
    );
    expect(contributors.get('gpu-network/addVisibilityGraphRecipe')).toBe('gpu-network');
    expect(contributors.get('GPUTransferMask')).toBe('computation-record');
    expect(contributors.get('GPUSnapshotTap')).toBe('computation-record');
  } finally {
    rmSync(directory, {recursive: true, force: true});
  }
});

it('keeps current explicit shared packers across package entries (ERG-106)', () => {
  const classes = collectClassParameters();
  const flow = classes.find(record => record.className === 'GPUFlowAggregation');
  expect(
    flow?.packers.some(packer => packer.call.startsWith('getGPUTimeWindowFilterParameterValues('))
  ).toBe(true);
  const rasterJoin = classes.find(record => record.className === 'GPURasterJoin');
  expect(
    rasterJoin?.packers.some(packer =>
      packer.call.startsWith('getGPUPolygonRasterizationParameterValues(')
    )
  ).toBe(true);
});
