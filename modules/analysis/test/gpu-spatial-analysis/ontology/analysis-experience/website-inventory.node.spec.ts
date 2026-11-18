// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {existsSync} from 'node:fs';
import {dirname, relative, resolve, sep} from 'node:path';
import {expect, it} from 'vitest';
import {
  selectInventoryContract,
  getInventoryOutputIssues
} from '../../../../../../website/scripts/inventory-contracts';
import {extractInventory} from '../../../../../../website/scripts/extract-inventory';
import {getGPUAnalysisOperatorCatalog} from '../../../../src/gpu-spatial-analysis/ontology/index';
import {entries} from '../../../../../../website/src/catalog/entries/geometry-construction-overlay';
import inventory from '../../../../../../website/src/catalog/generated/inventory.json';
import {entries as joinEntries} from '../../../../../../website/src/catalog/entries/cells-index-join';
import {entries as rasterEntries} from '../../../../../../website/src/catalog/entries/raster-embeddings-flow';

it('validates the committed mode-omitted visibility example against the reference overload', () => {
  const entry = entries.find(entry => entry.id === 'visibility-graph-recipe')!;
  const symbol = inventory.symbols.find(
    symbol => symbol.symbol === entry.symbol && symbol.importPath === entry.importPath
  )!;
  const contract = selectInventoryContract(symbol, entry.symbol, entry.example);
  expect(contract.props.find(prop => prop.name === 'mode')?.optional).toBe(true);
  expect(
    getInventoryOutputIssues(
      contract,
      entry.outputs.map(output => output.name)
    )
  ).toMatchObject({unknown: [], missing: []});
});

it('checks explicit CSR calls against CSR-only required return fields', () => {
  const symbol = inventory.symbols.find(
    symbol =>
      symbol.symbol === 'addVisibilityGraphRecipe' &&
      symbol.importPath.endsWith('/gpu-spatial-analysis')
  )!;
  const contract = selectInventoryContract(
    symbol,
    symbol.symbol,
    "addVisibilityGraphRecipe(graph, {mode: 'csr', positions, segments, maximumDistance: 1, edgeCapacity: 8})"
  );
  expect(contract.props.find(prop => prop.name === 'mode')?.optional).toBe(false);
  expect(getInventoryOutputIssues(contract, ['outputs', 'status', 'provenance']).missing).toEqual([
    'topology',
    'estimatedCommandNodeCount'
  ]);
  expect(
    getInventoryOutputIssues(contract, [
      'outputs',
      'status',
      'provenance',
      'topology',
      'estimatedCommandNodeCount'
    ]).missing
  ).toEqual([]);
  expect(() =>
    selectInventoryContract(
      symbol,
      symbol.symbol,
      "addVisibilityGraphRecipe(graph, {mode: 'unknown'})"
    )
  ).toThrow(/no unique overload/);
  expect(() =>
    selectInventoryContract(
      symbol,
      symbol.symbol,
      'addVisibilityGraphRecipe(graph, {mode: selectedMode})'
    )
  ).toThrow(/string literal/);
  expect(() =>
    selectInventoryContract(symbol, symbol.symbol, 'addVisibilityGraphRecipe(graph, {...options})')
  ).toThrow(/without spreads/);
});

it('reports a missing qualified visibility descriptor while both declarations remain public', async () => {
  const operators = getGPUAnalysisOperatorCatalog().filter(
    operator =>
      !(
        operator.exportName === 'addVisibilityGraphRecipe' &&
        operator.package.endsWith('/gpu-network')
      )
  );
  await expect(extractInventory(operators)).rejects.toThrow(
    '@luma-contributors/analysis/gpu-network:addVisibilityGraphRecipe'
  );
}, 60000);

it('selects overloaded defaults by their required options when mode alone is ambiguous', () => {
  const entry = entries.find(entry => entry.id === 'building-tessellation-recipe')!;
  const symbol = inventory.symbols.find(
    symbol => symbol.symbol === entry.symbol && symbol.importPath === entry.importPath
  )!;
  const contract = selectInventoryContract(symbol, entry.symbol, entry.example);
  expect(contract.props.some(prop => prop.name === 'capacity')).toBe(true);
  expect(contract.props.some(prop => prop.name === 'candidateRadius')).toBe(false);
  expect(
    getInventoryOutputIssues(
      contract,
      entry.outputs.map(output => output.name)
    )
  ).toMatchObject({unknown: [], missing: []});
});

it('keeps neighborStatus visible within the nearest-weights short-card limits', () => {
  const entry = joinEntries.find(entry => entry.id === 'nearest-feature-weights')!;
  expect(entry.params.length).toBeLessThanOrEqual(8);
  const status = entry.params.find(parameter => parameter.name === 'neighborStatus');
  expect(status).toBeDefined();
  expect(status!.description.length).toBeLessThanOrEqual(80);
  expect(status!.description).toContain('output.status.candidateOverflow');
});

it.each([
  'GPUDatumGridTransform',
  'GPUPointBuffer',
  'makePointCloudLocalPositions'
])('keeps %s inventory directory aligned with its relocated source', name => {
  const record = inventory.symbols.find(symbol => symbol.symbol === name)!;
  expect(record).toBeDefined();
  expect(existsSync(resolve(record.sourceFile)), record.sourceFile).toBe(true);
  const directory = relative('modules/analysis/src', dirname(record.sourceFile))
    .split(sep)
    .slice(0, 2)
    .join('/');
  expect(record.directory).toBe(directory);
});

it('keeps committed inventory paths, ports and kinds equal to extraction from current source', async () => {
  const extracted = await extractInventory();
  const project = (
    symbols: readonly {
      symbol: string;
      importPath: string;
      sourceFile: string;
      directory: string;
      props: readonly {name: string; optional: boolean}[];
      outputs: readonly {name: string; optional: boolean}[];
      dataKinds: {input: readonly string[]; output: readonly string[]};
    }[]
  ) =>
    symbols.map(symbol => ({
      identity: `${symbol.importPath}:${symbol.symbol}`,
      sourceFile: symbol.sourceFile,
      directory: symbol.directory,
      props: symbol.props.map(({name, optional}) => ({name, optional})),
      outputs: symbol.outputs.map(({name, optional}) => ({name, optional})),
      dataKinds: symbol.dataKinds
    }));
  expect(project(inventory.symbols)).toEqual(project(extracted.symbols));
  for (const [symbol, fields, expected] of [
    ['GPURasterEmbeddingNearestPixels', 'outputs', ['pixelRows']],
    ['GPURasterZonalStatistics', 'outputs', ['denseZoneRows']],
    ['GPUNearestFeatureJoin', 'outputs', ['neighborRows']],
    ['GPUNearestFeatureWeights', 'props', ['neighborRows', 'neighborStatus']]
  ] as const) {
    const record = inventory.symbols.find(record => record.symbol === symbol)!;
    expect(
      record[fields].map(field => field.name),
      symbol
    ).toEqual(expect.arrayContaining([...expected]));
  }
  const entry = rasterEntries.find(entry => entry.id === 'raster-embedding-nearest-pixels')!;
  const record = inventory.symbols.find(record => record.symbol === entry.symbol)!;
  expect(
    getInventoryOutputIssues(
      record,
      entry.outputs.map(output => output.name)
    )
  ).toMatchObject({unknown: [], missing: []});
}, 60000);
