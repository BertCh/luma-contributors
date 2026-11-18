// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/// <reference types="vite/client" />
/// <reference path="../../../../../../website/src/env.d.ts" />

import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {catalogPlugin} from '../../../../../../website/build/catalog-plugin';
import {describe, expect, it, vi} from 'vitest';
import {
  GPU_ANALYSIS_QUESTION_RECORDS,
  GPU_ANALYSIS_WEBSITE_DATA_TYPES,
  getGPUAnalysisOperatorCatalog,
  getGPUAnalysisOperatorQuestions,
  resolveGPUAnalysisOperatorKinds
} from '../../../../src/gpu-spatial-analysis/ontology/index';
import {
  attachCatalogAnalysisMetadata,
  CATALOG_HELPER_KINDS,
  getCatalogAnalysisMetadata,
  getCatalogDataTypes
} from '../../../../../../website/src/catalog/analysis-metadata';
import type {CatalogEntry} from '../../../../../../website/src/catalog/schema';
import {DATA_TYPES, QUESTIONS} from '../../../../../../website/src/catalog/taxonomy';
import inventory from '../../../../../../website/src/catalog/generated/inventory.json';

const entryGroups = import.meta.glob<{entries: CatalogEntry[]}>(
  '../../../../../../website/src/catalog/entries/*.ts',
  {eager: true}
);
const entries = Object.values(entryGroups).flatMap(group => group.entries);

vi.mock('virtual:catalog-summaries', async () => {
  const groups = import.meta.glob<{entries: CatalogEntry[]}>(
    '../../../../../../website/src/catalog/entries/*.ts',
    {eager: true}
  );
  const {attachCatalogAnalysisMetadata: attachMetadata} = await import(
    '../../../../../../website/src/catalog/analysis-metadata'
  );
  return {
    default: Object.entries(groups).flatMap(([path, group]) =>
      group.entries.map(entry => ({
        ...attachMetadata(entry),
        group: path.split('/').at(-1)!.replace('.ts', '')
      }))
    )
  };
});

import {
  FACETS,
  catalog,
  filterEntries,
  getEntriesBy,
  getFacetCounts,
  getNextSteps,
  getPreviousSteps,
  loadEntry
} from '../../../../../../website/src/catalog/index';

describe('shared website analysis metadata', () => {
  it('declares the public local-permutation replay factory as a recording helper', () => {
    expect(
      getCatalogAnalysisMetadata(
        'createLocalPermutationReplay',
        '@luma-contributors/analysis/gpu-spatial-analysis'
      )
    ).toEqual({
      questions: ['prepare-data'],
      dataKinds: {
        input: ['attributes', 'spatial-weights', 'parameters'],
        output: ['attributes', 'summary']
      }
    });
  });
  it('retains both visibility recipe identities through inventory extraction and metadata lookup', () => {
    const recipes = inventory.symbols.filter(item => item.symbol === 'addVisibilityGraphRecipe');
    expect(recipes.map(item => item.importPath).sort()).toEqual([
      '@luma-contributors/analysis/gpu-network',
      '@luma-contributors/analysis/gpu-spatial-analysis'
    ]);
    expect(new Set(recipes.map(item => item.sourceFile)).size).toBe(2);
    for (const recipe of recipes) {
      expect(getCatalogAnalysisMetadata(recipe.symbol, recipe.importPath).dataKinds).toEqual(
        recipe.dataKinds
      );
    }
    expect(() => getCatalogAnalysisMetadata('addVisibilityGraphRecipe')).toThrow(
      /ambiguous.*importPath/
    );
  });
  it('retains visibility overloads and assigns documentation and lenses by qualified identity', () => {
    for (const entryPoint of ['gpu-network', 'gpu-spatial-analysis']) {
      const recipe = inventory.symbols.find(
        item =>
          item.symbol === 'addVisibilityGraphRecipe' &&
          item.importPath === `@luma-contributors/analysis/${entryPoint}`
      )!;
      expect(recipe.docsSlug).toBe(
        entryPoint === 'gpu-network'
          ? 'api/gpu-network/visibility-graph'
          : 'api/gpu-spatial-analysis/urban-morphology'
      );
      expect(recipe.lensScenarios).toEqual([`${entryPoint}.visibility-graph-recipe`]);
      expect((recipe as unknown as {overloads?: unknown[]}).overloads).toHaveLength(2);
    }
  });
  it('does not silently discard the public centrality continuation factory', () => {
    expect(
      inventory.symbols.some(item => item.symbol === 'createNetworkCentralityContinuation')
    ).toBe(true);
  });
  it('uses the shared question ids, labels, descriptions and order, including the twelfth question', () => {
    expect(QUESTIONS).toEqual(
      GPU_ANALYSIS_QUESTION_RECORDS.map(question => ({
        id: question.id,
        label: question.label,
        description: question.summary
      }))
    );
    expect(FACETS.question).toBe(QUESTIONS);
    expect(QUESTIONS).toHaveLength(12);
    expect(DATA_TYPES.map(type => type.id)).toEqual([...GPU_ANALYSIS_WEBSITE_DATA_TYPES]);
  });

  it('covers every authored entry and extracted symbol, using the authoritative operator kinds', () => {
    const operators = new Map(
      getGPUAnalysisOperatorCatalog().map(operator => [
        `${operator.package}:${operator.exportName}`,
        operator
      ])
    );
    expect(entries.length).toBeGreaterThan(500);
    for (const item of inventory.symbols) {
      const metadata = getCatalogAnalysisMetadata(item.symbol, item.importPath);
      expect(item.questions, item.symbol).toEqual(metadata.questions);
      expect(item.dataKinds, item.symbol).toEqual(metadata.dataKinds);
      const operator = operators.get(`${item.importPath}:${item.symbol}`);
      if (operator) {
        const kinds = resolveGPUAnalysisOperatorKinds(operator);
        expect(metadata.dataKinds, item.symbol).toEqual({
          input: kinds.inputs,
          output: kinds.outputs
        });
        expect(metadata.questions, item.symbol).toEqual(getGPUAnalysisOperatorQuestions(operator));
        expect(item.ontology?.inputs, item.symbol).toEqual(kinds.inputs);
        expect(item.ontology?.outputs, item.symbol).toEqual(kinds.outputs);
      } else expect(Object.hasOwn(CATALOG_HELPER_KINDS, item.symbol), item.symbol).toBe(true);
    }
    for (const entry of entries) {
      const resolved = attachCatalogAnalysisMetadata(entry);
      expect(resolved.questions.length, entry.symbol).toBeGreaterThan(0);
      expect(resolved.interpretation?.summary, entry.symbol).toBe(entry.explanation);
      expect(Array.isArray(resolved.interpretation?.cautions), entry.symbol).toBe(true);
      expect(
        inventory.symbols.some(
          item => item.symbol === entry.symbol && item.importPath === entry.importPath
        ),
        entry.symbol
      ).toBe(true);
    }
    expect(() => getCatalogAnalysisMetadata('GPUUnknownWebsiteOperation')).toThrow(
      'shared analysis metadata'
    );
    expect(() => getCatalogAnalysisMetadata('constructor')).toThrow('shared analysis metadata');
  });

  it('documents named-method limitations without asserting unobserved run conditions', () => {
    const symbols = [
      'GPUKernelDensity',
      'GPUHotSpotAnalysis',
      'GPULocalMoran',
      'GPUQuantileLISA',
      'GPUSpatialClustering',
      'GPUKriging',
      'GPUInverseDistanceWeighting',
      'GPUOrdinaryLeastSquares',
      'GPUGeographicallyWeightedRegression',
      'addHotSpotAnalysisRecipe',
      'addSpaceTimeHotSpotsRecipe',
      'addClusterAndOutlineRecipe',
      'addPointsInPolygonsChoroplethRecipe',
      'addZonalDistributionRecipe',
      'addRateClusterMapRecipe',
      'addSpatialRegressionRecipe',
      'addDrainageNetworkRecipe'
    ];
    for (const symbol of symbols) {
      const entry = entries.find(candidate => candidate.symbol === symbol);
      expect(entry, symbol).toBeDefined();
      const interpretation = attachCatalogAnalysisMetadata(entry!).interpretation;
      expect(interpretation?.summary, symbol).toBeTruthy();
      expect(interpretation?.cautions.length, symbol).toBeGreaterThan(0);
    }
    const authored = {
      symbol: 'GPUKernelDensity',
      explanation: 'Existing explanation.',
      interpretation: {
        summary: 'Authored interpretation.',
        cautions: ['pseudo-p-values'] as const
      }
    };
    expect(attachCatalogAnalysisMetadata(authored).interpretation).toBe(authored.interpretation);
  });

  it('filters questions jointly with other facets and counts under the other selections', () => {
    const question = 'where-significant';
    const expected = catalog.filter(entry => entry.questions?.includes(question));
    expect(getEntriesBy('question', question)).toEqual(expected);
    expect(filterEntries({question, dataType: 'points'})).toEqual(
      expected.filter(entry => Object.values(getCatalogDataTypes(entry)).flat().includes('points'))
    );
    expect(filterEntries({question: 'unknown-question'})).toEqual([]);
    const counts = getFacetCounts({question, dataType: 'points'});
    for (const facet of QUESTIONS) {
      expect(counts.question[facet.id], facet.id).toBe(
        filterEntries({dataType: 'points', question: facet.id}).length
      );
    }
    expect(counts.data['points']).toBe(filterEntries({question, dataType: 'points'}).length);
  });

  it('discovers point-capable geometry measures from shared kinds despite legacy polygon-only ports', () => {
    const measures = catalog.find(entry => entry.symbol === 'GPUGeometryMeasures')!;
    const legacy = {
      ...measures,
      dataTypes: {input: ['polygons'] as const, output: ['table'] as const}
    };
    const attached = attachCatalogAnalysisMetadata(legacy);
    expect(attached.dataTypes.input).toEqual(['points', 'lines', 'polygons']);
    expect(filterEntries({dataType: 'points'}, [measures])).toEqual([measures]);
    expect(getFacetCounts({query: 'GPUGeometryMeasures'}).data['points']).toBe(1);
    const producer = {
      ...measures,
      id: 'point-producer',
      dataKinds: {input: [], output: ['points'] as const}
    };
    const consumer = {
      ...measures,
      id: 'value-consumer',
      dataKinds: {input: ['attributes'] as const, output: []}
    };
    expect(getNextSteps(producer, catalog.length)).toContainEqual(measures);
    expect(getPreviousSteps(consumer, catalog.length)).toContainEqual(measures);
  });

  it('does not advertise network assignment and cost tables as polygon producers', () => {
    for (const symbol of [
      'GPUNetworkServiceAreas',
      'GPUNetworkCostMatrix',
      'GPUNetworkCostTable'
    ]) {
      const entry = catalog.find(candidate => candidate.symbol === symbol)!;
      expect(getCatalogAnalysisMetadata(symbol).dataKinds.output).not.toContain('polygons');
      expect(filterEntries({dataType: 'polygons'}, [entry])).toEqual([]);
      expect(getCatalogDataTypes(entry).output).toEqual(['table']);
    }
  });

  it('projects every published port with the shared helpers and explicitly refines point clouds', () => {
    for (const entry of catalog) {
      expect(entry.dataTypes, entry.symbol).toEqual(getCatalogDataTypes(entry));
    }
    const cloud = getCatalogDataTypes({symbol: 'GPUPointCloudFeatures'});
    expect(cloud.input).toContain('point-cloud');
    expect(cloud.input).not.toContain('points');
    expect(getCatalogDataTypes({symbol: 'makePointCloudLocalPositions'})).toEqual({
      input: ['point-cloud'],
      output: ['point-cloud']
    });
    expect(getCatalogDataTypes({symbol: 'makeGPURasterMetadata'})).toEqual({input: [], output: []});
    expect(getCatalogDataTypes({symbol: 'GPUTerrainDerivatives'}).input).toEqual(
      expect.arrayContaining(['raster', 'terrain'])
    );
  });

  it('serializes shared metadata through the real build plugin while keeping interpretation in details', async () => {
    const plugin = catalogPlugin({
      entriesDirectory: fileURLToPath(
        new URL('../../../../../../website/src/catalog/entries', import.meta.url)
      )
    });
    const load = (typeof plugin.load === 'function' ? plugin.load : plugin.load!.handler) as (
      this: {addWatchFile: (file: string) => void},
      id: string
    ) => Promise<string>;
    const watchedFiles: string[] = [];
    const source = await load.call(
      {addWatchFile: file => watchedFiles.push(file)},
      '\0virtual:catalog-summaries'
    );
    const summaries = JSON.parse(
      source.replace(/^export default /, '').replace(/;$/, '')
    ) as CatalogEntry[];
    expect(summaries).toHaveLength(entries.length);
    expect(watchedFiles).toHaveLength(Object.keys(entryGroups).length);
    for (const summary of summaries) {
      expect(summary.questions, summary.symbol).toEqual(
        getCatalogAnalysisMetadata(summary.symbol, summary.importPath).questions
      );
      expect(summary.dataKinds, summary.symbol).toEqual(
        getCatalogAnalysisMetadata(summary.symbol, summary.importPath).dataKinds
      );
      expect(summary).not.toHaveProperty('interpretation');
      expect(summary).not.toHaveProperty('explanation');
    }
  });

  it('attaches the same metadata to lazy detail loads and build-time summaries', async () => {
    const entry = await loadEntry('local-morans-i');
    expect(entry?.questions).toEqual(getCatalogAnalysisMetadata('GPULocalMoran').questions);
    expect(entry?.interpretation?.cautions.length).toBeGreaterThan(0);
    expect(await loadEntry('missing-entry')).toBeUndefined();
    const exploreSource = readFileSync(
      new URL('../../../../../../website/src/routes/explore/index.tsx', import.meta.url),
      'utf8'
    );
    expect(exploreSource).toContain("question: readText(search['question'])");
    expect(exploreSource).toContain('question: search.question');
  });
});
