// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  compareRateClusterClasses,
  createRateClusterOverflowDataset
} from './curated-bindings-rates-fixtures';
import {runCatalogOperationAsync} from '../../src/analysis/catalog-bindings';
import type {CatalogFeatureCollection, CatalogPosition} from '../../src/analysis/catalog-geojson';
import {CURATED_RATE_CLUSTER_MAP_BINDING} from '../../src/curated/bindings/curated-rate-cluster-map-binding';
import {prepareGpuRateClusterMap} from '../../src/curated/bindings/curated-rate-cluster-map-gpu';
import {getKeplerCuratedAnalysis} from '../../src/curated/curated-analyses';
import {createKeplerCuratedAnalysisSample} from '../../src/curated/curated-analysis-samples';
import {
  buildKeplerCuratedAnalysisRequest,
  createKeplerCuratedAnalysisMapping,
  updateKeplerCuratedAnalysisMapping
} from '../../src/curated/curated-analysis-mapping';
import {profileKeplerFeatureCollection} from '../../src/profile/dataset-profile';
import {requireTestDevice} from '../../../analysis/test/utils/require-test-device';
import {computeEmpiricalBayesOracle} from '../../../analysis/test/gpu-spatial-analysis/rate-smoothing/empirical-bayes-oracle';
import {computeLocalPermutationOracle} from '../../../analysis/test/gpu-spatial-analysis/permutation-inference/permutation-oracle';
import {
  computeLocalMoranOracle,
  createSeededRandom,
  getQuadrant
} from '../../../analysis/test/gpu-spatial-analysis/spatial-autocorrelation/spatial-autocorrelation-oracle';
import {
  computeContiguityOracle,
  computeTransformOracle,
  type OraclePolygons
} from '../../../analysis/test/gpu-spatial-analysis/spatial-weights/spatial-weights-oracle';

/** The 7 by 6 scene from rate-cluster-map-recipe.spec.ts, with independently named role columns. */
function createRecipeDataset(): CatalogFeatureCollection {
  const random = createSeededRandom(21);
  const features: CatalogFeatureCollection['features'] = [];
  for (let row = 0; row < 6; row++) {
    for (let column = 0; column < 7; column++) {
      const population = 2000 + Math.floor(random() * 8000);
      const rate = column < 3 && row < 3 ? 0.08 : column >= 5 && row >= 4 ? 0.005 : 0.03;
      const cases = Math.round(population * rate * (0.9 + 0.2 * random()));
      const ring: CatalogPosition[] = [
        [column, row],
        [column + 1, row],
        [column + 1, row + 1],
        [column, row + 1],
        [column, row]
      ];
      features.push({
        id: `district-${features.length}`,
        geometry: {type: 'Polygon', coordinates: [ring]},
        properties: {reported: cases, exposed: population}
      });
    }
  }
  features[7 * 4 + 2].properties!['exposed'] = 0;
  // One multipart island with two disjoint shells; retain its own row and exclude it from inference.
  features.push({
    id: 'island',
    geometry: {
      type: 'MultiPolygon',
      coordinates: [
        [
          [
            [100, 100],
            [101, 100],
            [101, 101],
            [100, 101],
            [100, 100]
          ]
        ],
        [
          [
            [102, 100],
            [103, 100],
            [103, 101],
            [102, 101],
            [102, 100]
          ]
        ]
      ]
    },
    properties: {reported: 40, exposed: 2000}
  });
  return {type: 'FeatureCollection', features};
}

for (const dataset of ['sample', 'recipe-fixture'] as const) {
  it(`rate cluster map dispatches the public recipe on mapped ${dataset} data`, async ({
    annotate
  }) => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const areas =
      dataset === 'sample'
        ? createKeplerCuratedAnalysisSample('city-district-rates')['areas']
        : createRecipeDataset();
    const coordinateSystem = dataset === 'sample' ? 'EPSG:4326' : 'planar';
    const casesProperty = dataset === 'sample' ? 'cases' : 'reported';
    const populationProperty = dataset === 'sample' ? 'population' : 'exposed';
    const analysis = getKeplerCuratedAnalysis('rate-cluster-map')!;
    const profile = profileKeplerFeatureCollection(areas, {
      datasetId: 'independent-districts',
      label: dataset,
      coordinateSystem
    });
    let mapping = createKeplerCuratedAnalysisMapping(analysis, [profile]);
    for (const [role, column] of [
      ['cases', casesProperty],
      ['population', populationProperty]
    ])
      mapping = updateKeplerCuratedAnalysisMapping(
        mapping,
        {slot: 'areas', role, columns: [column]},
        analysis,
        [profile]
      );
    const request = buildKeplerCuratedAnalysisRequest(analysis, mapping, {
      parameterOverrides: {
        coordinateSystem,
        criterion: dataset === 'sample' ? 'queen' : 'rook',
        seed: 11,
        permutations: 999,
        significance: 0.05,
        multipleTesting: 'benjamini-hochberg'
      }
    });
    expect(request.inputDatasetIds).toEqual({areas: 'independent-districts'});
    const result = await runCatalogOperationAsync(
      CURATED_RATE_CLUSTER_MAP_BINDING,
      request.parameters,
      {areas},
      {device}
    );
    if (result.engine !== 'gpu')
      throw new Error(
        `Rate cluster ${dataset} ran ${result.engine}: ${result.fallbackReason}; ${JSON.stringify(result.diagnostics)}`
      );
    expect(result.dispatchedContributors).toEqual(
      CURATED_RATE_CLUSTER_MAP_BINDING.gpuContributorNames
    );
    const table = result.tables.find(candidate => candidate.id === 'rate-clusters')!;
    const prepared = prepareGpuRateClusterMap({areas}, request.parameters);
    // Independent brute-force polygon adjacency and existing recipe-fixture numerical oracles.
    const featureOfPolygon: number[] = [];
    const polygons: OraclePolygons = [];
    prepared.polygons.parts.forEach((parts, row) => {
      for (const rings of parts) {
        featureOfPolygon.push(row);
        polygons.push(rings.map(ring => ring.map(position => [position[0], position[1]])));
      }
    });
    const topology = computeContiguityOracle(
      polygons,
      request.parameters['criterion'] as 'queen' | 'rook',
      0,
      featureOfPolygon
    );
    const weights = {
      offsets: Uint32Array.from(topology.offsets),
      neighbors: Uint32Array.from(topology.neighbors),
      weights: Float32Array.from(computeTransformOracle(topology, 'row'))
    };
    const empiricalBayes = computeEmpiricalBayesOracle({
      events: prepared.events,
      populations: prepared.populations,
      clampPriorVariance: true
    });
    const actualRates = Float32Array.from(prepared.rows, row =>
      Number(table.columns['smoothed_rate'][row.sourceIndex])
    );
    const moran = computeLocalMoranOracle({weights, values: actualRates});
    const permutation = computeLocalPermutationOracle({
      weights,
      values: actualRates,
      statistic: 'localMoran',
      seed: 11,
      permutationCount: 999,
      significanceLevel: 0.05,
      maximumNeighbors: 64,
      alternative: 'two-sided',
      multipleTesting: 'benjamini-hochberg'
    });
    const mismatches: string[] = [];
    const checkNumber = (
      name: string,
      sourceIndex: number,
      expected: number,
      absolute: number,
      relative = 0
    ) => {
      const value = table.columns[name][sourceIndex];
      if (Number.isNaN(expected)) {
        if (value !== null)
          mismatches.push(`${name}[${sourceIndex}] expected null, received ${value}`);
      } else if (
        typeof value !== 'number' ||
        !Number.isFinite(value) ||
        Math.abs(value - expected) > absolute + relative * Math.abs(expected)
      )
        mismatches.push(
          `${name}[${sourceIndex}] ${value} != ${expected} (tolerance ${absolute} + ${relative} relative)`
        );
    };
    const effectiveSignificance = Number(result.summary['effectiveSignificance']);
    expect(effectiveSignificance).toBe(Math.fround(0.05));
    const labels = ['not-significant', 'HH', 'LH', 'LL', 'HL'];
    prepared.rows.forEach((row, position) => {
      const sourceIndex = row.sourceIndex;
      checkNumber('raw_rate', sourceIndex, empiricalBayes.raw[position], 2e-7, 2e-4);
      checkNumber('smoothed_rate', sourceIndex, empiricalBayes.smoothed[position], 2e-7, 5e-4);
      checkNumber('local_i', sourceIndex, moran.localI[position], 2e-3, 2e-3);
      checkNumber('spatial_lag', sourceIndex, moran.spatialLag[position], 2e-3, 2e-3);
      // At most three permutation draws (two-sided) plus 1% relative rounding tolerance.
      checkNumber('p_value', sourceIndex, permutation.pseudoPValues[position], 0.006, 0.01);
      checkNumber(
        'adjusted_p_value',
        sourceIndex,
        permutation.adjustedPValues[position],
        0.006,
        0.01
      );
      const neighbors = topology.offsets[position + 1] - topology.offsets[position];
      if (table.columns['neighbor_count'][sourceIndex] !== neighbors)
        mismatches.push(`neighbor_count[${sourceIndex}] != ${neighbors}`);
    });
    const classParity = compareRateClusterClasses(
      prepared.rows.map((row, position) => ({
        sourceIndex: row.sourceIndex,
        adjustedPValue: table.columns['adjusted_p_value'][row.sourceIndex] as number | null,
        oracleAdjustedPValue: permutation.adjustedPValues[position],
        actualClass: String(table.columns['cluster'][row.sourceIndex]),
        significantClass:
          labels[
            getQuadrant(actualRates[position] - moran.moments.mean, moran.spatialLag[position])
          ]
      })),
      effectiveSignificance
    );
    mismatches.push(...classParity.mismatches);
    // Persist boundary rows and coverage as a test annotation even on successful headless runs.
    // Include the same report in any thrown error; the headless runner drops console output.
    await annotate(JSON.stringify({dataset, ...classParity}), 'rate-cluster-class-parity');
    areas.features.forEach((feature, sourceIndex) => {
      if (
        table.columns['source_index'][sourceIndex] !== sourceIndex ||
        table.columns['source_id'][sourceIndex] !==
          (feature.id === undefined || feature.id === null ? null : String(feature.id))
      )
        mismatches.push(`identity[${sourceIndex}]`);
      if (table.columns['geometry'][sourceIndex] !== JSON.stringify(feature.geometry))
        mismatches.push(`publication geometry[${sourceIndex}]`);
      if (table.columns['status'][sourceIndex] !== prepared.statuses[sourceIndex])
        mismatches.push(`exclusion status[${sourceIndex}]`);
      if (prepared.statuses[sourceIndex] !== 'analysed') {
        for (const column of [
          'raw_rate',
          'smoothed_rate',
          'local_i',
          'p_value',
          'adjusted_p_value'
        ])
          if (table.columns[column][sourceIndex] !== null)
            mismatches.push(`excluded ${column}[${sourceIndex}]`);
        if (table.columns['cluster'][sourceIndex] !== 'not-tested')
          mismatches.push(`excluded class[${sourceIndex}]`);
      }
    });
    if (result.summary['testedCount'] !== classParity.testedCount)
      mismatches.push(
        `tested family ${result.summary['testedCount']} != ${classParity.testedCount}`
      );
    expect(table.rowCount).toBe(areas.features.length);
    expect(table.geometryColumn).toBe('geometry');
    expect(result.summary['minimumPseudoPValue']).toBe(2 / 1000);
    expect(result.summary['permutationAlternative']).toBe('two-sided-doubled-tail');
    expect(result.diagnostics.find(item => item.code === 'gpu-inference-method')).toBeDefined();
    expect(result.diagnostics.find(item => item.code === 'pseudo-p-values')?.message).toContain(
      'doubling the smaller tail'
    );
    if (mismatches.length)
      throw new Error(
        `Rate cluster ${dataset} recipe parity:\n${mismatches.join('\n')}\nClass comparison report: ${JSON.stringify(classParity)}`
      );

    // Canonical dispatch order must preserve inference under an independently reversed source order.
    const reversed = await runCatalogOperationAsync(
      CURATED_RATE_CLUSTER_MAP_BINDING,
      request.parameters,
      {areas: {type: 'FeatureCollection', features: [...areas.features].reverse()}},
      {device}
    );
    expect(reversed.engine).toBe('gpu');
    const reversedTable = reversed.tables[0];
    for (const column of ['smoothed_rate', 'local_i', 'p_value', 'adjusted_p_value', 'cluster'])
      expect(
        [...reversedTable.columns[column]].reverse(),
        `${dataset} row order ${column}`
      ).toEqual(table.columns[column]);
  }, 120000);
}

it('rate recipe marks constant rates untested and gives unsupported Sidak a visible CPU fallback', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const areas = createRecipeDataset();
  for (const feature of areas.features) feature.properties = {reported: 0, exposed: 1000};
  const parameters = {
    casesProperty: 'reported',
    populationProperty: 'exposed',
    coordinateSystem: 'planar',
    permutations: 99
  };
  const constant = await runCatalogOperationAsync(
    CURATED_RATE_CLUSTER_MAP_BINDING,
    parameters,
    {areas},
    {device}
  );
  if (constant.engine !== 'gpu')
    throw new Error(`Constant-rate GPU fallback: ${constant.fallbackReason}`);
  expect(constant.summary['testedCount']).toBe(0);
  expect(constant.tables[0].columns['p_value']).toEqual(
    new Array(areas.features.length).fill(null)
  );
  expect(new Set(constant.tables[0].columns['cluster'])).toEqual(new Set(['not-tested']));
  expect(constant.diagnostics.map(item => item.code)).toContain('constant-smoothed-rate');
  const fallback = await runCatalogOperationAsync(
    CURATED_RATE_CLUSTER_MAP_BINDING,
    {...parameters, multipleTesting: 'sidak'},
    {areas},
    {device}
  );
  expect(fallback.engine).toBe('cpu-reference');
  expect(fallback.fallbackReason).toContain('Sidak correction');
  expect(fallback.diagnostics.map(item => item.code)).toContain('gpu-fallback');
}, 120000);

it('rate recipe refuses a dense overflowing GPU topology and reports the CPU fallback', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const areas = createRateClusterOverflowDataset();
  const result = await runCatalogOperationAsync(
    CURATED_RATE_CLUSTER_MAP_BINDING,
    {
      casesProperty: 'reported',
      populationProperty: 'exposed',
      coordinateSystem: 'planar',
      permutations: 99
    },
    {areas},
    {device}
  );
  expect(result.engine).toBe('cpu-reference');
  expect(result.fallbackReason).toContain('GPU contiguity overflow');
  expect(result.diagnostics.find(item => item.code === 'gpu-fallback')).toBeDefined();
  expect(result.tables[0].rowCount).toBe(66);
}, 120000);

it('rate recipe leaves a constant connected population untested despite a varying island rate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const areas = createRecipeDataset();
  // Keep four touching squares and the isolated multipart area from the parity fixture.
  areas.features = [
    areas.features[0],
    areas.features[1],
    areas.features[7],
    areas.features[8],
    areas.features[42]
  ];
  areas.features.forEach((feature, row) => {
    feature.properties = {reported: row === 4 ? 900 : 100, exposed: 1000};
  });
  const result = await runCatalogOperationAsync(
    CURATED_RATE_CLUSTER_MAP_BINDING,
    {
      casesProperty: 'reported',
      populationProperty: 'exposed',
      coordinateSystem: 'planar',
      permutations: 99
    },
    {areas},
    {device}
  );
  if (result.engine !== 'gpu')
    throw new Error(`Island-spread regression ran ${result.engine}: ${result.fallbackReason}`);
  expect(result.summary['testedCount']).toBe(0);
  expect(result.summary['significantCount']).toBe(0);
  for (const column of ['local_i', 'spatial_lag', 'p_value', 'adjusted_p_value'])
    expect(result.tables[0].columns[column], column).toEqual(new Array(5).fill(null));
  expect(result.tables[0].columns['cluster']).toEqual(new Array(5).fill('not-tested'));
}, 120000);
