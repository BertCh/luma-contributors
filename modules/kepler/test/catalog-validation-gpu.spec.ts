// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {readFloat32Vector} from '../../../examples/deck/kepler-compute/src/kriging-diagnostics';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {
  buildVecchiaConditioningSets,
  computeVecchiaFisherScoringOnCPU,
  computeVecchiaLogLikelihoodOnCPU,
  computeVecchiaSpaceTimeLogLikelihoodOnCPU,
  getGPUVecchiaLogLikelihoodParameterValues,
  getGPUVecchiaSpaceTimeLogLikelihoodParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import {makeCatalogTableInput, type CatalogDataInput} from '../src/analysis/catalog-data-input';
import {
  parseCatalogFeatureCollection,
  type CatalogFeatureCollection,
  type CatalogOutputTable
} from '../src/analysis/catalog-geojson';
import {VALIDATION_BINDINGS} from '../src/analysis/catalog-validation-bindings';
import {makeValidationSpaceTimeCovariance} from '../src/analysis/catalog-validation-vecchia';
import {getCatalogSampleInputs} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {getKeplerOperationDescriptor} from '../src/descriptors/operation-descriptors';
import {planKeplerResultPresentation} from '../src/presentation/result-presentation';
import {fitKeplerVecchiaVariogram} from '../src/analysis/kriging-variogram-fit';
import {addKeplerAnalysisRecipe} from '../src/analysis/recipe-bindings';
import {
  restoreKeplerAnalysisRecipe,
  serializeKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe
} from '../src/recipes/analysis-recipe';
import {
  createInputBuffer,
  createOutputBuffer,
  importGraphBuffer,
  readFloat32,
  submitGraph
} from '@luma-contributors/analysis/testing';

async function runOperation(
  id: string,
  parameters: Record<string, unknown>,
  inputs: Record<string, CatalogDataInput>
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return runCatalogOperationAsync(getCatalogBinding(id)!, parameters, inputs, {device});
}

function points(rows: Record<string, unknown>[]): CatalogFeatureCollection {
  return {
    type: 'FeatureCollection',
    features: rows.map((properties, index) => ({
      geometry: {type: 'Point', coordinates: [index * 10, index % 3]},
      properties
    }))
  };
}
function readMetric(table: CatalogOutputTable, name: string): number {
  return table.columns['value'][table.columns['metric'].indexOf(name)] as number;
}

test('column-only scores return the same indexed values for table and feature routes', async () => {
  const rows = [
    {observed: 1, predicted: 2, variance: 0, member1: 2, member2: 2, pvalue: 0.01},
    {observed: 3, predicted: 3, variance: 0, member1: 3, member2: 3, pvalue: 0.2}
  ];
  const table: CatalogOutputTable = {
    id: 'forecasts',
    label: 'Forecasts',
    rowCount: 2,
    columns: Object.fromEntries(
      Object.keys(rows[0]).map(name => [name, rows.map(row => row[name as keyof typeof row])])
    ),
    columnTypes: Object.fromEntries(Object.keys(rows[0]).map(name => [name, 'float64']))
  };
  const input = makeCatalogTableInput(table, {maximumRows: 2048});
  for (const id of ['validation-gaussian-scores', 'validation-ensemble-scores', 'validation-fdr']) {
    const parameters =
      id === 'validation-ensemble-scores'
        ? {memberProperties: 'member1,member2'}
        : id === 'validation-fdr'
          ? {method: 'bonferroni', alpha: 0.05}
          : {};
    const tableResult = await runOperation(id, parameters, {points: input});
    const featureResult = await runOperation(id, parameters, {points: points(rows)});
    const output = tableResult.tables[0];
    expect(output.rowCount).toBe(2);
    expect(featureResult.tables[0].rowCount).toBe(2);
    expect(output.geometryColumn).toBeUndefined();
    const featureGeometryColumn = featureResult.tables[0].geometryColumn;
    expect(featureGeometryColumn).toBe('geometry');
    expect(
      featureResult.tables[0].columns[featureGeometryColumn!].map(value =>
        JSON.parse(String(value))
      )
    ).toEqual([
      {type: 'Point', coordinates: [0, 0]},
      {type: 'Point', coordinates: [10, 1]}
    ]);
    expect(output.columns['feature_index']).toEqual([0, 1]);
    for (const [name, values] of Object.entries(output.columns))
      expect(values).toEqual(featureResult.tables[0].columns[name]);
    // A zero-variance Gaussian and identical ensemble members are point forecasts:
    // CRPS is |prediction - observation| = [1, 0]. Bonferroni doubles the two p-values.
    const expected = id === 'validation-fdr' ? [0.02, 0.4] : [1, 0];
    const actual = output.columns[id === 'validation-fdr' ? 'adjusted_pvalue' : 'crps'];
    expect(actual).toHaveLength(2);
    actual.forEach((value, index) => {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeCloseTo(expected[index], 6);
    });
    if (id === 'validation-fdr') expect(output.columns['significant']).toEqual([1, 0]);
  }
});

test('B1 catalog variogram score counts both adjacent directions without cancellation', async () => {
  const result = await runOperation(
    'validation-ensemble-scores',
    {
      memberProperties: 'member1,member2',
      variogramOrder: 1
    },
    {
      points: points([
        {observed: 0, member1: 0, member2: 0},
        {observed: 10000001, member1: 10000000, member2: 10000000}
      ])
    }
  );
  // Each directed pair contributes (|10000001-0| - mean(|10000000-0|))^2 = 1.
  // The binding explicitly lists (0,1) and (1,0), so the exact total is 2.
  const score = readMetric(result.tables[1], 'variogram_score');
  expect(Number.isFinite(score)).toBe(true);
  expect(score).toBe(2);
});

test.each([
  {
    name: 'seam',
    bounds: [179.5, -0.125, 179.75, 0.125] as const,
    longitudes: [-179.875, -179.75, -179.625],
    latitudes: [0, 0, 0]
  },
  {
    name: 'pole',
    bounds: [-0.125, 89.375, 0.125, 89.625] as const,
    longitudes: [179.875, -90, 90],
    latitudes: [89.5, 89.5, 89.5]
  }
])('wrapped interpolation binding replays a constant field across the $name', async fixture => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const kind of ['idw-points', 'kriging-points'] as const) {
    const graph = new GPUCommandGraph(device);
    const longitudeBuffer = createInputBuffer(device, Float32Array.from(fixture.longitudes));
    const latitudeBuffer = createInputBuffer(device, Float32Array.from(fixture.latitudes));
    const sourceBuffer = createInputBuffer(device, Float32Array.of(7, 7, 7));
    const outputBuffer = createOutputBuffer(device, 1);
    const common = {
      longitudeColumn: 'longitude',
      latitudeColumn: 'latitude',
      coordinateSpace: 'longitude-latitude' as const,
      metric: 'great-circle' as const,
      units: 'meters' as const,
      longitudeSearch: 'wrapped' as const,
      grid: {bounds: fixture.bounds, width: 1, height: 1},
      neighborCount: 3,
      column: 'value'
    };
    const recipe: KeplerAnalysisRecipe = {
      schemaVersion: 1,
      algorithmVersion: 1,
      id: `wrapped-${kind}`,
      source: {datasetId: 'wrapped-source', revision: 1},
      population: 'all',
      backendPolicy: 'require-gpu',
      operation:
        kind === 'idw-points'
          ? {
              ...common,
              kind,
              searchRadius: 150000,
              power: 2,
              minimumNeighborCount: 3,
              taper: 'none'
            }
          : {
              ...common,
              kind,
              searchRadiusMeters: 150000,
              minimumNeighbors: 3,
              variogram: {model: 'exponential', nugget: 0, sill: 1, rangeMeters: 150000}
            }
    };
    try {
      const restored = restoreKeplerAnalysisRecipe(serializeKeplerAnalysisRecipe(recipe), {
        resolveDatasetRevision: () => 1
      });
      const result = addKeplerAnalysisRecipe(graph, restored, {
        source: recipe.source,
        columns: {
          longitude: importGraphBuffer(graph, 'longitude', longitudeBuffer, 'float32', 3),
          latitude: importGraphBuffer(graph, 'latitude', latitudeBuffer, 'float32', 3),
          value: importGraphBuffer(graph, 'value', sourceBuffer, 'float32', 3)
        },
        outputs: {values: importGraphBuffer(graph, 'prediction', outputBuffer, 'float32', 1)}
      });
      try {
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          const [prediction] = await readFloat32(outputBuffer, 1);
          // IDW's normalized weights and ordinary kriging's sum-to-one constraint preserve 7.
          expect(Number.isFinite(prediction)).toBe(true);
          expect(prediction).toBeCloseTo(7, 4);
        } finally {
          compiled.destroy();
        }
      } finally {
        result.destroyParameters();
      }
    } finally {
      longitudeBuffer.destroy();
      latitudeBuffer.destroy();
      sourceBuffer.destroy();
      outputBuffer.destroy();
    }
  }
});

for (const binding of VALIDATION_BINDINGS)
  test(`${binding.id} dispatches contributor and publishes declared result/diagnostic tables`, async () => {
    const sample = getCatalogSampleInputs(binding.id);
    const inputs = Object.fromEntries(
      binding.inputs.map(input => [input.name, parseCatalogFeatureCollection(sample[input.name])])
    );
    const result = await runOperation(binding.id, {}, inputs);
    expect(result.engine).toBe('gpu');
    expect(result.dispatchedContributors).toEqual(binding.contributorNames);
    expect(result.tables.length).toBeGreaterThanOrEqual(2);
    expect(
      result.tables.every(table => binding.outputs.some(output => output.tableId === table.id))
    ).toBe(true);
    expect(result.tables[0].rowCount).toBe(inputs[binding.inputs.at(-1)!.name].features.length);
    expect(result.tables.at(-1)!.columns['metric'].length).toBeGreaterThan(0);
    // AX-80: verify presentation metadata against real contributor output, not synthetic tables.
    const descriptor = getKeplerOperationDescriptor(binding.id)!;
    for (const output of descriptor.outputs) {
      const table = result.tables.find(candidate => candidate.id === output.tableId);
      if (!table)
        throw new Error(`${binding.id}: descriptor table ${output.tableId} was not published`);
      if (output.kind === 'summary') {
        if (!table.columns['metric'] || !table.columns['value'])
          throw new Error(`${binding.id}: ${output.tableId} must publish metric/value diagnostics`);
        continue;
      }
      const presentation = planKeplerResultPresentation(table, output);
      if (!presentation || presentation.column !== output.presentation?.column)
        throw new Error(
          `${binding.id}: ${output.tableId} has no usable ${output.presentation?.column} presentation; published columns: ${Object.keys(table.columns).join(', ')}`
        );
      const source = inputs[output.kindFrom!];
      if (table.rowCount !== source.features.length)
        throw new Error(
          `${binding.id}: ${output.tableId} rows must preserve the ${output.kindFrom} input`
        );
      if (
        output.tableId === 'validation-fdr' &&
        (presentation.classScheme !== 'significance-flag' || (presentation.unrecognized ?? 0) !== 0)
      )
        throw new Error(`${binding.id}: significant must use recognized significance-flag classes`);
    }
  });

test.each([
  'regression-gtwr',
  'regression-mgtwr'
])('%s recovers known coefficients and excludes a missing predictor', async id => {
  const source = points(
    Array.from({length: 24}, (_, index) => ({
      predictor: index === 23 ? null : Math.sin(index),
      observed: 3 + 2 * Math.sin(index),
      time: index
    }))
  );
  const result = await runOperation(
    id,
    {coordinateSystem: 'planar', bandwidth: 10000, timeScale: 0},
    {points: source}
  );
  const table = result.tables[0];
  for (let row = 0; row < 23; row++) {
    expect(table.columns['coef_intercept'][row]).toBeCloseTo(3, 3);
    expect(table.columns['coef_predictor'][row]).toBeCloseTo(2, 3);
  }
  expect(table.columns['coef_predictor'][23]).toBeNull();
  expect(table.columns['status'][23]).toBe(2);
});

test('space-time kriging preserves a constant field at off-time query points', async () => {
  const samples = points(Array.from({length: 12}, (_, index) => ({observed: 4, time: index})));
  const queries = points([{time: 2.5}, {time: 7.5}]);
  const result = await runOperation(
    'interpolation-space-time-kriging',
    {coordinateSystem: 'planar', range: 100, anisotropy: 10},
    {points: samples, queries}
  );
  expect(result.tables[0].columns['status']).toEqual([0, 0]);
  for (const prediction of result.tables[0].columns['prediction'] as number[])
    expect(prediction).toBeCloseTo(4, 4);
  for (const variance of result.tables[0].columns['variance'] as number[])
    expect(variance).toBeGreaterThan(0);
});

test.each([
  'likelihood',
  'fisher',
  'space-time'
])('Vecchia %s wrapper matches the existing library CPU twin including an excluded row', async kind => {
  const source = points(
    Array.from({length: 9}, (_, index) => ({
      observed: index === 8 ? null : Math.sin(index),
      time: index
    }))
  );
  const positions = Float32Array.from(
    source.features.flatMap(feature => feature.geometry!.coordinates as number[])
  );
  const observations = Float32Array.from(
    source.features.flatMap((feature, index) => [
      index === 8 ? NaN : (feature.properties!['observed'] as number),
      0,
      0,
      index === 8 ? 0 : 1
    ])
  );
  const times = Float32Array.from({length: 9}, (_, index) => index);
  const covariance = {model: 'exponential' as const, sill: 1, nugget: 0.05, range: 100};
  const neighborhoods = buildVecchiaConditioningSets({
    positions,
    neighborCount: 3,
    validity: Uint32Array.from({length: 9}, (_, index) => (index < 8 ? 1 : 0)),
    ...(kind === 'space-time' ? {times, timeScale: 2} : {})
  });
  const input = {
    positions,
    observations,
    neighborhoodOffsets: neighborhoods.offsets,
    neighborhoods: neighborhoods.neighbors,
    parameters:
      kind === 'space-time'
        ? getGPUVecchiaSpaceTimeLogLikelihoodParameterValues(
            makeValidationSpaceTimeCovariance(covariance, 2)
          )
        : getGPUVecchiaLogLikelihoodParameterValues(covariance)
  };
  const oracle =
    kind === 'fisher'
      ? computeVecchiaFisherScoringOnCPU(input)
      : kind === 'space-time'
        ? computeVecchiaSpaceTimeLogLikelihoodOnCPU({...input, times})
        : computeVecchiaLogLikelihoodOnCPU(input);
  const result = await runOperation(
    `validation-vecchia-${kind}`,
    {
      coordinateSystem: 'planar',
      neighborCount: 3,
      range: 100,
      ...(kind === 'space-time' ? {anisotropy: 2} : {})
    },
    {points: source}
  );
  expect(result.tables[0].columns['status']).toEqual(Array.from(oracle.siteStatus));
  const stride = kind === 'fisher' ? 10 : 1;
  for (let row = 0; row < 8; row++)
    expect(result.tables[0].columns['site_log_likelihood'][row]).toBeCloseTo(
      oracle.siteTerms[row * stride],
      3
    );
  if (kind === 'fisher')
    for (let row = 0; row < 8; row++)
      expect(result.tables[0].columns['score_log_range'][row]).toBeCloseTo(
        oracle.siteTerms[row * 10 + 2],
        3
      );
});

test('split conformal order statistic and unbounded statuses are visible', async () => {
  const calibration = points(
    Array.from({length: 9}, (_, index) => ({observed: index + 1, predicted: 0}))
  );
  const queries = points([{predicted: 10}, {predicted: null}]);
  const result = await runOperation(
    'validation-conformal',
    {miscoverage: 0.2},
    {calibration, queries}
  );
  expect(result.tables[0].columns['lower']).toEqual([2, null]);
  expect(result.tables[0].columns['upper']).toEqual([18, null]);
  expect(result.tables[0].columns['status']).toEqual([0, 4]);
  const unbounded = await runOperation(
    'validation-conformal',
    {miscoverage: 0.01},
    {calibration, queries: points([{predicted: 10}])}
  );
  expect(unbounded.tables[0].columns['lower']).toEqual([null]);
  expect(unbounded.tables[0].columns['upper']).toEqual([null]);
  expect(unbounded.tables[0].columns['status']).toEqual([3]);
});

test('geographic conformal intervals reproduce a constant calibration residual with equal local weights', async () => {
  const calibration = points(Array.from({length: 12}, () => ({observed: 1, predicted: 0})));
  const result = await runOperation(
    'validation-geo-conformal',
    {coordinateSystem: 'planar', bandwidth: 100000, miscoverage: 0.2},
    {calibration, queries: points([{predicted: 10}])}
  );
  expect(result.tables[0].columns['lower']).toEqual([9]);
  expect(result.tables[0].columns['upper']).toEqual([11]);
  expect(result.tables[0].columns['status']).toEqual([0]);
});

test('area of applicability distinguishes exact training predictors and extrapolation', async () => {
  const training = points(Array.from({length: 8}, (_, index) => ({predictor: index})));
  const result = await runOperation(
    'validation-area-of-applicability',
    {},
    {training, queries: points([{predictor: 0}, {predictor: 100}])}
  );
  expect(result.tables[0].columns['applicable']).toEqual([1, 0]);
  expect(result.tables[0].columns['dissimilarity'][0]).toBe(0);
  expect(result.tables[1].rowCount).toBe(8);
  expect(readMetric(result.tables[2], 'status')).toBe(0);
});

test('ensemble CRPS uses member-major inputs and fair correction; Gaussian zero variance is absolute error', async () => {
  const source = points([{observed: 0, member1: -1, member2: 1, predicted: 2, variance: 0}]);
  const empirical = await runOperation(
    'validation-ensemble-scores',
    {memberProperties: 'member1,member2'},
    {points: source}
  );
  expect(empirical.tables[0].columns['crps'][0]).toBeCloseTo(0.5, 6);
  const fair = await runOperation(
    'validation-ensemble-scores',
    {memberProperties: 'member1,member2', fair: true},
    {points: source}
  );
  expect(fair.tables[0].columns['crps'][0]).toBeCloseTo(0, 6);
  const gaussian = await runOperation('validation-gaussian-scores', {}, {points: source});
  expect(gaussian.tables[0].columns['crps'][0]).toBeCloseTo(2, 6);
});

test('FDR wrapper adjusts exactly the valid testing family with Benjamini-Hochberg', async () => {
  const result = await runOperation(
    'validation-fdr',
    {method: 'benjamini-hochberg'},
    {points: points([0.01, 0.04, 0.03, 0.2, null, -1].map(pvalue => ({pvalue})))}
  );
  const table = result.tables[0];
  [0.04, 0.053333333, 0.053333333, 0.2].forEach((value, index) =>
    expect(table.columns['adjusted_pvalue'][index]).toBeCloseTo(value, 6)
  );
  expect(table.columns['significant']).toEqual([1, 0, 0, 0, 0, 0]);
  expect(table.columns['adjusted_pvalue'].slice(4)).toEqual([null, null]);
  expect(readMetric(result.tables[1], 'tested_count')).toBe(4);
  const presentation = planKeplerResultPresentation(
    table,
    getKeplerOperationDescriptor('validation-fdr')!.outputs[0]
  );
  const nonsignificant = presentation?.legend.find(entry => entry.value === 0)?.count;
  const untested = presentation?.legend.find(entry => entry.label === 'Untested')?.count;
  if (nonsignificant !== 3 || untested !== 2 || presentation?.excludedRows?.join(',') !== '4,5') {
    throw new Error(
      `validation-fdr: expected 3 tested nonsignificant rows and 2 untested rows at indices 4,5; presentation: ${JSON.stringify(presentation)}`
    );
  }
});

test.each([
  'likelihood',
  'fisher'
])('ordinary-kriging %s fitting returns finite replayable covariance and convergence diagnostics', async method => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const longitudes = Float32Array.from({length: 16}, (_, index) => -74 + (index % 4) * 0.002);
  const latitudes = Float32Array.from(
    {length: 16},
    (_, index) => 40 + Math.floor(index / 4) * 0.002
  );
  const values = Float32Array.from({length: 16}, (_, index) => Math.sin(index * 1.4));
  const fit = await fitKeplerVecchiaVariogram(device, longitudes, latitudes, values, {
    method: method as 'likelihood' | 'fisher',
    initial: {model: 'exponential', sill: 1, nugget: 0.05, range: 300},
    maximumIterations: 5
  });
  expect(Number.isFinite(fit.criterion)).toBe(true);
  expect(fit.rangeMeters).toBeGreaterThan(0);
  expect(fit.diagnostics!['valid_rows']).toBe(16);
  expect(fit.diagnostics!['failed_sites']).toBe(0);
});

test('variogram readback preserves nullable interleaved chunk offsets rather than fitting null payloads', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const buffer = device.createBuffer({
    data: Float32Array.from([99, 10, 99, 999, 99, 30]),
    usage: Buffer.STORAGE | Buffer.COPY_SRC
  });
  const vector = new GPUVector({
    type: 'data',
    name: 'nullable',
    data: [
      new GPUData({
        buffer,
        format: 'float32',
        length: 3,
        byteOffset: 4,
        byteStride: 8,
        rowByteLength: 4,
        nullBitmap: Uint8Array.of(0b101)
      })
    ]
  });
  try {
    expect(Array.from(await readFloat32Vector(vector))).toEqual([10, NaN, 30]);
  } finally {
    buffer.destroy();
  }
});
