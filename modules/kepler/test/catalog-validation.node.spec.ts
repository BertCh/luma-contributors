// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  listCatalogEntries,
  normalizeCatalogParameters,
  parseCatalogInvocation,
  serializeCatalogInvocation,
  validateCatalogInputs,
  runCatalogOperationAsync
} from '../src/analysis/catalog-bindings';
import {parseCatalogFeatureCollection} from '../src/analysis/catalog-geojson';
import {VALIDATION_BINDINGS} from '../src/analysis/catalog-validation-bindings';
import {
  getValidationTemporalContext,
  makeValidationResult,
  readValidationColumn,
  readValidationMatrix,
  readValidationNames,
  readValidationPositions,
  readValidationTimes
} from '../src/analysis/catalog-validation-common';
import {getCatalogSampleInputs} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {parseKeplerKrigingSettings} from '../src/recipes/analysis-recipe';
import {
  evaluateValidationVecchia,
  fitValidationVecchia,
  isValidationVecchiaCovarianceUsable
} from '../src/analysis/catalog-validation-vecchia';
import {fitKeplerVecchiaVariogram} from '../src/analysis/kriging-variogram-fit';
import type {Device} from '@luma.gl/core';

const CONTRIBUTORS = [
  'GPUGeographicallyTemporallyWeightedRegression',
  'GPUMultiscaleGeographicallyTemporallyWeightedRegression',
  'GPUSpaceTimeKriging',
  'GPUVecchiaLogLikelihood',
  'GPUVecchiaFisherScoring',
  'GPUVecchiaSpaceTimeLogLikelihood',
  'GPUAreaOfApplicability',
  'GPUConformalPredictionIntervals',
  'GPUGeoConformalPredictionIntervals',
  'GPUEnsembleScores',
  'GPUGaussianPredictiveScores',
  'GPURasterFalseDiscoveryRateMask'
];

test('all twelve contributors are registered once, discoverable and GPU executable', () => {
  expect(VALIDATION_BINDINGS).toHaveLength(12);
  expect(VALIDATION_BINDINGS.flatMap(binding => binding.contributorNames).sort()).toEqual(
    [...CONTRIBUTORS].sort()
  );
  for (const binding of VALIDATION_BINDINGS) {
    expect(getCatalogBinding(binding.id)).toBe(binding);
    expect(binding.engine).toBe('gpu');
    expect(binding.runGpu).toBeTypeOf('function');
    expect(binding.run).toBeUndefined();
    expect(binding.outputs.some(output => output.tableId.endsWith('-diagnostics'))).toBe(true);
    const entries = listCatalogEntries().filter(entry =>
      binding.contributorNames.includes(entry.contributorName)
    );
    expect(entries.length).toBeGreaterThan(0);
    expect(
      entries.every(
        entry => entry.bindingIds.includes(binding.id) && entry.runnable === 'catalog-binding'
      )
    ).toBe(true);
  }
});

for (const binding of VALIDATION_BINDINGS)
  test(`${binding.id}: built-in sample, defaults and serialized invocation agree`, async () => {
    const samples = getCatalogSampleInputs(binding.id);
    const inputs = Object.fromEntries(
      binding.inputs.map(input => [input.name, parseCatalogFeatureCollection(samples[input.name])])
    );
    expect(() => validateCatalogInputs(binding, inputs)).not.toThrow();
    const defaults = getCatalogDefaultParameters(binding);
    expect(normalizeCatalogParameters(binding, defaults)).toEqual(defaults);
    const restored = parseCatalogInvocation(serializeCatalogInvocation(binding, defaults, inputs));
    expect(restored.binding).toBe(binding);
    expect(restored.invocation.parameters).toEqual(defaults);
    await expect(runCatalogOperationAsync(binding, defaults, inputs)).rejects.toThrow(
      'GPU device required'
    );
  });

test('input setup preserves missing rows, matrix order and subsecond time precision', () => {
  const collection = parseCatalogFeatureCollection(
    JSON.stringify({
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: {type: 'Point', coordinates: [100, 200]},
          properties: {first: 2, second: 7, time: 1700000000.25}
        },
        {
          type: 'Feature',
          geometry: {type: 'Point', coordinates: [110, 200]},
          properties: {first: null, second: 9, time: 1700000000.5}
        }
      ]
    })
  );
  expect(Array.from(readValidationColumn(collection, 'first'))).toEqual([2, NaN]);
  expect(Array.from(readValidationMatrix(collection, ['first', 'second']))).toEqual([2, 7, NaN, 9]);
  const values = {coordinateSystem: 'planar', timeOrigin: 1700000000, timeProperty: 'time'};
  expect(Array.from(readValidationTimes(collection, values))).toEqual([0.25, 0.5]);
  expect(getValidationTemporalContext(values).epoch).toBe(1700000000n);
  const query = {...collection, features: [collection.features[1]]};
  const [positions, queries] = readValidationPositions([collection, query], values);
  expect(Array.from(positions)).toEqual([0, 0, 10, 0]);
  expect(Array.from(queries)).toEqual([10, 0]);
  expect(() => readValidationNames('first,first')).toThrow('distinct');
  expect(() => readValidationNames('first,')).toThrow();
  expect(() => readValidationColumn(collection, '')).toThrow();
  expect(() => readValidationPositions([{...query, features: [{geometry: null}]}], values)).toThrow(
    'Point'
  );
});

test('capacity and parameter bounds reject malformed invocation before dispatch', () => {
  const binding = getCatalogBinding('regression-mgtwr')!;
  expect(() => normalizeCatalogParameters(binding, {iterationCount: 33})).toThrow();
  expect(() => normalizeCatalogParameters(binding, {timeOrigin: 1.5})).toThrow();
  expect(() => normalizeCatalogParameters(binding, {componentBandwidths: [0, 1]})).toThrow();
  const sample = parseCatalogFeatureCollection(getCatalogSampleInputs(binding.id)['points']);
  expect(() =>
    validateCatalogInputs(binding, {
      points: {...sample, features: Array(2049).fill(sample.features[0])}
    })
  ).toThrow('capacity');
});

test('diagnostic tables retain row identity, null nonfinite values and explicit unbounded statuses', () => {
  const collection = parseCatalogFeatureCollection(
    getCatalogSampleInputs('validation-conformal')['queries']
  );
  const result = makeValidationResult(
    'intervals',
    'Intervals',
    collection,
    {lower: [Infinity, NaN, 1, 2, 3, 4], status: [3, 4, 0, 0, 0, 0]},
    {threshold: NaN, calibration_count: 0},
    ['GPUConformalPredictionIntervals']
  );
  expect(result.tables[0].columns['feature_index']).toEqual([0, 1, 2, 3, 4, 5]);
  expect(result.tables[0].columns['lower']).toEqual([null, null, 1, 2, 3, 4]);
  expect(result.tables[0].columns['status']).toEqual([3, 4, 0, 0, 0, 0]);
  expect(result.tables[1].columns['value']).toEqual([null, 0]);
});

const kriging = {
  longitudeColumn: 'longitude',
  latitudeColumn: 'latitude',
  coordinateSpace: 'longitude-latitude',
  metric: 'great-circle',
  units: 'meters',
  grid: {bounds: [-74, 40, -73, 41], width: 2, height: 2},
  searchRadiusMeters: 1000,
  neighborCount: 8,
  minimumNeighbors: 3,
  variogram: {model: 'matern-3/2', nugget: Math.fround(0.1), sill: 1, rangeMeters: 500}
};

test.each([
  'vecchia-likelihood',
  'vecchia-fisher'
])('kriging replay preserves %s provenance and Matern parameters', method => {
  const settings = {
    ...kriging,
    variogramFit: {
      method,
      weighting: 'conditional-likelihood',
      binCount: 0,
      pairCount: 40,
      maximumLagMeters: 1000,
      criterion: 23
    }
  };
  expect(parseKeplerKrigingSettings(JSON.parse(JSON.stringify(settings)))).toEqual(settings);
  expect(() =>
    parseKeplerKrigingSettings({
      ...settings,
      variogramFit: {...settings.variogramFit, weighting: 'cressie'}
    })
  ).toThrow('weighting');
});

test('Vecchia grid fitting rejects unsupported sample sizes before touching the device', async () => {
  await expect(
    fitKeplerVecchiaVariogram({} as Device, [0], [0], [1], {
      method: 'fisher',
      initial: {model: 'exponential', sill: 1, range: 100}
    })
  ).rejects.toThrow('fewer than 3 valid samples (found 1)');
});

test.each([
  'likelihood',
  'fisher'
] as const)('Vecchia %s fitting diagnoses constant or insufficient valid observations before GPU work', async method => {
  const options = {
    method,
    initial: {model: 'exponential' as const, sill: 1, range: 100, nugget: 0.05}
  };
  await expect(
    fitKeplerVecchiaVariogram({} as Device, [0, 1, 2, 3], [0, 0, 0, 0], [4, 4, 4, NaN], options)
  ).rejects.toThrow('response has no variance among valid samples');
  await expect(
    fitKeplerVecchiaVariogram({} as Device, [0, 1, 2], [0, 0, 0], [1, 2, NaN], options)
  ).rejects.toThrow('fewer than 3 valid samples (found 2)');
  await expect(
    fitKeplerVecchiaVariogram({} as Device, [0, 1, 2], [0, 0, 0], [1e-100, 2e-100, 3e-100], options)
  ).rejects.toThrow('response has no variance at Float32 precision');
});

test('Vecchia candidate admission rejects nonfinite, zero-total, overflowed and underflowed covariance parameters', () => {
  const initial = {model: 'exponential' as const, sill: 1, range: 100, nugget: 0};
  expect(isValidationVecchiaCovarianceUsable(initial)).toBe(true);
  expect(isValidationVecchiaCovarianceUsable({...initial, sill: 0, nugget: 1})).toBe(true);
  for (const candidate of [
    {...initial, sill: 0},
    {...initial, sill: 1e-100},
    {...initial, sill: -1e-100, nugget: 1},
    {...initial, sill: NaN, range: NaN, nugget: NaN},
    {...initial, range: 1e-100},
    {...initial, range: Infinity},
    {...initial, nugget: -1},
    {...initial, sill: 3e38, nugget: 3e38}
  ])
    expect(isValidationVecchiaCovarianceUsable(candidate)).toBe(false);
});

const emptyVecchiaData = {
  positions: new Float32Array(),
  observations: new Float32Array(),
  neighborhoods: {offsets: new Uint32Array(), neighbors: new Uint32Array()},
  neighborCount: 1
};

test.each([
  'likelihood',
  'fisher'
] as const)('Vecchia %s search rejects zero-total Float32 candidates before accessing the GPU', async method => {
  const result = await fitValidationVecchia(
    {} as Device,
    emptyVecchiaData,
    {model: 'exponential', sill: 1e-100, range: 100, nugget: 0},
    method,
    4
  );
  expect(result.failed).toBe(true);
});

test('Vecchia direct evaluation rejects nonfinite covariance before creating GPU resources', async () => {
  await expect(
    evaluateValidationVecchia({} as Device, emptyVecchiaData, {
      model: 'exponential',
      sill: NaN,
      range: NaN,
      nugget: NaN
    })
  ).rejects.toThrow('finite positive range and total sill at Float32 precision');
});

test.each([
  'regression-gtwr',
  'regression-mgtwr'
])('%s rejects a predictor named intercept before dispatch instead of overwriting a coefficient', async id => {
  const binding = getCatalogBinding(id)!;
  const inputs = {points: parseCatalogFeatureCollection(getCatalogSampleInputs(id)['points'])};
  await expect(
    runCatalogOperationAsync(binding, {predictorProperties: 'intercept'}, inputs, {
      device: {} as Device
    })
  ).rejects.toThrow('predictor property "intercept" is reserved for the fitted intercept');
});

test.each([
  ['regression-gtwr', 'predictorProperties', 11],
  ['regression-mgtwr', 'predictorProperties', 11],
  ['validation-area-of-applicability', 'predictorProperties', 16],
  ['validation-ensemble-scores', 'memberProperties', 64]
] as const)('%s property lists allow every supported member while bounding individual names', (id, parameter, maximumNames) => {
  const binding = getCatalogBinding(id)!;
  const names = Array.from({length: maximumNames}, (_, index) => `member${index}`.padEnd(64, 'x'));
  const values = normalizeCatalogParameters(binding, {[parameter]: names.join(', ')});
  expect(readValidationNames(values[parameter], maximumNames)).toEqual(names);
  expect(() => readValidationNames([...names, 'extra'].join(','), maximumNames)).toThrow(
    'distinct names'
  );
  expect(() => readValidationNames(`${names[0]}x`, maximumNames)).toThrow('at most 64 characters');
});

test('64 short ensemble member names survive catalog parameter normalization', () => {
  const binding = getCatalogBinding('validation-ensemble-scores')!;
  const names = Array.from({length: 64}, (_, index) => `m${index}`);
  const values = normalizeCatalogParameters(binding, {memberProperties: names.join(',')});
  expect(readValidationNames(values['memberProperties'], 64)).toEqual(names);
});
