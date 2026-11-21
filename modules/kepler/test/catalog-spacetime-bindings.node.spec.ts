// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {transpileModule, ScriptTarget} from 'typescript';
import {expect, test} from 'vitest';
import {
  getWitnessLogLikelihood,
  getWitnessLikelihoodScore,
  HAWKES_WITNESS_FIRST_UPDATE,
  HAWKES_WITNESS_FINAL
} from './catalog-spacetime-hawkes-reference';
import {
  getGPUAnalysisOperator,
  getGPUAnalysisCapabilityCatalog
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {
  getCatalogBinding,
  getCatalogDefaultParameters,
  listCatalogEntries,
  normalizeCatalogParameters,
  runCatalogOperationAsync
} from '../src/analysis/catalog-bindings';
import {SPACETIME_BINDINGS} from '../src/analysis/catalog-spacetime-bindings';
import {
  makeSpacetimeGrid,
  makeSpacetimeNeighbors,
  makeSpacetimePointTable,
  readSpacetimePoints,
  readSpacetimeSeries
} from '../src/analysis/catalog-spacetime-common';
import {
  findNativeGeometryFields,
  makeCatalogPointColumns,
  readCatalogFeaturesFromNativeDataset
} from '../../../examples/deck/kepler-compute/src/analysis-catalog-input';
import {getSpacetimeCatalogSample} from '../../../examples/deck/kepler-compute/src/analysis-catalog-spacetime';
import {SPACETIME_TEST_PARAMETERS, SPACETIME_TEST_POINTS} from './catalog-spacetime-fixture';

const CONTRIBUTORS = [
  'GPUSpaceTimeKernelDensity',
  'GPUSpaceTimeRelativeRisk',
  'GPUAdaptiveBandwidthKernelDensity',
  'GPUSpaceTimeHawkes',
  'GPUExpectationBasedSpaceTimeScan',
  'GPUSpaceTimeCrossCorrelation',
  'GPUSpaceTimeLocalMoran',
  'GPUJacquezTest',
  'GPULocalKnox',
  'GPUNearRepeatKnox',
  'GPUSpaceTimeDiggleTest',
  'GPUSpaceTimeKFunction',
  'GPUPettittTest',
  'GPULocationChangePoints'
];

test('spacetime binds exactly the fourteen requested existing contributors', () => {
  expect(SPACETIME_BINDINGS.flatMap(binding => binding.contributorNames)).toEqual(CONTRIBUTORS);
  expect(new Set(SPACETIME_BINDINGS.map(binding => binding.id)).size).toBe(14);
});

for (const binding of SPACETIME_BINDINGS) {
  test(`${binding.id}: registered GPU contract, real capability and parameter mapping`, async () => {
    expect(getCatalogBinding(binding.id)).toBe(binding);
    expect(binding.engine).toBe('gpu');
    expect(binding.run).toBeUndefined();
    expect(binding.runGpu).toBeTypeOf('function');
    expect(binding.gpuContributorNames).toEqual(binding.contributorNames);
    const operator = getGPUAnalysisOperator(binding.contributorNames[0]);
    expect(operator, binding.contributorNames[0]).toBeDefined();
    expect(binding.capabilityIds).toContain(operator!.primaryCapability);
    for (const capability of binding.capabilityIds)
      expect(getGPUAnalysisCapabilityCatalog().some(candidate => candidate.id === capability)).toBe(
        true
      );
    expect(binding.inputs).toEqual([
      {
        name: 'points',
        label: 'Event points',
        accepts: ['point'],
        maximumFeatures: 512,
        maximumVertices: 512
      }
    ]);
    expect(
      binding.outputs.every(output => output.identityColumns.length > 0 && output.maximumRows > 0)
    ).toBe(true);
    expect(binding.limitation).toContain('512');
    expect(binding.declaredBehavior.join(' ')).toContain('relative seconds');
    const defaults = getCatalogDefaultParameters(binding);
    expect(normalizeCatalogParameters(binding, defaults)).toEqual(defaults);
    const raw = Object.fromEntries(
      binding.parameters.map(parameter => [
        parameter.name,
        SPACETIME_TEST_PARAMETERS[parameter.name] ?? parameter.defaultValue
      ])
    );
    const parameters = normalizeCatalogParameters(binding, raw);
    expect(parameters['timeProperty']).toBe('time');
    expect(parameters['timeUnit']).toBe('milliseconds');
    const prepared = readSpacetimePoints(SPACETIME_TEST_POINTS, parameters);
    expect(Array.from(prepared.times.slice(0, 8))).toEqual([0, 60, 120, 180, 240, 300, 360, 420]);
    expect(prepared.positions.slice(16, 18)).toEqual(Float32Array.from([2, 0]));
    expect(prepared.timeOrigin).toBe(1700000000000);
    for (const parameter of binding.parameters) {
      if (parameter.kind === 'number' || parameter.kind === 'integer') {
        expect(
          normalizeCatalogParameters(binding, {
            ...raw,
            [parameter.name]: String(parameters[parameter.name])
          })[parameter.name]
        ).toBe(parameters[parameter.name]);
        if (parameter.maximum !== undefined)
          expect(() =>
            normalizeCatalogParameters(binding, {...raw, [parameter.name]: parameter.maximum! + 1})
          ).toThrow();
        if (parameter.minimum !== undefined)
          expect(() =>
            normalizeCatalogParameters(binding, {...raw, [parameter.name]: parameter.minimum! - 1})
          ).toThrow();
      }
    }
    await expect(
      runCatalogOperationAsync(binding, raw, {points: SPACETIME_TEST_POINTS})
    ).rejects.toThrow();
    const fixture = JSON.parse(getSpacetimeCatalogSample(binding.id)!);
    expect(fixture.features.length).toBe(48);
    expect(readSpacetimePoints(fixture, defaults).count).toBe(48);
  });
}

test('new ontology contributors appear in panel discovery with real capability IDs', () => {
  const entries = listCatalogEntries({lensScenarios: []});
  for (const contributor of CONTRIBUTORS) {
    const entry = entries.find(candidate => candidate.identity.exportName === contributor)!;
    expect(entry.runnable).toBe('catalog-binding');
    expect(entry.capabilityId).toBe(getGPUAnalysisOperator(contributor)!.primaryCapability);
  }
});

test('epoch milliseconds and seconds map identically after subtracting their origins', () => {
  const milliseconds = readSpacetimePoints(SPACETIME_TEST_POINTS, SPACETIME_TEST_PARAMETERS);
  const secondsInput = structuredClone(SPACETIME_TEST_POINTS);
  for (const feature of secondsInput.features)
    feature.properties!['time'] = Number(feature.properties!['time']) / 1000;
  const seconds = readSpacetimePoints(secondsInput, {
    ...SPACETIME_TEST_PARAMETERS,
    timeUnit: 'seconds'
  });
  expect(seconds.times).toEqual(milliseconds.times);
  expect(seconds.positions).toEqual(milliseconds.positions);
  expect(seconds.timeOrigin).toBe(1700000000);
});

test('time and geometry mapping reject invalid rows, empty input and capacity overflow before GPU work', () => {
  for (const time of [undefined, '1700000000000', Number.NaN, Infinity]) {
    const collection = structuredClone(SPACETIME_TEST_POINTS);
    collection.features[2].properties!['time'] = time;
    expect(() => readSpacetimePoints(collection, SPACETIME_TEST_PARAMETERS)).toThrow(
      /finite numeric time/
    );
  }
  const invalidGeometry = structuredClone(SPACETIME_TEST_POINTS);
  invalidGeometry.features[0].geometry = {
    type: 'MultiPoint',
    coordinates: [
      [0, 0],
      [1, 1]
    ]
  };
  expect(() => readSpacetimePoints(invalidGeometry, SPACETIME_TEST_PARAMETERS)).toThrow(/Point/);
  expect(() =>
    readSpacetimePoints({type: 'FeatureCollection', features: []}, SPACETIME_TEST_PARAMETERS)
  ).toThrow(/512/);
  expect(() =>
    readSpacetimePoints(
      {type: 'FeatureCollection', features: Array(513).fill(SPACETIME_TEST_POINTS.features[0])},
      SPACETIME_TEST_PARAMETERS
    )
  ).toThrow(/512/);
});

test('series mapping sorts periods and joins every source row to the correct cell and period', () => {
  const collection = structuredClone(SPACETIME_TEST_POINTS);
  collection.features.reverse();
  const points = readSpacetimePoints(collection, SPACETIME_TEST_PARAMETERS);
  const series = readSpacetimeSeries(points, SPACETIME_TEST_PARAMETERS, true, true);
  expect(series.cellCount).toBe(6);
  expect(series.periodCount).toBe(8);
  expect(series.sourcePeriods.slice(0, 8)).toEqual([7, 6, 5, 4, 3, 2, 1, 0]);
  expect(Array.from(series.cube.slice(0, 8))).toEqual([6, 7, 6, 7, 18, 19, 18, 19]);
  expect(Array.from(series.baseline.slice(0, 8))).toEqual(Array(8).fill(6));
  expect(series.second[0]).toBe(7);
  expect(series.sourceRows[0]).toBe(7);
  const table = makeSpacetimePointTable(points, {period: series.sourcePeriods});
  expect(table.geometryColumn).toBe('geometry');
  expect(table.columns['source_index']).toEqual(Array.from({length: 48}, (_, index) => index));
  expect(table.columns['source_id'][0]).toBe('location-5-period-7');
  expect(table.columns['period'][0]).toBe(7);
});

test('series mapping refuses duplicates, gaps, irregular periods, moving locations and missing numeric columns', () => {
  const read = (collection: typeof SPACETIME_TEST_POINTS) =>
    readSpacetimeSeries(
      readSpacetimePoints(collection, SPACETIME_TEST_PARAMETERS),
      SPACETIME_TEST_PARAMETERS,
      true,
      true
    );
  const duplicate = structuredClone(SPACETIME_TEST_POINTS);
  duplicate.features[1] = duplicate.features[0];
  expect(() => read(duplicate)).toThrow(/duplicate/);
  const gap = structuredClone(SPACETIME_TEST_POINTS);
  gap.features.pop();
  expect(() => read(gap)).toThrow(/complete/);
  const irregular = structuredClone(SPACETIME_TEST_POINTS);
  irregular.features[0].properties!['time'] = 1700000000001;
  expect(() => read(irregular)).toThrow(/equally spaced/);
  const moving = structuredClone(SPACETIME_TEST_POINTS);
  moving.features[1].geometry = {type: 'Point', coordinates: [5, 5]};
  expect(() => read(moving)).toThrow(/constant/);
  const missing = structuredClone(SPACETIME_TEST_POINTS);
  delete missing.features[0].properties!['second'];
  expect(() => read(missing)).toThrow(/second/);
});

test('neighbor and grid host maps have deterministic ties and bounded source/grid identities', () => {
  const neighbors = makeSpacetimeNeighbors(
    Float32Array.from([0, 0, 1, 0, -1, 0, 0, 2]),
    Infinity,
    2
  );
  expect(Array.from(neighbors.neighbors.slice(0, 2))).toEqual([1, 2]);
  expect(Array.from(neighbors.distances.slice(0, 2))).toEqual([1, 1]);
  const points = readSpacetimePoints(SPACETIME_TEST_POINTS, SPACETIME_TEST_PARAMETERS);
  const {grid, bounds} = makeSpacetimeGrid(points, SPACETIME_TEST_PARAMETERS);
  expect(bounds).toEqual([-2, -2, 6, 4]);
  expect([grid.width, grid.height]).toEqual([2, 2]);
});

test('non-resident numeric point/time columns map with row identities and survive the stored field token', () => {
  const rows = [
    [-87.65, 41.88, 1700000000000, 4],
    [-87.64, 41.89, 1700000060000, 8]
  ];
  const dataset = {
    fields: ['longitude', 'latitude', 'time', 'value'].map(name => ({name})),
    dataContainer: {
      numRows: () => rows.length,
      valueAt: (row: number, column: number) => rows[row][column]
    }
  };
  expect(findNativeGeometryFields(dataset)).toEqual([]);
  for (const columns of [
    ['', ''],
    ['', 'latitude'],
    ['longitude', ''],
    ['missing', 'latitude']
  ] as const)
    expect(() =>
      readCatalogFeaturesFromNativeDataset(dataset, makeCatalogPointColumns(columns[0], columns[1]))
    ).toThrow('Choose both numeric point columns');
  const mapping = makeCatalogPointColumns('longitude', 'latitude');
  const points = readCatalogFeaturesFromNativeDataset(dataset, mapping);
  expect(points.features[0].id).toBe(0);
  expect(points.features[0].geometry).toEqual({type: 'Point', coordinates: [-87.65, 41.88]});
  expect(points.features[1].properties!['time']).toBe(1700000060000);
  expect(
    readSpacetimePoints(points, {...SPACETIME_TEST_PARAMETERS, coordinateSystem: 'EPSG:4326'}).times
  ).toEqual(Float32Array.from([0, 60]));
});

test('scan validates exact resident counts before float32 rounding and without a GPU allocation', async () => {
  const binding = getCatalogBinding('expectation-based-space-time-scan')!;
  const points = structuredClone(SPACETIME_TEST_POINTS);
  points.features[0].properties!['value'] = 16777214.75;
  const parameters = normalizeCatalogParameters(
    binding,
    Object.fromEntries(
      binding.parameters.map(parameter => [
        parameter.name,
        SPACETIME_TEST_PARAMETERS[parameter.name] ?? parameter.defaultValue
      ])
    )
  );
  await expect(binding.runGpu!({device: {} as never}, {points}, parameters)).rejects.toThrow(
    /integer counts/
  );
});

test('series periods recorded over months use their period spacing for the float32 time guard', () => {
  const collection = structuredClone(SPACETIME_TEST_POINTS);
  collection.features.forEach((feature, row) => {
    feature.properties!['time'] = 1700000000000 + (row % 8) * 30 * 86400000;
  });
  const binding = getCatalogBinding('pettitt-test')!;
  const parameters = normalizeCatalogParameters(binding, {coordinateSystem: 'planar'});
  const points = readSpacetimePoints(collection, parameters);
  const series = readSpacetimeSeries(points, parameters);
  expect(series.intervalSeconds).toBe(30 * 86400);
  expect(series.periodCount).toBe(8);
});

test('Hawkes catalog keeps an explicit budget and has an independent finite-window oracle', () => {
  const binding = getCatalogBinding('space-time-hawkes')!;
  // The catalog intentionally requests 20; it does not inherit the ordinary estimator default.
  expect(getCatalogDefaultParameters(binding)['maximumIterations']).toBe(20);
  expect(normalizeCatalogParameters(binding, {maximumIterations: 1})['maximumIterations']).toBe(1);
  expect(getWitnessLogLikelihood([2 / 81, 0.5, 0.5, 4])).toBeCloseTo(-16.974445977916307, 10);
  expect(getWitnessLogLikelihood(HAWKES_WITNESS_FIRST_UPDATE)).toBeCloseTo(-14.776229122727967, 10);
  expect(getWitnessLogLikelihood(HAWKES_WITNESS_FINAL)).toBeCloseTo(-12.526952698424633, 10);
  for (const score of getWitnessLikelihoodScore(HAWKES_WITNESS_FINAL)) {
    expect(Number.isFinite(score)).toBe(true);
    expect(Math.abs(score)).toBeLessThan(1e-5);
  }
});

test('Pettitt catalog evidence excludes untested cells and preserves source row alignment', () => {
  const source = readFileSync('modules/kepler/src/analysis/catalog-spacetime-series.ts', 'utf8');
  const start = source.indexOf('    const tested =');
  const end = source.indexOf("  if (kind === 'changes')", start);
  const body = transpileModule(source.slice(start, end).replace(/\n  }\s*$/, ''), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const read = new Function(
    'result',
    'series',
    `
    const cellCount = series.cellCount, periodCount = series.periodCount, points = {}, summary = {};
    const finiteSpacetimeValue = value => Number.isFinite(value) && value !== 0xffffffff ? value : null;
    const expand = entries => series.sourceCells.map(cell => finiteSpacetimeValue(entries[cell]));
    const makeSpacetimePointTable = (points, columns) => columns;
    ${body}
  `
  );
  // n=4, ranks [1,2,3,4]: K=4, p=2*exp(-6*16/(16*5))=.6023884.
  // The middle cell has one finite observation; the last is tested and constant.
  const series = {
    cellCount: 3,
    periodCount: 4,
    periods: [0, 1, 2, 3],
    cube: Float32Array.of(1, 2, 3, 4, NaN, 2, NaN, NaN, 7, 7, 7, 7),
    sourceCells: [2, 1, 0, 0],
    sourcePeriods: [3, 2, 1, 3]
  };
  const output = read(
    {
      statistic: Uint32Array.of(4, 0, 0),
      changeSlice: Uint32Array.of(2, 0xffffffff, 1),
      pValues: Float32Array.of(0.6023884, NaN, 1)
    },
    series
  );
  expect(output.summary['pValueMinimum']).toBeCloseTo(0.6023884, 6);
  expect(output.summary['testedLocations']).toBe(2);
  expect(output.tables[0]['p_value']).toEqual([
    1,
    null,
    Math.fround(0.6023884),
    Math.fround(0.6023884)
  ]);
  expect(output.tables[0]['change_slice']).toEqual([null, null, 2, 2]);
  expect(output.tables[0]['change_time']).toEqual([null, null, 2, 2]);
  expect(output.tables[0]['segment']).toEqual([null, null, 0, 1]);
  // Eligibility is already encoded by changeSlice: do not scan the source cube again.
  expect(() =>
    read(
      {statistic: Uint32Array.of(0), changeSlice: Uint32Array.of(1), pValues: Float32Array.of(1)},
      {
        cellCount: 1,
        periodCount: 2,
        periods: [0, 1],
        sourceCells: [0],
        sourcePeriods: [1],
        get cube() {
          throw new Error('source cube must not be reread');
        }
      }
    )
  ).not.toThrow();
  const untested = read(
    {
      statistic: Uint32Array.of(0),
      changeSlice: Uint32Array.of(0xffffffff),
      pValues: Float32Array.of(NaN)
    },
    {
      cellCount: 1,
      periodCount: 2,
      periods: [0, 1],
      cube: Float32Array.of(NaN, 1),
      sourceCells: [0, 0],
      sourcePeriods: [0, 1]
    }
  );
  expect(untested.summary['testedLocations']).toBe(0);
  expect(untested.summary['pValueMinimum']).toBe('unavailable');
  expect(untested.summary['statisticMaximum']).toBe('unavailable');
  expect(untested.tables[0]['segment']).toEqual([null, null]);
});

test('B6 queued Pettitt constant fixture normalizes only its declared parameters', () => {
  const source = readFileSync('modules/kepler/test/catalog-spacetime-gpu.spec.ts', 'utf8');
  const start = source.indexOf(
    '  const parameters = normalizeCatalogParameters(',
    source.indexOf('// Catalog rows are a complete finite rectangle')
  );
  const end = source.indexOf('\n  const output =', start);
  const body = transpileModule(source.slice(start, end), {
    compilerOptions: {target: ScriptTarget.ES2022}
  }).outputText;
  const binding = getCatalogBinding('pettitt-test')!;
  const resolve = new Function(
    'binding',
    'normalizeCatalogParameters',
    'SPACETIME_TEST_PARAMETERS',
    `${body} return parameters;`
  );
  expect(() =>
    resolve(binding, normalizeCatalogParameters, SPACETIME_TEST_PARAMETERS)
  ).not.toThrow();
});

test('Float32 closed-radius boundary retains pairs included by the uploaded distance', () => {
  const positions = Float32Array.from([0, 0, 0.6, 0.8]);
  const distance = Math.hypot(positions[2], positions[3]);
  expect(distance).toBeGreaterThan(1);
  expect(Math.fround(distance)).toBe(1);
  const pairs = makeSpacetimeNeighbors(positions, 1);
  expect(Array.from(pairs.neighbors)).toEqual([1, 0]);
  expect(Array.from(pairs.distances)).toEqual([1, 1]);
  expect(Array.from(makeSpacetimeNeighbors(positions, 1 - Number.EPSILON).neighbors)).toEqual([
    1, 0
  ]);
  expect(
    Array.from(makeSpacetimeNeighbors(Float32Array.from([0, 0, 1 + 2 ** -23, 0]), 1).neighbors)
  ).toEqual([]);
});
