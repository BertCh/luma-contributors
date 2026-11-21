// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {
  runRasterTimeSeries,
  type RasterTimeStack,
  type RasterTimeSeriesOperation
} from '../src/analysis/raster-timeseries';
import {createRasterTimeSeriesHost} from '../../../examples/deck/kepler-compute/src/raster-timeseries-host';
import {
  createNumericRasterSource,
  createNumericRasterTile
} from '../../../examples/deck/kepler-compute/src/raster-tile-ingest';
import * as arrow from 'apache-arrow';

import {
  makeRasterTimeSeriesFixtures,
  expectRasterTimeSeriesFixture
} from './raster-timeseries-fixtures';

function makeStack(): RasterTimeStack {
  const sliceCount = 24;
  return {
    cellCount: 3,
    sliceCount,
    bandCount: 1,
    times: Array.from(
      {length: sliceCount},
      (_, slice) => Date.UTC(2020, 0, 1) + slice * 32 * 86400000
    ),
    values: Float32Array.from({length: 3 * sliceCount}, (_, index) => {
      const cell = Math.floor(index / sliceCount);
      const slice = index % sliceCount;
      return cell === 2 ? NaN : 10 + cell * 5 + slice * 0.2 + Math.sin((slice * Math.PI) / 6);
    })
  };
}
async function getDevice() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  return device;
}

for (const fixture of makeRasterTimeSeriesFixtures()) {
  test(`raster-native ${fixture.name} matches numerical fixtures and statuses`, async () => {
    const result = await runRasterTimeSeries(
      await getDevice(),
      fixture.stack,
      fixture.operation,
      fixture.settings
    );
    expectRasterTimeSeriesFixture(result, fixture);
    expect(result.readbackBytes).toBeGreaterThan(0);
    for (const column of Object.values(result.columns)) {
      expect(column.values.length % column.components).toBe(0);
      if (column.scope === 'cell')
        expect(column.values.length / column.components).toBe(fixture.stack.cellCount);
    }
  });
}

test('rolling means and nodata agree with an independent trailing-window oracle', async () => {
  const stack = makeStack();
  stack.values[4] = NaN;
  const result = await runRasterTimeSeries(await getDevice(), stack, 'rolling-mean', {
    windowLength: 3
  });
  const actual = result.columns['rollingMean'].values;
  for (let cell = 0; cell < stack.cellCount; cell++)
    for (let slice = 0; slice < stack.sliceCount; slice++) {
      const window = Array.from(
        stack.values.slice(cell * stack.sliceCount + slice - 2, cell * stack.sliceCount + slice + 1)
      );
      const index = cell * stack.sliceCount + slice;
      if (slice < 2 || window.some(value => !Number.isFinite(value)))
        expect(Number.isNaN(actual[index])).toBe(true);
      else expect(actual[index]).toBeCloseTo(window.reduce((sum, value) => sum + value, 0) / 3, 4);
    }
});

test('climatology uses only the declared baseline and produces sample z scores', async () => {
  const stack = {
    cellCount: 1,
    sliceCount: 6,
    bandCount: 1,
    times: [0, 1, 2, 3, 4, 5],
    values: new Float32Array([1, 2, 3, 10, 11, NaN])
  };
  const result = await runRasterTimeSeries(await getDevice(), stack, 'climate-anomaly', {
    calibrationSliceCount: 3
  });
  // GPU float32 reductions, sqrt and normalization can round these derived values.
  for (const [name, expected] of [
    ['mean', [2]],
    ['standardDeviation', [1]],
    ['anomaly', [-1, 0, 1, 8, 9, NaN]]
  ] as const) {
    const actual = result.columns[name].values;
    expect(actual.length).toBe(expected.length);
    expected.forEach((value, index) => {
      if (Number.isNaN(value)) expect(Number.isNaN(actual[index])).toBe(true);
      else expect(Math.abs(actual[index] - value)).toBeLessThan(1e-6);
    });
  }
});

test('multiband medoid keeps a real acquisition and excludes incomplete observations', async () => {
  const result = await runRasterTimeSeries(
    await getDevice(),
    {
      cellCount: 1,
      sliceCount: 3,
      bandCount: 2,
      times: [0, 1, 2],
      values: new Float32Array([1, 10, 100, NaN, 3, 30])
    },
    'medoid'
  );
  expect(Array.from(result.columns['composite'].values)).toEqual([1, 10]);
  expect(result.columns['sliceIndex'].values[0]).toBe(0);
});

test('host publishes exact zonal values with acquisition provenance and source geometry', async () => {
  const device = await getDevice();
  const source = createNumericRasterSource({
    id: 'stack',
    kind: 'single-raster',
    crs: 'EPSG:4326',
    tiles: [
      createNumericRasterTile({
        id: 'tile',
        width: 2,
        height: 1,
        extent: [0, 0, 2, 1],
        bands: [1, 2, 3].map(value => ({values: new Float32Array([value, value * 10])}))
      })
    ]
  });
  const zones = arrow.tableFromArrays({
    geometry: [
      JSON.stringify({
        type: 'Polygon',
        coordinates: [
          [
            [0, 0],
            [2, 0],
            [2, 1],
            [0, 1],
            [0, 0]
          ]
        ]
      })
    ]
  });
  let published: arrow.Table | undefined;
  const host = createRasterTimeSeriesHost({
    getDevice: () => device,
    getSource: () => source,
    getPolygonSource: () => ({table: zones, geometryColumn: 'geometry'}),
    publishTable: async ({table}) => {
      published = table;
    }
  });
  const result = await host.run({
    rasterLayerId: 'stack',
    operation: 'rolling-mean',
    slices: [1, 2, 3].map((band, index) => ({
      time: new Date(Date.UTC(2020, 0, 1) + index * 86400000).toISOString(),
      bands: [band]
    })),
    polygonDatasetId: 'zones',
    outputComponent: 2
  });
  expect(result.engine).toBe('gpu');
  expect(published!.getChild('zonal_mean')!.get(0)).toBeCloseTo(11);
  expect(published!.getChild('zonal_count')!.get(0)).toBe(2);
  expect(published!.getChild('geometry')!.get(0)).toBe(zones.getChild('geometry')!.get(0));
  expect(JSON.parse(published!.schema.metadata.get('raster_timeseries')!).contributors).toContain(
    'GPURasterRollingMean'
  );
});

test('host refuses source changes during GPU execution', async () => {
  const device = await getDevice();
  let reads = 0;
  let publications = 0;
  const source = createNumericRasterSource({
    id: 'source',
    kind: 'single-raster',
    crs: 'EPSG:4326',
    tiles: [
      createNumericRasterTile({
        id: 'tile',
        width: 1,
        height: 1,
        extent: [0, 0, 1, 1],
        bands: [{values: new Float32Array([1])}]
      })
    ]
  });
  const changed = {...source, id: 'changed'};
  const host = createRasterTimeSeriesHost({
    getDevice: () => device,
    getSource: () => (++reads > 2 ? changed : source),
    getPolygonSource: () => undefined,
    publishTable: async () => {
      publications++;
    }
  });
  await expect(
    host.run({
      rasterLayerId: 'source',
      operation: 'snapshot-covariance' as RasterTimeSeriesOperation,
      slices: [0, 1].map(index => ({time: new Date(index).toISOString(), bands: [1]}))
    })
  ).rejects.toThrow(/source changed/);
  expect(publications).toBe(0);
});

test('snapshot covariance equals an independent centred Gram matrix and EOF excludes missing cells', async () => {
  const stack = makeStack();
  const device = await getDevice();
  const result = await runRasterTimeSeries(device, stack, 'snapshot-covariance');
  const matrix = result.columns['covariance'].values;
  const means = [0, 1].map(
    cell =>
      Array.from(stack.values.slice(cell * stack.sliceCount, (cell + 1) * stack.sliceCount)).reduce(
        (sum, value) => sum + value,
        0
      ) / stack.sliceCount
  );
  for (let row = 0; row < stack.sliceCount; row++)
    for (let column = 0; column < stack.sliceCount; column++) {
      const expected = [0, 1].reduce(
        (sum, cell) =>
          sum +
          (stack.values[cell * stack.sliceCount + row] - means[cell]) *
            (stack.values[cell * stack.sliceCount + column] - means[cell]),
        0
      );
      expect(matrix[row * stack.sliceCount + column]).toBeCloseTo(expected, 4);
    }
  expect(result.columns['keptCellCount'].values[0]).toBe(2);
  const projection = await runRasterTimeSeries(device, stack, 'eof');
  expect(projection.columns['status'].values[2]).toBe(1);
  expect(Number.isNaN(projection.columns['eofs'].values[4])).toBe(true);
});

for (const [distribution, zeroAtom, expectedProbability] of [
  ['gamma', false, 0],
  ['fisk', true, 1 / 3]
] as const) {
  test(`W5-18 explicit ${distribution} zeroAtom=${zeroAtom} controls the fitted atom`, async () => {
    const result = await runRasterTimeSeries(
      await getDevice(),
      {
        cellCount: 1,
        sliceCount: 3,
        bandCount: 1,
        times: [0, 1, 2],
        values: Float32Array.of(0, 1, 2)
      },
      'standardized-index',
      {calibrationSliceCount: 3, distribution, zeroAtom}
    );
    expect(result.columns['fit'].values).toHaveLength(3);
    expect(result.columns['index'].values).toHaveLength(3);
    expect(result.columns['counts'].values).toHaveLength(3);
    const probability = result.columns['fit'].values[2];
    expect(Number.isFinite(probability)).toBe(true);
    expect(probability).toBeCloseTo(expectedProbability, 6);
    expect(Array.from(result.columns['status'].values)).toEqual([0, 0, 0]);
    if (zeroAtom) {
      // Atom mass is 1/3, so the original zero has normal quantile Phi^-1(1/3).
      const index = result.columns['index'].values[0];
      expect(Number.isFinite(index)).toBe(true);
      expect(index).toBeCloseTo(-0.43072729929545756, 5);
    }
  });
}

for (const values of [
  [10000000, 10000001],
  [10000001, 10000000]
]) {
  test(`SCL-01 adapter sample deviation and calibration population moment for ${values.join(',')}`, async () => {
    const device = await getDevice();
    const stack = {
      cellCount: 1,
      sliceCount: 2,
      bandCount: 1,
      times: [0, 1],
      values: Float32Array.from(values)
    };
    const streaming = await runRasterTimeSeries(device, stack, 'streaming');
    // Independent binary64 closed form: mean 10000000.5, residuals +/-0.5, M2=0.5.
    // Streaming publishes SAMPLE standard deviation: sqrt(M2/(n-1)) = sqrt(0.5).
    for (const name of ['mean', 'standardDeviation', 'counts', 'streamingStatus'])
      expect(streaming.columns[name].values).toHaveLength(1);
    expect(streaming.streamingCountStatus).toHaveLength(1);
    const deviation = streaming.columns['standardDeviation'].values[0];
    expect(Number.isFinite(deviation)).toBe(true);
    expect(deviation).toBeCloseTo(0.7071067811865476, 5);
    const calibration = await runRasterTimeSeries(device, stack, 'standardized-calibration', {
      calibrationSliceCount: 2,
      distribution: 'fisk'
    });
    expect(calibration.columns['fit'].values).toHaveLength(3);
    expect(calibration.columns['counts'].values).toHaveLength(3);
    const shape = calibration.columns['fit'].values[0];
    const scale = calibration.columns['fit'].values[1];
    // APP Fisk uses POPULATION variance M2/n=.25, shape=pi*mean/sqrt(3*variance),
    // scale=2*mean^3/(variance+2*mean^2). Python math gives these literals.
    expect(Number.isFinite(shape)).toBe(true);
    expect(Number.isFinite(scale)).toBe(true);
    expect(Math.abs(shape / 36275989.09848373 - 1)).toBeLessThan(1e-5);
    expect(Math.abs(scale / 10000000.499999987 - 1)).toBeLessThan(1e-6);
    const impliedSquaredDeviations = 2 * ((Math.PI * 10000000.5) / (Math.sqrt(3) * shape)) ** 2;
    expect(Number.isFinite(impliedSquaredDeviations)).toBe(true);
    expect(impliedSquaredDeviations).toBeCloseTo(0.5, 5);
  });
}
