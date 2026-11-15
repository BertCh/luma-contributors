// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {afterEach, beforeEach, expect, test, vi} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {getGPUVectorFormatInfo, type GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {
  GPURasterStandardizedIndexCalibration,
  GPURasterStreamingAccumulator,
  GPURasterClimateRunAccumulator,
  getGPURasterStreamingAccumulatorParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import * as arrow from 'apache-arrow';
import {CatalogGpuSession} from '../src/analysis/catalog-gpu-session';
import {
  runRasterTimeSeries,
  getRasterTimeSeriesBinding,
  type RasterTimeSeriesResult
} from '../src/analysis/raster-timeseries';
import {createRasterTimeSeriesHost} from '../../../examples/deck/kepler-compute/src/raster-timeseries-host';
import {bindRasterTimeStack} from '../../../examples/deck/kepler-compute/src/raster-timeseries-stack';
import {RasterLiveController} from '../../../examples/deck/kepler-compute/src/raster-live-controller';

import {
  makeRasterTimeSeriesFixtures,
  expectRasterTimeSeriesFixture
} from './raster-timeseries-fixtures';

// Real wrappers, views and contributor constructors; only GPU allocation/submission/readback are stubbed.
const device = {
  type: 'webgpu',
  limits: {maxBufferSize: 1 << 28},
  isLost: false
} as unknown as Device;
const producers: GPUCommandNodeProducer[] = [];
const uploads = new Map<GraphDataView, Float32Array | Uint32Array | Int32Array>();
let serial = 0;
let onRun = () => {};
function createView<Format extends GPUVectorFormat>(
  session: CatalogGpuSession,
  format: Format,
  length: number
) {
  return session.graph.createDataView(
    session.graph.createTransientBuffer({
      id: `probe-${serial++}`,
      byteLength: length * getGPUVectorFormatInfo(format).byteLength,
      usage: Buffer.STORAGE
    }),
    {format, length}
  );
}
beforeEach(() => {
  producers.length = 0;
  uploads.clear();
  onRun = () => {};
  vi.spyOn(CatalogGpuSession.prototype, 'input').mockImplementation(function <
    Format extends GPUVectorFormat
  >(
    this: CatalogGpuSession,
    _name: string,
    values: Float32Array | Uint32Array | Int32Array,
    format: Format
  ) {
    const view = createView(
      this,
      format,
      (values.length * 4) / getGPUVectorFormatInfo(format).byteLength
    );
    uploads.set(view, values);
    return view;
  });
  vi.spyOn(CatalogGpuSession.prototype, 'output').mockImplementation(function <
    Format extends GPUVectorFormat
  >(this: CatalogGpuSession, _name: string, format: Format, length: number) {
    return {
      view: createView(this, format, length),
      buffer: {} as Buffer,
      length,
      components: getGPUVectorFormatInfo(format).byteLength / 4
    };
  });
  vi.spyOn(GPUCommandGraph.prototype, 'add').mockImplementation(producer => {
    producers.push(producer as GPUCommandNodeProducer);
  });
  vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => onRun());
  vi.spyOn(CatalogGpuSession.prototype, 'readFloat32').mockImplementation(async output =>
    new Float32Array(output.length * output.components).fill(1)
  );
  vi.spyOn(CatalogGpuSession.prototype, 'readUint32').mockImplementation(
    async output => new Uint32Array(output.length * output.components)
  );
});
afterEach(() => vi.restoreAllMocks());

function makeStack(values: number[]) {
  return {
    cellCount: 1,
    sliceCount: values.length,
    bandCount: 1,
    times: values.map((_, index) => index * 86400000),
    values: Float32Array.from(values)
  };
}
function makeLayer() {
  return {
    id: 'source',
    type: 'rasterTile',
    _loadedTiles: [
      {
        index: {x: 0, y: 0, z: 0},
        data: {images: {imageBands: [{data: new Float32Array([10]), width: 1, height: 1}]}}
      }
    ]
  };
}
const slices = [0, 1, 2].map(index => ({
  time: new Date(index * 86400000).toISOString(),
  bands: [1]
}));
const polygon = JSON.stringify({
  type: 'Polygon',
  coordinates: [
    [
      [-179, -80],
      [179, -80],
      [179, 80],
      [-179, 80],
      [-179, -80]
    ]
  ]
});
function makeHost(layer = makeLayer(), geometry = [polygon]) {
  const controller = new RasterLiveController({
    getLayers: () => [layer],
    getDevice: () => undefined
  });
  const table = arrow.tableFromArrays({geometry});
  const publications: arrow.Table[] = [];
  const host = createRasterTimeSeriesHost({
    getDevice: () => device,
    getSource: id => controller.getSource(id),
    getPolygonSource: id => (id === 'zones' ? {table, geometryColumn: 'geometry'} : undefined),
    publishTable: async ({table}) => {
      publications.push(table);
    }
  });
  return {host, layer, controller, publications};
}

test('shifted Fisk calibration bounds the shifted baseline, preserving every positive observation', async () => {
  await runRasterTimeSeries(
    device,
    makeStack([-5, -4, -3, -2, -1, 0]),
    'standardized-calibration',
    {distribution: 'fisk', location: -10, calibrationSliceCount: 6}
  );
  const contributor = producers.find(
    producer => producer instanceof GPURasterStandardizedIndexCalibration
  ) as GPURasterStandardizedIndexCalibration;
  expect(contributor).toBeInstanceOf(GPURasterStandardizedIndexCalibration);
  const parameters = uploads.get(contributor.props.parameters)!;
  expect(parameters[6]).toBe(10);
  expect(
    [-5, -4, -3, -2, -1, 0].filter(value => Math.fround(value - parameters[2]) <= parameters[6])
  ).toHaveLength(6);
});

test('tiny variances use the actual magnitude and bounded baseline headroom', async () => {
  const stack = makeStack([1e-6, 1.000001e-6, 1.000002e-6]);
  await runRasterTimeSeries(device, stack, 'climate-anomaly', {calibrationSliceCount: 3});
  const contributor = producers.find(
    producer => producer instanceof GPURasterStreamingAccumulator
  ) as GPURasterStreamingAccumulator;
  const parameters = uploads.get(contributor.props.parameters)!;
  const expected = getGPURasterStreamingAccumulatorParameterValues(
    {reset: true, maximumValueMagnitude: Math.max(...stack.values), maximumStreamSliceCount: 3},
    3
  );
  expect(Array.from(parameters)).toEqual(Array.from(expected));
  const mean = Array.from(stack.values).reduce((sum, value) => sum + value, 0) / 3;
  const moment = Array.from(stack.values).reduce((sum, value) => sum + (value - mean) ** 2, 0);
  expect(2 ** parameters[3]).toBeLessThan(moment * 1e-6);
});

test('climate runs explicitly exclude threshold equality', async () => {
  await runRasterTimeSeries(device, makeStack([0, 0, 0]), 'climate-runs');
  const contributor = producers.find(
    producer => producer instanceof GPURasterClimateRunAccumulator
  ) as GPURasterClimateRunAccumulator;
  expect(contributor.condition).toBe('above');
});

test('replacement bands at the same coordinates invalidate an executing run and retained results', async () => {
  const {host, layer, controller, publications} = makeHost();
  const before = controller.getSource('source');
  onRun = () => {
    layer._loadedTiles[0].data.images.imageBands[0].data = new Float32Array([100]);
  };
  await expect(
    host.run({
      rasterLayerId: 'source',
      operation: 'rolling-mean',
      slices,
      polygonDatasetId: 'zones',
      outputComponent: 2
    })
  ).rejects.toThrow(/source changed/);
  expect(controller.getSource('source')).not.toBe(before);
  expect(publications).toHaveLength(0);
  onRun = () => {};
  await host.run({
    rasterLayerId: 'source',
    operation: 'rolling-mean',
    slices,
    polygonDatasetId: 'zones',
    outputComponent: 2
  });
  expect(host.getLatest()).toBeDefined();
  layer._loadedTiles[0].data.images.imageBands[0].data = new Float32Array([200]);
  expect(host.getLatest()).toBeUndefined();
});

test('all-skipped zones are refused; mixed polygon support reports skipped rows', async () => {
  const point = JSON.stringify({type: 'Point', coordinates: [0, 0]});
  const invalid = makeHost(makeLayer(), [point]);
  await expect(
    invalid.host.run({
      rasterLayerId: 'source',
      operation: 'rolling-mean',
      slices,
      polygonDatasetId: 'zones',
      outputComponent: 2
    })
  ).rejects.toThrow(/polygon/);
  expect(invalid.publications).toHaveLength(0);
  const mixed = makeHost(makeLayer(), [point, polygon]);
  const result = await mixed.host.run({
    rasterLayerId: 'source',
    operation: 'rolling-mean',
    slices,
    polygonDatasetId: 'zones',
    outputComponent: 2
  });
  expect(result.skippedRowCount).toBe(1);
  expect(mixed.publications[0].getChild('zonal_count')!.get(0)).toBeNull();
  expect(mixed.publications[0].getChild('zonal_count')!.get(1)).toBe(1);
});

test('impossible calendar dates are refused before normalization; leap dates and offsets work', () => {
  const {controller} = makeHost();
  for (const time of ['2020-02-30T00:00:00Z', '2020-04-31T00:00:00Z', '2019-02-29T12:00:00+05:30'])
    expect(() =>
      bindRasterTimeStack('source', [slices[0], {...slices[1], time}], id =>
        controller.getSource(id)
      )
    ).toThrow(/calendar/);
  for (const time of ['2020-02-29T12:00:00+05:30', '2020-03-01T00:00:00-03:00'])
    expect(
      bindRasterTimeStack('source', [slices[0], {...slices[1], time}], id =>
        controller.getSource(id)
      ).stack.times[1]
    ).toBe(Date.parse(time));
});

test('EOF global outputs publish without zones; EOF cell output still needs zones', async () => {
  for (const outputName of ['covariance', 'explainedVarianceRatio', 'keptCellCount']) {
    const {host, publications} = makeHost();
    const result = await host.run({rasterLayerId: 'source', operation: 'eof', slices, outputName});
    expect(result.published).toBe(true);
    expect(publications[0].schema.fields.map(field => field.name)).not.toContain('geometry');
  }
  const {host} = makeHost();
  await expect(host.run({rasterLayerId: 'source', operation: 'eof', slices})).rejects.toThrow(
    /polygon/
  );
});

for (const fixture of makeRasterTimeSeriesFixtures()) {
  test(`GPU-free ${fixture.name} constructs the declared contributors`, async () => {
    await runRasterTimeSeries(device, fixture.stack, fixture.operation, fixture.settings);
    expect(producers.map(producer => producer.constructor.name)).toEqual(
      getRasterTimeSeriesBinding(fixture.operation).contributors
    );
  });
}

test('numerical fixture checks refuse NaN outputs even when statuses and counts are finite', () => {
  for (const fixture of makeRasterTimeSeriesFixtures()) {
    const columns: RasterTimeSeriesResult['columns'] = Object.fromEntries([
      ...Object.entries(fixture.numerical).map(([name, expected]) => [
        name,
        {values: new Float32Array(expected.length).fill(NaN), components: 1, scope: 'cell' as const}
      ]),
      ...Object.entries(fixture.integers ?? {}).map(([name, expected]) => [
        name,
        {values: Uint32Array.from(expected), components: 1, scope: 'cell' as const}
      ])
    ]);
    if (Object.keys(fixture.integers ?? {}).length) {
      // The former matrix guard accepts these finite status/count words despite unusable numerical arrays.
      expect(
        Object.values(columns).some(column => Array.from(column.values).some(Number.isFinite))
      ).toBe(true);
    }
    expect(() =>
      expectRasterTimeSeriesFixture(
        {columns, contributors: [], readbackBytes: 0, settings: {}},
        fixture
      )
    ).toThrow();
  }
});

test('replacement mask buffers invalidate retained source validity', async () => {
  const original = makeLayer();
  const layer = {
    ...original,
    _loadedTiles: [
      {
        ...original._loadedTiles[0],
        data: {
          images: {
            ...original._loadedTiles[0].data.images,
            imageMask: {data: new Uint8Array([1]), width: 1, height: 1}
          }
        }
      }
    ]
  };
  const {host, controller} = makeHost(layer);
  await host.run({
    rasterLayerId: 'source',
    operation: 'rolling-mean',
    slices,
    polygonDatasetId: 'zones',
    outputComponent: 2
  });
  expect(host.getLatest()).toBeDefined();
  layer._loadedTiles[0].data.images.imageMask.data = new Uint8Array([0]);
  expect(host.getLatest()).toBeUndefined();
  expect(controller.getSource('source')!.tiles[0].bands[0].validity[0]).toBe(0);
});

test('source generation guard remains active during asynchronous publication', async () => {
  const layer = makeLayer();
  const controller = new RasterLiveController({
    getLayers: () => [layer],
    getDevice: () => undefined
  });
  const table = arrow.tableFromArrays({geometry: [polygon]});
  let guarded = false;
  const host = createRasterTimeSeriesHost({
    getDevice: () => device,
    getSource: id => controller.getSource(id),
    getPolygonSource: () => ({table, geometryColumn: 'geometry'}),
    publishTable: async ({isCurrent}) => {
      expect(isCurrent?.()).toBe(true);
      await Promise.resolve();
      layer._loadedTiles[0].data.images.imageBands[0].data = new Float32Array([100]);
      guarded = isCurrent?.() === false;
    }
  });
  await expect(
    host.run({
      rasterLayerId: 'source',
      operation: 'rolling-mean',
      slices,
      polygonDatasetId: 'zones',
      outputComponent: 2
    })
  ).rejects.toThrow(/source changed during publication/);
  expect(guarded).toBe(true);
  expect(host.getLatest()).toBeUndefined();
});

test.each([
  'replace',
  'remove',
  'remap'
])('polygon source %s invalidates asynchronous time-series publication', async change => {
  const {controller} = makeHost();
  const table = arrow.tableFromArrays({geometry: [polygon], alternate: [polygon]});
  let polygons: {table: arrow.Table; geometryColumn: string} | undefined = {
    table,
    geometryColumn: 'geometry'
  };
  let guarded = false;
  const host = createRasterTimeSeriesHost({
    getDevice: () => device,
    getSource: id => controller.getSource(id),
    getPolygonSource: () => polygons,
    publishTable: async ({isCurrent}) => {
      expect(isCurrent?.()).toBe(true);
      await Promise.resolve();
      polygons =
        change === 'remove'
          ? undefined
          : {
              table: change === 'replace' ? arrow.tableFromArrays({geometry: [polygon]}) : table,
              geometryColumn: change === 'remap' ? 'alternate' : 'geometry'
            };
      guarded = isCurrent?.() === false;
    }
  });
  await expect(
    host.run({
      rasterLayerId: 'source',
      operation: 'rolling-mean',
      slices,
      polygonDatasetId: 'zones',
      outputComponent: 2
    })
  ).rejects.toThrow(/source changed during publication/);
  expect(guarded).toBe(true);
  expect(host.getLatest()).toBeUndefined();
});
