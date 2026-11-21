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
  GPURasterRollingMean,
  GPURasterStandardizedIndex,
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

vi.mock(
  '@luma-contributors/analysis/raster-analysis',
  () => import('../../analysis/src/raster-analysis')
);

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

test('W5-17 admits window one and preserves two-cell physical width with one active slice', async () => {
  await runRasterTimeSeries(
    device,
    {
      cellCount: 2,
      sliceCount: 2,
      bandCount: 1,
      times: [0, 1],
      // Cell 0/1 live samples at byte offsets 0/8; the other entries are padding.
      values: Float32Array.of(12, -999, 22, -999)
    },
    'rolling-mean',
    {windowLength: 1, activeSliceCount: 1}
  );
  const contributor = producers.find(
    producer => producer instanceof GPURasterRollingMean
  ) as GPURasterRollingMean;
  expect(contributor.props.windowLength).toBe(1);
  expect(contributor.props.chunkSliceCount).toBe(2);
  expect(uploads.get(contributor.props.parameters)![1]).toBe(1);
  expect(Array.from(uploads.get(contributor.props.values as GraphDataView)!)).toEqual([
    12, -999, 22, -999
  ]);
  await expect(
    runRasterTimeSeries(device, makeStack([12, 22]), 'rolling-mean', {windowLength: 129})
  ).rejects.toThrow(/128/);
});

for (const [distribution, zeroAtom] of [
  ['gamma', false],
  ['fisk', true],
  ['gamma', undefined],
  ['fisk', undefined]
] as const) {
  test(`W5-18 shares ${distribution} zeroAtom=${zeroAtom} between calibration and evaluation`, async () => {
    await runRasterTimeSeries(device, makeStack([0, 1, 2]), 'standardized-index', {
      calibrationSliceCount: 3,
      distribution,
      ...(zeroAtom === undefined ? {} : {zeroAtom})
    });
    const calibration = producers.find(
      producer => producer instanceof GPURasterStandardizedIndexCalibration
    ) as GPURasterStandardizedIndexCalibration;
    const evaluation = producers.find(
      producer => producer instanceof GPURasterStandardizedIndex
    ) as GPURasterStandardizedIndex;
    expect(calibration.props.zeroAtom).toBe(zeroAtom ?? distribution === 'gamma');
    expect(evaluation.props.zeroAtom).toBe(calibration.props.zeroAtom);
  });
}
test('W5-18 rejects invalid atom choices and the removed zeroInflated prop', async () => {
  for (const zeroAtom of ['yes', 1, null]) {
    await expect(
      runRasterTimeSeries(device, makeStack([0, 1, 2]), 'standardized-index', {
        calibrationSliceCount: 3,
        zeroAtom
      } as never)
    ).rejects.toThrow(/zeroAtom.*boolean/);
  }
  await expect(
    runRasterTimeSeries(device, makeStack([0, 1, 2]), 'standardized-index', {
      calibrationSliceCount: 3,
      zeroAtom: true,
      zeroInflated: false
    })
  ).rejects.toThrow(/zeroInflated was removed; use zeroAtom/);
});

test('SCL-06 retains sticky count status and suppresses only affected published slots', async () => {
  vi.mocked(CatalogGpuSession.prototype.readUint32).mockImplementation(async output => {
    const accumulator = producers.find(
      producer => producer instanceof GPURasterStreamingAccumulator
    ) as GPURasterStreamingAccumulator;
    return output.view === accumulator.props.state.status
      ? Uint32Array.of(1, 0)
      : Uint32Array.of(3, 3);
  });
  const result = await runRasterTimeSeries(
    device,
    {
      cellCount: 2,
      sliceCount: 3,
      bandCount: 1,
      times: [0, 1, 2],
      values: Float32Array.of(1, 2, 3, 4, 5, 6)
    },
    'climate-anomaly',
    {calibrationSliceCount: 3}
  );
  const accumulator = producers.find(
    producer => producer instanceof GPURasterStreamingAccumulator
  ) as GPURasterStreamingAccumulator;
  expect(accumulator.props.state.status).toBeDefined();
  expect(result.streamingCountStatus).toEqual([
    {complete: false, overflow: true},
    {complete: true, overflow: false}
  ]);
  expect(Array.from(result.columns['streamingStatus'].values)).toEqual([1, 0]);
  for (const name of ['mean', 'standardDeviation']) {
    expect(Number.isNaN(result.columns[name].values[0])).toBe(true);
    expect(result.columns[name].values[1]).toBe(1);
  }
  expect(Array.from(result.columns['anomaly'].values).map(Number.isNaN)).toEqual([
    true,
    true,
    true,
    false,
    false,
    false
  ]);
});

test('SCL-06 maps a monthly count loss to its cell and UTC-month observations', async () => {
  vi.mocked(CatalogGpuSession.prototype.readUint32).mockImplementation(async output => {
    const accumulator = producers.find(
      producer => producer instanceof GPURasterStreamingAccumulator
    ) as GPURasterStreamingAccumulator;
    const result = new Uint32Array(output.length);
    if (output.view === accumulator.props.state.status) result[13] = 1; // cell 1, February
    return result;
  });
  const result = await runRasterTimeSeries(
    device,
    {
      cellCount: 2,
      sliceCount: 3,
      bandCount: 1,
      times: [Date.UTC(2020, 0, 1), Date.UTC(2020, 1, 1), Date.UTC(2021, 1, 1)],
      values: Float32Array.of(1, 2, 3, 4, 5, 6)
    },
    'climate-anomaly',
    {slotCount: 12, calibrationSliceCount: 3}
  );
  expect(result.streamingCountStatus?.filter(status => !status.complete)).toHaveLength(1);
  expect(Array.from(result.columns['anomaly'].values).map(Number.isNaN)).toEqual([
    false,
    false,
    false,
    false,
    true,
    true
  ]);
  expect(Number.isNaN(result.columns['mean'].values[13])).toBe(true);
  expect(result.columns['mean'].values[1]).toBe(1);
});

for (const certificate of [Uint32Array.of(1), Uint32Array.of(0, 0, 0)]) {
  test(`B3 refuses ${certificate.length} count certificates for two cells`, async () => {
    vi.mocked(CatalogGpuSession.prototype.readUint32).mockImplementation(async output => {
      const accumulator = producers.find(
        producer => producer instanceof GPURasterStreamingAccumulator
      ) as GPURasterStreamingAccumulator;
      return output.view === accumulator.props.state.status ? certificate : Uint32Array.of(3, 3);
    });
    await expect(
      runRasterTimeSeries(
        device,
        {
          cellCount: 2,
          sliceCount: 3,
          bandCount: 1,
          times: [0, 1, 2],
          values: Float32Array.of(1, 2, 3, 4, 5, 6)
        },
        'climate-anomaly',
        {calibrationSliceCount: 3}
      )
    ).rejects.toThrow(/streamingStatus.*2.*received/);
  });
}

test('B4 validates only live rolling samples and still refuses an oversized live sample', async () => {
  const stack = {
    cellCount: 2,
    sliceCount: 3,
    bandCount: 1,
    times: [0, 1, 2],
    values: Float32Array.of(2, 6, 1e22, 20, 60, 1e22)
  };
  await runRasterTimeSeries(device, stack, 'rolling-mean', {windowLength: 1, activeSliceCount: 2});
  const rolling = producers.find(
    producer => producer instanceof GPURasterRollingMean
  ) as GPURasterRollingMean;
  expect(uploads.get(rolling.props.parameters)![1]).toBe(2);
  const liveOversize = {...stack, values: Float32Array.of(1e22, 6, 0, 20, 60, 0)};
  await expect(
    runRasterTimeSeries(device, liveOversize, 'rolling-mean', {
      windowLength: 1,
      activeSliceCount: 2
    })
  ).rejects.toThrow(/magnitude exceeds 1e15/);
  for (const activeSliceCount of [-1, 1.5, 4]) {
    await expect(
      runRasterTimeSeries(device, {...stack, values: new Float32Array(6)}, 'rolling-mean', {
        windowLength: 1,
        activeSliceCount
      })
    ).rejects.toThrow(/activeSliceCount/);
  }
});

test('S1 suppression uses the submitted slot indices after caller dates change during readback', async () => {
  const times = [Date.UTC(2020, 0, 1), Date.UTC(2020, 1, 1), Date.UTC(2021, 1, 1)];
  vi.mocked(CatalogGpuSession.prototype.readUint32).mockImplementation(async output => {
    const accumulator = producers.find(
      producer => producer instanceof GPURasterStreamingAccumulator
    ) as GPURasterStreamingAccumulator;
    const result = new Uint32Array(output.length);
    if (output.view === accumulator.props.state.status) {
      result[1] = 1;
      times[1] = Date.UTC(2020, 2, 1);
    }
    return result;
  });
  const result = await runRasterTimeSeries(
    device,
    {cellCount: 1, sliceCount: 3, bandCount: 1, times, values: Float32Array.of(1, 2, 3)},
    'climate-anomaly',
    {slotCount: 12, calibrationSliceCount: 3}
  );
  expect(Array.from(result.columns['anomaly'].values).map(Number.isNaN)).toEqual([
    false,
    true,
    true
  ]);
});

test('S2 reuses immutable decoded certificates for repeated complete and loss slots', async () => {
  vi.mocked(CatalogGpuSession.prototype.readUint32).mockImplementation(async output => {
    const accumulator = producers.find(
      producer => producer instanceof GPURasterStreamingAccumulator
    ) as GPURasterStreamingAccumulator;
    return output.view === accumulator.props.state.status
      ? Uint32Array.of(0, 0, 1, 1)
      : new Uint32Array(output.length);
  });
  const result = await runRasterTimeSeries(
    device,
    {cellCount: 4, sliceCount: 2, bandCount: 1, times: [0, 1], values: new Float32Array(8)},
    'streaming'
  );
  const statuses = result.streamingCountStatus!;
  expect(statuses).toHaveLength(4);
  expect(statuses[0]).toBe(statuses[1]);
  expect(statuses[2]).toBe(statuses[3]);
  expect(Object.isFrozen(statuses[0])).toBe(true);
  expect(Object.isFrozen(statuses[2])).toBe(true);
});

for (const operation of [
  'climate-anomaly',
  'standardized-calibration',
  'standardized-index'
] as const) {
  test(`EARLIER calibration default is bounded by a short ${operation} stack`, async () => {
    await runRasterTimeSeries(device, makeStack([0, 1, 2]), operation, {zeroAtom: false});
    const calibration = producers.find(
      producer =>
        producer instanceof GPURasterStandardizedIndexCalibration ||
        producer instanceof GPURasterStreamingAccumulator
    ) as GPURasterStandardizedIndexCalibration | GPURasterStreamingAccumulator;
    expect(calibration.props.chunkSliceCount).toBe(3);
    await expect(
      runRasterTimeSeries(device, makeStack([0, 1, 2]), operation, {calibrationSliceCount: 4})
    ).rejects.toThrow(/calibrationSliceCount/);
  });
}
