// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {GPUVector} from '@luma.gl/graph/gpu-data';
import {NullDevice} from '@luma.gl/core/test-utils';
import * as measureWidening from '../../../examples/deck/kepler-compute/src/resident-chart-measure-widening';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {getKeplerAggregationLayout} from '../src/aggregation';
import {
  AggregationRuntime,
  getNativeAggregationSettings,
  prepareAggregationProjection,
  getNativeAggregationTopology,
  makeAggregateFootprints,
  type NativeAggregationLayer
} from '../../../examples/deck/kepler-compute/src/aggregation-runtime';

function layer(
  overrides: Partial<NativeAggregationLayer['config']['visConfig']> = {}
): NativeAggregationLayer {
  return {
    id: 'grid',
    type: 'grid',
    config: {
      columnMode: 'points',
      columns: {lat: {value: 'latitude'}, lng: {value: 'longitude'}},
      colorScale: 'quantize',
      visConfig: {
        worldUnitSize: 1,
        coverage: 1,
        colorAggregation: 'average',
        colorRange: {colors: ['#000000', '#ffffff']},
        ...overrides
      }
    }
  };
}

describe('native Kepler aggregation controls and projection', () => {
  test('a small radius over distant valid-coordinate outliers requires a dense lattice beyond the cap', () => {
    const projection = prepareAggregationProjection([-139, -122], [37, 47]);
    const layout = getKeplerAggregationLayout(
      getNativeAggregationTopology(projection, {kind: 'hexagon', sizeMeters: 50})
    );
    expect(layout.cellCount).toBeGreaterThan(65536);
    // Native sparse occupied-bin aggregation would need at most these two occupied cells.
    expect(layout.cellCount).toBeGreaterThan(projection.positions.length / 2);
  });
  test('converts native kilometers to metric cell size and maps count, sum and average', () => {
    const grid = layer({worldUnitSize: 0.25});
    expect(getNativeAggregationSettings(grid)).toMatchObject({sizeMeters: 250, statistic: 'count'});
    grid.config.colorField = {name: 'value'};
    expect(getNativeAggregationSettings(grid)).toMatchObject({
      statistic: 'mean',
      weightColumn: 'value'
    });
    grid.config.visConfig.colorAggregation = 'sum';
    expect(getNativeAggregationSettings(grid).statistic).toBe('sum');
    grid.config.visConfig.colorAggregation = 'median';
    // Statistics beyond count/sum/mean use the advanced style plan; the fast-path statistic stays count.
    expect(getNativeAggregationSettings(grid).advanced?.color.aggregation).toBe('median');
  });
  test('rejects unsupported visible controls instead of rendering with different semantics', () => {
    expect(() => getNativeAggregationSettings(layer({enable3d: true}))).not.toThrow();
    expect(getNativeAggregationSettings(layer({percentile: [5, 95]})).advanced).toBeDefined();
    expect(() => getNativeAggregationSettings(layer({percentile: [NaN, 95]}))).toThrow(
      /percentile/
    );
    expect(() => getNativeAggregationSettings(layer({coverage: -1}))).toThrow(/coverage/);
    expect(() => getNativeAggregationSettings(layer({worldUnitSize: 0}))).toThrow(/positive/);
    const grid = layer();
    grid.config.colorScale = 'weird';
    expect(() => getNativeAggregationSettings(grid)).toThrow(/weird/);
    grid.config.colorScale = 'quantize';
    // Aggregate count labels are not drawn, but the layer still renders.
    grid.config.textLabel = [{field: {name: 'count'}}];
    expect(() => getNativeAggregationSettings(grid)).not.toThrow();
  });
  test('uses latitude-dependent meters rather than degree units and keeps source invalid rows', () => {
    const equator = prepareAggregationProjection([-0.005, 0.005, NaN], [0, 0, 0]);
    const northern = prepareAggregationProjection([-0.005, 0.005], [60, 60]);
    const equatorWidth = equator.positions[2] - equator.positions[0];
    const northernWidth = northern.positions[2] - northern.positions[0];
    expect(equatorWidth).toBeGreaterThan(1100);
    expect(equatorWidth).toBeLessThan(1120);
    expect(northernWidth / equatorWidth).toBeCloseTo(0.5, 5);
    expect(Number.isNaN(equator.positions[4])).toBe(true);
    // Rows outside the Web Mercator domain are skipped like native Kepler; global spans are allowed.
    expect(prepareAggregationProjection([179, -179], [0, 0]).geographicBounds).toEqual([
      -179, 0, 179, 0
    ]);
    const outside = prepareAggregationProjection([0, 1], [89, 10]);
    expect(Number.isNaN(outside.positions[0])).toBe(true);
    expect(outside.geographicBounds).toEqual([1, 10, 1, 10]);
    expect(() => prepareAggregationProjection([0], [89])).toThrow(/finite data/);
  });
  test('anchors meter grids globally and pads hexagons without losing edge points', () => {
    const projection = prepareAggregationProjection([-74.01, -74, -73.99], [40.7, 40.71, 40.72]);
    const grid = getNativeAggregationTopology(projection, {kind: 'grid', sizeMeters: 250});
    expect(grid.kind).toBe('grid');
    if (grid.kind !== 'grid') throw new Error('Expected grid');
    const commonSize = projection.commonUnitsPerMeter * 250;
    const nativeAnchor = Math.fround(
      Math.floor(projection.commonOrigin[0] / commonSize) * commonSize
    );
    const latticeOffset =
      ((nativeAnchor - projection.commonOrigin[0]) / projection.commonUnitsPerMeter -
        grid.origin[0]) /
      250;
    expect(latticeOffset).toBeCloseTo(Math.round(latticeOffset), 7);
    const layout = getKeplerAggregationLayout(grid);
    expect(layout.bounds[0]).toBeLessThanOrEqual(projection.bounds[0]);
    expect(layout.bounds[2]).toBeGreaterThan(projection.bounds[2]);
    const hexagon = getNativeAggregationTopology(projection, {kind: 'hexagon', sizeMeters: 250});
    expect(hexagon.kind).toBe('hexagon');
    if (hexagon.kind !== 'hexagon') throw new Error('Expected hexagon');
    expect(hexagon.assignment).toBe('d3');
    expect(hexagon.bounds[0]).toBeLessThan(projection.bounds[0]);
    expect(hexagon.bounds[1]).toBeLessThan(projection.bounds[1]);
    expect(hexagon.bounds[2]).toBeGreaterThan(projection.bounds[2]);
  });
  test('coverage shrinks footprints about cell centers without changing aggregate row identity', () => {
    const layout = getKeplerAggregationLayout({
      kind: 'grid',
      origin: [0, 0],
      cellSize: [100, 200],
      gridSize: [1, 1]
    });
    expect(Array.from(makeAggregateFootprints(layout, 0.5))).toEqual([
      25, 50, 0, 1, 75, 50, 0, 1, 75, 150, 0, 1, 25, 150, 0, 1
    ]);
  });
});

test('integer aggregates await the existing exact cast, reuse it, and release it after retirement', async () => {
  const device = new NullDevice({});
  const makeVector = <Format extends 'uint32' | 'float32'>(
    format: Format,
    data: Uint32Array | Float32Array
  ) =>
    new GPUVector({
      type: 'buffer',
      name: format,
      format,
      length: 2,
      ownsBuffer: true,
      buffer: device.createBuffer({data, usage: Buffer.STORAGE})
    });
  const source = makeVector('uint32', new Uint32Array([1980, 2000]));
  const mask = makeVector('uint32', new Uint32Array([1, 1]));
  const widened = makeVector('float32', new Float32Array([1980, 2000]));
  const destroy = vi.fn(() => widened.destroy());
  const cast = vi
    .spyOn(measureWidening, 'widenResidentChartMeasure')
    .mockResolvedValue({vector: widened, destroy});
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1024});
  const runtime = new AggregationRuntime({
    device,
    longitude: [-74, -74],
    latitude: [40.7, 40.7],
    selectionMask: mask,
    columns: {yearbuilt: {values: source}},
    memoryRegistry: registry
  });
  const aggregate = layer();
  aggregate.config.colorField = {name: 'yearbuilt'};
  try {
    expect(runtime.resolve(aggregate)).toEqual({});
    await vi.waitFor(() => {
      // A tiny registry deliberately refuses the program after its input has passed admission.
      // This checks the asynchronous boundary without launching a GPU or allocating a large program.
      expect(runtime.resolve(aggregate).reason).toMatch(/budget/i);
    });
    runtime.resolve(aggregate);
    expect(cast).toHaveBeenCalledOnce();
    expect(cast.mock.calls[0]?.[0].vector).toBe(source);
    expect(runtime.accountedResidentByteLength).toBe(runtime.uploadBytes + 8);
    await runtime.destroy(Promise.resolve());
    expect(destroy).toHaveBeenCalledOnce();
    expect(source.data[0].buffer.destroyed).toBe(false);
    expect(registry.memoryUsage.totalBytes).toBe(0);
  } finally {
    cast.mockRestore();
    await runtime.destroy(Promise.resolve());
    source.destroy();
    mask.destroy();
    device.destroy();
  }
});

test('published aggregation preserves nullable source slots and empty upload batches behind a shared fence', async () => {
  const device = new NullDevice({});
  const mask = new GPUVector({
    type: 'buffer',
    name: 'selection',
    format: 'uint32',
    length: 4,
    ownsBuffer: true,
    buffer: device.createBuffer({data: new Uint32Array([1, 1, 1, 1]), usage: Buffer.STORAGE})
  });
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1024});
  const runtime = new AggregationRuntime({
    device,
    longitude: [-74, null, -74, -74],
    latitude: [40.7, 0, undefined, 40.7],
    sourceChunkLengths: [2, 0, 2],
    selectionMask: mask,
    coordinateColumns: {lat: 'north', lng: 'east'},
    memoryRegistry: registry
  });
  expect(runtime.positions.data.map(chunk => chunk.length)).toEqual([2, 0, 2]);
  expect(Array.from(runtime.projection.positions).map(Number.isNaN)).toEqual([
    false,
    false,
    true,
    true,
    true,
    true,
    false,
    false
  ]);
  expect(runtime.projection.geographicBounds).toEqual([-74, 40.7, -74, 40.7]);
  expect(runtime.uploadBytes).toBe(36);
  expect(registry.memoryUsage.reservedBytes).toEqual({cpu: 32, gpu: 36});
  let release!: () => void;
  const fence = new Promise<void>(resolve => {
    release = resolve;
  });
  const pending = runtime.destroy(fence);
  expect(runtime.destroy(Promise.resolve())).toBe(pending);
  expect(runtime.accountedResidentByteLength).toBe(36);
  expect(runtime.positions.data.every(chunk => !chunk.buffer.destroyed)).toBe(true);
  release();
  await pending;
  expect(runtime.accountedResidentByteLength).toBe(0);
  expect(registry.memoryUsage.totalBytes).toBe(0);
  expect(mask.data[0].buffer.destroyed).toBe(false);
  mask.destroy();
  device.destroy();
});

test('published aggregation rejects malformed chunks before allocation and rolls back partial uploads', () => {
  const device = new NullDevice({});
  const mask = new GPUVector({
    type: 'buffer',
    name: 'selection',
    format: 'uint32',
    length: 2,
    ownsBuffer: true,
    buffer: device.createBuffer({data: new Uint32Array([1, 1]), usage: Buffer.STORAGE})
  });
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1024});
  const props = {
    device,
    longitude: [-74, -74],
    latitude: [40.7, 40.7],
    selectionMask: mask,
    memoryRegistry: registry
  };
  const original = device.createBuffer.bind(device);
  const created: Buffer[] = [];
  const allocation = vi.spyOn(device, 'createBuffer').mockImplementation(options => {
    if (created.length === 1) throw new Error('Injected upload failure');
    const buffer = original(options);
    created.push(buffer);
    return buffer;
  });
  expect(() => new AggregationRuntime({...props, sourceChunkLengths: [1]})).toThrow(/chunks/);
  expect(allocation).not.toHaveBeenCalled();
  expect(() => new AggregationRuntime({...props, sourceChunkLengths: [1, 1]})).toThrow(/Injected/);
  expect(created.every(buffer => buffer.destroyed)).toBe(true);
  expect(registry.memoryUsage.totalBytes).toBe(0);
  allocation.mockRestore();
  mask.destroy();
  device.destroy();
});

test('published aggregation device loss settles a stranded source retirement without releasing borrowed masks', async () => {
  const device = new NullDevice({});
  let lose!: () => void;
  Object.defineProperty(device, 'lost', {
    value: new Promise<void>(resolve => {
      lose = resolve;
    })
  });
  const mask = new GPUVector({
    type: 'buffer',
    name: 'selection',
    format: 'uint32',
    length: 1,
    ownsBuffer: true,
    buffer: device.createBuffer({data: new Uint32Array([1]), usage: Buffer.STORAGE})
  });
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 1024});
  const runtime = new AggregationRuntime({
    device,
    longitude: [-74],
    latitude: [40.7],
    selectionMask: mask,
    memoryRegistry: registry
  });
  const pending = runtime.destroy(new Promise<void>(() => {}));
  lose();
  await pending;
  expect(registry.memoryUsage.totalBytes).toBe(0);
  expect(runtime.positions.data[0].buffer.destroyed).toBe(true);
  expect(mask.data[0].buffer.destroyed).toBe(false);
  mask.destroy();
  device.destroy();
});
