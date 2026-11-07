// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import * as arrow from 'apache-arrow';
import {expect, test, vi} from 'vitest';
import {prepareArrowTableForGPUAnalytics} from '@luma-contributors/analysis/analysis-core';
import {NullDevice} from '@luma.gl/core/test-utils';
import {KeplerDatasetRegistry} from '../src/runtime/dataset-registry';
import {
  estimateTabularPointPositionBytes,
  prepareTabularPointPositions,
  validateTabularPointPositions,
  makeGPUVectorsFromTabularPointPositions
} from '../../../examples/deck/kepler-compute/src/point-position-source';

const roles = {longitude: 'east', latitude: 'north', altitude: 'height'};
function prepare(values: readonly (readonly (number | null)[])[] = [[0, -2], [], [null, 5]]) {
  const batches = values.map(heights => {
    const table = new arrow.Table({
      east: arrow.vectorFromArray(
        heights.map(() => -73),
        new arrow.Float32()
      ),
      north: arrow.vectorFromArray(
        heights.map(() => 40),
        new arrow.Float32()
      ),
      height: arrow.vectorFromArray([...heights], new arrow.Float32())
    });
    return arrow.tableFromIPC(arrow.tableToIPC(table, 'stream')).batches[0];
  });
  return prepareArrowTableForGPUAnalytics(new arrow.Table(batches[0].schema, batches));
}

test('tabular positions preserve empty batches, XY source identity and independent altitude validity', () => {
  const source = prepare();
  const result = prepareTabularPointPositions(source, roles);
  expect(result.batches.map(batch => batch.rowCount)).toEqual([2, 0, 2]);
  expect(result.batches.map(batch => batch.sourceRowOffset)).toEqual([0, 2, 2]);
  expect(result.batches.map(batch => [...batch.positions])).toEqual([
    [-73, 40, -73, 40],
    [],
    [-73, 40, -73, 40]
  ]);
  expect(result.batches.map(batch => [...batch.positions3D!])).toEqual([
    [-73, 40, 0, -73, 40, -2],
    [],
    [-73, 40, 0, -73, 40, 5]
  ]);
  expect(result.batches.map(batch => [...batch.altitudeValidity!])).toEqual([[1, 1], [], [1, 1]]);
  expect(source.table.getChild('height')?.get(2)).toBeNull();
  expect(result.bounds).toEqual([-73, 40, -73, 40]);
  expect(estimateTabularPointPositionBytes([2, 0, 2], true)).toEqual({
    cpuByteLength: 96,
    gpuByteLength: 120
  });
  expect(result.typedStorageByteLength).toBe(96);
  expect(result.gpuByteLength).toBe(120);
  const flat = prepareTabularPointPositions(source, {longitude: 'east', latitude: 'north'});
  expect(flat.batches[0].positions3D).toBeUndefined();
  expect(flat.gpuByteLength).toBe(40);
});

test('null and out-of-range coordinates are hidden and counted instead of failing preparation', () => {
  const table = new arrow.Table({
    east: arrow.vectorFromArray([1, -183, null, 3], new arrow.Float32()),
    north: arrow.vectorFromArray([2, 10, 5, 95], new arrow.Float32())
  });
  const source = prepareArrowTableForGPUAnalytics(table);
  const result = prepareTabularPointPositions(source, {longitude: 'east', latitude: 'north'});
  expect(result.invalidRowCount).toBe(3);
  expect(result.bounds).toEqual([1, 2, 1, 2]);
  const [positions] = result.batches.map(batch => [...batch.positions]);
  expect(positions.slice(0, 2)).toEqual([1, 2]);
  expect(positions.slice(2).every(Number.isNaN)).toBe(true);
  expect(() =>
    validateTabularPointPositions(source, {longitude: 'east', latitude: 'north'})
  ).not.toThrow();
});

test('altitude uses explicit Float64 normalization and rejects implicit integer precision loss', () => {
  const table = new arrow.Table({
    east: arrow.makeVector(new Float32Array([1])),
    north: arrow.makeVector(new Float32Array([2])),
    height: arrow.makeVector(new Float64Array([0.1]))
  });
  expect(() => prepareArrowTableForGPUAnalytics(table, {float64: {mode: 'exact'}})).toThrow();
  const normalized = prepareArrowTableForGPUAnalytics(table, {
    float64: {mode: 'round', maximumAbsoluteError: 1e-7, maximumRelativeError: 1e-6}
  });
  expect(prepareTabularPointPositions(normalized, roles).batches[0].positions3D![2]).toBe(
    Math.fround(0.1)
  );
  const integer = prepareArrowTableForGPUAnalytics(
    new arrow.Table({
      east: arrow.makeVector(new Float32Array([1])),
      north: arrow.makeVector(new Float32Array([2])),
      height: arrow.makeVector(new Int32Array([16777217]))
    })
  );
  expect(() => prepareTabularPointPositions(integer, roles)).toThrow(
    /explicit Float32 normalization/
  );
  expect(() => validateTabularPointPositions(integer, roles)).toThrow(
    /explicit Float32 normalization/
  );
  expect(() => prepareTabularPointPositions(normalized, {...roles, altitude: 'missing'})).toThrow(
    /selected normalized numeric/
  );
});

test('point upload reserves its exact plan, retains every batch and releases owned buffers only after the fence', async () => {
  const device = new NullDevice({});
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 120});
  const prepared = prepareTabularPointPositions(prepare(), roles);
  const vectors = makeGPUVectorsFromTabularPointPositions(device, prepared, {
    memoryRegistry: registry
  });
  expect(vectors.positions.format).toBe('float32x2');
  expect(vectors.positions3D?.format).toBe('float32x3');
  expect(vectors.altitudeValidity?.format).toBe('uint32');
  expect(vectors.positions3D?.data.map(chunk => chunk.length)).toEqual([2, 0, 2]);
  expect(registry.memoryUsage.reservedBytes.gpu).toBe(120);
  const buffers = [
    ...vectors.positions.data,
    ...vectors.positions3D!.data,
    ...vectors.altitudeValidity!.data
  ].map(chunk => chunk.buffer);
  let release!: () => void;
  const destroyed = vectors.destroy(
    new Promise<void>(resolve => {
      release = resolve;
    })
  );
  expect(buffers.every(buffer => !buffer.destroyed)).toBe(true);
  expect(vectors.gpuByteLength).toBe(120);
  release();
  await destroyed;
  expect(buffers.every(buffer => buffer.destroyed)).toBe(true);
  expect(vectors.gpuByteLength).toBe(0);
  expect(registry.memoryUsage.totalBytes).toBe(0);
  await registry.destroy();
  device.destroy();
});

test('position upload admission and partial allocation failure leave no buffers or reservations', async () => {
  const device = new NullDevice({});
  const prepared = prepareTabularPointPositions(prepare(), roles);
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 119});
  const create = vi.spyOn(device, 'createBuffer');
  expect(() =>
    makeGPUVectorsFromTabularPointPositions(device, prepared, {memoryRegistry: registry})
  ).toThrow(/budget/);
  expect(create).not.toHaveBeenCalled();
  await registry.destroy();
  create.mockRestore();
  const admitted = new KeplerDatasetRegistry({memoryBudgetBytes: 120});
  const allocate = device.createBuffer.bind(device);
  const buffers: ReturnType<typeof allocate>[] = [];
  vi.spyOn(device, 'createBuffer').mockImplementation(props => {
    if (buffers.length === 4) throw new Error('Injected position upload failure');
    const buffer = allocate(props);
    buffers.push(buffer);
    return buffer;
  });
  expect(() =>
    makeGPUVectorsFromTabularPointPositions(device, prepared, {memoryRegistry: admitted})
  ).toThrow(/Injected/);
  expect(buffers.every(buffer => buffer.destroyed)).toBe(true);
  expect(admitted.memoryUsage.totalBytes).toBe(0);
  await admitted.destroy();
  device.destroy();
});

test('pinned native Arrow and row-array point accessors differ for null altitude', () => {
  const source = readFileSync('node_modules/@kepler.gl/layers/dist/esm/layer-utils.js', 'utf8');
  const start = source.indexOf('export function createGeoArrowPointVector(');
  const end = source.indexOf('\n}', start) + 2;
  expect(start).toBeGreaterThan(-1);
  // Execute the installed helper body with its actual Arrow dependency, avoiding unrelated Deck imports.
  const native = new Function(
    'arrow',
    'GEOARROW_METADATA_KEY',
    'GEOARROW_EXTENSIONS',
    `${source.slice(start, end).replace('export function', 'function')};return createGeoArrowPointVector;`
  )(arrow, 'ARROW:extension:name', {POINT: 'geoarrow.point'});
  const accessorSource = readFileSync(
    'node_modules/@kepler.gl/layers/dist/esm/point-layer/point-layer.js',
    'utf8'
  );
  const accessorStart = accessorSource.indexOf('export var pointPosAccessor =');
  const accessorEnd = accessorSource.indexOf('export var geojsonPosAccessor', accessorStart);
  const accessor = new Function(
    `${accessorSource.slice(accessorStart, accessorEnd).replace('export var', 'var')};return pointPosAccessor;`
  )();
  const rows = [
    [1, 2, null],
    [1, 2, 0],
    [1, 2, -2],
    [1, 2, NaN],
    [1, 2, Infinity]
  ];
  const container = {
    numRows: () => rows.length,
    valueAt: (row: number, column: number) => rows[row][column]
  };
  const position = accessor({lng: {fieldIdx: 0}, lat: {fieldIdx: 1}, altitude: {fieldIdx: 2}})(
    container
  );
  const vector: arrow.Vector = native(container, position);
  expect([...vector.data[0].children[0].values].slice(0, 9)).toEqual([1, 2, 0, 1, 2, 0, 1, 2, -2]);
  expect(vector.data[0].children[0].nullCount).toBe(0);
  expect(rows.map((_, index) => position({index}).every(Number.isFinite))).toEqual([
    false,
    true,
    true,
    false,
    false
  ]);
});

test('captured device loss settles point upload retirement even when the caller fence is stranded', async () => {
  const device = new NullDevice({});
  let resolveLoss!: (value: {reason: 'destroyed'; message: string}) => void;
  const loss = new Promise<{reason: 'destroyed'; message: string}>(resolve => {
    resolveLoss = resolve;
  });
  Object.defineProperty(device, 'lost', {value: loss});
  const registry = new KeplerDatasetRegistry({memoryBudgetBytes: 120});
  const vectors = makeGPUVectorsFromTabularPointPositions(
    device,
    prepareTabularPointPositions(prepare(), roles),
    {memoryRegistry: registry}
  );
  const destroyed = vectors.destroy(new Promise<void>(() => {}));
  expect(registry.memoryUsage.reservedBytes.gpu).toBe(120);
  resolveLoss({reason: 'destroyed', message: 'test loss'});
  await destroyed;
  expect(vectors.positions.data.every(chunk => chunk.buffer.destroyed)).toBe(true);
  expect(vectors.positions3D!.data.every(chunk => chunk.buffer.destroyed)).toBe(true);
  expect(registry.memoryUsage.totalBytes).toBe(0);
  await registry.destroy();
  device.destroy();
});
