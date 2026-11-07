// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {GPUKeplerAggregation} from '../src/aggregation';
import oracle from './fixtures/d3-hexbin-oracle.json';

it('d3 assignment uses positive-infinity half rounding instead of WGSL ties-to-even', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = makeVector(device, 'float32x2', [oracle.float32HalfTie.point]);
  const aggregation = new GPUKeplerAggregation({
    device,
    positions,
    topology: {
      kind: 'hexagon',
      bounds: [0, 0, 2, 2],
      radius: oracle.float32HalfTie.radius,
      assignment: 'd3'
    }
  });
  try {
    submit(device, aggregation);
    const counts = await readVector(aggregation.counts);
    const [column, row] = oracle.float32HalfTie.cell;
    expect(column).toBe(1);
    expect(counts[row * aggregation.layout.gridSize[0] + column]).toBe(1);
    expect(counts[0]).toBe(0);
    expect(counts.reduce((total, count) => total + count, 0)).toBe(1);
  } finally {
    aggregation.destroy();
    positions.destroy();
  }
});

it('resident d3 mode matches independent native hex counts and signed sums across chunks and mask updates', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The explicit x=sqrt(3)/2 fixture is below the exact boundary in f64 but rounds to the
  // boundary after f32 division. CPU/GPU arithmetic cannot promise identity there. Exactly
  // representable vertical half ties remain in this GPU oracle.
  const cases = oracle.cases.filter(([x, y], index) => x >= 0 && y >= 0 && index !== 7);
  const coordinates = cases.flatMap(([x, y]) => [x, y]);
  const weightValues = cases.map((_, index) => (index % 9) - 4);
  const maskValues = cases.map((_, index) => (index % 5 ? 3 : 0));
  const positions = makeVector(device, 'float32x2', [
    coordinates.slice(0, 128),
    [],
    coordinates.slice(128)
  ]);
  const weights = makeVector(device, 'float32', [
    weightValues.slice(0, 33),
    weightValues.slice(33)
  ]);
  const selectionMask = makeVector(device, 'uint32', [maskValues]);
  const aggregation = new GPUKeplerAggregation({
    device,
    positions,
    weights,
    selectionMask,
    statistic: 'sum',
    topology: {kind: 'hexagon', bounds: [0, 0, 16, 16], radius: 1, assignment: 'd3'}
  });
  try {
    for (const selected of [maskValues, maskValues.map(() => 1)]) {
      selectionMask.data[0].buffer.write(Uint32Array.from(selected));
      submit(device, aggregation);
      const expectedCounts = new Array<number>(aggregation.layout.cellCount).fill(0);
      const expectedSums = new Array<number>(aggregation.layout.cellCount).fill(0);
      for (let index = 0; index < cases.length; index++) {
        if (!selected[index]) continue;
        const [, , column, row] = cases[index];
        const cell = row * aggregation.layout.gridSize[0] + column;
        expectedCounts[cell]++;
        expectedSums[cell] += weightValues[index];
      }
      expect(await readVector(aggregation.counts)).toEqual(expectedCounts);
      expect(await readVector(aggregation.values)).toEqual(expectedSums);
      expect(await readVector(aggregation.overflow)).toEqual([0]);
    }
  } finally {
    aggregation.destroy();
    positions.destroy();
    weights.destroy();
    selectionMask.destroy();
  }
});

it('resident rectangular grid matches pinned counts, signed sums, missing-weight policy and inclusive edges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = makeVector(device, 'float32x2', [
    [10, 20, 11, 21, 12, 20],
    [],
    [14, 28, 9, 20, 13, 25, NaN, 21, 10, 27]
  ]);
  const weights = makeVector(device, 'float32', [
    [2, -1],
    [0, 3, 100, NaN, 5, Infinity]
  ]);
  const selectionMask = makeVector(device, 'uint32', [[7], [], [1, 1, 1, 1], [1, 1, 1]]);
  const validity = makeVector(device, 'uint32', [[1, 1, 1, 1, 1, 1, 1, 1]]);
  const inputs = [positions, weights, selectionMask, validity];
  const readSpies = inputs.flatMap(vector =>
    vector.data.map(chunk => vi.spyOn(chunk.buffer, 'readAsync'))
  );
  const aggregation = new GPUKeplerAggregation({
    device,
    positions,
    weights,
    selectionMask,
    validity,
    statistic: 'mean',
    topology: {kind: 'grid', origin: [10, 20], cellSize: [2, 4], gridSize: [2, 2]}
  });
  const compileSpy = vi.spyOn(GPUCommandGraph.prototype, 'compile');
  try {
    const polygons = aggregation.getPolygonGeometry();
    const countBuffer = aggregation.counts.data[0].buffer;
    expect(polygons.selectionMask).toBe(aggregation.counts);
    expect(aggregation.memberLookup).toBe('not-materialized');
    submit(device, aggregation);
    expect(await readVector(aggregation.counts)).toEqual([2, 1, 1, 2]);
    expect(await readVector(aggregation.sums!)).toEqual([1, 0, 0, 3]);
    expect(await readVector(aggregation.values)).toEqual([0.5, 0, 0, 1.5]);
    expect(await readVector(aggregation.centers)).toEqual([11, 22, 13, 22, 11, 26, 13, 26]);
    const residentBytes = aggregation.accountedResidentByteLength;
    // Explicit null validity excludes the whole row from both numerator and denominator.
    validity.data[0].buffer.write(new Uint32Array([1, 1, 1, 1, 1, 0, 1, 0]));
    submit(device, aggregation);
    expect(await readVector(aggregation.counts)).toEqual([2, 1, 0, 1]);
    expect(await readVector(aggregation.means!)).toEqual([0.5, 0, 0, 3]);
    for (const chunk of selectionMask.data)
      chunk.buffer.write(new Uint32Array(Math.max(1, chunk.length)));
    submit(device, aggregation);
    expect(await readVector(aggregation.counts)).toEqual([0, 0, 0, 0]);
    expect(await readVector(aggregation.values)).toEqual([0, 0, 0, 0]);
    expect(aggregation.counts.data[0].buffer).toBe(countBuffer);
    expect(aggregation.getPolygonGeometry()).toBe(polygons);
    expect(aggregation.accountedResidentByteLength).toBe(residentBytes);
    expect(compileSpy).toHaveBeenCalledTimes(1);
    expect(aggregation.readbackByteLength).toBe(0);
    expect(await readVector(aggregation.overflow)).toEqual([0]);
    for (const spy of readSpies) expect(spy).not.toHaveBeenCalled();
    // Four CCW corners and local-chunk triangle references per stable aggregate ID.
    expect(Array.from(polygons.positions.data[0].valueOffsets!)).toEqual([0, 4, 8, 12, 16]);
    const polygonBytes = await polygons.positions.data[0].buffer.readAsync(0, 64);
    expect(Array.from(new Float32Array(polygonBytes.buffer, polygonBytes.byteOffset, 16))).toEqual([
      10, 20, 0, 1, 12, 20, 0, 1, 12, 24, 0, 1, 10, 24, 0, 1
    ]);
    const rowBytes = await polygons.rowIndices.data[0].buffer.readAsync(0, 64);
    expect(Array.from(new Uint32Array(rowBytes.buffer, rowBytes.byteOffset, 16))).toEqual([
      0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3
    ]);
  } finally {
    compileSpy.mockRestore();
    aggregation.destroy();
    expect(aggregation.accountedResidentByteLength).toBe(0);
    for (const vector of inputs)
      for (const chunk of vector.data) expect(chunk.buffer.destroyed).toBe(false);
    readSpies.forEach(spy => spy.mockRestore());
    inputs.forEach(vector => vector.destroy());
  }
});

it('resident hex grid keeps odd-r aggregate identities and weighted/unweighted selection across chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = makeVector(device, 'float32x2', [
    [0, 0, 0.1, 0.1],
    [],
    [Math.sqrt(3) * 2, 0, Math.sqrt(3), 3, 20, 20]
  ]);
  const weights = makeVector(device, 'float32', [
    [2, -3, 0],
    [4, 100]
  ]);
  const selectionMask = makeVector(device, 'uint32', [[1, 1, 1, 1, 1]]);
  const topology = {kind: 'hexagon', bounds: [0, 0, 8, 6], radius: 2} as const;
  const weighted = new GPUKeplerAggregation({
    device,
    positions,
    weights,
    selectionMask,
    topology,
    statistic: 'sum'
  });
  const unweighted = new GPUKeplerAggregation({device, positions, selectionMask, topology});
  try {
    submit(device, weighted);
    submit(device, unweighted);
    const counts = [2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
    expect(await readVector(weighted.counts)).toEqual(counts);
    expect(await readVector(unweighted.values)).toEqual(counts);
    expect(await readVector(weighted.values)).toEqual([-1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]);
    selectionMask.data[0].buffer.write(new Uint32Array([0, 1, 0, 1, 1]));
    submit(device, weighted);
    expect(await readVector(weighted.counts)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
    expect(await readVector(weighted.values)).toEqual([-3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]);
    expect(await readVector(weighted.overflow)).toEqual([0]);
    const polygon = weighted.getPolygonGeometry();
    expect(polygon.positions.data[0].valueLength).toBe(72);
    expect(polygon.indices.data[0].valueLength).toBe(144);
    expect(polygon.selectionMask).toBe(weighted.counts);
  } finally {
    weighted.destroy();
    unweighted.destroy();
    positions.destroy();
    weights.destroy();
    selectionMask.destroy();
  }
});

it('empty source produces stable all-zero aggregate cells', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = makeVector(device, 'float32x2', [[]]);
  const aggregation = new GPUKeplerAggregation({
    device,
    positions,
    topology: {kind: 'grid', origin: [0, 0], cellSize: [1, 1], gridSize: [2, 1]}
  });
  try {
    submit(device, aggregation);
    expect(await readVector(aggregation.counts)).toEqual([0, 0]);
    expect(await readVector(aggregation.values)).toEqual([0, 0]);
  } finally {
    aggregation.destroy();
    positions.destroy();
  }
});

function makeVector<Format extends 'float32x2' | 'float32' | 'uint32'>(
  device: Device,
  format: Format,
  chunks: number[][]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name: 'input',
    format,
    data: chunks.map(values => {
      const data = format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values);
      return new GPUData({
        format,
        length: values.length / (format === 'float32x2' ? 2 : 1),
        ownsBuffer: true,
        buffer: device.createBuffer({
          ...(data.length ? {data} : {byteLength: 8}),
          usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
        })
      });
    })
  });
}

function submit(device: Device, aggregation: GPUKeplerAggregation): void {
  const encoder = device.createCommandEncoder();
  aggregation.encode(encoder);
  device.submit(encoder.finish());
}

async function readVector(
  vector: GPUVector<'float32x2' | 'float32' | 'uint32'>
): Promise<number[]> {
  const chunk = vector.data[0];
  const values = chunk.length * (vector.format === 'float32x2' ? 2 : 1);
  const bytes = await chunk.buffer.readAsync(0, values * 4);
  return Array.from(
    vector.format === 'uint32'
      ? new Uint32Array(bytes.buffer, bytes.byteOffset, values)
      : new Float32Array(bytes.buffer, bytes.byteOffset, values)
  );
}
