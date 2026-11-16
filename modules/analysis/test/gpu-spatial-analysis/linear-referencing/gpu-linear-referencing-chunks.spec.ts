// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPULinearReferencing} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-linear-referencing';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  createVectorView,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createPaddedQueryFixture} from './gpu-linear-referencing-chunks.test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function runReferencing(
  device: Device,
  chunked: boolean,
  spherical: boolean,
  candidateCapacity = 64,
  chunkLengths = [1, 3, 2]
) {
  const fixture = createPaddedQueryFixture(device);
  const {graph, input, output} = fixture;
  const positions = input('positions', 'float32x2', new Float32Array([0, 0, 10, 0]));
  const lineOffsets = input('line-offsets', 'uint32', new Uint32Array([0, 2]));
  const radius = input('radius', 'float32', new Float32Array([100]));
  const points = [
    [2, 3],
    [6, -2],
    [5, 0],
    [-2, 1],
    [15, -3],
    [Number.NaN, 0]
  ];
  const total = chunkLengths.reduce((sum, length) => sum + length, 0);
  const lengths = chunked ? chunkLengths : [total];
  const queryChunks: GraphDataView<'float32x2'>[] = [];
  const measureChunks: GraphDataView<'float32'>[] = [];
  const distanceChunks: GraphDataView<'float32'>[] = [];
  const footChunks: GraphDataView<'float32x2'>[] = [];
  const lineChunks: GraphDataView<'uint32'>[] = [];
  let rowBase = 0;
  for (const [index, length] of lengths.entries()) {
    queryChunks.push(
      input(
        `queries-${index}`,
        'float32x2',
        new Float32Array(Array.from({length}, (_, row) => points[(rowBase + row) % 6]).flat())
      )
    );
    measureChunks.push(output(`measures-${index}`, 'float32', length));
    distanceChunks.push(output(`distances-${index}`, 'float32', length));
    footChunks.push(output(`foot-${index}`, 'float32x2', length));
    lineChunks.push(output(`lines-${index}`, 'uint32', length));
    rowBase += length;
  }
  const overflow = output('overflow', 'uint32', 1);
  const required = output('required', 'uint32', 1);
  graph.add(
    new GPULinearReferencing({
      positions,
      lineOffsets,
      radius,
      candidateCapacity,
      spatialContext: spherical
        ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle', sphereRadius: 1}
        : undefined,
      queryPositions: chunked
        ? createVectorView('queries', 'float32x2', queryChunks)
        : queryChunks[0],
      output: {
        measures: chunked
          ? createVectorView('measures', 'float32', measureChunks)
          : measureChunks[0],
        distances: chunked
          ? createVectorView('distances', 'float32', distanceChunks)
          : distanceChunks[0],
        footPoints: chunked ? createVectorView('foot', 'float32x2', footChunks) : footChunks[0],
        lineIndices: chunked ? createVectorView('lines', 'uint32', lineChunks) : lineChunks[0],
        status: {candidateOverflow: overflow, requiredCandidateCount: required}
      }
    })
  );
  const compiled = graph.compile();
  try {
    return await fixture.submitTwice(compiled, async () => ({
      measures: await fixture.readRows(measureChunks),
      distances: await fixture.readRows(distanceChunks),
      foot: await fixture.readRows(footChunks),
      lines: await fixture.readRows(lineChunks),
      overflow: (await fixture.read(overflow))[0],
      required: (await fixture.read(required))[0]
    }));
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
}

for (const spherical of [false, true])
  it(`matches packed projection and an equatorial-line oracle (spherical=${spherical})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const packed = await runReferencing(device, false, spherical);
    const chunked = await runReferencing(device, true, spherical);
    expect(chunked).toEqual(packed);
    expect(chunked.overflow).toBe(0);
    expect(chunked.required).toBe(spherical ? 0 : 5);
    expect(chunked.lines).toEqual([0, 0, 0, 0, 0, 0xffffffff]);
    const radians = Math.PI / 180;
    for (const [index, longitude] of [2, 6, 5, 0, 10].entries()) {
      expect(chunked.measures[index]).toBeCloseTo(
        longitude * (spherical ? radians : 1),
        spherical ? 5 : 6
      );
      expect(chunked.foot[index * 2]).toBeCloseTo(longitude, 4);
      expect(chunked.foot[index * 2 + 1]).toBeCloseTo(0, 4);
    }
    expect(chunked.distances[0]).toBeCloseTo(spherical ? 3 * radians : 3, 5);
    expect(chunked.distances[3]).toBeCloseTo(
      spherical ? Math.acos(Math.cos(radians) * Math.cos(2 * radians)) : Math.sqrt(5),
      5
    );
  });

it('merges planar candidate counts and overflow once across chunks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const packed = await runReferencing(device, false, false, 1);
  const chunked = await runReferencing(device, true, false, 1);
  expect(chunked).toEqual(packed);
  expect(chunked.overflow).toBe(1);
  expect(chunked.required).toBe(5);
  expect(chunked.lines[0]).toBe(0);
  // Projection and length reconstruction use float32 arithmetic: 2 can round to its
  // immediately preceding float32 value (2 - 2^-23). Allow two ulps at this magnitude.
  expect(Math.abs(chunked.measures[0] - Math.fround(2))).toBeLessThanOrEqual(2 ** -21);
  expect(chunked.lines.slice(1)).toEqual(Array(5).fill(0xffffffff));
});

it('projects four million spherical queries beyond the old packed scratch limit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    format: Format,
    data: Float32Array | Uint32Array,
    length: number
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const rowCount = 4_000_000;
  const points = new Float32Array(rowCount * 2);
  for (let row = 0; row < rowCount; row++) points[row * 2] = 5;
  const queries = input('queries', 'float32x2', points, rowCount);
  const outputBuffer = createOutputBuffer(device, rowCount);
  buffers.push(outputBuffer);
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'float32', rowCount);
  const lengths = [1_000_000, 1_500_000, 1_500_000];
  const split = <Format extends 'float32' | 'float32x2'>(
    name: string,
    view: GraphDataView<Format>
  ) => {
    let rowBase = 0;
    return createVectorView(
      name,
      view.format,
      lengths.map(length => {
        const chunk = graph.createDataView(view.buffer, {
          format: view.format,
          length,
          byteOffset: rowBase * view.rowByteLength
        });
        rowBase += length;
        return chunk;
      })
    );
  };
  const overflow = createOutputBuffer(device, 1);
  buffers.push(overflow);
  graph.add(
    new GPULinearReferencing({
      positions: input('positions', 'float32x2', new Float32Array([0, 0, 10, 0]), 2),
      lineOffsets: input('line-offsets', 'uint32', new Uint32Array([0, 2]), 2),
      radius: input('radius', 'float32', new Float32Array([1]), 1),
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        sphereRadius: 1
      },
      queryPositions: split('queries', queries),
      output: {
        measures: split('measures', output),
        status: {candidateOverflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const row of [0, 999_999, 1_000_000, 2_500_000, rowCount - 1]) {
      const bytes = await outputBuffer.readAsync(row * 4, 4);
      expect(new Float32Array(bytes.buffer, bytes.byteOffset, 1)[0]).toBeCloseTo(
        (5 * Math.PI) / 180,
        6
      );
    }
    expect(await readUint32(overflow, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

for (const spherical of [false, true])
  it(`preserves offsets, padding and scalar counts across 255/256/257-row dispatches (spherical=${spherical})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const lengths = [255, 256, 257];
    const packed = await runReferencing(device, false, spherical, 1024, lengths);
    const chunked = await runReferencing(device, true, spherical, 1024, lengths);
    expect(chunked).toEqual(packed);
    expect(chunked.overflow).toBe(0);
    expect(chunked.required).toBe(spherical ? 0 : 640);
    for (let row = 0; row < 768; row++) {
      expect(chunked.lines[row]).toBe(row % 6 === 5 ? 0xffffffff : 0);
      if (row % 6 !== 5)
        expect(chunked.measures[row]).toBeCloseTo(
          [2, 6, 5, 0, 10][row % 6] * (spherical ? Math.PI / 180 : 1),
          5
        );
    }
  });
