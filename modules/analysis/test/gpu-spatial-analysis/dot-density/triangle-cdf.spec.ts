// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getDotTriangleCumulativeAreaNodes} from '../../../src/gpu-spatial-analysis/dot-density/dot-density-triangle-cdf';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

async function computeCumulativeAreas(
  triangleAreas: Float32Array,
  offsets: Uint32Array,
  valid: Uint32Array,
  totals: Float32Array,
  viewWordOffset = 0
): Promise<number[]> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'float32' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const padded =
      values instanceof Float32Array
        ? new Float32Array(values.length + viewWordOffset)
        : new Uint32Array(values.length + viewWordOffset);
    padded.set(values, viewWordOffset);
    const buffer = createInputBuffer(device, padded);
    buffers.push(buffer);
    const backing = importGraphBuffer(graph, name, buffer, format, padded.length);
    return graph.createDataView(backing.buffer, {
      format,
      byteOffset: viewWordOffset * 4,
      length: values.length
    });
  };
  const outputLength = viewWordOffset + triangleAreas.length + totals.length;
  const output = createOutputBuffer(device, outputLength);
  buffers.push(output);
  const sentinel = 123456789;
  output.write(new Uint32Array(outputLength).fill(sentinel));
  const backing = importGraphBuffer(graph, 'cdf', output, 'uint32', outputLength);
  graph.add(
    getDotTriangleCumulativeAreaNodes(graph, {
      id: 'cdf',
      operation: 'GPUDotDensity',
      triangleAreas: input('triangles', triangleAreas, 'float32'),
      featureTriangleOffsets: input('offsets', offsets, 'uint32'),
      featureValid: input('validity', valid, 'uint32'),
      featureAreas: input('totals', totals, 'float32'),
      cumulativeAreas: graph.createDataView(backing.buffer, {
        format: 'uint32',
        byteOffset: viewWordOffset * 4,
        length: triangleAreas.length + totals.length
      })
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const fullResult = await readUint32(output, outputLength);
    expect(fullResult.slice(0, viewWordOffset)).toEqual(Array(viewWordOffset).fill(sentinel));
    const result = fullResult.slice(viewWordOffset);
    expect(Array.from(result.slice(triangleAreas.length))).toEqual(
      Array(totals.length).fill(sentinel)
    );
    return result.slice(0, triangleAreas.length);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

it('matches closed-form prefixes across empty features, repeated offsets, invalid areas and padding', async () => {
  // 0 empty; 1 has 2 triangles; 2/3 empty; 4 spans scan blocks; 5 invalid; 6/7 empty;
  // 8 contains non-finite areas; 9 has non-finite total; then terminal padding.
  const offsets = new Uint32Array([0, 0, 2, 2, 2, 1026, 1028, 1028, 1028, 1031, 1032]);
  const areas = new Float32Array(1034).fill(1);
  areas.set([1, 3]);
  areas.set([2, NaN, Infinity, 1], 1028);
  const valid = new Uint32Array(10).fill(1);
  valid[5] = 0;
  const totals = new Float32Array([0, 4, 0, 0, 1024, 2, 0, 0, 2, Infinity]);
  const actual = await computeCumulativeAreas(areas, offsets, valid, totals);
  const expected = new Uint32Array(areas.length);
  expected.set([1_000_000_000, 4_000_000_000]);
  // Every equal triangle contributes the same quantized integer, independent of prefix order.
  const contribution = Math.trunc(Math.fround(Math.fround(1 / 1024) * 4_000_000_000));
  for (let rank = 1; rank <= 1024; rank++) expected[rank + 1] = rank * contribution;
  expected.set([4_000_000_000, 4_000_000_000, 4_000_000_000], 1028);
  expect(Array.from(actual)).toEqual(Array.from(expected));
});

it('saturates monotonically after repeated carries and resets at the next nonempty feature', async () => {
  // Deliberately underestimated totals stress the CDF even when the upstream area reduction errs.
  // Every unit triangle quantizes to 4e9, so the second and all later prefixes saturate.
  const areas = new Float32Array(1030).fill(1);
  const offsets = new Uint32Array([0, 0, 1025, 1025, 1030]);
  const actual = await computeCumulativeAreas(
    areas,
    offsets,
    new Uint32Array([1, 1, 1, 1]),
    new Float32Array([0, 1, 0, 1])
  );
  const expected = new Uint32Array(1030).fill(0xffffffff);
  expected[0] = 4_000_000_000;
  expected[1025] = 4_000_000_000;
  expect(Array.from(actual)).toEqual(Array.from(expected));
});

it('scans a million triangles in one feature against an independent integer oracle', async () => {
  // The old route put this entire prefix on one invocation. Fixed size: about 28 MiB of scratch.
  const count = 1_048_576;
  const actual = await computeCumulativeAreas(
    new Float32Array(count).fill(1),
    new Uint32Array([0, count]),
    new Uint32Array([1]),
    new Float32Array([count])
  );
  const contribution = Math.trunc(Math.fround(4_000_000_000 / count));
  for (const row of [0, 255, 256, 65535, 65536, count - 1]) {
    expect(actual[row]).toBe((row + 1) * contribution);
  }
}, 120000);

it('preserves CDF boundaries at scan block and hierarchy edges with offsets, empties and carries', async () => {
  const offsets = new Uint32Array([0, 0, 255, 255, 256, 257, 65535, 65535, 65536, 65537, 65537]);
  const featureCount = offsets.length - 1;
  const areas = new Float32Array(65539).fill(1);
  const totals = new Float32Array(featureCount);
  const expected = new Uint32Array(areas.length);
  for (let feature = 0; feature < featureCount; feature++) {
    const count = offsets[feature + 1] - offsets[feature];
    const denominator = feature % 2 === 0 ? 1 : 2 ** Math.ceil(Math.log2(Math.max(count, 1)));
    totals[feature] = denominator;
    const contribution = Math.trunc(Math.fround(4_000_000_000 / denominator));
    for (let rank = 0; rank < count; rank++)
      expected[offsets[feature] + rank] = Math.min(0xffffffff, (rank + 1) * contribution);
  }
  const result = await computeCumulativeAreas(
    areas,
    offsets,
    new Uint32Array(featureCount).fill(1),
    totals,
    3
  );
  expect(result).toEqual(Array.from(expected));
});
