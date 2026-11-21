// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {getColumnSumNodes} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-inference-kernels';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('column sums skip the first column and include a partial block with offset views (SSR-126)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const rows = 4099;
  const matrix = new Float32Array(3 + 3 * rows).fill(99);
  matrix.fill(2, 3 + rows, 3 + 2 * rows);
  matrix.set([3, 5, 7], 3 + rows + 4096);
  matrix.fill(-1, 3 + 2 * rows);
  matrix.set([8, -4, 2], 3 + 2 * rows + 4096);
  const inputBuffer = createInputBuffer(device, matrix);
  const outputBuffer = createOutputBuffer(device, 5);
  outputBuffer.write(Float32Array.of(1234, 1234, 1234, 1234, 1234));
  const input = importGraphBuffer(graph, 'matrix', inputBuffer, 'float32', matrix.length);
  const output = importGraphBuffer(graph, 'totals', outputBuffer, 'float32', 5);
  const nodes = getColumnSumNodes(graph, {
    id: 'literal-column-sums',
    operation: 'GPUGlobalSpatialStatistics',
    matrix: graph.createDataView(input.buffer, {
      format: 'float32',
      length: 3 * rows,
      byteOffset: 12
    }),
    rows,
    blockCount: 2,
    first: 1,
    count: 2,
    totals: graph.createDataView(output.buffer, {
      format: 'float32',
      length: 2,
      byteOffset: 8
    })
  });
  graph.add({id: 'literal-column-reduction', getCommandNodes: () => nodes});
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const result = await readFloat32(outputBuffer, 5);
    // Independent integer sums: 4096*2 + 3+5+7 = 8207;
    // 4096*(-1) + 8-4+2 = -4090. Prefix/suffix words must stay untouched.
    expect(result.every(Number.isFinite)).toBe(true);
    expect(result).toEqual([1234, 1234, 8207, -4090, 1234]);
  } finally {
    compiled.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
  }
});
