// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUNeighborhoodSummary} from '../../../src/gpu-spatial-analysis/neighborhood-summary/gpu-neighborhood-summary';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';

useCPUExecutorTestHooks();

for (const maximumNeighbors of [1, 8]) {
  it(`B6 scales both central squares and their denominator at capacity ${maximumNeighbors}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const offsets = createTransientView(graph, 'offsets', 'uint32', 4);
      const neighbors = createTransientView(graph, 'neighbors', 'uint32', 2);
      const weights = createTransientView(graph, 'weights', 'float32', 2);
      const values = createTransientView(graph, 'values', 'float32', 3);
      const table = createTransientView(graph, 'table', 'float32', 9);
      const nodes = new GPUNeighborhoodSummary({
        weights: {offsets, neighbors, weights},
        values,
        maximumNeighbors,
        statistics: ['sum', 'mean', 'standardDeviation'],
        output: {table}
      }).getCommandNodes(graph);
      const memory = createCPUGraphMemory();
      memory.write(offsets, Uint32Array.of(0, 2, 2, 2));
      memory.write(neighbors, Uint32Array.of(1, 2));
      memory.write(weights, Float32Array.of(2 ** 125, 2 ** 126));
      memory.write(values, Float32Array.of(0, -4, 2));
      executeCommandNodesOnCPU(nodes, memory);
      const actual = memory.readFloat32(table);
      // Exact ratio 1:2: weighted sum=0, mean=0, variance=(16+2*4)/3=8.
      expect(actual[0]).toBe(0);
      expect(actual[1]).toBe(0);
      expect(Number.isFinite(actual[2])).toBe(true);
      expect(Math.abs(actual[2] - Math.sqrt(8))).toBeLessThan(2e-7);
      for (const row of [1, 2]) {
        expect(actual[3 * row]).toBe(0); // Empty weighted sum is zero.
        expect(Number.isNaN(actual[3 * row + 1])).toBe(true);
        expect(Number.isNaN(actual[3 * row + 2])).toBe(true);
      }
    } finally {
      device.destroy();
    }
  });
}
