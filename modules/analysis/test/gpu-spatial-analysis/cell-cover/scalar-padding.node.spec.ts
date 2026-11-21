// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCellCover} from '../../../src/gpu-spatial-analysis/cell-cover';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const family of ['quadbin', 'h3', 's2'] as const) {
  it(`clears only the first diagnostic word (${family})`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const scalar = (length = 1) =>
      createTransientView(graph, `scalar-${serial++}`, 'uint32', length);
    const invalidCount = scalar(2);
    const uncertainCount = scalar(2);
    const nodes = new GPUCellCover({
      family,
      resolution: 2,
      candidateCapacity: 64,
      edgeSlabs: false,
      candidateEnumeration: 'bounds',
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      featureOffsets: scalar(2),
      polygonOffsets: scalar(2),
      ringOffsets: scalar(2),
      output: {
        cells: createTransientView(graph, 'cells', 'uint32x2', 8),
        featureIds: scalar(8),
        status: {
          count: scalar(),
          overflow: scalar(),
          candidateOverflow: scalar(),
          invalidCount,
          uncertainCount
        }
      }
    }).getCommandNodes(graph);
    const memory = createCPUGraphMemory();
    memory.write(invalidCount, Uint32Array.of(99, 0x12345678));
    memory.write(uncertainCount, Uint32Array.of(99, 0x12345678));
    try {
      const writers = nodes.filter(
        node =>
          /-clear-(invalidCount|uncertainCount)$/.test(node.id) || node.id.endsWith('-diagnostics')
      );
      expect(writers).toHaveLength(family === 's2' ? 2 : 4);
      executeCommandNodesOnCPU(writers, memory);
      expect(Array.from(memory.readUint32(invalidCount))).toEqual([0, 0x12345678]);
      expect(Array.from(memory.readUint32(uncertainCount))).toEqual([0, 0x12345678]);
    } finally {
      device.destroy();
    }
  });
}
