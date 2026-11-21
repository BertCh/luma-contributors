// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {GPUCoverageSimplification} from '../../../src/gpu-spatial-analysis/polygon-coverage-simplification';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOutputOracleFixture} from '../outline-geometry/output-oracle-fixture';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
useCPUExecutorTestHooks();

it('B1 restores originals when only the final detection round first exceeds pair capacity', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const statistics of [false, true]) {
      const fixture = createOutputOracleFixture(device);
      const {column, output, graph} = fixture;
      try {
        // Two polygons partition [0,6]x[0,4] along a zigzag, with a triangular hole left of it.
        // All 17 stored vertices must survive whenever detection cannot certify a shortcut.
        const coordinates = Float32Array.of(
          0,
          0,
          2,
          0,
          2,
          1,
          4,
          2,
          2,
          3,
          2,
          4,
          0,
          4,
          2.75,
          1.9,
          2.75,
          2.1,
          3.1,
          2,
          2,
          0,
          6,
          0,
          6,
          4,
          2,
          4,
          2,
          3,
          4,
          2,
          2,
          1
        );
        const offsets = output('offsets', 'uint32', 4);
        const required = output('required', 'uint32', 1);
        const stats = statistics ? output('stats', 'uint32', 4) : undefined;
        const contributor = new GPUCoverageSimplification({
          positions: column('input', 'float32x2', coordinates),
          ringOffsets: column('rings', 'uint32', Uint32Array.of(0, 7, 10, 17)),
          polygonOffsets: column('polygons', 'uint32', Uint32Array.of(0, 2, 3)),
          featureOffsets: column('features', 'uint32', Uint32Array.of(0, 1, 2)),
          parameters: column('parameters', 'float32', Float32Array.of(5, 0, 0, 0)),
          simplifyBoundary: true,
          topologyIterationCount: 1,
          topologyPairCapacity: 8,
          output: {
            positions: output('result', 'float32x2', 17),
            ringOffsets: offsets,
            topologyStatistics: stats,
            status: {overflow: output('overflow', 'uint32', 1), requiredCount: required}
          }
        });
        const memory = createCPUGraphMemory();
        for (const entry of fixture.columns) memory.write(entry.view, entry.values);
        const nodes = contributor.getCommandNodes(graph);
        let workStats: GraphDataView | undefined;
        for (const node of nodes)
          for (const resource of node.resources ?? []) {
            if (
              'buffer' in resource &&
              'buffer' in resource.buffer &&
              resource.buffer.buffer.id === 'coverage-simplification-topology-work'
            )
              workStats = resource.buffer;
          }
        expect(workStats).toBeDefined();
        const finalRound = nodes.findIndex(node =>
          node.id.startsWith('coverage-simplification-round-1-')
        );
        expect(finalRound).toBeGreaterThan(0);
        executeCommandNodesOnCPU(nodes.slice(0, finalRound), memory);
        expect(memory.readUint32(workStats!)[3]).toBe(0);
        executeCommandNodesOnCPU(nodes.slice(finalRound), memory);
        expect(memory.readUint32(workStats!)[3]).toBe(1);
        expect(Array.from(memory.readUint32(required)), `statistics=${statistics}`).toEqual([17]);
        expect(Array.from(memory.readUint32(offsets))).toEqual([0, 7, 10, 17]);
        if (stats) expect(memory.readUint32(stats)[3]).toBe(1);
      } finally {
        fixture.destroy();
      }
    }
  } finally {
    device.destroy();
  }
});
