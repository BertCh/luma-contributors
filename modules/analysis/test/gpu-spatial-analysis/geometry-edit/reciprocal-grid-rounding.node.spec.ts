// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {SNAP_SOURCE} from '../../../src/gpu-spatial-analysis/geometry-edit/geometry-edit-kernels';
import {getCoverageSnapKeyWGSL} from '../../../src/gpu-spatial-analysis/polygon-coverage-topology/coverage-topology-kernels';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_FLOAT_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const reciprocal of [false, true]) {
  for (const [spacing, coordinate, expectedCell, expectedPosition] of [
    [41, 20.5, 1, 41],
    [7, 58720228, 8388604, 58720228],
    [1, 0.4999999701976776, 0, 0],
    [1, -0.5, 0, 0]
  ]) {
    it(`B2 preserves the exact cell at spacing ${spacing}, coordinate ${coordinate}, reciprocal=${reciprocal}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const positions = createTransientView(graph, 'positions', 'float32x2', 2);
      const offsets = createTransientView(graph, 'offsets', 'uint32', 2);
      const snapped = createTransientView(graph, 'snapped', 'float32', 2);
      const keys = createTransientView(graph, 'keys', 'uint32', 2);
      const rewrite = (source: string) =>
        reciprocal
          ? source
              .replaceAll('/ gridSize', '* (1.0 / gridSize)')
              .replaceAll('/ SNAP_TOLERANCE', '* (1.0 / SNAP_TOLERANCE)')
              .replaceAll('/ spacing', '* (1.0 / spacing)')
          : source;
      const common = [
        {name: 'positions', view: positions, type: 'f32' as const, access: 'read' as const}
      ];
      const nodes = [
        createWGSLKernelNode(graph, {
          id: 'cleanup',
          operation: 'probe',
          variant: 'cleanup',
          invocationCount: 2,
          bindings: [...common, {name: 'result', view: snapped, type: 'f32', access: 'read_write'}],
          declarations: rewrite(SNAP_SOURCE),
          body: `result[resultOffset + index] = readSnappedPoint(index, ${spacing}.0).x;`
        }),
        createWGSLKernelNode(graph, {
          id: 'coverage',
          operation: 'probe',
          variant: 'coverage',
          invocationCount: 2,
          bindings: [
            ...common,
            {name: 'ringOffsets', view: offsets, type: 'u32', access: 'read'},
            {name: 'result', view: keys, type: 'u32', access: 'read_write'}
          ],
          declarations: rewrite(`${IS_FINITE_FLOAT_WGSL}\n${getCoverageSnapKeyWGSL(spacing, 1)}`),
          body: 'result[resultOffset + index] = quantise(positions[positionsOffset + 2u * index], getSnapOrigin().x);'
        })
      ];
      const memory = createCPUGraphMemory();
      memory.write(positions, Float32Array.of(0, 0, coordinate, 0));
      memory.write(offsets, Uint32Array.of(0, 2));
      executeCommandNodesOnCPU(nodes, memory);
      // Exact rational answers: 20.5/41=1/2, 58720228/7=8388604, no host quotient oracle.
      expect(Number.isFinite(memory.readFloat32(snapped)[1])).toBe(true);
      expect(memory.readFloat32(snapped)[1] + 0).toBe(expectedPosition);
      expect(memory.readUint32(keys)[1]).toBe((expectedCell ^ 0x80000000) >>> 0);
      device.destroy();
    });
  }
}
