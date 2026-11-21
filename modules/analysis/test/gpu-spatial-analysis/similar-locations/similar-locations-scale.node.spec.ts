// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUSimilarLocations,
  getGPUSimilarLocationsParameterValues
} from '../../../src/gpu-spatial-analysis/similar-locations';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';

useCPUExecutorTestHooks();
for (const scale of [1, 2 ** -80, 2 ** 127]) {
  it(`keeps actual z-score moment/reference/distance kernels in a finite frame at ${scale}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const values = createTransientView(graph, 'values', 'float32', 3);
      const selection = createTransientView(graph, 'selection', 'uint32', 3);
      const parameters = createTransientView(graph, 'parameters', 'float32', 5);
      const distances = createTransientView(graph, 'distances', 'float32', 3);
      const nodes = new GPUSimilarLocations({
        id: 'scale',
        values,
        columnCount: 1,
        resultCapacity: 2,
        referenceMask: selection,
        parameters,
        output: {
          ranks: createTransientView(graph, 'ranks', 'uint32', 3),
          distances,
          topIds: createTransientView(graph, 'top', 'uint32', 2),
          status: {count: createTransientView(graph, 'count', 'uint32', 1)}
        }
      }).getCommandNodes(graph);
      const memory = createCPUGraphMemory();
      memory.write(
        values,
        Float32Array.from(scale === 2 ** 127 ? [-scale, scale, 0] : [0, 2 * scale, scale])
      );
      memory.write(selection, Uint32Array.of(1, 0, 0));
      memory.write(parameters, getGPUSimilarLocationsParameterValues({resultCount: 2}, 1));
      for (const suffix of ['validate', 'column-statistics', 'reference', 'distance']) {
        let node = nodes.find(candidate => candidate.id === `scale-${suffix}`)!;
        if (suffix === 'column-statistics' || suffix === 'reference') {
          const record = getWGSLKernelNodeRecord(graph, node.id)!;
          node = createWGSLKernelNode(graph, {
            id: `${node.id}-serial`,
            operation: 'ScaleReview',
            invocationCount: 1,
            workgroupSize: 2,
            bindings: record.bindings,
            body: record.body,
            declarations: record
              .declarations!.replace('const LANES: u32 = 256u;', 'const LANES: u32 = 1u;')
              .replaceAll('workgroupBarrier();', '')
          });
        }
        executeCommandNodesOnCPU([node], memory);
      }
      const actual = memory.readFloat32(distances);
      expect(Number.isNaN(actual[0])).toBe(true);
      // Population variance 2*d^2/3: standardized distances are sqrt(6), sqrt(3/2).
      for (const [row, expected] of [
        [1, Math.sqrt(6)],
        [2, Math.sqrt(1.5)]
      ]) {
        expect(Number.isFinite(actual[row])).toBe(true);
        expect(Math.abs(actual[row] - expected)).toBeLessThan(3e-7);
      }
      expect(actual[2]).toBeLessThan(actual[1]);
    } finally {
      device.destroy();
    }
  });
}
