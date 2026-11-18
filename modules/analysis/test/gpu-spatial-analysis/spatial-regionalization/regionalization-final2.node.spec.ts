// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {computeWardRegionsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-ward-regions-operation';
import {GPUWardRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createWardLossPath} from './regionalization-final1-fixtures';

for (const mergeSchedule of ['sequential', 'mutual-nearest'] as const) {
  for (const invalid of [false, true]) {
    it(`spatial-3:G10 zero-round Ward status agrees across CPU and emitted WGSL: ${mergeSchedule}, invalid=${invalid}`, () => {
      const input = createWardLossPath(false);
      if (invalid) input.values[1] = NaN;
      const graph = new GPUCommandGraph(createNullWebGPUDevice());
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `ward-zero-${serial++}`, format, length);
      const weights = {
        offsets: view('uint32', 4),
        neighbors: view('uint32', 4),
        weights: view('float32', 4)
      };
      const values = view('float32', 3);
      const parameters = view('uint32', 1);
      const output = {
        labels: view('uint32', 3),
        objective: view('float32', 1),
        status: view('uint32', 4)
      };
      const options = {
        standardize: false,
        targetRegionCount: 2,
        maximumIterations: 0,
        mergeSchedule
      };
      const contributor = new GPUWardRegions({...options, weights, values, parameters, output});
      // No merge can run at this budget. Execute the real initialization/finalization kernels;
      // adjacency preprocessing and label publication cannot change these four status words.
      const nodes = contributor
        .getCommandNodes(graph)
        .filter(node => /-(initialize-status|initialize-regions|finalize)$/.test(node.id));
      expect(nodes).toHaveLength(3);
      const memory = createCPUGraphMemory();
      memory.write(weights.offsets, input.offsets);
      memory.write(weights.neighbors, input.neighbors);
      memory.write(values, input.values);
      executeCommandNodesOnCPU(nodes, memory);
      const expected = invalid ? [0, 0, 0, 1] : [0, 0, 1, 0];
      expect(Array.from(computeWardRegionsOnCPU(input, options).status)).toEqual(expected);
      expect(Array.from(memory.readUint32(output.status))).toEqual(expected);
    });
  }
}
