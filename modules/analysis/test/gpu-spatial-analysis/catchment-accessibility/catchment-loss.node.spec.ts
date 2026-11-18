// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCatchmentAccessibility} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const method of ['2sfca', '3sfca'] as const) {
  for (const role of ['demand', 'facility'] as const) {
    it.each([
      'overflow',
      'candidateOverflow'
    ] as const)(`${method} suppresses ${role} %s and recovers`, flagName => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      const floats = (name: string, values: number[]) => {
        const view = createTransientView(graph, name, 'float32', values.length);
        memory.write(view, Float32Array.from(values));
        return view;
      };
      const integers = (name: string, values: number[]) => {
        const view = createTransientView(graph, name, 'uint32', values.length);
        memory.write(view, Uint32Array.from(values));
        return view;
      };
      try {
        const loss = integers('loss', [1]);
        const forwardOffsets = integers('offsets', role === 'demand' ? [0, 1] : [0, 2]);
        const reverseOffsets = integers(
          'reverse-offsets',
          role === 'facility' ? [0, 1, 1] : [0, 1, 2]
        );
        const status = {
          overflow: flagName === 'overflow' ? loss : undefined,
          candidateOverflow: flagName === 'candidateOverflow' ? loss : undefined
        };
        const output = {
          accessibility: floats('access', [99]),
          ratios: floats('ratios', [99, 99]),
          reachableFacilities: integers('reachable', [99]),
          validity: integers('validity', [99])
        };
        const nodes = new GPUCatchmentAccessibility({
          method,
          supply: floats('supply', [1, 1]),
          demand: floats('demand', [1]),
          demandWeights: {
            offsets: forwardOffsets,
            neighbors: integers('neighbors', [0, 1]),
            weights: floats('weights', [1, 1]),
            status: role === 'demand' ? status : undefined
          },
          facilityWeights: {
            offsets: reverseOffsets,
            neighbors: integers('reverse-neighbors', [0, 0]),
            weights: floats('reverse-weights', [1, 1]),
            status: role === 'facility' ? status : undefined
          },
          output
        }).getCommandNodes(graph);
        executeCommandNodesOnCPU(nodes, memory);
        expect(memory.readFloat32(output.accessibility)[0]).toBeNaN();
        const ratiosDependOnLoss = role === 'facility' || method === '3sfca';
        if (ratiosDependOnLoss)
          expect(Array.from(memory.readFloat32(output.ratios)).every(Number.isNaN)).toBe(true);
        else expect(Array.from(memory.readFloat32(output.ratios))).toEqual([1, 1]);
        expect(Array.from(memory.readUint32(output.reachableFacilities))).toEqual([
          role === 'demand' ? 0xffffffff : 2
        ]);
        expect(Array.from(memory.readUint32(output.validity))).toEqual([1]);
        memory.write(loss, Uint32Array.of(0));
        memory.write(forwardOffsets, Uint32Array.of(0, 2));
        memory.write(reverseOffsets, Uint32Array.of(0, 1, 2));
        executeCommandNodesOnCPU(nodes, memory);
        expect(memory.readFloat32(output.accessibility)[0]).toBeCloseTo(2, 6);
        expect(Array.from(memory.readFloat32(output.ratios))).toEqual(
          method === '2sfca' ? [1, 1] : [2, 2]
        );
        expect(Array.from(memory.readUint32(output.reachableFacilities))).toEqual([2]);
      } finally {
        device.destroy();
      }
    });
  }
}
