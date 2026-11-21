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
      'candidateOverflow',
      'both'
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
          overflow: flagName !== 'candidateOverflow' ? loss : undefined,
          candidateOverflow: flagName !== 'overflow' ? loss : undefined
        };
        const output = {
          accessibility: floats('access', [99]),
          ratios: floats('ratios', [99, 99]),
          reachableFacilities: integers('reachable', [99]),
          demandValidity: integers('validity', [99])
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
        expect(Array.from(memory.readUint32(output.demandValidity))).toEqual([1]);
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

for (const method of ['2sfca', '3sfca'] as const) {
  it(`${method} suppresses dependent results while costsComplete is 0 and recovers (NET-101)`, () => {
    // One demand location (population 1) reaching two facilities of supply 1, all weights 1. By
    // hand: 2SFCA R = [1, 1], A = 2; 3SFCA splits the population over the two facilities, so
    // R = [2, 2] and A = 2 * (1/2) * 2 = 2. A costsComplete word of 0 (the cost-weights
    // status.complete of an unfinished or short-horizon matrix) must suppress A, R and the count.
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
      const costsComplete = integers('costs-complete', [0]);
      const output = {
        accessibility: floats('access', [99]),
        ratios: floats('ratios', [99, 99]),
        reachableFacilities: integers('reachable', [99]),
        demandValidity: integers('validity', [99])
      };
      expect(
        () =>
          new GPUCatchmentAccessibility({
            method,
            supply: floats('bad-supply', [1, 1]),
            demand: floats('bad-demand', [1, 1]),
            facilityWeights: {
              offsets: integers('bad-offsets', [0, 0, 0]),
              neighbors: integers('bad-neighbors', [0]),
              weights: floats('bad-weights', [1])
            },
            costsComplete: integers('two-rows', [1, 1]),
            output: {accessibility: floats('bad-access', [0, 0])}
          })
      ).toThrow(/costsComplete must contain exactly one row/);
      const nodes = new GPUCatchmentAccessibility({
        method,
        supply: floats('supply', [1, 1]),
        demand: floats('demand', [1]),
        demandWeights: {
          offsets: integers('offsets', [0, 2]),
          neighbors: integers('neighbors', [0, 1]),
          weights: floats('weights', [1, 1])
        },
        facilityWeights: {
          offsets: integers('reverse-offsets', [0, 1, 2]),
          neighbors: integers('reverse-neighbors', [0, 0]),
          weights: floats('reverse-weights', [1, 1])
        },
        costsComplete,
        output
      }).getCommandNodes(graph);
      executeCommandNodesOnCPU(nodes, memory);
      expect(memory.readFloat32(output.accessibility)[0]).toBeNaN();
      expect(Array.from(memory.readFloat32(output.ratios)).every(Number.isNaN)).toBe(true);
      expect(Array.from(memory.readUint32(output.reachableFacilities))).toEqual([0xffffffff]);
      expect(Array.from(memory.readUint32(output.demandValidity))).toEqual([1]);
      memory.write(costsComplete, Uint32Array.of(1));
      executeCommandNodesOnCPU(nodes, memory);
      const [accessibility] = memory.readFloat32(output.accessibility);
      expect(Number.isFinite(accessibility)).toBe(true);
      expect(accessibility).toBeCloseTo(2, 6);
      expect(Array.from(memory.readFloat32(output.ratios))).toEqual(
        method === '2sfca' ? [1, 1] : [2, 2]
      );
      expect(Array.from(memory.readUint32(output.reachableFacilities))).toEqual([2]);
    } finally {
      device.destroy();
    }
  });
}
