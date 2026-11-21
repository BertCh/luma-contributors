// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUNetworkAccessibility} from '../../../src/gpu-network/network-accessibility/gpu-network-accessibility';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';
import {GPUNetworkCostWeights} from '../../../src/gpu-network/network-cost-weights/gpu-network-cost-weights';

for (const method of ['2sfca', '3sfca'] as const) {
  for (const publicStatus of [false, true]) {
    it(`${method} propagates supplied route loss into nested catchment and recovers, status=${publicStatus}`, () => {
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
      const costs = floats('costs', [2, 7]);
      const costsComplete = integers('converged', [1]);
      const costLimit = floats('limit', [2]);
      const output = floats('accessibility', [99, 99]);
      const ratios = floats('ratios', [99]);
      const complete = publicStatus ? integers('complete', [99]) : undefined;
      // The weights producer uses cooperative workgroups. Seed its hand-worked all-one CSR;
      // execute the real parent certificate and child scoring/suppression kernels below.
      const weights = vi
        .spyOn(GPUNetworkCostWeights.prototype, 'getCommandNodes')
        .mockImplementation(function (this: GPUNetworkCostWeights) {
          const outputWeights = this.outputWeights;
          memory.write(
            outputWeights.offsets,
            Uint32Array.from(this.props.orientation === 'rows' ? [0, 1] : [0, 1, 1])
          );
          memory.write(
            outputWeights.neighbors,
            Uint32Array.from(this.props.orientation === 'rows' ? [0, 1] : [0, 0])
          );
          memory.write(outputWeights.weights!, Float32Array.of(1, 1));
          return [];
        });
      try {
        const nodes = new GPUNetworkAccessibility({
          costs,
          costsComplete,
          costLimit,
          opportunityWeights: floats('supply', [12]),
          parameters: floats('parameters', [5, 0, 0, 1]),
          status: complete ? {complete} : undefined,
          catchment: {method, demand: floats('demand', [3, 4]), output, ratios}
        }).getCommandNodes(graph);
        // Supply 12, demand [3,4], costs [2,7], cutoff 5: R=12/3=4, A=[4,0].
        // A public certificate carries loss while leaving those incumbents usable.
        for (const [converged, horizon] of [
          [1, 2],
          [0, 5],
          [1, 5],
          [1, NaN]
        ]) {
          memory.write(costsComplete, Uint32Array.of(converged));
          memory.write(costLimit, Float32Array.of(horizon));
          executeCommandNodesOnCPU(nodes, memory);
          const certified = converged !== 0 && horizon >= 5;
          if (publicStatus || certified) {
            expect(Array.from(memory.readFloat32(output))).toEqual([4, 0]);
            expect(Array.from(memory.readFloat32(ratios))).toEqual([4]);
          } else {
            expect(Array.from(memory.readFloat32(output))).toEqual([NaN, NaN]);
            expect(Array.from(memory.readFloat32(ratios))).toEqual([NaN]);
          }
          if (complete)
            expect(Array.from(memory.readUint32(complete))).toEqual([certified ? 1 : 0]);
        }
      } finally {
        weights.mockRestore();
        device.destroy();
      }
    });
  }
}

for (const evidence of ['costsComplete', 'costLimit'] as const) {
  it(`suppresses score-only outputs without a public certificate for supplied ${evidence}`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const costs = createTransientView(graph, 'score-costs', 'float32', 2);
    const supply = createTransientView(graph, 'score-supply', 'float32', 1);
    const parameters = createTransientView(graph, 'score-parameters', 'float32', 4);
    const cumulative = createTransientView(graph, 'cumulative', 'float32', 2);
    const gravity = createTransientView(graph, 'gravity', 'float32', 2);
    const costsComplete =
      evidence === 'costsComplete'
        ? createTransientView(graph, 'converged', 'uint32', 1)
        : undefined;
    const costLimit =
      evidence === 'costLimit' ? createTransientView(graph, 'limit', 'float32', 1) : undefined;
    memory.write(costs, Float32Array.of(2, 7));
    memory.write(supply, Float32Array.of(12));
    memory.write(parameters, Float32Array.of(5, 0, 0, 1));
    try {
      const nodes = new GPUNetworkAccessibility({
        costs,
        opportunityWeights: supply,
        parameters,
        costsComplete,
        costLimit,
        cumulative,
        gravity
      }).getCommandNodes(graph);
      for (const complete of [false, true]) {
        if (costsComplete) memory.write(costsComplete, Uint32Array.of(complete ? 1 : 0));
        if (costLimit) memory.write(costLimit, Float32Array.of(complete ? 5 : 2));
        executeCommandNodesOnCPU(nodes, memory);
        expect(Array.from(memory.readFloat32(cumulative))).toEqual(complete ? [12, 0] : [NaN, NaN]);
        expect(Array.from(memory.readFloat32(gravity))).toEqual(complete ? [12, 0] : [NaN, NaN]);
      }
    } finally {
      device.destroy();
    }
  });
}
