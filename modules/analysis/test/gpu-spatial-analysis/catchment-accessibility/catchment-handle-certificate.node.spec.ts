// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCatchmentAccessibility} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-catchment-accessibility';
import {GPUHuffTradeAreas} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {GPUNetworkCostWeights} from '../../../src/gpu-network/network-cost-weights/gpu-network-cost-weights';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const role of ['rows', 'columns'] as const)
  for (const companion of [false, true]) {
    it(`honors ${role} handle completeness independently of candidate loss, companion ${companion}`, () => {
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
        const costs = floats('costs', [0, 2]),
          parameters = floats('parameters', [2, 0, 0, 1]);
        const converters = (['rows', 'columns'] as const).map(
          orientation =>
            new GPUNetworkCostWeights({
              id: `converter-${orientation}`,
              costs,
              rowCount: 1,
              orientation,
              parameters,
              costsComplete: integers(`search-${orientation}`, [1]),
              output: {
                offsets: integers(
                  `offsets-${orientation}`,
                  orientation === 'rows' ? [0, 2] : [0, 1, 2]
                ),
                neighbors: integers(
                  `neighbors-${orientation}`,
                  orientation === 'rows' ? [0, 1] : [0, 0]
                ),
                weights: floats(`weights-${orientation}`, [1, 1]),
                status: {
                  overflow: integers(`overflow-${orientation}`, [0]),
                  complete: integers(`complete-${orientation}`, [orientation === role ? 0 : 1])
                }
              }
            })
        );
        const handles = converters.map(converter => ({
          ...converter.outputWeights,
          status: {
            ...converter.outputWeights.status,
            ...(companion
              ? {candidateOverflow: integers(`candidate-${converter.orientation}`, [0])}
              : {})
          }
        }));
        const demand = floats('demand', [1, 1]);
        const accessibility = floats('accessibility', [99, 99]),
          expectedDemand = floats('expected-demand', [99]);
        const tradeArea = integers('trade-area', [99, 99]);
        const catchment = new GPUCatchmentAccessibility({
          method: '2sfca',
          supply: floats('supply', [2]),
          demand,
          facilityWeights: handles[0],
          demandWeights: handles[1],
          output: {accessibility}
        });
        const huff = new GPUHuffTradeAreas({
          attractiveness: floats('attractiveness', [1]),
          demand,
          facilityWeights: handles[0],
          demandWeights: handles[1],
          output: {tradeArea, expectedDemand}
        });
        const catchmentNodes = catchment.getCommandNodes(graph),
          huffNodes = huff.getCommandNodes(graph);
        executeCommandNodesOnCPU(catchmentNodes, memory);
        executeCommandNodesOnCPU(huffNodes, memory);
        expect(Array.from(memory.readFloat32(accessibility))).toEqual([NaN, NaN]);
        expect(Array.from(memory.readFloat32(expectedDemand))).toEqual([NaN]);
        if (role === 'columns')
          expect(Array.from(memory.readUint32(tradeArea))).toEqual([0xffffffff, 0xffffffff]);
        // Same nodes, only the upstream certificate changes; supply 2 / two unit demands = 1.
        for (const converter of converters)
          memory.write(converter.outputWeights.status!.complete!, Uint32Array.of(1));
        executeCommandNodesOnCPU(catchmentNodes, memory);
        executeCommandNodesOnCPU(huffNodes, memory);
        expect(Array.from(memory.readFloat32(accessibility))).toEqual([1, 1]);
        const capturedDemand = memory.readFloat32(expectedDemand)[0];
        expect(Number.isFinite(capturedDemand)).toBe(true);
        expect(capturedDemand).toBeCloseTo(2, 6);
        expect(Array.from(memory.readUint32(tradeArea))).toEqual([0, 0]);
      } finally {
        device.destroy();
      }
    });
  }
