// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUHuffTradeAreas} from '../../../src/gpu-spatial-analysis/catchment-accessibility/gpu-huff-trade-areas';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {HUFF_REVIEW_CASES} from './huff-review-cases';

it.each([
  'demand',
  'facility'
] as const)('suppresses Huff results for incomplete %s weights and recovers', role => {
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
    const overflow = integers('overflow', [1]);
    const forwardOffsets = integers('offsets', role === 'demand' ? [0, 1] : [0, 2]);
    const reverseOffsets = integers('reverse-offsets', role === 'facility' ? [0, 1, 1] : [0, 1, 2]);
    const probabilities = floats('probabilities', [99, 99]);
    const tradeArea = integers('trade-area', [99]);
    const expectedDemand = floats('expected-demand', [99, 99]);
    const nodes = new GPUHuffTradeAreas({
      attractiveness: floats('attractiveness', [1, 1]),
      demand: floats('demand', [10]),
      demandWeights: {
        offsets: forwardOffsets,
        neighbors: integers('neighbors', [0, 1]),
        weights: floats('weights', [1, 1]),
        status:
          role === 'demand'
            ? {
                overflow,
                candidateOverflow: overflow,
                requiredCount: integers('required-count', [2])
              }
            : undefined
      },
      facilityWeights: {
        offsets: reverseOffsets,
        neighbors: integers('reverse-neighbors', [0, 0]),
        weights: floats('reverse-weights', [1, 1]),
        status:
          role === 'facility'
            ? {
                overflow,
                candidateOverflow: overflow,
                requiredCount: integers('required-count', [2])
              }
            : undefined
      },
      output: {probabilities, tradeArea, expectedDemand}
    }).getCommandNodes(graph);
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readFloat32(expectedDemand)).every(Number.isNaN)).toBe(true);
    if (role === 'demand') {
      expect(Array.from(memory.readFloat32(probabilities)).every(Number.isNaN)).toBe(true);
      expect(Array.from(memory.readUint32(tradeArea))).toEqual([0xffffffff]);
    } else {
      expect(Array.from(memory.readFloat32(probabilities))).toEqual([0.5, 0.5]);
    }
    memory.write(overflow, Uint32Array.of(0));
    memory.write(forwardOffsets, Uint32Array.of(0, 2));
    memory.write(reverseOffsets, Uint32Array.of(0, 1, 2));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readFloat32(probabilities))).toEqual([0.5, 0.5]);
    expect(Array.from(memory.readFloat32(expectedDemand))).toEqual([5, 5]);
    expect(Array.from(memory.readUint32(tradeArea))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

it.each(HUFF_REVIEW_CASES)('Huff review: $name', testCase => {
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
    const probabilities = floats('probabilities', [99, 99]);
    const tradeArea = integers('trade-area', [99]);
    const expectedDemand = floats('expected-demand', [99, 99]);
    executeCommandNodesOnCPU(
      new GPUHuffTradeAreas({
        attractiveness: floats('attractiveness', testCase.attractiveness),
        parameters:
          testCase.exponent === undefined ? undefined : floats('parameters', [testCase.exponent]),
        demand: floats('demand', [testCase.demand]),
        demandWeights: {
          offsets: integers('offsets', [0, 2]),
          neighbors: integers('neighbors', [0, 1]),
          weights: floats('weights', testCase.weights)
        },
        facilityWeights: {
          offsets: integers('reverse-offsets', [0, 1, 2]),
          neighbors: integers('reverse-neighbors', [0, 0]),
          weights: floats('reverse-weights', testCase.weights)
        },
        output: {probabilities, tradeArea, expectedDemand}
      }).getCommandNodes(graph),
      memory
    );
    expect(Array.from(memory.readUint32(tradeArea))).toEqual([testCase.winner]);
    const shares = memory.readFloat32(probabilities);
    const captured = memory.readFloat32(expectedDemand);
    for (let facility = 0; facility < 2; facility++) {
      expect(shares[facility]).toBeCloseTo(testCase.probabilities[facility], 6);
      if (testCase.captured[facility] === 0) expect(captured[facility]).toBe(0);
      else
        expect(Math.abs(captured[facility] / testCase.captured[facility] - 1)).toBeLessThan(1e-4);
    }
  } finally {
    device.destroy();
  }
});

it('zeros removed Huff edges on resubmission', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  try {
    const offsets = createTransientView(graph, 'offsets', 'uint32', 2);
    const neighbors = createTransientView(graph, 'neighbors', 'uint32', 2);
    const attractiveness = createTransientView(graph, 'attractiveness', 'float32', 2);
    const weights = createTransientView(graph, 'weights', 'float32', 2);
    const probabilities = createTransientView(graph, 'probabilities', 'float32', 2);
    const tradeArea = createTransientView(graph, 'trade-area', 'uint32', 1);
    memory.write(offsets, Uint32Array.of(0, 2));
    memory.write(neighbors, Uint32Array.of(0, 1));
    memory.write(weights, Float32Array.of(1, 1));
    memory.write(attractiveness, Float32Array.of(1, 1));
    memory.write(probabilities, Float32Array.of(99, 99));
    const nodes = new GPUHuffTradeAreas({
      attractiveness,
      demandWeights: {offsets, neighbors, weights},
      output: {tradeArea, probabilities}
    }).getCommandNodes(graph);
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readFloat32(probabilities))).toEqual([0.5, 0.5]);
    memory.write(offsets, Uint32Array.of(0, 1));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readFloat32(probabilities))).toEqual([1, 0]);
    memory.write(offsets, Uint32Array.of(0, 0));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readFloat32(probabilities))).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});

it.each([
  {scale: 1, exponent: 10, weights: [1, 1], captured: [1, 1024]},
  {scale: 10000, exponent: 10, weights: [1, 1], captured: [1, 1024]},
  {scale: 1e-20, exponent: 10, weights: [1, 1], captured: [1, 1024]},
  {scale: 10000, exponent: 3e38, weights: [1, 1], captured: [0, 1025]},
  {scale: 10000, exponent: 10, weights: [2, 1], captured: [2050 / 1026, (1025 * 1024) / 1026]},
  {scale: 10000, exponent: 10, weights: [0, 1], captured: [0, 1025]}
])('normalizes Huff powers before overflow: $scale, $exponent, $weights', ({
  scale,
  exponent,
  weights,
  captured
}) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  function input(name: string, values: Float32Array): GraphDataView<'float32'>;
  function input(name: string, values: Uint32Array): GraphDataView<'uint32'>;
  function input(name: string, values: Float32Array | Uint32Array): GraphDataView {
    const view = createTransientView(
      graph,
      name,
      values instanceof Float32Array ? 'float32' : 'uint32',
      values.length
    );
    memory.write(view, values);
    return view;
  }
  try {
    const probabilities = createTransientView(graph, 'probabilities', 'float32', 2);
    const tradeArea = createTransientView(graph, 'trade-area', 'uint32', 1);
    const tradeAreaProbability = createTransientView(graph, 'trade-probability', 'float32', 1);
    const expectedDemand = createTransientView(graph, 'expected-demand', 'float32', 2);
    executeCommandNodesOnCPU(
      new GPUHuffTradeAreas({
        attractiveness: input('attractiveness', Float32Array.of(scale, 2 * scale)),
        parameters: input('parameters', Float32Array.of(exponent)),
        demand: input('demand', Float32Array.of(1025)),
        demandWeights: {
          offsets: input('offsets', Uint32Array.of(0, 2)),
          neighbors: input('neighbors', Uint32Array.of(0, 1)),
          weights: input('weights', Float32Array.from(weights))
        },
        facilityWeights: {
          offsets: input('reverse-offsets', Uint32Array.of(0, 1, 2)),
          neighbors: input('reverse-neighbors', Uint32Array.of(0, 0)),
          weights: input('reverse-weights', Float32Array.from(weights))
        },
        output: {probabilities, tradeArea, tradeAreaProbability, expectedDemand}
      }).getCommandNodes(graph),
      memory
    );
    expect(Array.from(memory.readUint32(tradeArea))).toEqual([1]);
    const shares = memory.readFloat32(probabilities);
    expect(Number.isFinite(shares[0]) && Number.isFinite(shares[1])).toBe(true);
    expect(shares[0]).toBeCloseTo(captured[0] / 1025, 6);
    expect(shares[1]).toBeCloseTo(captured[1] / 1025, 6);
    expect(memory.readFloat32(tradeAreaProbability)[0]).toBeCloseTo(captured[1] / 1025, 6);
    expect(memory.readFloat32(expectedDemand)[0]).toBeCloseTo(captured[0], 3);
    expect(memory.readFloat32(expectedDemand)[1]).toBeCloseTo(captured[1], 2);
  } finally {
    device.destroy();
  }
});
