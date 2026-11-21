// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {addSpatialRateRecipe} from '../../../src/gpu-spatial-analysis/recipes/rate-mapping-recipes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it.each([
  undefined,
  'own-rate'
] as const)('W5-17 island policy %s excludes negative-event donors', islandPolicy => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const events = createTransientView(graph, 'events', 'float32', 2);
  const populations = createTransientView(graph, 'populations', 'float32', 2);
  const weights = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 3),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 2),
    weights: createTransientView(graph, 'weights', 'float32', 2)
  };
  try {
    const result = addSpatialRateRecipe(graph, {events, populations, weights, islandPolicy});
    expectTypeOf(result.intermediates['pooledRates']).toEqualTypeOf<GraphDataView<'float32'>>();
    const memory = createCPUGraphMemory();
    memory.write(events, Float32Array.of(6, -2));
    memory.write(populations, Float32Array.of(30, 10));
    memory.write(weights.offsets, Uint32Array.of(0, 1, 2));
    memory.write(weights.neighbors, Uint32Array.of(1, 0));
    memory.write(weights.weights, Float32Array.of(1, 1));
    // Independent pooled input to the final policy pass: eligible focus 6/30=.2;
    // the negative-event focus is invalid. This isolates the recipe's donor decision.
    memory.write(result.intermediates['pooledRates'], Float32Array.of(0.2, NaN));
    const nodes = (graph as unknown as {nodes: GPUCommandNode[]}).nodes;
    executeCommandNodesOnCPU(
      nodes.filter(node => node.id === 'spatial-rate-finish'),
      memory
    );
    const actual = memory.readFloat32(result.outputs.spatialRates);
    if (islandPolicy === 'own-rate') {
      expect(Number.isFinite(actual[0])).toBe(true);
      expect(actual[0]).toBeCloseTo(0.2, 6);
    } else expect(actual[0]).toBeNaN();
    expect(actual[1]).toBeNaN();
  } finally {
    device.destroy();
  }
});

it('B1 zero-weight slots do not prevent the default island policy', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const events = createTransientView(graph, 'events', 'float32', 2);
  const populations = createTransientView(graph, 'populations', 'float32', 2);
  const weights = {
    offsets: createTransientView(graph, 'offsets', 'uint32', 3),
    neighbors: createTransientView(graph, 'neighbors', 'uint32', 1),
    weights: createTransientView(graph, 'weights', 'float32', 1)
  };
  try {
    const result = addSpatialRateRecipe(graph, {events, populations, weights});
    const memory = createCPUGraphMemory();
    memory.write(events, Float32Array.of(4, 8));
    memory.write(populations, Float32Array.of(10, 10));
    memory.write(weights.offsets, Uint32Array.of(0, 1, 1));
    memory.write(weights.neighbors, Uint32Array.of(1));
    memory.write(weights.weights, Float32Array.of(0));
    // The smoother's own-rate fallback at an island is 4/10=.4, 8/10=.8.
    memory.write(result.intermediates['pooledRates'], Float32Array.of(0.4, 0.8));
    const nodes = (graph as unknown as {nodes: GPUCommandNode[]}).nodes;
    executeCommandNodesOnCPU(
      nodes.filter(node => node.id === 'spatial-rate-finish'),
      memory
    );
    expect(Array.from(memory.readFloat32(result.outputs.spatialRates))).toEqual([NaN, NaN]);
  } finally {
    device.destroy();
  }
});
