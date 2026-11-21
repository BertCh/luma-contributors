// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {addVisibilityGraphRecipe as publicVisibilityRecipe} from '@luma-contributors/analysis/gpu-network';
import {
  getNetworkVisibilityGraphExecutionPlan,
  addVisibilityGraphRecipe
} from '../../../src/gpu-network/visibility-graph/visibility-graph';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const [count, laneCount, maximumIterations, expectedPivots, expectedCommands] of [
  [256, 1, 256, 23, 31970],
  [10000, 3, 8, 30, 2098],
  [257, 32, 16, 32, 1202]
] as const) {
  it(`SCL-04 selects feasible automatic visibility pivots for ${count} positions and ${laneCount} lanes`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const result = addVisibilityGraphRecipe(graph, {
        mode: 'csr',
        positions: createTransientView(graph, 'positions', 'float32x2', count),
        segments: createTransientView(graph, 'segments', 'float32x4', 1),
        maximumDistance: 1.1,
        edgeCapacity: 4 * count,
        laneCount,
        maximumIterations
      });
      // 1058 fixed commands + batches*(5*rounds+64); 24 one-lane batches at
      // 256 rounds would require 33314 commands, while 23 require 31970.
      expect(result.provenance.pivotCount).toBe(expectedPivots);
      expect(result.estimatedCommandNodeCount).toBe(expectedCommands);
      expect(result.provenance.pivotCount % laneCount).toBe(0);
      expect(result.estimatedCommandNodeCount).toBeLessThanOrEqual(32768);
      expect(result.provenance.pivotCount).toBeLessThan(count);
    } finally {
      device.destroy();
    }
  });
}

for (const [count, laneCount, maximumIterations, pivotCount, commands] of [
  [256, 1, 256, 2, 3746],
  [10000, 3, 8, 30, 2098]
] as const) {
  it(`SCL-04 preserves explicit ${pivotCount}-pivot plan and SCL-16 preflight agrees with construction`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const props = {
      mode: 'csr' as const,
      positions: createTransientView(graph, 'positions', 'float32x2', count),
      segments: createTransientView(graph, 'segments', 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 4 * count,
      laneCount,
      maximumIterations,
      pivotCount
    };
    const state = graph as unknown as {nodes: unknown[]; buffers: Map<string, unknown>};
    const bufferCount = state.buffers.size;
    try {
      const plan = publicVisibilityRecipe.getExecutionPlan(graph, props);
      expect(state.buffers.size).toBe(bufferCount);
      expect(state.nodes).toHaveLength(0);
      // By hand: 1026 adjacency + 32 fixed + batches * (5*rounds+64).
      expect(plan.estimatedCommandNodeCount).toBe(commands);
      expect(plan.route).toBe('sampled');
      expect(plan.pivotCount).toBe(pivotCount);
      expect(plan.bindingBytes['centrality lane scratch']).toBe(laneCount * count * 12);
      expect(plan.constraints.every(constraint => constraint.satisfied)).toBe(true);
      const result = addVisibilityGraphRecipe(graph, props);
      expect(result.provenance.executionPlan).toEqual(plan);
      expect(state.nodes.length).toBeLessThanOrEqual(commands);
    } finally {
      device.destroy();
    }
  });
}

it('SCL-16 preserves exact rejection and explains an impossible automatic smallest candidate without graph work', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    mode: 'csr' as const,
    positions: createTransientView(graph, 'positions', 'float32x2', 256),
    segments: createTransientView(graph, 'segments', 'float32x4', 1),
    maximumDistance: 1.1,
    edgeCapacity: 1024,
    laneCount: 1,
    maximumIterations: 256
  };
  const state = graph as unknown as {nodes: unknown[]; buffers: Map<string, unknown>};
  const before = state.buffers.size;
  try {
    expect(() =>
      getNetworkVisibilityGraphExecutionPlan(graph, {...props, metricsMode: 'exact'})
    ).toThrow(/command node count/);
    expect(() => getNetworkVisibilityGraphExecutionPlan(graph, {...props, laneCount: 257})).toThrow(
      /smallest candidate pivotCount 257 failed: pivotCount/
    );
    expect(state.buffers.size).toBe(before);
    expect(state.nodes).toHaveLength(0);
    const exact = getNetworkVisibilityGraphExecutionPlan(graph, {
      ...props,
      laneCount: 256,
      metricsMode: 'exact'
    });
    expect(exact.route).toBe('exact');
    expect(exact.pivotCount).toBe(256);
  } finally {
    device.destroy();
  }
});

it('SCL-04 adapts inferred lanes to device binding bytes and names an impossible smallest plan', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    mode: 'csr' as const,
    positions: createTransientView(graph, 'positions', 'float32x2', 256),
    segments: createTransientView(graph, 'segments', 'float32x4', 1),
    maximumDistance: 1.1,
    edgeCapacity: 1024,
    maximumIterations: 90
  };
  try {
    Object.assign(device.limits, {maxStorageBufferBindingSize: 24000});
    const plan = getNetworkVisibilityGraphExecutionPlan(graph, props);
    // At most floor(24000/(256*12)) = 7 lanes; 4 is the largest divisor of 32.
    expect(plan.laneCount).toBe(4);
    expect(plan.pivotCount).toBe(32);
    expect(Object.values(plan.bindingBytes).every(bytes => bytes <= 24000)).toBe(true);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 16000});
    // Shared centrality CSR bytes: (2*(256+1) + 4*1024)*4 = 18440, even at one pivot.
    expect(() => getNetworkVisibilityGraphExecutionPlan(graph, props)).toThrow(
      /smallest candidate pivotCount 1 failed: centrality adjacency needs 18440 binding bytes; limit 16000/
    );
  } finally {
    device.destroy();
  }
});
