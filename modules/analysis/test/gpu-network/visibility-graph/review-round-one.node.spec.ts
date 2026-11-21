// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import type {GPUAnalysisExecutionPlanner} from '@luma-contributors/analysis/analysis-core';
import {
  addVisibilityGraphRecipe,
  type GPUNetworkVisibilityGraphCSRProps
} from '@luma-contributors/analysis/gpu-network';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createFixture(count = 8, segmentCount = 1) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props: GPUNetworkVisibilityGraphCSRProps = {
    mode: 'csr',
    positions: createTransientView(graph, 'positions', 'float32x2', count),
    segments: createTransientView(graph, 'segments', 'float32x4', segmentCount),
    maximumDistance: 1.1,
    edgeCapacity: 2 * count,
    laneCount: 1,
    maximumIterations: count
  };
  const state = graph as unknown as {nodes: unknown[]; buffers: Map<string, {byteLength: number}>};
  return {device, graph, props, state};
}

for (const name of [
  'approximation',
  'zeroWeightCycle',
  'converged',
  'iterationLimitReached'
] as const) {
  for (const route of ['preflight', 'construction'] as const) {
    it(`B2 ${route} refuses malformed ${name} before graph mutation`, () => {
      const {device, graph, props, state} = createFixture();
      try {
        for (const [format, length] of [
          ['uint32', 2],
          ['float32', 1]
        ] as const) {
          const malformed = createTransientView(graph, `${name}-${format}`, format, length);
          const before = state.buffers.size;
          const invalid = {
            ...props,
            output: {status: {[name]: malformed}}
          } as GPUNetworkVisibilityGraphCSRProps;
          const invoke = () =>
            route === 'preflight'
              ? addVisibilityGraphRecipe.getExecutionPlan(graph, invalid)
              : addVisibilityGraphRecipe(graph, invalid);
          expect(invoke).toThrow(new RegExp(`output.status.${name}`));
          expect(state.nodes).toHaveLength(0);
          expect(state.buffers.size).toBe(before);
        }
      } finally {
        device.destroy();
      }
    });
  }
}

it('B3 odd-round frontier bytes match actual declarations at the device boundary', () => {
  const {device, graph, props, state} = createFixture(31);
  try {
    const plan = addVisibilityGraphRecipe.getExecutionPlan(graph, props);
    // Two phases, floor(31/2)+1 slots each, 256 bytes per dispatch slot.
    expect(plan.bindingBytes['centrality round state']).toBe(8192);
    Object.assign(device.limits, {maxStorageBufferBindingSize: 8192});
    expect(() => addVisibilityGraphRecipe(graph, props)).not.toThrow();
    expect(Math.max(...Array.from(state.buffers.values(), buffer => buffer.byteLength))).toBe(8192);
  } finally {
    device.destroy();
  }
});

it('B4 exposes catalog-compatible planning and reports dominant adjacency bindings', () => {
  const {device, graph, props, state} = createFixture(7, 2000);
  try {
    const planner: GPUAnalysisExecutionPlanner<GPUNetworkVisibilityGraphCSRProps, GPUCommandGraph> =
      addVisibilityGraphRecipe.executionPlanner;
    const plan = planner.getExecutionPlan(props, graph);
    expect(plan.feasible).toBe(true);
    expect(Array.isArray(plan.accuracyImplications)).toBe(true);
    expect(plan.accuracyImplications.length).toBeGreaterThan(0);
    expect(
      plan.constraints.find(constraint => constraint.quantity === 'adjacency BVH node bounds')
        ?.received
    ).toBe(32760);
    const ordinary = addVisibilityGraphRecipe.getExecutionPlan(graph, props);
    // 2048 BVH leaves -> (2*2048-1)*8 = 32760 bytes in each node-bounds binding.
    expect(ordinary.bindingBytes['adjacency BVH node bounds']).toBe(32760);
    expect(state.nodes).toHaveLength(0);
    addVisibilityGraphRecipe(graph, props);
    expect(Math.max(...Array.from(state.buffers.values(), buffer => buffer.byteLength))).toBe(
      32760
    );
  } finally {
    device.destroy();
  }
});

for (const mode of ['reference', 'unknown'] as const) {
  it(`S2 CSR preflight refuses mode ${mode} without graph mutation`, () => {
    const {device, graph, props, state} = createFixture(4);
    const before = state.buffers.size;
    try {
      expect(() =>
        addVisibilityGraphRecipe.getExecutionPlan(graph, {
          ...props,
          mode
        } as unknown as GPUNetworkVisibilityGraphCSRProps)
      ).toThrow(/mode.*csr.*received/);
      expect(state.nodes).toHaveLength(0);
      expect(state.buffers.size).toBe(before);
    } finally {
      device.destroy();
    }
  });
}
