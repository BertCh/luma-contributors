// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTransitionDiagnostics} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics';
import {createTransitionDiagnosticsSummaryNodes} from '../../../src/gpu-spatial-analysis/distribution-dynamics/transition-diagnostics-summary';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

const CONDITION_COUNT = 65537;

/** Condition 0 has [2^24, 3*2^24]; every other condition has [1, 3]; origin 1 is empty. */
function createMixedMagnitudeCounts(): Uint32Array {
  const values = new Uint32Array(CONDITION_COUNT * 4);
  for (let condition = 0; condition < CONDITION_COUNT; condition++) {
    values[condition * 4] = condition === 0 ? 2 ** 24 : 1;
    values[condition * 4 + 1] = condition === 0 ? 3 * 2 ** 24 : 3;
  }
  return values;
}

it('pools 65537 mixed-magnitude conditions with bounded integer tiles', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const counts = createTransientView(graph, 'counts', 'uint32', CONDITION_COUNT * 4);
  const values = createMixedMagnitudeCounts();
  try {
    const {nodes, pooledCounts} = createTransitionDiagnosticsSummaryNodes(graph, {
      id: 'pool',
      counts,
      classCount: 2,
      conditionCount: CONDITION_COUNT,
      minimumTransitions: 5,
      output: {
        steadyStates: createTransientView(graph, 'steady', 'float32', (CONDITION_COUNT + 1) * 2)
      }
    });
    expect(nodes.some(node => node.workload?.operation === 'GPUSegmentedReduction')).toBe(false);
    const memory = createCPUGraphMemory();
    memory.write(counts, values);
    executeCommandNodesOnCPU(nodes, memory);
    // Closed form: [2^24+65536, 3*(2^24+65536)], exactly representable in f32.
    expect(Array.from(memory.readFloat32(pooledCounts!))).toEqual([16842752, 50528256, 0, 0]);
    memory.write(counts, new Uint32Array(values.length));
    executeCommandNodesOnCPU(nodes, memory);
    expect(Array.from(memory.readFloat32(pooledCounts!))).toEqual([0, 0, 0, 0]);
  } finally {
    device.destroy();
  }
}, 120000);

it('keeps the homogeneity graph for 65537 conditions within the command-node allowance', () => {
  // The GPU regression `B5 identical mixed-magnitude regimes` (transition-diagnostics-reduction
  // .spec.ts) checks the published [0, 1, 0, 1, 65536]; public reductions use barriers that the
  // interpreter cannot run, so this case checks only the bounded structure.
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const counts = createTransientView(graph, 'counts', 'uint32', CONDITION_COUNT * 4);
  const homogeneity = createTransientView(graph, 'homogeneity', 'float32', 5);
  const stateSparsity = createTransientView(graph, 'sparsity', 'uint32', 2);
  try {
    const contributor = new GPUTransitionDiagnostics({
      counts,
      classCount: 2,
      conditionCount: CONDITION_COUNT,
      output: {homogeneity, stateSparsity}
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.some(node => node.workload?.operation === 'GPUSegmentedReduction')).toBe(false);
    // ceil(log256(65537)) = 3 tile levels plus one publication for each of the two condition pools.
    expect(nodes.filter(node => /-(pool-conditions|count-regimes)-/.test(node.id))).toHaveLength(8);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});
