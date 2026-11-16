// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

import {createTrajectoryCounterOverflowFixture} from './trajectory-counter-overflow-fixture';
import {expectTrajectoryCounterOverflowReadback} from './trajectory-counter-overflow-assertions';

it('wires a saturating reservation into the production encounter counter', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const fixture = createTrajectoryCounterOverflowFixture(graph);
    const scanSource = getGPUCommandNodeKernelSources([fixture.nodes[1]])[0].source;
    expect(scanSource).toContain('let slot = reserveTrajectoryCount();');
    expect(scanSource).toContain('atomicCompareExchangeWeak');
    expect(scanSource).toContain('currentCount == TRAJECTORY_COUNT_MAXIMUM');
    expect(scanSource).not.toContain('atomicAdd(&state[stateOffset], 1u)');
    // wgsl_reflect does not execute compare-exchange. The headless sibling executes this
    // production scan from the seeded count and checks both saturation and untouched slots.
    for (const kernel of getGPUCommandNodeKernelSources(fixture.nodes)) {
      const source = kernel.source.replace(/\/\*[\s\S]*?\*\/|\/\/[^\n]*/g, '');
      expect(
        (source.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
          WGSL_RESERVED_WORDS.has(identifier)
        )
      ).toEqual([]);
    }
  } finally {
    device.destroy();
  }
});

it('executes the already-saturated production path without wrapping into earlier slots', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const fixture = createTrajectoryCounterOverflowFixture(graph);
    const memory = createCPUGraphMemory();
    executeCommandNodesOnCPU([fixture.nodes[0]], memory);
    // This path never calls compare-exchange, which wgsl_reflect does not implement.
    // The headless sibling additionally crosses the boundary from 0xfffffffe concurrently.
    memory.write(fixture.state, new Uint32Array([0xffffffff]));
    executeCommandNodesOnCPU(fixture.nodes.slice(1), memory);
    expectTrajectoryCounterOverflowReadback({
      state: Array.from(memory.readUint32(fixture.state)),
      pairs: Array.from(memory.readUint32(fixture.pairs)),
      overflow: Array.from(memory.readUint32(fixture.overflow))
    });
  } finally {
    device.destroy();
  }
});
