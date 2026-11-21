// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUBrownianBridgeUtilization} from '../../../src/gpu-spatial-analysis/trajectory-home-range/gpu-brownian-bridge-utilization';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {createEmptyBridgeInputs} from './empty-bridge-inputs.test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

for (const offsets of [[0], [0, 0]]) {
  it(`t1 empty bridge declares imported inputs and overwrites poisoned density for offsets ${offsets}`, () => {
    const device = createNullWebGPUDevice();
    const fixture = new RecipeTestFixture(device, 't1-empty-bridge');
    try {
      const inputs = createEmptyBridgeInputs(fixture, Uint32Array.from(offsets));
      for (const view of [
        inputs.positions,
        inputs.timestamps,
        inputs.motionVariances,
        inputs.locationErrors
      ])
        expect(view.length).toBe(0);
      const density = fixture.output('density', 'float32', 6);
      const nodes = new GPUBrownianBridgeUtilization({
        ...inputs,
        gridSize: [3, 2],
        output: {density: density.view}
      }).getCommandNodes(fixture.graph);
      const memory = createCPUGraphMemory();
      memory.write(density.view, Float32Array.of(1, 2, 3, 4, 5, 6));
      executeCommandNodesOnCPU(nodes, memory);
      // No fix pairs means no positive-duration segment: all six cell densities are zero.
      expect(Array.from(memory.readFloat32(density.view))).toEqual([0, 0, 0, 0, 0, 0]);
    } finally {
      fixture.destroy();
      device.destroy();
    }
  });
}
