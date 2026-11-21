// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it} from 'vitest';
import {GEODESIC_WGSL} from '../../../src/gpu-spatial-analysis/geometry-measures/geodesic-wgsl';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

useCPUExecutorTestHooks();
const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
it('GEO-121 preserves both polar half-turn endpoint branches in actual interpolation WGSL', () => {
  const graph = new GPUCommandGraph(device),
    memory = createCPUGraphMemory();
  const positions = createTransientView(graph, 'polar-positions', 'float32x2', 10);
  executeCommandNodesOnCPU(
    [
      createWGSLKernelNode(graph, {
        id: 'polar-half-turn',
        operation: 'GPUGeodesicPairs',
        invocationCount: 10,
        bindings: [{name: 'positions', view: positions, type: 'f32', access: 'read_write'}],
        declarations: GEODESIC_WGSL,
        body: `let start = vec2<f32>(0.0,80.0);
let end = vec2<f32>(select(180.0,-180.0,index >= 5u),80.0);
let fraction = f32(index % 5u) * 0.25;
let position = geodesicInterpolate(start,end,geodesicCentralAngle(start,end),fraction);
positions[positionsOffset+2u*index] = position.x;
positions[positionsOffset+2u*index+1u] = position.y;`
      })
    ],
    memory
  );
  const actual = memory.readFloat32(positions);
  expect(actual).toHaveLength(20);
  for (let line = 0; line < 2; line++) {
    // The shortest path joins the two meridians at the north pole: 5-degree steps.
    for (const [row, latitude] of [80, 85, 90, 85, 80].entries()) {
      expect(Number.isFinite(actual[10 * line + 2 * row + 1])).toBe(true);
      expect(Math.abs(actual[10 * line + 2 * row + 1] - latitude)).toBeLessThan(0.001);
    }
    const before = actual[10 * line + 6],
      endpoint = actual[10 * line + 8];
    expect(Number.isFinite(before)).toBe(true);
    expect(Number.isFinite(endpoint)).toBe(true);
    expect(endpoint).toBe(line === 0 ? 180 : -180);
    expect(Math.abs(before - endpoint)).toBeLessThan(0.01);
  }
});
