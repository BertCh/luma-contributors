// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getWrappedRectangleScanWGSL} from '../../../src/gpu-spatial-analysis/spatial-interpolation/grid-neighbor-scan';
import {
  getSeparationWGSL,
  resolveInterpolationSeparation
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/separation-wgsl';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('PTS-117 wrapped cells cover both longitude intervals once and whole polar caps', () => {
  const device = createNullWebGPUDevice();
  for (const [longitude, latitude, width, expected] of [
    [179.9375, 0, 8, [1, 1, 0, 0, 0, 0, 1, 1]],
    [-179.9375, 0, 8, [1, 1, 0, 0, 0, 0, 1, 1]],
    [179.9375, 0, 1, [1]],
    [0, 89.9, 8, [1, 1, 1, 1, 1, 1, 1, 1]]
  ] as const) {
    const graph = new GPUCommandGraph(device);
    const cellOffsets = createTransientView(graph, 'offsets', 'uint32', width + 1);
    const sortedIds = createTransientView(graph, 'rows', 'uint32', width);
    const visits = createTransientView(graph, 'visits', 'uint32', width);
    const node = createWGSLKernelNode(graph, {
      id: 'scan',
      operation: 'WrappedScanProbe',
      invocationCount: 1,
      bindings: [
        {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
        {name: 'sortedIds', view: sortedIds, type: 'u32', access: 'read'},
        {name: 'visits', view: visits, type: 'u32', access: 'read_write'}
      ],
      declarations: `${getSeparationWGSL(resolveInterpolationSeparation('scan', {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}, 'wrapped'))}
const DOMAIN_MIN: vec2f = vec2f(-180.0, -90.0);
const DOMAIN_MAX: vec2f = vec2f(180.0, 90.0);
const INDEX_WIDTH: u32 = ${width}u;
fn getCoordinate(value: f32, minimum: f32, maximum: f32, count: u32) -> u32 {
  return min(u32(clamp((value - minimum) / (maximum - minimum), 0.0, 1.0) * f32(count)), count - 1u);
}`,
      body: `let center = vec2f(${longitude}, ${latitude});
let radius = 50000.0;
let rowLow = 0u; let rowHigh = 0u;
let columnLow = 0u; let columnHigh = ${width - 1}u;
${getWrappedRectangleScanWGSL('visits[visitsOffset + sample] += 1u;')}`
    });
    const memory = createCPUGraphMemory();
    memory.write(
      cellOffsets,
      Uint32Array.from({length: width + 1}, (_, index) => index)
    );
    memory.write(
      sortedIds,
      Uint32Array.from({length: width}, (_, index) => index)
    );
    memory.write(visits, new Uint32Array(width));
    executeCommandNodesOnCPU([node], memory);
    // Eight 45-degree columns: the seam cap plus one-column slack touches only 0,1,6,7.
    // One coarse column belongs to both intervals but is visited once; a polar cap touches all.
    expect(Array.from(memory.readUint32(visits))).toEqual(expected);
  }
  device.destroy();
});

it('PTS-117 planar separation keeps its planar scan when wrapping is requested', () => {
  expect(resolveInterpolationSeparation('planar', undefined, 'wrapped').wrappedLongitude).toBe(
    false
  );
});
