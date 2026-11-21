// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeIsolinesOnCPU} from '../../../src/raster-analysis/isolines/cpu-isolines-operation';
import {getGPUIsolinesParameterValues} from '../../../src/raster-analysis/isolines/isolines-parameters';

for (const scene of [
  {segments: 2, vertices: 3, expected: [2, 2, 0, 0, 1, 3]},
  {segments: 2, vertices: 2, expected: [2, 2, 0, 1, 0, 0]},
  {segments: 1, vertices: 3, expected: [1, 2, 1, 1, 0, 0]}
]) {
  it(`isolines count complete ramp chains independently (segments=${scene.segments}, vertices=${scene.vertices})`, () => {
    const grid = {
      width: 2,
      height: 3,
      affine: [1, 0, 0, 0, 1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const result = computeIsolinesOnCPU(
      {
        values: Float32Array.of(0, 2, 0, 2, 0, 2),
        levels: Float32Array.of(1),
        parameters: getGPUIsolinesParameterValues({levelCount: 1, extent: [0, 0, 2, 3]}, grid)
      },
      {grid, segmentCapacity: scene.segments, vertexCapacity: scene.vertices}
    );
    // Two marching cells each cross the level once: two joined segments, one three-vertex
    // open chain. A short segment buffer suppresses stitching; a short vertex buffer drops
    // the whole chain. All six expected words come from that geometry and capacities.
    expect([
      result.count[0],
      result.requiredCount[0],
      result.overflow[0],
      result.polylineOverflow[0],
      result.polylineCount[0],
      result.vertexCount[0]
    ]).toEqual(scene.expected);
  });
}
