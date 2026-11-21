// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeNormalizeGeometryOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-normalize-geometry-operation';
import {computeOffsetCurveComponentsOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-offset-curve-components-operation';

it('CPU normalization status matches independently classified leaf conditions', () => {
  for (const frame of [
    {point: [1, 2], kind: 0, status: 0},
    {point: [NaN, 2], kind: 0, status: 1},
    {point: [1, 2], kind: 8, status: 2}
  ]) {
    const result = computeNormalizeGeometryOnCPU({
      positions: Float32Array.from(frame.point),
      nodes: Uint32Array.of(frame.kind, 0, 1, 0),
      children: new Uint32Array(0)
    });
    expect(Array.from(result.status)).toEqual([frame.status]);
  }
});
it('CPU offset status matches the one-component two-vertex oracle in every word', () => {
  for (const vertexCapacity of [1, 2]) {
    const result = computeOffsetCurveComponentsOnCPU(
      {
        original: {positions: Float32Array.of(0, 0, 2, 0), lineOffsets: Uint32Array.of(0, 2)},
        raw: {positions: Float32Array.of(0, 1, 2, 1), lineOffsets: Uint32Array.of(0, 2)}
      },
      {
        distance: 1,
        tolerance: 0,
        vertexCapacity,
        componentCapacity: 1,
        segmentCapacity: 4,
        intersectionCapacity: 4
      }
    );
    expect(Array.from(result.status)).toEqual(
      vertexCapacity === 1 ? [0, 0, 1, 2, 1, 0, 0, 0] : [1, 2, 1, 2, 0, 0, 0, 0]
    );
  }
});
