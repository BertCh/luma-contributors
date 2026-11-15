// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGreatCircleArcs,
  getGPUGreatCircleArcsParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {createLineLineFixture} from '@luma-contributors/analysis/test/gpu-spatial-analysis/line-segmentize/line-test-utils';

it.each([
  {longitude: 179.9713592529297, missingInterior: false},
  {longitude: 180, missingInterior: true}
])('emits the two-segment arc to longitude $longitude without whole-arc rejection', async ({
  longitude,
  missingInterior
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createLineLineFixture(device, {
    inputs: {
      sources: {values: new Float32Array([0, 0]), format: 'float32x2'},
      targets: {values: new Float32Array([longitude, 0]), format: 'float32x2'}
    },
    capacity: 3,
    lineCapacity: 1,
    withVertexColumns: false,
    withApproximation: true,
    createContributor: (inputs, parameters, output) =>
      new GPUGreatCircleArcs({
        sources: inputs['sources'] as GraphDataView<'float32x2'>,
        targets: inputs['targets'] as GraphDataView<'float32x2'>,
        maximumSegmentCount: 2,
        parameters,
        output
      })
  });
  try {
    const result = await fixture.run(
      getGPUGreatCircleArcsParameterValues({
        maximumSegmentLength: 0,
        minimumSegments: 2
      })
    );
    // Three vertices are reserved for both pairs. The 1e-6-radian interpolator guard affects
    // only the interior position; it does not discard the line, change its count or flag overflow.
    expect(result.count).toBe(1);
    expect(result.requiredCount).toBe(1);
    expect(result.vertexCount).toBe(3);
    expect(result.requiredVertexCount).toBe(3);
    expect(result.lineOffsets).toEqual([0, 3]);
    expect(result.overflow).toBe(0);
    expect(result.approximation).toBe(0); // Only the subdivision length cap sets this flag.
    expect(result.positions[0]).toEqual([0, 0]);
    expect(result.positions[2][1]).toBe(0);
    if (missingInterior) {
      // At exactly 180 degrees either seam sign represents the unwrapped target.
      expect(Math.abs(result.positions[2][0])).toBe(180);
      expect(result.positions[1].every(Number.isNaN)).toBe(true);
    } else {
      expect(result.positions[2][0]).toBe(longitude);
      expect(result.positions[1].every(Number.isFinite)).toBe(true);
      // The equatorial short arc has its midpoint at longitude / 2 in exact arithmetic.
      // Check its hemisphere without claiming a near-antipodal float32 accuracy envelope.
      expect(result.positions[1][0]).toBeGreaterThan(0);
      expect(result.positions[1][0]).toBeLessThan(180);
      expect(Math.abs(result.positions[1][1])).toBeLessThan(1e-5);
    }
  } finally {
    fixture.destroy();
  }
});
