// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeOffsetCurveComponentsOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-offset-curve-components-operation';
import {
  expectCompleteObliqueOffset,
  PROJECTED_OBLIQUE_OFFSET_CASES
} from './offset-curve-components-test-utils';

it.each(PROJECTED_OBLIQUE_OFFSET_CASES)('retains the complete oblique offset: $name', fixture => {
  // The normal (-300,400)/500 times 5 is (-3,4); translation changes neither endpoint.
  const result = computeOffsetCurveComponentsOnCPU(
    {
      original: {positions: fixture.originalPositions, lineOffsets: Uint32Array.of(0, 2)},
      raw: {positions: fixture.expectedPositions, lineOffsets: Uint32Array.of(0, 2)}
    },
    {distance: 5, tolerance: fixture.tolerance, vertexCapacity: 8, componentCapacity: 4}
  );
  expectCompleteObliqueOffset(result, fixture.expectedPositions);
});

it.each(PROJECTED_OBLIQUE_OFFSET_CASES)('retains the reversed oblique offset: $name', fixture => {
  const reverse = (values: Float32Array) =>
    Float32Array.of(values[2], values[3], values[0], values[1]);
  const result = computeOffsetCurveComponentsOnCPU(
    {
      original: {positions: reverse(fixture.originalPositions), lineOffsets: Uint32Array.of(0, 2)},
      raw: {positions: reverse(fixture.expectedPositions), lineOffsets: Uint32Array.of(0, 2)}
    },
    {distance: -5, tolerance: fixture.tolerance, vertexCapacity: 8, componentCapacity: 4}
  );
  expectCompleteObliqueOffset(result, reverse(fixture.expectedPositions));
});

it('distinguishes rounded duplicate publication from loss of the oblique interior', () => {
  const positions = Float32Array.of(-3, 4, 397, 304);
  const complete = {
    positions,
    components: Uint32Array.of(0, 2, 0, 0),
    sourceEdges: Uint32Array.of(0, 0),
    status: Uint32Array.of(1, 2, 1, 2, 0, 0, 0, 0)
  };
  expectCompleteObliqueOffset(complete, positions);
  for (const broken of [
    {...complete, positions: Float32Array.of(-3, 4, -3, 4)},
    {...complete, status: Uint32Array.of(0, 0, 0, 0, 0, 0, 0, 0)},
    {
      ...complete,
      positions: Float32Array.of(-3, 4, -3, 4, 397, 304, 397, 304),
      status: Uint32Array.of(1, 4, 1, 4, 0, 0, 0, 0)
    },
    {...complete, positions: Float32Array.of(-3, 4, 397, NaN)}
  ])
    expect(() => expectCompleteObliqueOffset(broken, positions)).toThrow();
});
