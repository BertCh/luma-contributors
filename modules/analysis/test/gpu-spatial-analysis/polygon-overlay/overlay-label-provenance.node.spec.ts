// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeLabelPointOnCPU} from '../../../src/gpu-spatial-analysis/label-point/cpu-label-point-operation';

it('an interior label point cannot enumerate dissolved polygon source membership', () => {
  // Adjacent parcels [0,1] x [0,4] and [1,4] x [0,4] dissolve into this square.
  const result = computeLabelPointOnCPU({
    positions: Float32Array.of(0, 0, 4, 0, 4, 4, 0, 4),
    ringOffsets: Uint32Array.of(0, 4),
    polygonOffsets: Uint32Array.of(0, 1),
    featureOffsets: Uint32Array.of(0, 1)
  });
  expect(Array.from(result.degenerate)).toEqual([0]);
  expect(Array.from(result.points)).toEqual([2, 2]);
  const sourceIntervals = [
    [0, 1],
    [1, 4]
  ];
  const pointSources = sourceIntervals.flatMap(([minimum, maximum], row) =>
    result.points[0] > minimum && result.points[0] < maximum ? [row] : []
  );
  // Positive-area intersection is closed form for these rectangles; both contribute area.
  const areaSources = sourceIntervals.flatMap(([minimum, maximum], row) =>
    Math.min(4, maximum) > Math.max(0, minimum) ? [row] : []
  );
  expect(pointSources).toEqual([1]);
  expect(areaSources).toEqual([0, 1]);
  expect(pointSources).not.toEqual(areaSources);
});
