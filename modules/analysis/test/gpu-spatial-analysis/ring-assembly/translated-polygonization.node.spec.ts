// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getSegmentPolygonizationDiagnostics} from '../../../src/analysis-core/ring-assembly/segment-polygonization-diagnostics';

it('FLT-01 preserves binary64 face area under translation', () => {
  // Closed form: (1/16)^2 = 1/256. Keep binary64 input: float32 cannot store this square.
  const origin = 10_000_000;
  const end = origin + 1 / 16;
  const result = getSegmentPolygonizationDiagnostics({
    segments: [
      [origin, origin, end, origin],
      [end, origin, end, end],
      [end, end, origin, end],
      [origin, end, origin, origin]
    ]
  });
  expect(result.closedRingSegmentIndices).toEqual([0, 1, 2, 3]);
  expect(result.closedRings).toHaveLength(1);
  expect(result.closedRings[0].signedArea).toBe(0.00390625);
});
