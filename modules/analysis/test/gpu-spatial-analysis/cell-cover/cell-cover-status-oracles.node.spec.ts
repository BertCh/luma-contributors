// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {coverQuadbinOnCPU, flattenCoverFeatures, hasCrossingCoverEdge} from './cell-cover-oracle';

// This existing test-side host reference has no status outputs or capacity option.
// At resolution 2, the rectangle spans column 0 and rows 0..1. Both centers
// (-135,79.1713) and (-135,40.9799) are inside, so its accepted count is exactly 2.
// Row 0 has Morton prefix 0000; row 1 has prefix 0010. Keys use the Quadbin
// header, resolution 2, and 48 low one-bits. The unsplit ring has 340-degree
// longitude edges, making the independent invalid-feature count exactly 1.
it('host cover reference accepts exactly two analytic tile centers and rejects the unsplit seam', () => {
  for (const north of [85, 90]) {
    const ring = [
      [-170, 10],
      [-100, 10],
      [-100, north],
      [-170, north]
    ];
    const result = coverQuadbinOnCPU(flattenCoverFeatures([[[ring]]]), 2, 'center');
    expect(result.cells).toEqual([0x4820ffffffffffffn, 0x4822ffffffffffffn]);
    expect(result.featureRows).toEqual([0, 0]);
    expect(hasCrossingCoverEdge([ring])).toBe(false);
  }
  const ring = [
    [170, 10],
    [-170, 10],
    [-170, 85],
    [170, 85]
  ];
  expect(hasCrossingCoverEdge([ring])).toBe(true);
  expect(coverQuadbinOnCPU(flattenCoverFeatures([[[ring]]]), 2, 'center').cells).toEqual([]);
});
