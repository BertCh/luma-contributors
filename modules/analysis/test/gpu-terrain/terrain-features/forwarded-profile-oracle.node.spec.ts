// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {findProfilePeaksOracle} from './profile-peaks-oracle';

it('rounds the inclusive prominence minimum to its packed float32 precision', () => {
  // 2+2^-24 rounds to 2 in float32; peak prominence is exactly 2-0=2.
  const result = findProfilePeaksOracle(Float32Array.of(0, 2, 0), undefined, [0, 3], {
    window: 1,
    minimumSide: 1,
    minimumProminence: 2 + 2 ** -24
  });
  expect([...result.mask]).toEqual([0, 1, 0]);
  expect(result.peaks.map(peak => peak.row)).toEqual([1]);
});
