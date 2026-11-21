// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {integrateReference} from './flow-map-reference';

it('review equal-time slices always select the first velocity in the binary64 reference', () => {
  for (const startTime of [4, 5]) {
    const values = integrateReference({
      start: Float32Array.of(2, 0),
      end: Float32Array.of(6, 0),
      width: 1,
      height: 1,
      extent: [0, 0, 100, 100],
      seeds: [[8, 8]],
      startTime,
      timeStep: 1,
      sliceTimes: [4, 4],
      stepCount: 1
    });
    // Equal slices select weight zero. Constant v=(2,0) for one second maps (8,8) -> (10,8).
    expect(values).toEqual([[10, 8]]);
  }
});
