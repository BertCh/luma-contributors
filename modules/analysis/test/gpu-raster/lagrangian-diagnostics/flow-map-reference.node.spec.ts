// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {integrateReference} from './flow-map-reference';

for (const missingEndLanding of [true, false]) {
  it(`B3 two-slice FlowMap reference checks backward landing: missing end ${missingEndLanding}`, () => {
    const start = Float32Array.of(0, 2, 0, 2, 0, 2, 0, 2, 0, 2);
    const end = start.slice();
    if (missingEndLanding) end.set([NaN, NaN], 8);
    // Signed y spacing: y=3.5 -> midpoint 2 -> landing .5 in row 4.
    // The start slice is finite everywhere. Both slices are required even at a clamped time.
    expect(
      integrateReference({
        start,
        end,
        width: 1,
        height: 5,
        extent: [0, 5, 1, -1],
        seeds: [[0.5, 3.5]],
        startTime: 1.5,
        timeStep: -1.5,
        sliceTimes: [0, 1.5],
        stepCount: 1
      })
    ).toEqual([missingEndLanding ? undefined : [0.5, 0.5]]);
  });
}
