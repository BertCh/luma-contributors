// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpaceTimeHawkesOnCPU} from '../../../src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation';

for (const maximumIterations of [1, 3]) {
  it(`Hawkes CPU uses the hand-derived Poisson stopping decision at budget ${maximumIterations}`, () => {
    // Equal times preclude parents: first update background 1 -> 0.2, second update unchanged.
    // Relative change 0.8 exceeds 1e-4 at update 1; update 2 has zero relative change.
    const result = computeSpaceTimeHawkesOnCPU(
      {
        positions: Float32Array.of(0, 0, 1, 1),
        times: Float32Array.of(2, 2),
        parameters: Float32Array.of(1, 10, 10, 10, 1e-4, 1, 0.5, 1, 1)
      },
      {maximumIterations}
    );
    expect(Array.from(result.converged)).toEqual([maximumIterations === 1 ? 0 : 1]);
    expect(Array.from(result.iterationLimitReached)).toEqual([maximumIterations === 1 ? 1 : 0]);
    expect(Array.from(result.iterationCount)).toEqual([maximumIterations === 1 ? 1 : 2]);
  });
}
