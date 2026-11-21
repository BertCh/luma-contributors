// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Assertions shared with the GPU fixture, so a corrupt middle row cannot escape qualification. */
export function expectKMeansMajorityDistances(distances: readonly number[]): void {
  expect(distances).toHaveLength(255);
  // Exact majority residual squared = (1/255)^2; independently derived for the GPU fixture.
  for (let row = 0; row < 254; row++) {
    expect(Number.isFinite(distances[row]), `row ${row}`).toBe(true);
    expect(distances[row], `row ${row}`).toBeCloseTo(0.000015378700499807765, 8);
  }
}
