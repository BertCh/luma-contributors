// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Hand-enumerable conditional randomization, without a random-generator reference:
 * mean = 2; centered x = [-1, 3, -1, -1]; sum of squares = 12.
 * Row 1 has one slot of weight 2. Its three possible donors (0, 2, 3) all equal -1,
 * so the complete conditional distribution is {-6, -6, -6}, with probability 1/3 each.
 * The observed raw Moran term is 3 * (2 * -1) = -6, and the public scaled value is
 * (n - 1) * -6 / 12 = -1.5. Row 0's observed raw term is 1, exposing a wrong pin.
 * Row 3 is a kept island: it stays in the donor population but cannot be tested/replayed.
 */
export function createLocalReplayLiteralFixture() {
  return {
    weights: {
      offsets: Uint32Array.of(0, 1, 2, 3, 3),
      neighbors: Uint32Array.of(2, 0, 0),
      weights: Float32Array.of(1, 2, 1)
    },
    values: Float32Array.of(1, 5, 1, 1)
  };
}
