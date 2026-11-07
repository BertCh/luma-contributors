// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU Ripley K curves, used only to make the observed and simulated input curves of the envelope
// test (the operation is given curves; it never synthesizes them).

/** K(r) - pi r^2 for the given radii, without edge correction, in a window of `area`. */
export function computeKDeviation(
  positions: Float32Array,
  radii: Float32Array,
  area: number
): Float32Array {
  const count = positions.length / 2;
  const curve = new Float32Array(radii.length);
  for (let first = 0; first < count; first++) {
    for (let second = first + 1; second < count; second++) {
      const distance = Math.hypot(
        positions[first * 2] - positions[second * 2],
        positions[first * 2 + 1] - positions[second * 2 + 1]
      );
      radii.forEach((radius, index) => {
        if (distance <= radius) curve[index] += 2;
      });
    }
  }
  return Float32Array.from(
    curve,
    (pairs, index) => (area * pairs) / (count * (count - 1)) - Math.PI * radii[index] ** 2
  );
}
