// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Orthonormal associated Legendre values including Condon-Shortley phase. Normalization is
 * carried through the recurrence, avoiding factorial overflow at degree 39. Derived from DLMF
 * 14.10.3 and 14.30.1; diagonal from Rodrigues (14.7.8). These are Q_l^m = N_lm P_l^m.
 * @internal
 */
export const FITTING_SPHERICAL_HARMONIC_WGSL = /* wgsl */ `
fn normalizedLegendre(degree: u32, order: u32, latitude: f32) -> f32 {
  let vertical = sin(latitude);
  // cos(latitude), rather than sqrt(1 - sin(latitude)^2), preserves small polar offsets.
  let radial = max(0.0, cos(latitude));
  var diagonal = 0.28209479177387814;
  for (var diagonalOrder = 1u; diagonalOrder <= order; diagonalOrder++) {
    let value = f32(diagonalOrder);
    diagonal *= -sqrt((2.0 * value + 1.0) / (2.0 * value)) * radial;
  }
  if (degree == order) { return diagonal; }
  var previous = diagonal;
  var current = sqrt(2.0 * f32(order) + 3.0) * vertical * diagonal;
  for (var nextDegree = order + 2u; nextDegree <= degree; nextDegree++) {
    let value = f32(nextDegree);
    let orderSquared = f32(order * order);
    let previousSquared = (value - 1.0) * (value - 1.0);
    let multiplier = sqrt((4.0 * value * value - 1.0) / (value * value - orderSquared));
    let predecessor = sqrt((previousSquared - orderSquared) / (4.0 * previousSquared - 1.0));
    let next = multiplier * (vertical * current - predecessor * previous);
    previous = current;
    current = next;
  }
  return current;
}
`;
