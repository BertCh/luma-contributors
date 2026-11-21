// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Positive Gauss-Legendre pairs for sixteen-point integration on each smooth chord segment. */
const CHORD_NODES = [
  0.09501250983763744, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
  0.755404408355003, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499
];
const CHORD_WEIGHTS = [
  0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
  0.1246289712555339, 0.09515851168249278, 0.06225352393864789, 0.027152459411754095
];

/** Integrate a clipped thin disk cap with x=1-t². Splits occur where a rectangle edge hits
 * the circle, so every quadrature segment is smooth; remaining width survives endpoint rounding.
 */
export function getDensityClippedCapMass(
  power: 1 | 2,
  low: number,
  high: number,
  acrossLow: number,
  acrossHigh: number
): number {
  const start = Math.sqrt(Math.max(0, 1 - Math.min(1, high)));
  const end = Math.sqrt(Math.max(0, 1 - low));
  const transition = (coordinate: number) => {
    const magnitude = Math.min(1, Math.abs(coordinate));
    return Math.max(
      start,
      Math.min(end, magnitude / Math.sqrt(1 + Math.sqrt((1 - magnitude) * (1 + magnitude))))
    );
  };
  const first = transition(acrossLow);
  const second = transition(acrossHigh);
  const cuts = [start, Math.min(first, second), Math.max(first, second), end];
  const chord = (time: number) => {
    const remaining = time * time * (2 - time * time);
    const edge = Math.sqrt(remaining);
    const bottom = Math.max(-edge, acrossLow);
    const top = Math.min(edge, acrossHigh);
    if (!(top > bottom)) return 0;
    const width = top - bottom;
    const midpoint = bottom + width / 2;
    const squared = midpoint * midpoint;
    const complement = (edge - Math.abs(midpoint)) * (edge + Math.abs(midpoint));
    const integral =
      power === 1
        ? width * (complement - (width * width) / 12)
        : width *
          (complement * complement +
            ((6 * squared - 2 * remaining) * width * width) / 12 +
            width ** 4 / 80);
    return 2 * time * Math.max(0, integral);
  };
  let total = 0;
  for (let piece = 0; piece < 3; piece++) {
    const halfWidth = (cuts[piece + 1] - cuts[piece]) / 2;
    const midpoint = cuts[piece] + halfWidth;
    for (let index = 0; index < CHORD_NODES.length; index++) {
      const offset = halfWidth * CHORD_NODES[index];
      total +=
        halfWidth * CHORD_WEIGHTS[index] * (chord(midpoint - offset) + chord(midpoint + offset));
    }
  }
  return ((power + 1) / Math.PI) * total;
}

/** Float32 counterpart of the clipped-cap integral, including explicit edge-transition splits. */
export function getDensityClippedCapWGSL(power: 1 | 2): string {
  return /* wgsl */ `
fn getCapTransition(coordinate: f32, start: f32, end: f32) -> f32 {
  let magnitude = min(1.0, abs(coordinate));
  return clamp(magnitude / sqrt(1.0 + sqrt((1.0 - magnitude) * (1.0 + magnitude))), start, end);
}
fn getClippedCapChord(time: f32, acrossLow: f32, acrossHigh: f32) -> f32 {
  let remaining = time * time * (2.0 - time * time);
  let edge = sqrt(remaining);
  let bottom = max(-edge, acrossLow);
  let top = min(edge, acrossHigh);
  if (!(top > bottom)) { return 0.0; }
  let width = top - bottom;
  let midpoint = bottom + 0.5 * width;
  let squared = midpoint * midpoint;
  let complement = (edge - abs(midpoint)) * (edge + abs(midpoint));
  let integral = ${power === 1 ? 'width * (complement - width * width / 12.0)' : 'width * (complement * complement + (6.0 * squared - 2.0 * remaining) * width * width / 12.0 + width * width * width * width / 80.0)'};
  return 2.0 * time * max(0.0, integral);
}
fn getClippedCapMass(low: f32, high: f32, acrossLow: f32, acrossHigh: f32) -> f32 {
  let start = sqrt(max(0.0, 1.0 - min(1.0, high)));
  let end = sqrt(max(0.0, 1.0 - low));
  let first = getCapTransition(acrossLow, start, end);
  let second = getCapTransition(acrossHigh, start, end);
  let cuts = array<f32, 4>(start, min(first, second), max(first, second), end);
  let nodes = array<f32, 8>(${CHORD_NODES.join(', ')});
  let weights = array<f32, 8>(${CHORD_WEIGHTS.join(', ')});
  var total = 0.0;
  for (var piece = 0u; piece < 3u; piece++) {
    let halfWidth = (cuts[piece + 1u] - cuts[piece]) * 0.5;
    let midpoint = cuts[piece] + halfWidth;
    for (var index = 0u; index < 8u; index++) {
      let offset = halfWidth * nodes[index];
      total += halfWidth * weights[index] * (getClippedCapChord(midpoint - offset, acrossLow, acrossHigh) + getClippedCapChord(midpoint + offset, acrossLow, acrossHigh));
    }
  }
  return ${(power + 1) / Math.PI} * total;
}
`;
}
