// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Lower-bound statements of `GPUTrackSimilaritySearch` for the measures beyond the dynamic time
 * warping and Frechet endpoint bound. Each bound is computed in `O(n + m)` from the original
 * vertices, in the planar metric, and never exceeds the measure it bounds (up to the f32 slack
 * built into each statement, which scales with the number of summed terms).
 *
 * @internal
 */

/** Measures of the search with a bounding-box or sum bound handled here. @internal */
export type TrackSimilaritySearchBoundMetric =
  | 'hausdorff'
  | 'edit-distance-on-real-sequence'
  | 'edit-distance-with-real-penalty'
  | 'symmetric-segment-path';

/** Relative rounding slack of one f32 addition, doubled. */
const ADDITION_SLACK = '1.2e-7';

/** WGSL declarations the bound statements use. @internal */
export const TRACK_SIMILARITY_BOUND_DECLARATIONS = /* wgsl */ `
// Euclidean distance from a point to an axis-aligned box. The polyline of a track lies inside the
// box of its vertices, so this never exceeds the distance from the point to the polyline.
fn boxDistance(point: vec2<f32>, lower: vec2<f32>, upper: vec2<f32>) -> f32 {
  return length(max(max(lower - point, point - upper), vec2<f32>(0.0)));
}`;

/**
 * WGSL statements that assign `bounds[boundsOffset + index]`. They run after the caller defined
 * `startA`, `endA`, `startB`, `endB`, `loadA` and `loadB`. @internal
 */
export function getTrackSimilaritySearchBoundStatements(
  metric: TrackSimilaritySearchBoundMetric,
  tolerance: number,
  gapPoint: readonly [number, number]
): string {
  const boxes = `var lowerA = loadA(startA); var upperA = lowerA; var lowerB = loadB(startB); var upperB = lowerB;
for (var row = startA; row < endA; row++) { let position = loadA(row); lowerA = min(lowerA, position); upperA = max(upperA, position); }
for (var row = startB; row < endB; row++) { let position = loadB(row); lowerB = min(lowerB, position); upperB = max(upperB, position); }`;
  const countA = '(endA - startA)';
  const countB = '(endB - startB)';
  switch (metric) {
    case 'hausdorff':
      // Every vertex of one track lies on that track, so the Hausdorff distance is at least the
      // distance from any vertex to the box of the other track.
      return `${boxes}
var farthest = 0.0;
for (var row = startA; row < endA; row++) { farthest = max(farthest, boxDistance(loadA(row), lowerB, upperB)); }
for (var row = startB; row < endB; row++) { farthest = max(farthest, boxDistance(loadB(row), lowerA, upperA)); }
bounds[boundsOffset + index] = farthest;`;
    case 'symmetric-segment-path':
      // The mean of distances to the polyline is at least the mean of distances to its box.
      return `${boxes}
var sumA = 0.0; var sumB = 0.0;
for (var row = startA; row < endA; row++) { sumA += boxDistance(loadA(row), lowerB, upperB); }
for (var row = startB; row < endB; row++) { sumB += boxDistance(loadB(row), lowerA, upperA); }
let slack = max(0.0, 1.0 - f32(${countA} + ${countB}) * ${ADDITION_SLACK});
bounds[boundsOffset + index] = 0.5 * (sumA / f32(${countA}) + sumB / f32(${countB})) * slack;`;
    case 'edit-distance-on-real-sequence':
      // A vertex with no vertex of the other track within the tolerance cannot match, and the
      // cheapest script costs at least max(n, m) minus the matchable vertices of the poorer side.
      // The box widened by the tolerance contains every point within the tolerance of the track.
      return `${boxes}
let reach = vec2<f32>(f32(${tolerance}) * 1.000001);
var possibleA = 0u; var possibleB = 0u;
for (var row = startA; row < endA; row++) { let position = loadA(row); if (all(position >= lowerB - reach) && all(position <= upperB + reach)) { possibleA++; } }
for (var row = startB; row < endB; row++) { let position = loadB(row); if (all(position >= lowerA - reach) && all(position <= upperA + reach)) { possibleB++; } }
bounds[boundsOffset + index] = f32(max(${countA}, ${countB}) - min(possibleA, possibleB));`;
    case 'edit-distance-with-real-penalty':
      // ERP is a metric, and its distance to the empty sequence is the sum of the gap distances,
      // so the difference of the two sums is a lower bound (reverse triangle inequality).
      return `let gap = vec2<f32>(${gapPoint[0]}, ${gapPoint[1]});
var sumA = 0.0; var sumB = 0.0;
for (var row = startA; row < endA; row++) { sumA += length(loadA(row) - gap); }
for (var row = startB; row < endB; row++) { sumB += length(loadB(row) - gap); }
bounds[boundsOffset + index] = max(0.0, abs(sumA - sumB) - f32(${countA} + ${countB}) * ${ADDITION_SLACK} * (sumA + sumB));`;
  }
}
