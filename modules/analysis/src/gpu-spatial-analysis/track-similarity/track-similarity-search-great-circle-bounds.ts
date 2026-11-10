// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import type {TrackSimilaritySearchBoundMetric} from './track-similarity-search-bounds';

/**
 * Lower-bound statements of `GPUTrackSimilaritySearch` in a great-circle context. The planar
 * bounding-box bounds do not hold for longitude and latitude, so the box measures (Hausdorff,
 * SSPD) fall back to the trivial bound 0 (no pruning, never wrong) and the others use bounds that
 * only need the point distance to be a metric: the count difference for EDR (every extra vertex
 * costs one edit) and the difference of the sums of gap distances for ERP.
 *
 * @internal
 */

/** WGSL declarations of a great-circle bound kernel: the haversine and the sphere radius. @internal */
export function getTrackSimilarityGreatCircleBoundDeclarations(sphereRadius: number): string {
  return `${GEODESIC_WGSL}
const SPHERE_RADIUS: f32 = ${sphereRadius};
fn pointDistance(a: vec2<f32>, b: vec2<f32>) -> f32 { return geodesicHaversineDistance(a, b, SPHERE_RADIUS); }`;
}

/** Relative rounding slack of one f32 addition, doubled. */
const ADDITION_SLACK = '1.2e-7';

/**
 * WGSL statements that assign `bounds[boundsOffset + index]` for the box-bounded measures in a
 * great-circle context. They run after the caller defined `startA`, `endA`, `startB`, `endB`,
 * `loadA` and `loadB`. @internal
 */
export function getTrackSimilarityGreatCircleBoundStatements(
  metric: TrackSimilaritySearchBoundMetric,
  gapPoint: readonly [number, number]
): string {
  switch (metric) {
    case 'hausdorff':
    case 'symmetric-segment-path':
      return 'bounds[boundsOffset + index] = 0.0;';
    case 'edit-distance-on-real-sequence':
      return `let countA = endA - startA; let countB = endB - startB;
bounds[boundsOffset + index] = f32(max(countA, countB) - min(countA, countB));`;
    case 'edit-distance-with-real-penalty':
      return `let gap = vec2<f32>(${gapPoint[0]}, ${gapPoint[1]});
var sumA = 0.0; var sumB = 0.0;
for (var row = startA; row < endA; row++) { sumA += pointDistance(loadA(row), gap); }
for (var row = startB; row < endB; row++) { sumB += pointDistance(loadB(row), gap); }
bounds[boundsOffset + index] = max(0.0, abs(sumA - sumB) - f32((endA - startA) + (endB - startB)) * ${ADDITION_SLACK} * (sumA + sumB));`;
  }
}
