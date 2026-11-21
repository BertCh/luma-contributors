// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GET_NAN_WGSL,
  IS_FINITE_VALUE_WGSL,
  IS_NAN_BITS_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {getRectangleScanWGSL} from '../spatial-interpolation/grid-neighbor-scan';

/** Density value checks; prepend to the published grid-coordinate WGSL where required. */
export const DENSITY_VALUE_WGSL = `
${IS_FINITE_VALUE_WGSL}
${IS_NAN_BITS_WGSL}// WGSL rejects NaN constants, so build one from a runtime bit pattern.
${GET_NAN_WGSL}
`;

/** Compile-time point eligibility domain shared by fixed and adaptive density. */
export function getDensityDomainConstantsWGSL(
  bounds: readonly [number, number, number, number]
): string {
  return `const DOMAIN_MIN: vec2f = vec2f(${getWGSLFloatLiteral(bounds[0])}, ${getWGSLFloatLiteral(bounds[1])});
const DOMAIN_MAX: vec2f = vec2f(${getWGSLFloatLiteral(bounds[2])}, ${getWGSLFloatLiteral(bounds[3])});`;
}

/** Index bounds of a density support square, expanded one cell for float32 binning roundoff.
 * The caller retains its exact distance predicate, point eligibility and edge correction.
 */
export function getDensityNeighborBoundsWGSL(): string {
  return `let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    // Widen by one index cell per side to absorb f32 rounding in the binning pass; the exact
    // distance test rejects the extra candidates.
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);`;
}

/** Adaptive density's scan at an eligible pilot point or raster cell. */
export function getDensityNeighborScanWGSL(visit: string): string {
  return `let queryMin = center - vec2f(radius);
    let queryMax = center + vec2f(radius);
    if (all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX)) {
      ${getDensityNeighborBoundsWGSL()}
      ${getRectangleScanWGSL(visit)}
    }`;
}

/** Rectangle membership in the original encoded coordinate space, before subtracting an origin. */
export const DENSITY_RECTANGLE_MEMBERSHIP_WGSL = `
fn isInsideDensityRectangle(position: vec2f, minimum: vec2f, maximum: vec2f) -> bool {
  return all(position >= minimum) && all(position <= maximum);
}
`;
