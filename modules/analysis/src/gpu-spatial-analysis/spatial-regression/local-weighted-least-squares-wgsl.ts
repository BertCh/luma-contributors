// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE} from './ordinary-least-squares-parameters';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {getGridLookupWGSL} from './gwr-grid-index';

/**
 * Shared pieces of the local weighted least squares of the GWR family (GWR, generalized GWR, the
 * nonstationarity test): the kernel weight and the equilibrated Cholesky solve of the normal
 * equations. Every operator that accumulates `X'WX` over a neighborhood uses these two instead of
 * its own copy, so a numerical fix lands once. The neighborhood loop with neighborhood loop with its optional grid index
 * (`getCandidateLoopWGSL`, whose accumulator body each operator supplies) and the planar or
 * great-circle distance (`getLocalDistanceWGSL`) live here too. Internal; not exported from the package index.
 */

/**
 * WGSL of the kernel weight, `getLocalKernelWeight(distance, bandwidth, isBisquare) -> f32`:
 * bisquare `(1 - (d/h)^2)^2` with exactly 0 for `d >= h`, otherwise the Gaussian
 * `exp(-0.5 (d/h)^2)`. With a `gaussianCutoff` the Gaussian is exactly 0 beyond
 * `gaussianCutoff * h` and the module also declares `GAUSSIAN_CUTOFF`.
 *
 * @internal
 */
export function getLocalKernelWeightWGSL(gaussianCutoff: number | null = null): string {
  return /* wgsl */ `${
    gaussianCutoff === null
      ? ''
      : `// Gaussian weights vanish beyond this many bandwidths (compile-time gaussianCutoff).
const GAUSSIAN_CUTOFF: f32 = ${getWGSLFloatLiteral(gaussianCutoff)};
`
  }
fn getLocalKernelWeight(distance: f32, bandwidth: f32, isBisquare: bool) -> f32 {
  let ratio = distance / bandwidth;
  if (isBisquare) {
    if (distance >= bandwidth) {
      return 0.0;
    }
    let t = 1.0 - ratio * ratio;
    return t * t;
  }${
    gaussianCutoff === null
      ? ''
      : `
  if (distance > GAUSSIAN_CUTOFF * bandwidth) {
    return 0.0;
  }`
  }
  return exp(-0.5 * ratio * ratio);
}
`;
}

/**
 * WGSL of the diagonally equilibrated Cholesky solve of a symmetric positive definite `size x size`
 * system held in function-local arrays (lower triangle, row-major). Requires
 * `getCholeskyWGSL(size)` (which declares the shared `factorDenseCholesky_size` of `analysis-core`)
 * and `isFiniteValue` in the same module. Equilibration (`s_i = 1 / sqrt(A_ii)`) makes the
 * factorization insensitive to the scale of each design column.
 *
 * - `factorEquilibrated_size(a, scale) -> bool` fills `scale` and factors `S A S` with
 *   `factorDenseCholesky_size(a, size, true, limit^2)`; false when an entry is not finite, a
 *   diagonal is not positive, or the pivot ratio of the unit-diagonal matrix is below the square of
 *   `GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE` (the OLS limit, a factor pivot of `1e-3`: a column
 *   within `1 - 1e-6` of the span of the earlier ones, so GWR reports `SINGULAR` where OLS reports
 *   status 1).
 * - `solveEquilibrated_size(a, scale, vector)` solves `A x = vector` in place.
 * - `inverseDiagonalEquilibrated_size(a, scale, column) -> f32` is `(A^-1)[column][column]`.
 *
 * @internal
 */
export function getEquilibratedCholeskyWGSL(size: number): string {
  const minimumPivotRatio = getWGSLFloatLiteral(GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE ** 2);
  return /* wgsl */ `
fn factorEquilibrated_${size}(a: ptr<function, array<f32, ${size * size}>>, scale: ptr<function, array<f32, ${size}>>) -> bool {
  let factorization = factorDenseCholesky_${size}(a, ${size}u, true, ${minimumPivotRatio});
  *scale = factorization.scale;
  return factorization.status == 0u;
}

fn solveEquilibrated_${size}(a: ptr<function, array<f32, ${size * size}>>, scale: ptr<function, array<f32, ${size}>>, vector: ptr<function, array<f32, ${size}>>) {
  var factorization = DenseCholesky_${size}(*scale, ${size}u, 0u, 1.0);
  solveDenseCholesky_${size}(a, &factorization, vector);
}

fn inverseDiagonalEquilibrated_${size}(a: ptr<function, array<f32, ${size * size}>>, scale: ptr<function, array<f32, ${size}>>, column: u32) -> f32 {
  return choleskyInverseDiagonal_${size}(a, column) * (*scale)[column] * (*scale)[column];
}
`;
}

/**
 * WGSL of `getDistance(a, b)` and `getSquaredDistance(a, b)` for the local regressions: planar
 * Euclidean, or the great-circle distance in the units of `sphereRadius` for `(longitude, latitude)`
 * degrees positions (declares the geodesic helpers too).
 *
 * @internal
 */
export function getLocalDistanceWGSL(sphereRadius: number | null): string {
  if (sphereRadius === null) {
    return `fn getDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let delta = pointB - pointA;
  return sqrt(dot(delta, delta));
}

fn getSquaredDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let delta = pointB - pointA;
  return dot(delta, delta);
}`;
  }
  return `${GEODESIC_WGSL}
// Great-circle distance in the units of the sphere radius; positions are (longitude, latitude).
fn getDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  return geodesicHaversineDistance(pointA, pointB, ${getWGSLFloatLiteral(sphereRadius)});
}

fn getSquaredDistance(pointA: vec2f, pointB: vec2f) -> f32 {
  let distance = getDistance(pointA, pointB);
  return distance * distance;
}`;
}

/**
 * @internal
 *
 * Loop over the candidate rows of a location (`origin`, `bandwidth` in scope) running `body` with
 * `row` bound. Without a grid every row is visited in order. With one, bounded fixed-bandwidth
 * fits (bisquare, fixed) visit only the sorted contents of the cells covering the bandwidth, one
 * contiguous slot range per grid row; other encodings still visit every row.
 */
export function getCandidateLoopWGSL(
  body: string,
  hasGrid: boolean,
  /** WGSL boolean expression over `row` that is true for rows the loop may visit. */
  isCandidate = 'rowValid[rowValidOffset + row] != 0u'
): string {
  if (!hasGrid) {
    return `for (var row = 0u; row < ROW_COUNT; row++) {
    if (!(${isCandidate})) {
      continue;
    }
    ${body}
  }`;
  }
  return `let useGrid = isGridBandwidth(bandwidth);
  let range = select(vec4u(0u, 0u, 0u, 0u), getGridRange(origin, getSupportRadius(bandwidth)), useGrid);
  let gridRowCount = select(1u, range.w - range.z + 1u, useGrid);
  for (var gridRowStep = 0u; gridRowStep < gridRowCount; gridRowStep++) {
    var firstSlot = 0u;
    var endSlot = ROW_COUNT;
    if (useGrid) {
      let cellRow = (range.z + gridRowStep) * INDEX_WIDTH;
      firstSlot = cellOffsets[cellOffsetsOffset + cellRow + range.x];
      endSlot = cellOffsets[cellOffsetsOffset + cellRow + range.y + 1u];
    }
    for (var slot = firstSlot; slot < endSlot; slot++) {
      var row = slot;
      if (useGrid) {
        row = sortedIds[sortedIdsOffset + slot];
      }
      if (!(${isCandidate})) {
        continue;
      }
      ${body}
    }
  }`;
}

/**
 * Grid support for the candidate loop: bandwidth reach (`isGridBandwidth`, `getSupportRadius`) and
 * the focal-row ordering helpers. Needs `getIsBisquare()` (and `GAUSSIAN_CUTOFF` with a cutoff) and
 * the lookup of `gwr-grid-index.ts`.
 *
 * @internal
 */
export function getLocalGridSupportWGSL(
  indexGridSize: readonly [number, number],
  gaussianCutoff: number | null
): string {
  return /* wgsl */ `
${getGridLookupWGSL(indexGridSize)}

// Bisquare has bounded support (fixed or adaptive), so the grid is exact for it; the Gaussian
// kernel is bounded only with a compile-time cutoff.
fn isGridBandwidth(bandwidth: f32) -> bool {
  return ${gaussianCutoff === null ? 'getIsBisquare()' : 'true'} && bandwidth > 0.0 && isFiniteBits(bandwidth);
}

// Distance beyond which the weight is exactly 0.
fn getSupportRadius(bandwidth: f32) -> f32 {
  ${gaussianCutoff === null ? 'return bandwidth;' : 'return select(bandwidth * GAUSSIAN_CUTOFF, bandwidth, getIsBisquare());'}
}

// Included rows in grid-cell order: invocation \`slot\` fits the slot-th indexed row.
fn getFocalCount() -> u32 {
  return min(cellOffsets[cellOffsetsOffset + INDEX_WIDTH * INDEX_HEIGHT], ROW_COUNT);
}

fn getFocalRow(slot: u32) -> u32 {
  return sortedIds[sortedIdsOffset + slot];
}
`;
}
