// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL of the edit-distance dynamic programs (EDR and ERP) and of the symmetric segment-path
 * distance (SSPD) of `GPUTrackSimilarity`.
 *
 * The edit-distance programs use the layout of the dynamic time warping program: one 256-lane
 * workgroup per pair, the longer (densified) track along the lanes in strips, the carried last
 * row of a strip in workgroup memory. Their table has one extra row and column for the empty
 * prefix, so the boundary values (insertion and deletion prefixes) come out of the same
 * recurrence. Both expect the shared track helpers of the contributor in scope.
 *
 * @internal
 */

import {COUPLED_METRIC_WGSL} from './track-similarity-dynamic-programs';

/** Which edit-distance recurrence a program evaluates. @internal */
export type EditDistanceKind = 'edr' | 'erp';

/** WGSL literal of an f32 value, always with a decimal point. */
function getFloatLiteral(value: number): string {
  const text = String(Math.fround(value));
  return /[.e]/.test(text) ? text : `${text}.0`;
}

/**
 * Declarations of an edit-distance kernel.
 *
 * EDR (Chen, Ozsu and Oria 2005): a pair of vertices matches when their distance is at most
 * `tolerance`; a match costs 0, a mismatch (substitution), an insertion and a deletion cost 1.
 * ERP (Chen and Ng 2004): substitution costs the distance of the two vertices, and an insertion
 * or deletion costs the distance of the vertex to the fixed `gapPoint`.
 * @internal
 */
export function getEditDistanceDeclarations(
  kind: EditDistanceKind,
  lanes: number,
  shortCapacity: number,
  tolerance: number,
  gapPoint: readonly [number, number]
): string {
  const gapCost =
    kind === 'edr'
      ? 'return 1.0;'
      : `var point = densifiedPointB(startB, countB, row);
  if (isLong != swapped) { point = densifiedPointA(startA, countA, row); }
  return finishMetric(pointMetric(point, GAP_POINT));`;
  const substitutionCost =
    kind === 'edr'
      ? `return select(1.0, 0.0, finishMetric(coupledMetric(swapped, startA, countA, startB, countB, rowLong, rowShort)) <= f32(${getFloatLiteral(tolerance)}));`
      : 'return finishMetric(coupledMetric(swapped, startA, countA, startB, countB, rowLong, rowShort));';
  return `const WORKGROUP_SIZE: u32 = ${lanes}u;
const LANES: u32 = ${lanes}u;
const GAP_POINT: vec2<f32> = vec2<f32>(${getFloatLiteral(gapPoint[0])}, ${getFloatLiteral(gapPoint[1])});
var<workgroup> olderDiagonal: array<f32, ${lanes}>;
var<workgroup> previousDiagonal: array<f32, ${lanes}>;
var<workgroup> boundaryRow: array<f32, ${shortCapacity + 1}>;
var<workgroup> sharedLongCount: u32;
var<workgroup> sharedShortCount: u32;
${COUPLED_METRIC_WGSL}
// Cost of inserting or deleting densified vertex row of the long track (isLong) or of the short one.
fn gapCost(swapped: bool, startA: u32, countA: u32, startB: u32, countB: u32, row: u32, isLong: bool) -> f32 {
  ${gapCost}
}
fn substitutionCost(swapped: bool, startA: u32, countA: u32, startB: u32, countB: u32, rowLong: u32, rowShort: u32) -> f32 {
  ${substitutionCost}
}`;
}

/**
 * Body of an edit-distance kernel: the cost of the cheapest edit script between the densified
 * vertex sequences. The caller's prologue defines `pair`, `lane`, `skipped`, the track ranges and
 * `notANumber`. @internal
 */
export function getEditDistanceBody(prologue: string): string {
  return `${prologue}
  let infinity = 3.402823466e+38;
  let denseA = densifiedCount(countA);
  let denseB = densifiedCount(countB);
  let swapped = denseB > denseA;
  if (lane == 0u) {
    sharedLongCount = max(denseA, denseB);
    sharedShortCount = min(denseA, denseB);
  }
  // Barrier-bearing loops need bounds the compiler can prove uniform across the workgroup.
  let longCount = workgroupUniformLoad(&sharedLongCount);
  let shortCount = workgroupUniformLoad(&sharedShortCount);
  // Table rows and columns 0 hold the empty prefix; vertex k is row or column k + 1.
  let rowCount = longCount + 1u;
  let columnCount = shortCount + 1u;
  let stripCount = (rowCount + LANES - 1u) / LANES;
  var finalValue = infinity;
  for (var strip = 0u; strip < stripCount; strip++) {
    let firstRow = strip * LANES;
    let rows = min(LANES, rowCount - firstRow);
    let diagonalCount = rows + columnCount - 1u;
    olderDiagonal[lane] = infinity;
    previousDiagonal[lane] = infinity;
    workgroupBarrier();
    for (var diagonal = 0u; diagonal < diagonalCount; diagonal++) {
      var value = infinity;
      var column = 0u;
      let isActive = lane < rows && diagonal >= lane && diagonal - lane < columnCount;
      if (isActive) {
        column = diagonal - lane;
        let row = firstRow + lane;
        var up = infinity;
        var upLeft = infinity;
        var left = infinity;
        if (lane > 0u) {
          up = previousDiagonal[lane - 1u];
          upLeft = olderDiagonal[lane - 1u];
        } else if (strip > 0u) {
          up = boundaryRow[column];
          if (column > 0u) { upLeft = boundaryRow[column - 1u]; }
        }
        if (column > 0u) { left = previousDiagonal[lane]; }
        if (row == 0u && column == 0u) {
          value = 0.0;
        } else if (row == 0u) {
          value = left + gapCost(swapped, startA, countA, startB, countB, column - 1u, false);
        } else if (column == 0u) {
          value = up + gapCost(swapped, startA, countA, startB, countB, row - 1u, true);
        } else {
          value = min(
            upLeft + substitutionCost(swapped, startA, countA, startB, countB, row - 1u, column - 1u),
            min(
              up + gapCost(swapped, startA, countA, startB, countB, row - 1u, true),
              left + gapCost(swapped, startA, countA, startB, countB, column - 1u, false)
            )
          );
        }
        if (row == longCount && column == shortCount) { finalValue = value; }
      }
      workgroupBarrier();
      olderDiagonal[lane] = previousDiagonal[lane];
      previousDiagonal[lane] = value;
      if (isActive && lane == LANES - 1u) { boundaryRow[column] = value; }
      workgroupBarrier();
    }
  }
  if (!skipped && longCount > 0u && lane == longCount % LANES) {
    distances[distancesOffset + pair] = finalValue;
  } else if (skipped && lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = notANumber;
  }`;
}

/** Declarations of the SSPD kernel; the contributor adds `WORKGROUP_SIZE`. @internal */
export function getSymmetricSegmentPathDeclarations(lanes: number): string {
  const nearest = (
    side: 'A' | 'B'
  ) => `// Metric value from a point to the polyline of track ${side}: every segment, in order.
fn nearestMetricTo${side}(point: vec2<f32>, start: u32, count: u32) -> f32 {
  if (count == 1u) { return pointMetric(point, load${side}(start)); }
  var nearest = 3.402823466e+38;
  for (var segment = 0u; segment + 1u < count; segment++) {
    nearest = min(nearest, pointSegmentMetric(point, load${side}(start + segment), load${side}(start + segment + 1u)));
  }
  return nearest;
}`;
  return `const WORKGROUP_SIZE: u32 = ${lanes}u;
var<workgroup> partials: array<f32, ${lanes}>;
${nearest('A')}
${nearest('B')}`;
}

/**
 * Body of the SSPD kernel (Besse et al. 2016): the mean over the original vertices of A of the
 * distance to the polyline of B, averaged with the same mean from B to A. Each lane sums its
 * strided vertices in order, then the lanes are added by a fixed binary tree, so the f32 result
 * is deterministic. @internal
 */
export function getSymmetricSegmentPathBody(prologue: string, lanes: number): string {
  return `${prologue}
  var sumAToB = 0.0;
  for (var vertex = lane; vertex < countA; vertex += WORKGROUP_SIZE) {
    sumAToB += finishMetric(nearestMetricToB(loadA(startA + vertex), startB, countB));
  }
  partials[lane] = sumAToB;
  workgroupBarrier();
  for (var stride = ${lanes / 2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = partials[lane] + partials[lane + stride]; }
    workgroupBarrier();
  }
  let totalAToB = partials[0];
  workgroupBarrier();
  var sumBToA = 0.0;
  for (var vertex = lane; vertex < countB; vertex += WORKGROUP_SIZE) {
    sumBToA += finishMetric(nearestMetricToA(loadB(startB + vertex), startA, countA));
  }
  partials[lane] = sumBToA;
  workgroupBarrier();
  for (var stride = ${lanes / 2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = partials[lane] + partials[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u && pair < PAIR_COUNT) {
    distances[distancesOffset + pair] = select(0.5 * (totalAToB / f32(countA) + partials[0] / f32(countB)), notANumber, skipped);
  }`;
}
