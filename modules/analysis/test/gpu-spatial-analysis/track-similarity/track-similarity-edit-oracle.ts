// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU references in float64 for the edit-distance measures of `GPUTrackSimilarity` and the lower
 * bounds of `GPUTrackSimilaritySearch`, written from the definitions in the papers (see
 * `test/oracles/track-similarity-edit-measures.py`). Planar only.
 */

import {createRandom} from '../trajectory-interpolation/trajectory-interpolation-fixtures';
import type {Polyline} from './track-similarity-oracle';

type Point = readonly [number, number];

function measureDistance(first: Point, second: Point): number {
  return Math.hypot(first[0] - second[0], first[1] - second[1]);
}

/** EDR (Chen, Ozsu and Oria 2005) with a metric-ball match of radius `tolerance`. */
export function computeEditDistanceOnRealSequenceOracle(
  a: Polyline,
  b: Polyline,
  tolerance: number
): number {
  // Table over suffixes: row index = remaining vertices of a.
  let next = Array.from({length: b.length + 1}, (_, columnIndex) => b.length - columnIndex);
  for (let rowIndex = a.length - 1; rowIndex >= 0; rowIndex--) {
    const current = new Array<number>(b.length + 1);
    current[b.length] = a.length - rowIndex;
    for (let columnIndex = b.length - 1; columnIndex >= 0; columnIndex--) {
      const substitution = measureDistance(a[rowIndex], b[columnIndex]) <= tolerance ? 0 : 1;
      current[columnIndex] = Math.min(
        next[columnIndex + 1] + substitution,
        next[columnIndex] + 1,
        current[columnIndex + 1] + 1
      );
    }
    next = current;
  }
  return next[0];
}

/** ERP (Chen and Ng 2004) with the fixed gap point `gap`. */
export function computeEditDistanceWithRealPenaltyOracle(
  a: Polyline,
  b: Polyline,
  gap: Point
): number {
  const gapB = b.map(point => measureDistance(point, gap));
  let next = new Array<number>(b.length + 1).fill(0);
  for (let columnIndex = b.length - 1; columnIndex >= 0; columnIndex--) {
    next[columnIndex] = next[columnIndex + 1] + gapB[columnIndex];
  }
  for (let rowIndex = a.length - 1; rowIndex >= 0; rowIndex--) {
    const gapA = measureDistance(a[rowIndex], gap);
    const current = new Array<number>(b.length + 1);
    current[b.length] = next[b.length] + gapA;
    for (let columnIndex = b.length - 1; columnIndex >= 0; columnIndex--) {
      current[columnIndex] = Math.min(
        next[columnIndex + 1] + measureDistance(a[rowIndex], b[columnIndex]),
        next[columnIndex] + gapA,
        current[columnIndex + 1] + gapB[columnIndex]
      );
    }
    next = current;
  }
  return next[0];
}

function measureDistanceToPolyline(point: Point, polyline: Polyline): number {
  if (polyline.length === 1) {
    return measureDistance(point, polyline[0]);
  }
  let nearest = Infinity;
  for (let index = 0; index + 1 < polyline.length; index++) {
    const [x0, y0] = polyline[index];
    const [dx, dy] = [polyline[index + 1][0] - x0, polyline[index + 1][1] - y0];
    const lengthSquared = dx * dx + dy * dy;
    const fraction = lengthSquared
      ? Math.min(1, Math.max(0, ((point[0] - x0) * dx + (point[1] - y0) * dy) / lengthSquared))
      : 0;
    nearest = Math.min(
      nearest,
      Math.hypot(point[0] - x0 - fraction * dx, point[1] - y0 - fraction * dy)
    );
  }
  return nearest;
}

/** SSPD (Besse et al. 2016): the mean of the two mean vertex-to-polyline distances. */
export function computeSymmetricSegmentPathOracle(a: Polyline, b: Polyline): number {
  const mean = (from: Polyline, to: Polyline) =>
    from.reduce((sum, point) => sum + measureDistanceToPolyline(point, to), 0) / from.length;
  return 0.5 * (mean(a, b) + mean(b, a));
}

/** Distance from a point to the axis-aligned box `[lower, upper]`. */
function measureDistanceToBox(point: Point, lower: Point, upper: Point): number {
  return Math.hypot(
    Math.max(lower[0] - point[0], point[0] - upper[0], 0),
    Math.max(lower[1] - point[1], point[1] - upper[1], 0)
  );
}

function getBox(polyline: Polyline): {lower: Point; upper: Point} {
  return {
    lower: [Math.min(...polyline.map(p => p[0])), Math.min(...polyline.map(p => p[1]))],
    upper: [Math.max(...polyline.map(p => p[0])), Math.max(...polyline.map(p => p[1]))]
  };
}

/** Lower bound of the Hausdorff distance: the largest vertex distance to the other track's box. */
export function computeHausdorffLowerBoundOracle(a: Polyline, b: Polyline): number {
  const [boxA, boxB] = [getBox(a), getBox(b)];
  return Math.max(
    ...a.map(point => measureDistanceToBox(point, boxB.lower, boxB.upper)),
    ...b.map(point => measureDistanceToBox(point, boxA.lower, boxA.upper))
  );
}

/** Lower bound of SSPD: the same bound with means. */
export function computeSymmetricSegmentPathLowerBoundOracle(a: Polyline, b: Polyline): number {
  const [boxA, boxB] = [getBox(a), getBox(b)];
  const mean = (polyline: Polyline, box: {lower: Point; upper: Point}) =>
    polyline.reduce((sum, point) => sum + measureDistanceToBox(point, box.lower, box.upper), 0) /
    polyline.length;
  return 0.5 * (mean(a, boxB) + mean(b, boxA));
}

/** Lower bound of EDR: max(n, m) minus the fewer matchable vertices (box widened by the tolerance). */
export function computeEditDistanceOnRealSequenceLowerBoundOracle(
  a: Polyline,
  b: Polyline,
  tolerance: number
): number {
  const [boxA, boxB] = [getBox(a), getBox(b)];
  const count = (polyline: Polyline, box: {lower: Point; upper: Point}) =>
    polyline.filter(
      point =>
        point[0] >= box.lower[0] - tolerance &&
        point[0] <= box.upper[0] + tolerance &&
        point[1] >= box.lower[1] - tolerance &&
        point[1] <= box.upper[1] + tolerance
    ).length;
  return Math.max(a.length, b.length) - Math.min(count(a, boxB), count(b, boxA));
}

/** Lower bound of ERP: the difference of the two sums of gap distances. */
export function computeEditDistanceWithRealPenaltyLowerBoundOracle(
  a: Polyline,
  b: Polyline,
  gap: Point
): number {
  const sum = (polyline: Polyline) =>
    polyline.reduce((total, point) => total + measureDistance(point, gap), 0);
  return Math.abs(sum(a) - sum(b));
}

/** Random walks with a random offset, lengths, spread and scale, plus degenerate pairs. */
export function createLowerBoundPairs(seed: number): [Polyline, Polyline][] {
  const random = createRandom(seed);
  const walk = (count: number, originX: number, originY: number, step: number): Polyline => {
    let [x, y] = [originX, originY];
    return Array.from({length: count}, () => {
      x += (random() - 0.5) * step;
      y += (random() - 0.5) * step;
      return [x, y] as [number, number];
    });
  };
  const pairs: [Polyline, Polyline][] = [];
  for (let index = 0; index < 400; index++) {
    const offset = index % 4 === 0 ? 0 : random() * (index % 4 === 3 ? 200 : 15);
    const first = walk(1 + Math.floor(random() * 40), 0, 0, 1 + random() * 6);
    const second = walk(1 + Math.floor(random() * 40), offset, offset * 0.5, 1 + random() * 6);
    pairs.push([first, second]);
  }
  const line: Polyline = [
    [0, 0],
    [5, 5],
    [10, 0]
  ];
  pairs.push([line, line], [line, [[0, 0]]], [[[3, 3]], [[3, 3]]], [[[0, 0]], [[100, 100]]]);
  return pairs;
}
