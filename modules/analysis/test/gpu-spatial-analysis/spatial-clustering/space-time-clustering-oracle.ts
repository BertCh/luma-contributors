// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Double-precision CPU references for space-time clustering (ST-DBSCAN and ST-OPTICS), written from
 * the definitions of Birant and Kut (2007) and Ankerst et al. (1999), independent of the GPU code
 * and of the non-temporal oracle. A pair is related when its spatial distance is within the
 * spatial radius and `|dt|` is within the temporal radius; every point relates to itself.
 */

export type SpaceTimePoints = {
  /** `x, y` per point. */
  positions: readonly number[];
  times: readonly number[];
};

function isValidPoint(points: SpaceTimePoints, row: number): boolean {
  return (
    Number.isFinite(points.positions[2 * row]) &&
    Number.isFinite(points.positions[2 * row + 1]) &&
    Number.isFinite(points.times[row])
  );
}

function getDistance(points: SpaceTimePoints, first: number, second: number): number {
  return Math.hypot(
    points.positions[2 * first] - points.positions[2 * second],
    points.positions[2 * first + 1] - points.positions[2 * second + 1]
  );
}

function isRelatedInTime(points: SpaceTimePoints, first: number, second: number, radius: number) {
  return Math.abs(points.times[first] - points.times[second]) <= radius;
}

export type SpaceTimeDBSCANResult = {
  /** Compact cluster IDs by ascending smallest core row, or -1 for noise. */
  labels: number[];
  coreFlags: number[];
  /** Smallest core row of the point's cluster, or -1. */
  rootRows: number[];
};

/** Brute-force ST-DBSCAN: closed spatial and temporal radii, the point counts itself. */
export function clusterSpaceTimeDBSCAN(
  points: SpaceTimePoints,
  spatialRadius: number,
  timeRadius: number,
  minimumSamples: number
): SpaceTimeDBSCANResult {
  const count = points.times.length;
  const neighbors: number[][] = [];
  for (let row = 0; row < count; row++) {
    const list: number[] = [];
    if (isValidPoint(points, row)) {
      for (let other = 0; other < count; other++) {
        if (
          isValidPoint(points, other) &&
          getDistance(points, row, other) <= spatialRadius &&
          isRelatedInTime(points, row, other, timeRadius)
        ) {
          list.push(other);
        }
      }
    }
    neighbors.push(list);
  }
  const core = neighbors.map(list => list.length >= minimumSamples);
  const root: number[] = new Array(count).fill(-1);
  for (let row = 0; row < count; row++) {
    if (!core[row] || root[row] >= 0) {
      continue;
    }
    // Rows are visited in ascending order, so the first core row reaching a component is its smallest.
    root[row] = row;
    const stack = [row];
    while (stack.length > 0) {
      const current = stack.pop()!;
      for (const other of neighbors[current]) {
        if (core[other] && root[other] < 0) {
          root[other] = row;
          stack.push(other);
        }
      }
    }
  }
  const rootRows = root.slice();
  for (let row = 0; row < count; row++) {
    if (!core[row]) {
      const roots = neighbors[row].filter(other => core[other]).map(other => root[other]);
      rootRows[row] = roots.length > 0 ? Math.min(...roots) : -1;
    }
  }
  const compact = new Map<number, number>();
  [...new Set(rootRows.filter(value => value >= 0))]
    .sort((first, second) => first - second)
    .forEach((value, index) => compact.set(value, index));
  return {
    labels: rootRows.map(value => (value < 0 ? -1 : compact.get(value)!)),
    coreFlags: core.map(Number),
    rootRows
  };
}

export type SpaceTimeOPTICSResult = {
  ordering: number[];
  coreDistances: number[];
  reachability: number[];
  predecessors: number[];
  /** DBSCAN extraction at the extraction distance; -1 is noise. */
  labels: number[];
};

/** Brute-force ST-OPTICS with (reachability, row) frontier ties and sklearn's DBSCAN extraction. */
export function orderSpaceTimeOPTICS(
  points: SpaceTimePoints,
  maximumDistance: number,
  timeRadius: number,
  minimumSamples: number,
  extractionDistance: number
): SpaceTimeOPTICSResult {
  const count = points.times.length;
  const valid = Array.from({length: count}, (_, row) => isValidPoint(points, row));
  const related = (first: number, second: number) =>
    valid[first] &&
    valid[second] &&
    getDistance(points, first, second) <= maximumDistance &&
    isRelatedInTime(points, first, second, timeRadius);
  const coreDistances = Array.from({length: count}, (_, row) => {
    const distances: number[] = [];
    for (let other = 0; other < count; other++) {
      if (related(row, other)) {
        distances.push(getDistance(points, row, other));
      }
    }
    distances.sort((first, second) => first - second);
    return distances.length >= minimumSamples ? distances[minimumSamples - 1] : Infinity;
  });
  const reachability: number[] = new Array(count).fill(Infinity);
  const predecessors: number[] = new Array(count).fill(-1);
  const processed = valid.map(isValid => !isValid);
  const ordering: number[] = [];
  for (;;) {
    let focus = -1;
    for (let row = 0; row < count; row++) {
      if (!processed[row] && (focus < 0 || reachability[row] < reachability[focus])) {
        focus = row;
      }
    }
    if (focus < 0) {
      break;
    }
    processed[focus] = true;
    ordering.push(focus);
    if (!Number.isFinite(coreDistances[focus])) {
      continue;
    }
    for (let other = 0; other < count; other++) {
      if (!processed[other] && related(focus, other)) {
        const candidate = Math.max(coreDistances[focus], getDistance(points, focus, other));
        if (candidate < reachability[other]) {
          reachability[other] = candidate;
          predecessors[other] = focus;
        }
      }
    }
  }
  const labels: number[] = new Array(count).fill(-1);
  let cluster = -1;
  for (const row of ordering) {
    const farReachability = reachability[row] > extractionDistance;
    const nearCore = coreDistances[row] <= extractionDistance;
    if (farReachability && nearCore) {
      cluster++;
    }
    labels[row] = farReachability && !nearCore ? -1 : cluster;
  }
  return {ordering, coreDistances, reachability, predecessors, labels};
}
