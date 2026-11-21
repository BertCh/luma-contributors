// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

export type TraversalReadback = {
  counts: ArrayLike<number>;
  lengths: ArrayLike<number>;
  costs: ArrayLike<number>;
  vertices: ArrayLike<number>;
  edges: ArrayLike<number>;
};

/** Shared padding check for query-major, route-major edge-slot path output. */
export function expectTraversalPadding(
  rows: TraversalReadback,
  queryCount: number,
  routeCount: number,
  pathLength: number
): void {
  expect(rows.counts.length, 'query count rows').toBe(queryCount);
  const routeRows = queryCount * routeCount;
  expect(rows.lengths.length, 'route length rows').toBe(routeRows);
  expect(rows.costs.length, 'route cost rows').toBe(routeRows);
  expect(rows.vertices.length, 'vertex slots').toBe(routeRows * pathLength);
  expect(rows.edges.length, 'edge slots').toBe(routeRows * pathLength);
  const vertices = Array.from(rows.vertices);
  const edges = Array.from(rows.edges);
  for (let route = 0; route < rows.lengths.length; route++) {
    const length = rows.lengths[route];
    expect(
      vertices
        .slice(route * pathLength + length, (route + 1) * pathLength)
        .every(value => value === 0xffffffff)
    ).toBe(true);
    expect(
      edges
        .slice(route * pathLength + Math.max(0, length - 1), (route + 1) * pathLength)
        .every(value => value === 0xffffffff)
    ).toBe(true);
    if (route % routeCount >= rows.counts[Math.floor(route / routeCount)]) {
      expect(length).toBe(0);
      expect(rows.costs[route]).toBe(Infinity);
    }
  }
}

/** Status certificate for the hand-enumerated six-route fixture. */
export function expectYenEnumerationStatus(
  rows: {
    counts: ArrayLike<number>;
    converged: ArrayLike<number>;
    iterationLimitReached: ArrayLike<number>;
    candidateOverflow: ArrayLike<number>;
  },
  maximumPathLength: number,
  candidateCapacity: number
): void {
  // Direct [3] is certified first. With two vertex slots, every replacement is overlong,
  // so only that route survives. It has exactly one spur; an empty capacity-one pool can
  // retain its replacement and certify route two before multiple spurs overflow the pool.
  // All replacement costs are 2; their edge identities have no promised global tie order.
  const pathStop = maximumPathLength === 2;
  const poolStop = candidateCapacity === 1;
  expect(Array.from(rows.counts)).toEqual([pathStop ? 1 : poolStop ? 2 : 6, 0]);
  expect(Array.from(rows.converged)).toEqual([Number(!pathStop && !poolStop)]);
  expect(Array.from(rows.iterationLimitReached)).toEqual([Number(pathStop)]);
  expect(Array.from(rows.candidateOverflow)).toEqual([Number(poolStop)]);
}
