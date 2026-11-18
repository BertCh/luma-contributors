// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUNetworkPointRoutingInput} from '../../../src/gpu-network/network-optimization/cpu-network-point-routing-operation';

export type RoutingReviewFixture = {
  name: string;
  input: CPUNetworkPointRoutingInput;
  maximumRelaxations?: number;
  expectedCosts: number[];
  expectedStatus: number[];
  expectedPaths?: number[][];
};

export function getRoutingReviewFixtures(): RoutingReviewFixture[] {
  return [
    ...[-1, NaN].map(closure => ({
      name: `R2 closure ${closure}`,
      input: {
        costs: Float32Array.of(2, 4, closure),
        turns: Uint32Array.of(0, 1, 0, 2, 2, 1),
        turnPenalties: Float32Array.of(1, 0, 0),
        queries: Uint32Array.of(0, 1, 0, 2, 2, 1),
        fractions: Float32Array.of(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
      },
      expectedCosts: [4, Infinity, Infinity],
      expectedStatus: [0, 1, 1]
    })),
    {
      name: 'R3 partial terminal parent after absorption',
      input: {
        costs: Float32Array.of(0, 0, 2 ** 60),
        turns: Uint32Array.of(0, 2, 0, 1, 1, 2),
        turnPenalties: Float32Array.of(2, 0, 1),
        queries: Uint32Array.of(0, 2),
        fractions: Float32Array.of(1, 0)
      },
      expectedCosts: [1],
      expectedStatus: [0],
      expectedPaths: [[0, 1, 2]]
    },
    {
      name: 'R5 zero budget with a turn and direct same-edge travel',
      maximumRelaxations: 0,
      input: {
        costs: Float32Array.of(2, 10),
        turns: Uint32Array.of(0, 1),
        turnPenalties: Float32Array.of(7),
        queries: Uint32Array.of(0, 1, 0, 0),
        fractions: Float32Array.of(0.5, 0.25, 0.25, 0.75)
      },
      expectedCosts: [Infinity, 1],
      expectedStatus: [2, 2]
    },
    {
      name: 'R6 nonzero intermediate penalty',
      input: {
        costs: Float32Array.of(0, 2, 8),
        turns: Uint32Array.of(0, 1, 1, 2),
        turnPenalties: Float32Array.of(7, 0),
        queries: Uint32Array.of(0, 2),
        fractions: Float32Array.of(1, 0.5)
      },
      expectedCosts: [13],
      expectedStatus: [0],
      expectedPaths: [[0, 1, 2]]
    }
  ];
}

export function traceRoutingReviewPath(
  parents: ArrayLike<number>,
  source: number,
  destination: number,
  edgeCount: number,
  query: number
): number[] {
  const reversed = [destination];
  let cursor = destination;
  while (cursor !== source && reversed.length <= edgeCount) {
    cursor = parents[query * edgeCount + cursor];
    if (cursor >= edgeCount) break;
    reversed.push(cursor);
  }
  return reversed.reverse();
}
