// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  expectTraversalPadding,
  expectYenEnumerationStatus
} from '../alternative-routes/traversal-enumeration-assertions';
import {computeKShortestPathsOnCPU} from '../../../src/gpu-network/k-shortest-paths/index';
import {
  enumerateAlternativeRoutes,
  getTraversalEnumerationInput,
  readTraversalRoutes
} from '../alternative-routes/alternative-routes-operation-test-utils';

it('independent DFS pins every multigraph edge path and Yen certified prefixes', () => {
  const input = getTraversalEnumerationInput();
  const reference = enumerateAlternativeRoutes(input, 8);
  // Hand enumeration: direct [3]; two parallel 0->1 slots each continue via 1->2->3
  // or 1->3; finally 0->2->3. Vertex 4 is isolated.
  expect(readTraversalRoutes(reference, 6)).toEqual([
    {edges: [3], cost: 0.5},
    {edges: [0, 4, 6], cost: 2},
    {edges: [0, 5], cost: 2},
    {edges: [1, 4, 6], cost: 2},
    {edges: [1, 5], cost: 2},
    {edges: [2, 6], cost: 2}
  ]);
  for (const [maximumPathLength, candidateCapacity] of [
    [5, 64],
    [2, 64],
    [5, 1]
  ]) {
    const result = computeKShortestPathsOnCPU(input, {
      routeCount: 8,
      maximumPathLength,
      candidateCapacity
    });
    const count = result.counts[0];
    const actual = readTraversalRoutes(result, count, maximumPathLength);
    const all = readTraversalRoutes(reference, 6);
    expect(actual.map(route => route.cost)).toEqual(all.slice(0, count).map(route => route.cost));
    for (const route of actual) expect(all).toContainEqual(route);
    expect(new Set(actual.map(route => JSON.stringify(route.edges))).size).toBe(count);
    expectYenEnumerationStatus(result, maximumPathLength, candidateCapacity);
    expectTraversalPadding(result, 2, 8, maximumPathLength);
  }
});
