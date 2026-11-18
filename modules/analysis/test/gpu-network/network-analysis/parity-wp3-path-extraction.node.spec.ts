// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {extractPathsOracle} from './network-path-extraction-oracle';
import {getPathExtractionExpectedStatus, routingFixtures} from './parity-wp3-routing-fixture';

it('separates walk truncation from final capacity in parity status expectations', () => {
  // A retained root needs one node, while a different path hits its walk budget. There is ample
  // final capacity: growing it cannot recover that suppressed path.
  expect(getPathExtractionExpectedStatus(1, 16, true)).toEqual([1, 1, 0, 1]);
  expect(getPathExtractionExpectedStatus(5, 3, false)).toEqual([3, 5, 1, 0]);
  expect(getPathExtractionExpectedStatus(5, 3, true)).toEqual([3, 5, 1, 1]);
  expect(getPathExtractionExpectedStatus(0, 16, false)).toEqual([0, 0, 0, 0]);
});

for (const {input, cases} of routingFixtures) {
  for (const fixture of cases) {
    it(`matches independent bounded-walk status for ${fixture.case}/short`, () => {
      const maximumPathLength = 2;
      const nodeCapacity = Math.max(1, fixture.paths.flat().length);
      const edgeCapacity = Math.max(1, fixture.edgePaths.flat().length);
      const found = fixture.paths.map(path => path.length > 0 && path.length <= maximumPathLength);
      const truncated = fixture.paths.some(path => path.length > maximumPathLength);
      const nodeRequirement = fixture.paths.reduce(
        (total, path, row) => total + (found[row] ? path.length : 0),
        0
      );
      const edgeRequirement = fixture.edgePaths.reduce(
        (total, path, row) => total + (found[row] ? path.length : 0),
        0
      );
      const actual = extractPathsOracle({
        predecessors: fixture.predecessors,
        costs: fixture.costs,
        targets: fixture.targets,
        maxPathLength: maximumPathLength,
        capacity: nodeCapacity,
        csr: {
          offsets: Uint32Array.from(input.offsets),
          neighbors: Uint32Array.from(input.neighbors),
          weights: Float32Array.from(input.weights)
        },
        edgeCapacity
      });
      expect([
        actual.count,
        actual.requiredCount,
        actual.overflow,
        actual.candidateOverflow
      ]).toEqual(getPathExtractionExpectedStatus(nodeRequirement, nodeCapacity, truncated));
      expect([
        actual.edgeCount,
        actual.edgeTotalCount,
        actual.edgeOverflow,
        Number(actual.truncated)
      ]).toEqual(getPathExtractionExpectedStatus(edgeRequirement, edgeCapacity, truncated));
    });
  }
}
