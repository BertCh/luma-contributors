// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';

/** Hand-worked two-by-two queen-grid oracle for affine [1, 1, 0, 0, -1, 0]. */
export function expectUniformShearEdgeSlots(actual: {
  edges: readonly (readonly [number, number])[];
  conductances: readonly number[];
}): void {
  // Four slots per cell: east, south, south-east, south-west. Each off-grid slot
  // is a self edge of the owning cell, not an arbitrary zero-conductance edge.
  const expectedEdges = [
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 0],
    [1, 1],
    [1, 3],
    [1, 1],
    [1, 2],
    [2, 3],
    [2, 2],
    [2, 2],
    [2, 2],
    [3, 3],
    [3, 3],
    [3, 3],
    [3, 3]
  ];
  // A*(1,1)=(2,-1), A*(-1,1)=(0,-1): the diagonal conductances are 1/sqrt(5), 1.
  const expectedConductances = [
    1, 0.7071067811865476, 0.4472135954999579, 0, 0, 0.7071067811865476, 0, 1, 1, 0, 0, 0, 0, 0, 0,
    0
  ];
  expect(actual.edges).toHaveLength(16);
  expect(actual.conductances).toHaveLength(16);
  expect(actual.edges).toEqual(expectedEdges);
  actual.conductances.forEach((conductance, slot) => {
    expect(Number.isFinite(conductance), `slot ${slot}`).toBe(true);
    expect(conductance, `slot ${slot}`).toBeCloseTo(expectedConductances[slot]!, 6);
  });
}
