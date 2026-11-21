// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectRetainedCounterClockwiseTriangle} from './concave-hull-directed-oracle';

function createReadback() {
  // Hand orientation: (0,0),(1,0),(0,1) is counterclockwise. Its directed boundary is
  // 0->1, 1->2, 2->0. The asymmetric CORE-103 triangle has the same positive sign.
  return {
    converged: 1,
    capacity: 1,
    triangles: [0, 1, 2],
    triangleValidity: [1],
    edges: [0, 1, 1, 2, 2, 0],
    edgeValidity: [1, 1, 1]
  };
}

it('accepts counterclockwise cyclic corners and permuted directed boundary rows', () => {
  for (const triangles of [
    [0, 1, 2],
    [1, 2, 0],
    [2, 0, 1]
  ]) {
    expect(() =>
      expectRetainedCounterClockwiseTriangle({
        ...createReadback(),
        triangles,
        edges: [2, 0, 0, 1, 1, 2]
      })
    ).not.toThrow();
  }
});

for (const reversed of ['triangle', 'edges', 'both']) {
  it(`rejects reversed ${reversed} in the CORE-103 hull readback`, () => {
    const readback = createReadback();
    if (reversed !== 'edges') readback.triangles = [0, 2, 1];
    if (reversed !== 'triangle') readback.edges = [1, 0, 2, 1, 0, 2];
    expect(() => expectRetainedCounterClockwiseTriangle(readback)).toThrow();
  });
}
