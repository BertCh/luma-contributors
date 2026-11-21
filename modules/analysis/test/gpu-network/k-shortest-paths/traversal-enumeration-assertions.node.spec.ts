// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  expectTraversalPadding,
  expectYenEnumerationStatus,
  type TraversalReadback
} from '../alternative-routes/traversal-enumeration-assertions';

function emptyReadback(): TraversalReadback {
  // Two queries, eight requested routes, five slots per route. Empty queries still own padding.
  return {
    counts: [0, 0],
    lengths: new Array(16).fill(0),
    costs: new Array(16).fill(Infinity),
    vertices: new Array(80).fill(0xffffffff),
    edges: new Array(80).fill(0xffffffff)
  };
}

it('accepts the exact empty two-query layout', () => {
  expect(() => expectTraversalPadding(emptyReadback(), 2, 8, 5)).not.toThrow();
});

it('rejects deleting every padded row of an unreachable query', () => {
  const rows = emptyReadback();
  expect(() =>
    expectTraversalPadding(
      {
        ...rows,
        lengths: new Array(8).fill(0),
        costs: new Array(8).fill(Infinity),
        vertices: new Array(40).fill(0xffffffff),
        edges: new Array(40).fill(0xffffffff)
      },
      2,
      8,
      5
    )
  ).toThrow();
});

for (const field of ['counts', 'lengths', 'costs', 'vertices', 'edges'] as const) {
  for (const extra of [false, true]) {
    it(`rejects ${extra ? 'extra' : 'missing'} ${field} slots`, () => {
      const rows = emptyReadback();
      const values = Array.from(rows[field]);
      if (extra) values.push(values[values.length - 1]);
      else values.pop();
      expect(() => expectTraversalPadding({...rows, [field]: values}, 2, 8, 5)).toThrow();
    });
  }
}

it('rejects discarding the certified second route before candidate overflow', () => {
  // Direct [3] has one spur. Empty capacity-one pool can retain and certify its replacement.
  expect(() =>
    expectYenEnumerationStatus(
      {counts: [1, 0], converged: [0], iterationLimitReached: [0], candidateOverflow: [1]},
      5,
      1
    )
  ).toThrow();
});

it('rejects publishing a second route after the path-length stop', () => {
  expect(() =>
    expectYenEnumerationStatus(
      {counts: [2, 0], converged: [0], iterationLimitReached: [1], candidateOverflow: [0]},
      2,
      64
    )
  ).toThrow();
});

it.each([
  [2, 64],
  [5, 1]
])('rejects conflating both stop flags at length %i and pool %i', (length, capacity) => {
  expect(() =>
    expectYenEnumerationStatus(
      {
        counts: [length === 2 ? 1 : 2, 0],
        converged: [0],
        iterationLimitReached: [1],
        candidateOverflow: [1]
      },
      length,
      capacity
    )
  ).toThrow();
});
