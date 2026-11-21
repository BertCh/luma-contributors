// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectUniformShearEdgeSlots} from './raster-resistance-shear.test-utils';

function createOracle() {
  // East/south/south-east/south-west, four slots per cell. A maps those steps to
  // (1,0), (1,-1), (2,-1), (0,-1); unit resistance gives reciprocal lengths.
  const edges: [number, number][] = [
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
  const conductances = [
    1, 0.7071067811865476, 0.4472135954999579, 0, 0, 0.7071067811865476, 0, 1, 1, 0, 0, 0, 0, 0, 0,
    0
  ];
  return {edges, conductances};
}

it('S2 accepts the independently worked full slot table', () => {
  expect(() => expectUniformShearEdgeSlots(createOracle())).not.toThrow();
});

it.each([
  'extra-self-edge',
  'missing-unused-slot',
  'trailing-conductance',
  'wrong-self-cell',
  'non-finite'
] as const)('S2 rejects %s instead of accepting an incomplete slot assertion', mutation => {
  const actual = createOracle();
  if (mutation === 'extra-self-edge') {
    actual.edges.push([3, 3]);
    actual.conductances.push(0);
  }
  if (mutation === 'missing-unused-slot') {
    actual.edges.pop();
    actual.conductances.pop();
  }
  if (mutation === 'trailing-conductance') actual.conductances.push(0);
  if (mutation === 'wrong-self-cell') actual.edges[3] = [3, 3];
  if (mutation === 'non-finite') actual.conductances[0] = NaN;
  expect(() => expectUniformShearEdgeSlots(actual)).toThrow();
});
