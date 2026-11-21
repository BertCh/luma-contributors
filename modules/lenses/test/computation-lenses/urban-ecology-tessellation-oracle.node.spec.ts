// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {expectLocalTessellationQuadrants} from './urban-ecology-tessellation.test-utils';

// Sites (-1,-1), (1,-1), (1,1), (-1,1) have bisectors x=0 and y=0 inside [-2,2]^2.
const QUADRANTS = [
  [-2, -2, 0, -2, 0, 0, -2, 0],
  [0, -2, 2, -2, 2, 0, 0, 0],
  [0, 0, 2, 0, 2, 2, 0, 2],
  [-2, 0, 0, 0, 0, 2, -2, 2]
];

function fixture() {
  const cells = [4, 4, 0, 0, 4, 4, 0, 1, 4, 4, 0, 2, 4, 4, 0, 3];
  const positions = new Array<number>(128).fill(0);
  for (const [piece, vertices] of QUADRANTS.entries()) positions.splice(piece * 32, 8, ...vertices);
  return {cells, positions};
}

test('B4: the oracle accepts every quadrant with cyclic vertex order and permuted piece slots', () => {
  const {cells, positions} = fixture();
  expectLocalTessellationQuadrants(cells, positions);
  const permuted = new Array<number>(128).fill(0);
  const reordered: number[] = [];
  for (const [piece, sample] of [2, 0, 3, 1].entries()) {
    const vertices = QUADRANTS[sample];
    permuted.splice(piece * 32, 8, ...vertices.slice(2), ...vertices.slice(0, 2));
    reordered.push(4, 4, 0, sample);
  }
  expectLocalTessellationQuadrants(reordered, permuted);
  cells[0] = 5;
  cells[1] = 5;
  positions.splice(0, 10, -2, -2, -2, -2, 0, -2, 0, 0, -2, 0);
  expectLocalTessellationQuadrants(cells, positions);
});

test('B4: four copies of one quadrant cannot certify the enclosure union', () => {
  const {cells, positions} = fixture();
  for (let piece = 0; piece < 4; piece++) positions.splice(piece * 32, 8, ...QUADRANTS[0]);
  expect(() => expectLocalTessellationQuadrants(cells, positions)).toThrow();
});

test.each([-1, 1])('B4: changing both buffer scalar lengths by %s fails', difference => {
  const {cells, positions} = fixture();
  const changedCells = difference < 0 ? cells.slice(0, -1) : [...cells, 0];
  const changedPositions = difference < 0 ? positions.slice(0, -1) : [...positions, 0];
  expect(() => expectLocalTessellationQuadrants(changedCells, changedPositions)).toThrow();
});

test('B4: duplicated sample IDs and non-finite active vertices fail', () => {
  const {cells, positions} = fixture();
  cells[7] = 0;
  expect(() => expectLocalTessellationQuadrants(cells, positions)).toThrow();
  cells[7] = 1;
  positions[0] = NaN;
  expect(() => expectLocalTessellationQuadrants(cells, positions)).toThrow();
});
