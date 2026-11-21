// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {relaxTerrainFieldRound} from '../../../src/gpu-terrain/hydrology/cpu-terrain-flow-operation';

it('R9 downward phase crosses frozen tile boundaries from a controlled intermediate state', () => {
  const width = 3,
    height = 64;
  const surface = new Float64Array(width * height);
  const eligible = new Uint8Array(width * height);
  for (let row = 0; row < height; row++) eligible[row * width + 1] = 1;
  const initial = new Float64Array(width * height).fill(Infinity);
  initial[1] = 0;
  const frozen = initial.slice();
  const upward = initial.slice();
  const swept = initial.slice();
  // Only the top cell is initially visible. This fixture distinguishes down from tiled/up;
  // the corner fixture below separately distinguishes down from both horizontal sweeps.
  relaxTerrainFieldRound(frozen, surface, eligible, width, 1, 96, true);
  relaxTerrainFieldRound(upward, surface, eligible, width, 1, 100, true);
  relaxTerrainFieldRound(swept, surface, eligible, width, 1, 99, true);
  // A width-one corridor has exactly row unit-cost edges from its sole seeded outlet.
  // A tiled replacement cannot even reach the second tile.
  expect(frozen[32 * width + 1]).toBe(Infinity);
  expect(upward[32 * width + 1]).toBe(Infinity);
  for (let row = 0; row < height; row++) {
    expect(Number.isFinite(swept[row * width + 1])).toBe(true);
    expect(swept[row * width + 1]).toBe(row);
  }
});

it('B3 downward phase crosses a tile corner that horizontal sweeps cannot finish', () => {
  const width = 32;
  // Coordinates are (column,row): the only path runs NE from (14,17) through
  // (15,16) to (16,15), crossing both a tile-column and tile-row boundary.
  const cells = [17 * width + 14, 16 * width + 15, 15 * width + 16];
  const surface = new Float64Array(width * width);
  const eligible = new Uint8Array(surface.length);
  for (const cell of cells) eligible[cell] = 1;
  const initial = new Float64Array(surface.length).fill(Infinity);
  initial[cells[0]] = 0;
  for (const [iteration, expected] of [
    [97, [0, 1, Infinity]],
    [98, [0, 1, Infinity]],
    [99, [0, 1, 2]]
  ] as const) {
    const values = initial.slice();
    relaxTerrainFieldRound(values, surface, eligible, width, 1, iteration, true);
    // Each eligible neighbor costs one; an unavailable halo cannot publish the last edge.
    expect(cells.map(cell => values[cell])).toEqual(expected);
  }
});
