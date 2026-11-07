// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import * as arrow from 'apache-arrow';
import {
  MAXIMUM_NEIGHBOR_HOVER_ROWS,
  readPointNeighborPositions
} from '../../../examples/deck/kepler-compute/src/point-neighbor-hover';

function makeTable() {
  return new arrow.Table({
    lng: arrow.vectorFromArray([0, 1, 2, Number.NaN], new arrow.Float64()),
    lat: arrow.vectorFromArray([10, 11, 12, 13], new arrow.Float64()),
    altitude: arrow.vectorFromArray([5, 6, 7, 8], new arrow.Float64()),
    neighbors: arrow.vectorFromArray(
      [[1, 2], [0], [], [0, 3]],
      new arrow.List(new arrow.Field('item', new arrow.Int32()))
    )
  });
}

describe('point neighbor hover', () => {
  it('lists neighbors followed by the hovered row', () => {
    const table = makeTable();
    expect(
      readPointNeighborPositions(table, 0, {lng: 'lng', lat: 'lat', neighbors: 'neighbors'})
    ).toEqual([
      [1, 11, 0],
      [2, 12, 0],
      [0, 10, 0]
    ]);
  });

  it('drops invalid coordinates and reads altitude', () => {
    const table = makeTable();
    expect(
      readPointNeighborPositions(table, 3, {
        lng: 'lng',
        lat: 'lat',
        altitude: 'altitude',
        neighbors: 'neighbors'
      })
    ).toEqual([[0, 10, 5]]);
  });

  it('shows only the hovered row without a neighbors column and bounds the list', () => {
    const table = makeTable();
    expect(readPointNeighborPositions(table, 1, {lng: 'lng', lat: 'lat'})).toEqual([[1, 11, 0]]);
    expect(MAXIMUM_NEIGHBOR_HOVER_ROWS).toBeGreaterThan(0);
  });
});
