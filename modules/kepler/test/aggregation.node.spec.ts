// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getKeplerAggregationLayout} from '../src/aggregation';
import {
  getGPUPointDensityD3HexagonCell,
  getGPUPointDensityHexagonCell
} from '@luma-contributors/graph/gpu-spatial-analysis';
import oracle from './fixtures/d3-hexbin-oracle.json';

it('optional d3 hex assignment matches an independent d3-hexbin fixture including interior disagreements and ties', () => {
  let cubeDisagreements = 0;
  for (const [x, y, column, row] of oracle.cases) {
    expect(getGPUPointDensityD3HexagonCell(x, y, 0, 0, 1)).toEqual([column, row]);
    const cube = getGPUPointDensityHexagonCell(x, y, 0, 0, 1);
    if (cube[0] !== column || cube[1] !== row) cubeDisagreements++;
  }
  expect(cubeDisagreements).toBeGreaterThan(3);
  const [halfTieX, halfTieY] = oracle.float32HalfTie.point;
  expect(
    getGPUPointDensityD3HexagonCell(halfTieX, halfTieY, 0, 0, oracle.float32HalfTie.radius)
  ).toEqual(oracle.float32HalfTie.cell);
  expect(getGPUPointDensityD3HexagonCell(5.209384176131452, 3.932550949642261, 0, 0, 1)).toEqual([
    3, 3
  ]);
  expect(getGPUPointDensityHexagonCell(5.209384176131452, 3.932550949642261, 0, 0, 1)).toEqual([
    3, 2
  ]);
});

it('aggregation layout keeps the explicit grid origin and rectangular cell dimensions', () => {
  expect(
    getKeplerAggregationLayout({
      kind: 'grid',
      origin: [-10, 20],
      cellSize: [2, 3],
      gridSize: [4, 5]
    })
  ).toEqual({
    kind: 'grid',
    bounds: [-10, 20, -2, 35],
    cellSize: [2, 3],
    gridSize: [4, 5],
    cellCount: 20
  });
});

it('hex layout reserves a stable odd-r lattice over inclusive clipping bounds', () => {
  const layout = getKeplerAggregationLayout({kind: 'hexagon', bounds: [0, 0, 8, 6], radius: 2});
  expect(layout).toEqual({
    kind: 'hexagon',
    bounds: [0, 0, 8, 6],
    radius: 2,
    gridSize: [4, 3],
    cellCount: 12
  });
});

it('aggregation rejects nonfinite, empty, fractional and oversized fixed topology', () => {
  for (const gridSize of [
    [0, 1],
    [1.5, 2],
    [65536, 65536]
  ] as const) {
    expect(() =>
      getKeplerAggregationLayout({kind: 'grid', origin: [0, 0], cellSize: [1, 1], gridSize})
    ).toThrow();
  }
  expect(() =>
    getKeplerAggregationLayout({
      kind: 'grid',
      origin: [Infinity, 0],
      cellSize: [1, 1],
      gridSize: [1, 1]
    })
  ).toThrow();
  expect(() =>
    getKeplerAggregationLayout({kind: 'hexagon', bounds: [0, 0, 1, 1], radius: 0})
  ).toThrow();
  expect(() =>
    getKeplerAggregationLayout({
      kind: 'grid',
      origin: [1e9, 0],
      cellSize: [1, 1],
      gridSize: [100, 1]
    })
  ).toThrow(/rebase/);
});
