// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {createNullWebGPUDevice} from '@luma-contributors/analysis/testing';
import {GPUKeplerAggregation} from '../src/aggregation/gpu-kepler-aggregation';
import {
  computePointDensityField,
  computePointDensityFieldGpu
} from '../src/analysis/catalog-platform3-b-grids';
import {getCatalogBinding, runCatalogOperation} from '../src/analysis/catalog-bindings';
import {
  createPointDensityBoundarySample,
  POINT_DENSITY_BOUNDARY_GRID,
  POINT_DENSITY_BOUNDARY_COUNTS,
  POINT_DENSITY_BOUNDARY_VALUES,
  createPointDensityCatalogPoints,
  createPointDensityNormalizedCatalogPoints,
  POINT_DENSITY_NORMALIZED_CPU_COLUMNS,
  POINT_DENSITY_CATALOG_VALUES
} from './point-density-grid-coordinate-fixture';

const WIDE_UNIT = 2 ** 122;
const WIDE_GRID = {minX: -55 * WIDE_UNIT, minY: 0, cellSize: 5 * WIDE_UNIT, columns: 22, rows: 1};
const REVIEW_UNIT = 2 ** 124;
const REVIEW_GRID = {minX: -2 * REVIEW_UNIT, minY: 0, cellSize: REVIEW_UNIT, columns: 3, rows: 1};
const REVIEW_SAMPLE = {
  count: 3,
  x: Float64Array.of(-1, 0, 1),
  y: new Float64Array(3),
  value: Float64Array.of(1, 2, 4),
  sourceIndex: Int32Array.of(0, 1, 2),
  partIndex: new Int32Array(3),
  droppedCount: 0
};

it('catalog CPU puts the review grid edge at zero and counts the points on either side', () => {
  // The edges are -2u, -u, 0 and u. Zero belongs to column 2, so columns 1 and 2
  // get weights [1] and [2, 4] respectively. The old width limit admits this grid (3u < 4u).
  const sums = computePointDensityField(REVIEW_SAMPLE, REVIEW_GRID, 'sum');
  const means = computePointDensityField(REVIEW_SAMPLE, REVIEW_GRID, 'mean');
  expect(Array.from(sums.counts!)).toEqual([0, 1, 2]);
  expect(Array.from(sums.values)).toEqual([0, 1, 6]);
  expect(Array.from(means.values)).toEqual([0, 1, 3]);
});

it('catalog GPU refuses the review grid even though its extent is below 2^126', async () => {
  const device = createNullWebGPUDevice();
  try {
    await expect(
      computePointDensityFieldGpu({device}, REVIEW_SAMPLE, REVIEW_GRID, 'mean', true)
    ).rejects.toThrow(/catalog-point-density bounds.*below 2\^125/);
  } finally {
    device.destroy();
  }
});

it('resident GPU refuses the review grid even though its extent is below 2^126', () => {
  const device = createNullWebGPUDevice();
  const positions = makeReviewVector(device, 'float32x2', [-1, 0, 0, 0, 1, 0]);
  const weights = makeReviewVector(device, 'float32', [1, 2, 4]);
  try {
    expect(() => {
      const aggregation = new GPUKeplerAggregation({
        device,
        positions,
        weights,
        statistic: 'mean',
        topology: {
          kind: 'grid',
          origin: [-2 * REVIEW_UNIT, 0],
          cellSize: [REVIEW_UNIT, REVIEW_UNIT],
          gridSize: [3, 1]
        }
      });
      aggregation.destroy();
    }).toThrow(/kepler-aggregation bounds.*below 2\^125/);
  } finally {
    weights.destroy();
    positions.destroy();
    device.destroy();
  }
});

function makeReviewVector<Format extends 'float32x2' | 'float32'>(
  device: ReturnType<typeof createNullWebGPUDevice>,
  format: Format,
  values: readonly number[]
): GPUVector<Format> {
  return new GPUVector({
    type: 'data',
    name: format,
    format,
    ownsData: true,
    data: [
      new GPUData({
        format,
        ownsBuffer: true,
        length: values.length / (format === 'float32x2' ? 2 : 1),
        buffer: device.createBuffer({data: Float32Array.from(values), usage: Buffer.STORAGE})
      })
    ]
  });
}

it.each([
  // Edge 13 is -55t + 13 * 5t = 10t, with neighbors one float32 step to either side.
  {grid: WIDE_GRID, edgeColumn: 13, edgeBits: 0x7e200000},
  // Edge 2 is -13t + 2 * 7t = t. The finite float32 width 21t still needs exact edges.
  {
    grid: {minX: -13 * WIDE_UNIT, minY: 0, cellSize: 7 * WIDE_UNIT, columns: 3, rows: 1},
    edgeColumn: 2,
    edgeBits: 0x7c800000
  }
])('catalog CPU places a wide grid edge and its neighbors at column $edgeColumn', ({
  grid,
  edgeColumn,
  edgeBits
}) => {
  const positions = new Float32Array(Uint32Array.of(edgeBits - 1, edgeBits, edgeBits + 1).buffer);
  const sample = {
    count: 3,
    x: Float64Array.from(positions),
    y: new Float64Array(3),
    value: Float64Array.of(1, 2, 4),
    sourceIndex: Int32Array.of(0, 1, 2),
    partIndex: new Int32Array(3),
    droppedCount: 0
  };
  const result = computePointDensityField(sample, grid, 'sum');
  // The tie and upper neighbor share a column, with weights 2 and 4.
  const counts = new Uint32Array(grid.columns);
  counts.set([1, 2], edgeColumn - 1);
  const sums = new Float64Array(grid.columns);
  sums.set([1, 6], edgeColumn - 1);
  expect(result.counts).toEqual(counts);
  expect(result.values).toEqual(sums);
});

it('catalog GPU rejects a float32 grid extent of 2^126 before starting the session', async () => {
  const device = createNullWebGPUDevice();
  try {
    await expect(
      computePointDensityFieldGpu(
        {device},
        createPointDensityBoundarySample(),
        {minX: 0, minY: 0, cellSize: 2 ** 125, columns: 2, rows: 1},
        'count',
        false
      )
    ).rejects.toThrow(/catalog-point-density bounds.*below 2\^125/);
  } finally {
    device.destroy();
  }
});

it('resident GPU aggregation rejects wider grids before building the graph', () => {
  const device = createNullWebGPUDevice();
  const buffer = device.createBuffer({data: Float32Array.of(0, 0), usage: Buffer.STORAGE});
  const positions = new GPUVector({
    type: 'data',
    name: 'wide-grid-positions',
    format: 'float32x2',
    ownsData: false,
    data: [new GPUData({buffer, format: 'float32x2', length: 1, ownsBuffer: false})]
  });
  try {
    expect(() => {
      const aggregation = new GPUKeplerAggregation({
        device,
        positions,
        topology: {
          kind: 'grid',
          origin: [WIDE_GRID.minX, WIDE_GRID.minY],
          cellSize: [WIDE_GRID.cellSize, WIDE_GRID.cellSize],
          gridSize: [WIDE_GRID.columns, WIDE_GRID.rows]
        }
      });
      aggregation.destroy();
    }).toThrow(/kepler-aggregation bounds.*below 2\^125/);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('catalog CPU uses nominal round-to-nearest binary32 arithmetic on a 41-column unit grid', () => {
  // The plausible upload-only fix still assigns x=2 to column 2.
  expect(Math.floor(Math.fround(2) / 1)).toBe(2);
  const result = runCatalogOperation(
    getCatalogBinding('point-density-grid')!,
    {coordinateSystem: 'planar', cellSize: 1, statistic: 'sum', weightProperty: 'weight'},
    {points: createPointDensityNormalizedCatalogPoints()}
  );
  const columns = result.tables[0].columns;
  expect(columns['cell_id']).toEqual(['0:0', '0:1', '0:40']);
  expect(columns['column']).toEqual(POINT_DENSITY_NORMALIZED_CPU_COLUMNS);
  expect(columns['row']).toEqual([0, 0, 0]);
  expect(columns['sample_count']).toEqual([1, 1, 1]);
  expect(columns['value']).toEqual([1, 5, 11]);
});

it('a WGSL-permitted quotient error explains column 39 without changing the CPU oracle', () => {
  // Independently enumerated in grid-coordinate-bump.oracle.py: this quotient is within
  // 2.5 ULP of exact 40/41. It is an example of a permitted evaluation, not a claim about
  // Metal's instruction sequence. The lead observed column 39 on hardware.
  const quotient = 0.9756096601486206;
  const quotientUnit = 2 ** -24;
  expect(Math.abs(quotient - 40 / 41) / quotientUnit).toBeLessThanOrEqual(2.5);
  expect(Math.fround(quotient * 41)).toBe(39.999996185302734);
  expect(Math.trunc(Math.fround(quotient * 41))).toBe(39);
  expect(Math.trunc(Math.fround(Math.fround(40 / 41) * 41))).toBe(40);
});

it('the boundary sample uses the actual inclusive top-right corner instead of interior y=2', () => {
  // Independent permitted division: this quotient is 2/3 ULP below exact 2/3.
  // In the old (4,2) fixture it puts weight 10 in row 1, column 3: flat cell 7.
  const quotient = 0.6666666269302368;
  expect(Math.abs(quotient - 2 / 3) / 2 ** -24).toBeLessThanOrEqual(2.5);
  expect(Math.fround(quotient * 3)).toBe(1.9999998807907104);
  expect(Math.trunc(Math.fround(quotient * 3)) * 4 + 3).toBe(7);
  const sample = createPointDensityBoundarySample();
  const grid = POINT_DENSITY_BOUNDARY_GRID;
  expect(sample.x[2]).toBe(grid.minX + grid.columns * grid.cellSize);
  expect(sample.y[2]).toBe(grid.minY + grid.rows * grid.cellSize);
});

it.each([
  'count',
  'sum',
  'mean'
] as const)('point-density CPU %s bins uploaded coordinates and rejects outside rows', statistic => {
  const result = computePointDensityField(
    createPointDensityBoundarySample(),
    POINT_DENSITY_BOUNDARY_GRID,
    statistic
  );
  expect(Array.from(result.counts!)).toEqual(POINT_DENSITY_BOUNDARY_COUNTS);
  expect(Array.from(result.values)).toEqual(POINT_DENSITY_BOUNDARY_VALUES[statistic]);
});

it.each([
  'count',
  'sum',
  'mean'
] as const)('catalog CPU %s publishes uploaded boundary cells', statistic => {
  const result = runCatalogOperation(
    getCatalogBinding('point-density-grid')!,
    {coordinateSystem: 'planar', cellSize: 1, statistic, weightProperty: 'weight'},
    {points: createPointDensityCatalogPoints()}
  );
  const columns = result.tables[0].columns;
  expect(columns['cell_id']).toEqual(['0:0', '0:1']);
  expect(columns['column']).toEqual([0, 1]);
  expect(columns['row']).toEqual([0, 0]);
  expect(columns['sample_count']).toEqual([1, 2]);
  expect(columns['value']).toEqual(POINT_DENSITY_CATALOG_VALUES[statistic]);
});
