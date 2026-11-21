// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Device} from '@luma.gl/core';
import {GPUCommandGraph, GPUGridBinning} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {importGraphBuffer} from '../../analysis/src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../analysis/test/utils/gpu-contributor-test-utils';
import {computePointDensityFieldGpu} from '../src/analysis/catalog-platform3-b-grids';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';
import {
  createPointDensityBoundarySample,
  POINT_DENSITY_BOUNDARY_GRID,
  POINT_DENSITY_BOUNDARY_COUNTS,
  POINT_DENSITY_BOUNDARY_VALUES,
  createPointDensityCatalogPoints,
  createPointDensityNormalizedCatalogPoints,
  POINT_DENSITY_NORMALIZED_CPU_COLUMNS,
  POINT_DENSITY_NORMALIZED_GPU_COLUMNS,
  POINT_DENSITY_CATALOG_VALUES
} from './point-density-grid-coordinate-fixture';

async function readPublishedNormalizedGridCounts(device: Device): Promise<number[]> {
  const graph = new GPUCommandGraph(device);
  const positions = createInputBuffer(device, Float32Array.of(0, 0, 2, 0, 40, 0));
  const counts = createOutputBuffer(device, 41);
  counts.write(new Uint32Array(41).fill(99));
  graph.add(
    new GPUGridBinning({
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 3),
      output: importGraphBuffer(graph, 'counts', counts, 'uint32', 41),
      bounds: [0, 0, 41, 1],
      gridSize: [41, 1]
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return Array.from(await readUint32(counts, 41));
  } finally {
    compiled.destroy();
    positions.destroy();
    counts.destroy();
  }
}

it('catalog GPU follows the published builder within WGSL rounding bounds on a 41-column unit grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runCatalogOperationAsync(
    getCatalogBinding('point-density-grid')!,
    {coordinateSystem: 'planar', cellSize: 1, statistic: 'sum', weightProperty: 'weight'},
    {points: createPointDensityNormalizedCatalogPoints()},
    {device}
  );
  expect(result.engine).toBe('gpu');
  const builderCounts = await readPublishedNormalizedGridCounts(device);
  const builderColumns = builderCounts.flatMap((count, column) => (count ? [column] : []));
  // Independent quotient-error enumeration permits [1,2] for x=2 and [39,40] for x=40.
  // Keep the CPU oracle fixed, and require exact agreement with the public builder on this
  // device. This catches a catalog-only physical-cell floor on the lead's Metal device.
  expect(POINT_DENSITY_NORMALIZED_GPU_COLUMNS).toContainEqual(builderColumns);
  expect(builderCounts).toEqual(
    Array.from({length: 41}, (_, column) => (builderColumns.includes(column) ? 1 : 0))
  );
  console.log('point-density grid rounding', {
    nominalCpuColumns: POINT_DENSITY_NORMALIZED_CPU_COLUMNS,
    publishedBuilderColumns: builderColumns
  });
  const columns = result.tables[0].columns;
  expect(columns['cell_id']).toEqual(builderColumns.map(column => `0:${column}`));
  expect(columns['column']).toEqual(builderColumns);
  expect(columns['row']).toEqual([0, 0, 0]);
  expect(columns['sample_count']).toEqual([1, 1, 1]);
  // Each cell has one exactly represented integer weight: no float32 tolerance is needed.
  expect(columns['value']).toEqual([1, 5, 11]);
});

it.each([
  'count',
  'sum',
  'mean'
] as const)('point-density GPU %s matches independent boundary literals', async statistic => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await computePointDensityFieldGpu(
    {device},
    createPointDensityBoundarySample(),
    POINT_DENSITY_BOUNDARY_GRID,
    statistic,
    true
  );
  // Exact integer weights and at most two additions require no float32 tolerance.
  expect(Array.from(result.counts!)).toEqual(POINT_DENSITY_BOUNDARY_COUNTS);
  expect(Array.from(result.values)).toEqual(POINT_DENSITY_BOUNDARY_VALUES[statistic]);
});

it.each([
  'count',
  'sum',
  'mean'
] as const)('catalog GPU %s publishes literal uploaded boundary cells', async statistic => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runCatalogOperationAsync(
    getCatalogBinding('point-density-grid')!,
    {coordinateSystem: 'planar', cellSize: 1, statistic, weightProperty: 'weight'},
    {points: createPointDensityCatalogPoints()},
    {device}
  );
  expect(result.engine).toBe('gpu');
  const columns = result.tables[0].columns;
  expect(columns['cell_id']).toEqual(['0:0', '0:1']);
  expect(columns['column']).toEqual([0, 1]);
  expect(columns['row']).toEqual([0, 0]);
  expect(columns['sample_count']).toEqual([1, 2]);
  expect(columns['value']).toEqual(POINT_DENSITY_CATALOG_VALUES[statistic]);
});
