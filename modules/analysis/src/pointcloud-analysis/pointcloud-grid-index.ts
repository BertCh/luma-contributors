// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUSort} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateGPUStorageBindingByteLength,
  validateMaximumCount,
  validateRequirement
} from '../analysis-core/kernels/gpu-contributor-utils';
import {getPointCloudKernelNodes} from './pointcloud-kernel-nodes';
import {
  GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS,
  GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS
} from './pointcloud-limits';

export type PointCloudGrid = {
  origin: readonly [number, number, number];
  dimensions: readonly [number, number, number];
  cellSize: number;
};

/**
 * Throws unless a 3-D search grid is usable: finite origin and cell size, integer axis counts up to
 * `GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS` and a total below `GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS`.
 *
 * @param id Contributor ID, for the message.
 * @param name Prop name of the grid, for the message.
 * @param grid Grid to check.
 * @internal
 */
export function validatePointCloudSearchGrid(id: string, name: string, grid: PointCloudGrid): void {
  validateRequirement(
    id,
    grid.origin.every(value => Number.isFinite(Math.fround(value))),
    `${name}.origin must hold finite numbers`
  );
  for (const [axis, dimension] of grid.dimensions.entries()) {
    validateCapacity(
      id,
      `${name}.dimensions[${axis}]`,
      dimension,
      1,
      GPU_POINT_CLOUD_MAXIMUM_GRID_AXIS_CELLS
    );
  }
  validateFiniteNumber(id, `${name}.cellSize`, grid.cellSize);
  validateRequirement(
    id,
    Math.fround(grid.cellSize) > 0,
    `${name}.cellSize must be positive in float32`
  );
  validateMaximumCount(
    id,
    `${name} cell count`,
    grid.dimensions.reduce((product, value) => product * value, 1),
    GPU_POINT_CLOUD_MAXIMUM_GRID_CELLS
  );
}

/** The row direction of a raster grid: its `rowDirection`, `'south'` when the prop is left out. */
export function getPointCloudRowDirection(grid: {
  rowDirection?: 'north' | 'south';
}): 'north' | 'south' {
  return grid.rowDirection ?? 'south';
}

/**
 * @internal Exact reusable grid index. Raster mode ignores Z only for the cell key; `rowDirection`
 * `'south'` (raster mode only) counts rows down from the top edge.
 *
 * This stays a local index because the published `GPUGridIndex` (`@luma.gl/graph/gpu-core`) cannot
 * serve the five callers. Missing, to be requested upstream:
 * 1. A cell size. `GPUGridIndex` takes `bounds` and `gridSize` and divides the extent, so cell
 *    edges are `(max - min) / gridSize`, not `origin + k * cellSize` in float32. Point-cloud
 *    results (search cells, SMRF and surface rasters, the `cellIndex == index` check of the raster
 *    kernels) depend on the exact float32 cell of every point.
 * 2. A half-open domain. It accepts points on the upper bound; the callers drop them.
 * 3. Ascending point order inside a cell. IDs within a cell are unspecified (atomic scatter); the
 *    neighbour walk breaks distance ties by row ID and sums floats in cell order, so it needs the
 *    stable sort. The GWR family adds a rank sort for the same reason (`gwr-grid-index.ts`).
 * 4. A 3-D grid over a raster mode that ignores Z, and float32x3 positions with a 2-D grid.
 * 5. Grids above 2^24 cells per axis or with cell offsets larger than the caller's storage
 *    binding are rejected here with a named limit; the published class names none.
 * 6. The offsets and sorted IDs as plain read-only views for the caller's own kernels (it needs
 *    `count` and `overflow` rows and a caller-sized `objectIds` capacity).
 */
export function createPointCloudGridIndex<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  positions: GraphDataView<'float32x3'>,
  grid: PointCloudGrid,
  raster = false,
  rowDirection: 'north' | 'south' = 'north'
) {
  validatePointCloudSearchGrid(id, 'grid', grid);
  const cells = grid.dimensions.reduce((product, value) => product * value, 1);
  // The cell offsets hold one uint32 per cell plus one closing entry.
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'grid cell offsets',
    (cells + 1) * 4,
    'reduce grid.dimensions or increase grid.cellSize'
  );
  const allocate = (name: string, length = positions.length) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const keys = allocate('keys');
  const values = allocate('values');
  const sortedKeys = allocate('sorted-keys');
  const sortedIndices = allocate('sorted-indices');
  const offsets = allocate('offsets', cells + 1);
  const south = raster && rowDirection === 'south';
  // A raster point's coordinate is its offset from the origin in cell units in both directions, so
  // membership is the same half-open test. Rows of a south raster count down from the top edge:
  // row height - 1 - floor(y), in integers, the arithmetic of getPointCloudCellSource, so the key
  // of a point equals its raster cell index (`height - y` in float32 would round up just below a
  // row boundary).
  const rasterOffset = `(position.xy-vec2f(${grid.origin.slice(0, 2).map(Math.fround).join(',')}))/${Math.fround(grid.cellSize)}`;
  const coordinate = raster
    ? `vec3f(${rasterOffset},0.0)`
    : `(position-vec3f(${grid.origin.map(Math.fround).join(',')}))/${Math.fround(grid.cellSize)}`;
  const bounds = `vec3f(${grid.dimensions.map(value => `${value}.0`).join(',')})`;
  const inside = `all(coordinate>=vec3f(0.0))&&all(coordinate<${bounds})`;
  const cell = south
    ? `vec3u(u32(floor(coordinate.x)),${grid.dimensions[1] - 1}u-u32(floor(coordinate.y)),u32(floor(coordinate.z)))`
    : 'vec3u(floor(coordinate))';
  const nodes = getPointCloudKernelNodes(graph, {
    id: `${id}-keys`,
    length: positions.length,
    inputs: {positions},
    outputs: {keys, values},
    body: `let position=readPositions(index);let coordinate=${coordinate};writeValues(index,index);writeKeys(index,0xffffffffu);if(all(abs(position)<=vec3f(3.402823e38))&&${inside}){let cell=${cell};writeKeys(index,(cell.z*${grid.dimensions[1]}u+cell.y)*${grid.dimensions[0]}u+cell.x);}`
  });
  nodes.push(
    ...new GPUSort({
      id: `${id}-sort`,
      keys,
      values,
      outputKeys: sortedKeys,
      outputValues: sortedIndices,
      // Reserve a distinct largest masked key for the unchanged 0xffffffff invalid-point sentinel.
      keyBits: Math.ceil(Math.log2(cells + 1))
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...getPointCloudKernelNodes(graph, {
      id: `${id}-offsets`,
      length: cells + 1,
      inputs: {sortedKeys},
      outputs: {offsets},
      body: `var lower=0u;var upper=${positions.length}u;while(lower<upper){let middle=lower+(upper-lower)/2u;if(readSortedKeys(middle)<index){lower=middle+1u;}else{upper=middle;}}writeOffsets(index,lower);`
    })
  );
  return {nodes, sortedIndices, offsets, coordinate, inside};
}
