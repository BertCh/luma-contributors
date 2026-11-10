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

/** @internal Exact reusable grid index. Raster mode ignores Z only for the cell key. */
export function createPointCloudGridIndex<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  positions: GraphDataView<'float32x3'>,
  grid: PointCloudGrid,
  raster = false
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
  const coordinate = raster
    ? `vec3f((position.xy-vec2f(${grid.origin.slice(0, 2).map(Math.fround).join(',')}))/${Math.fround(grid.cellSize)},0.0)`
    : `(position-vec3f(${grid.origin.map(Math.fround).join(',')}))/${Math.fround(grid.cellSize)}`;
  const inside = `all(coordinate>=vec3f(0.0))&&all(coordinate<vec3f(${grid.dimensions.map(value => `${value}.0`).join(',')}))`;
  const nodes = getPointCloudKernelNodes(graph, {
    id: `${id}-keys`,
    length: positions.length,
    inputs: {positions},
    outputs: {keys, values},
    body: `let position=readPositions(index);let coordinate=${coordinate};writeValues(index,index);writeKeys(index,0xffffffffu);if(all(abs(position)<=vec3f(3.402823e38))&&${inside}){let cell=vec3u(floor(coordinate));writeKeys(index,(cell.z*${grid.dimensions[1]}u+cell.y)*${grid.dimensions[0]}u+cell.x);}`
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
