// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUSort} from '@luma.gl/graph/gpu-core';
import {getSpatialOperationNodes} from '@luma.gl/graph';

export type PointCloudGrid = {
  origin: readonly [number, number, number];
  dimensions: readonly [number, number, number];
  cellSize: number;
};

/** @internal Exact reusable grid index. Raster mode ignores Z only for the cell key. */
export function createPointCloudGridIndex<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  positions: GraphDataView<'float32x3'>,
  grid: PointCloudGrid,
  raster = false
) {
  assert(grid.origin.every(value => Number.isFinite(Math.fround(value))));
  assert(
    grid.dimensions.every(value => Number.isSafeInteger(value) && value > 0 && value <= 0x1000000)
  );
  assert(Math.fround(grid.cellSize) > 0 && Number.isFinite(Math.fround(grid.cellSize)));
  const cells = grid.dimensions.reduce((product, value) => product * value, 1);
  assert(cells < 0xffffffff && (cells + 1) * 4 <= graph.device.limits.maxStorageBufferBindingSize);
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
  const nodes = getSpatialOperationNodes(graph, {
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
    ...getSpatialOperationNodes(graph, {
      id: `${id}-offsets`,
      length: cells + 1,
      inputs: {sortedKeys},
      outputs: {offsets},
      body: `var lower=0u;var upper=${positions.length}u;while(lower<upper){let middle=lower+(upper-lower)/2u;if(readSortedKeys(middle)<index){lower=middle+1u;}else{upper=middle;}}writeOffsets(index,lower);`
    })
  );
  return {nodes, sortedIndices, offsets, coordinate, inside};
}
