// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getSpatialAnalysisPackedKernelNodes} from '../geographic-distribution/spatial-analysis-packed-kernel';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import {createPointNeighborGridIndexNodes} from '../../analysis-core/kernels/point-neighbor-grid-index';
import {GET_ORDERED_KEY_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Bounds the adaptive cell table to about 4 MiB even when the sample capacity is in the millions. */
const BUILDING_TESSELLATION_INDEX_MAXIMUM_DIMENSION = 1024;

/** A coordinate-rank grid and two sorted coordinate orders share one binding. @internal */
export type BuildingTessellationIndex = {
  data: GraphDataView<'uint32'>;
  rowCount: number;
  dimension: number;
  rowsPerBin: number;
};

/** Exact packed layout, also used by preflight before declaring any resources. @internal */
export function getBuildingTessellationIndexShape(rowCount: number) {
  const dimension = Math.min(
    BUILDING_TESSELLATION_INDEX_MAXIMUM_DIMENSION,
    Math.max(1, Math.ceil(Math.sqrt(rowCount)))
  );
  const cellCount = dimension * dimension;
  return {
    dimension,
    cellCount,
    rowsPerBin: Math.ceil(rowCount / dimension),
    byteLength: (5 * rowCount + cellCount + 1) * 4
  };
}

/**
 * Builds a rank grid: each axis bin holds a bounded slice of the coordinate order, so distant
 * outliers consume ranks instead of stretching every cell. The shorter sorted interval provides
 * a second exact search when correlated coordinates concentrate points in a few rank cells.
 * @internal
 */
export function addBuildingTessellationIndex<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {id: string; positions: GraphDataView<'float32x2'>; count: GraphDataView<'uint32x4'>}
): BuildingTessellationIndex {
  const {id, positions, count} = props;
  const rowCount = positions.length;
  const {dimension, cellCount, rowsPerBin, byteLength} =
    getBuildingTessellationIndexShape(rowCount);
  const data = createTransientView(graph, `${id}-packed`, 'uint32', byteLength / 4);
  const keys = createTransientView(graph, `${id}-keys`, 'uint32', rowCount);
  const rows = createTransientView(graph, `${id}-rows`, 'uint32', rowCount);
  // The two fixed axes reuse scratch; dispatch/pipeline count never depends on building count.
  for (const axis of [0, 1]) {
    // Separate outputs avoid overlapping aligned sort bindings for small or odd capacities.
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys-${axis}`, 'uint32', rowCount);
    const sortedRows = createTransientView(graph, `${id}-sorted-rows-${axis}`, 'uint32', rowCount);
    graph.add(
      getSpatialAnalysisPackedKernelNodes(graph, {
        id: `${id}-prepare-${axis}`,
        operation: 'BuildingTessellationIndex',
        invocationCount: rowCount,
        declarations: GET_ORDERED_KEY_WGSL + IS_FINITE_VALUE_WGSL,
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'count', view: count, type: 'u32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'rows', view: rows, type: 'u32', access: 'read_write'}
        ],
        body: `let point=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
let valid=index<count[countOffset]&&isFiniteValue(point.x)&&isFiniteValue(point.y);
keys[keysOffset+index]=select(0xffffffffu,getOrderedKey(point[${axis}u]),valid);
rows[rowsOffset+index]=index;`
      })
    );
    graph.add(
      new GPUSort({
        id: `${id}-sort-${axis}`,
        keys,
        values: rows,
        outputKeys: sortedKeys,
        outputValues: sortedRows
      }).getCommandNodes(graph)
    );
    graph.add(
      getSpatialAnalysisPackedKernelNodes(graph, {
        id: `${id}-pack-${axis}`,
        operation: 'BuildingTessellationIndex',
        invocationCount: rowCount,
        bindings: [
          {name: 'keys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'rows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'pointIndex', view: data, type: 'u32', access: 'read_write'}
        ],
        body: `pointIndex[pointIndexOffset+${2 * axis * rowCount}u+index]=keys[keysOffset+index];
pointIndex[pointIndexOffset+${(2 * axis + 1) * rowCount}u+index]=rows[rowsOffset+index];`
      })
    );
  }
  const bins = createTransientView(graph, `${id}-bins`, 'uint32x2', rowCount);
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-ranks`,
      operation: 'BuildingTessellationIndex',
      invocationCount: rowCount,
      bindings: [
        {name: 'pointIndex', view: data, type: 'u32', access: 'read'},
        {name: 'bins', view: bins, type: 'u32', access: 'read_write'}
      ],
      body: `let horizontalRow=pointIndex[pointIndexOffset+${rowCount}u+index];
let verticalRow=pointIndex[pointIndexOffset+${3 * rowCount}u+index];
bins[binsOffset+2u*horizontalRow]=index/${rowsPerBin}u;
bins[binsOffset+2u*verticalRow+1u]=index/${rowsPerBin}u;`
    })
  );
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-cells`,
      operation: 'BuildingTessellationIndex',
      invocationCount: rowCount,
      bindings: [
        {name: 'bins', view: bins, type: 'u32', access: 'read'},
        {name: 'count', view: count, type: 'u32', access: 'read'},
        {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
        {name: 'rows', view: rows, type: 'u32', access: 'read_write'}
      ],
      body: `keys[keysOffset+index]=select(${cellCount}u,bins[binsOffset+2u*index]+${dimension}u*bins[binsOffset+2u*index+1u],index<count[countOffset]);
rows[rowsOffset+index]=index;`
    })
  );
  const grid = createPointNeighborGridIndexNodes(graph, {
    id: `${id}-grid`,
    rowCount,
    cellKeys: keys,
    sourceRows: rows,
    cellCount
  });
  graph.add(grid.nodes);
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-pack-grid`,
      operation: 'BuildingTessellationIndex',
      invocationCount: Math.max(rowCount, cellCount + 1),
      bindings: [
        {name: 'offsets', view: grid.cellOffsets, type: 'u32', access: 'read'},
        {name: 'rows', view: grid.sortedRows, type: 'u32', access: 'read'},
        {name: 'pointIndex', view: data, type: 'u32', access: 'read_write'}
      ],
      body: `if(index<${cellCount + 1}u){pointIndex[pointIndexOffset+${4 * rowCount}u+index]=offsets[offsetsOffset+index];}
if(index<${rowCount}u){pointIndex[pointIndexOffset+${4 * rowCount + cellCount + 1}u+index]=rows[rowsOffset+index];}`
    })
  );
  return {data, rowCount, dimension, rowsPerBin};
}

/** Shared exact inclusive search bounds. Both signs of zero are included at numeric endpoints. @internal */
export function getBuildingTessellationIndexDeclarations(index: BuildingTessellationIndex): string {
  const {rowCount} = index;
  return (
    GET_ORDERED_KEY_WGSL +
    getBinarySearchWGSL('pointIndex') +
    `
fn getCandidateRanges(center:vec2<f32>,radius:f32)->vec4<u32>{
 let lower=center-vec2<f32>(radius);let upper=center+vec2<f32>(radius);
 // Infinite endpoints are valid ordered search bounds; NaNs are excluded by the caller.
 let horizontalFirst=lowerBoundPointIndex(0u,${rowCount}u,select(getOrderedKey(lower.x),0x7fffffffu,lower.x==0.0));
 let horizontalEnd=upperBoundPointIndex(0u,${rowCount}u,select(getOrderedKey(upper.x),0x80000000u,upper.x==0.0));
 let verticalFirst=lowerBoundPointIndex(${2 * rowCount}u,${3 * rowCount}u,select(getOrderedKey(lower.y),0x7fffffffu,lower.y==0.0));
 let verticalEnd=upperBoundPointIndex(${2 * rowCount}u,${3 * rowCount}u,select(getOrderedKey(upper.y),0x80000000u,upper.y==0.0));
 return vec4<u32>(horizontalFirst,horizontalEnd,verticalFirst-${2 * rowCount}u,verticalEnd-${2 * rowCount}u);
}
fn getCandidateInterval(center:vec2<f32>,radius:f32)->vec3<u32>{
 let ranges=getCandidateRanges(center,radius);
 if(ranges.w-ranges.z<ranges.y-ranges.x){return vec3<u32>(ranges.z+${2 * rowCount}u,ranges.w+${2 * rowCount}u,${rowCount}u);}
 return vec3<u32>(ranges.x,ranges.y,${rowCount}u);
}
`
  );
}

/** Candidate rows from adaptive rank cells; queries never depend on the global coordinate extent. @internal */
export function getBuildingTessellationGridWalk(
  index: BuildingTessellationIndex,
  body: string,
  beforeRow: string
): string {
  const {dimension, rowsPerBin, rowCount} = index;
  const offsetBase = 4 * rowCount;
  const rowBase = offsetBase + dimension * dimension + 1;
  return `let ranges=getCandidateRanges(center,searchRadius);
 if(ranges.x==ranges.y||ranges.z==ranges.w){return vec2<u32>(0u);}
 let lower=vec2<u32>(ranges.x,ranges.z)/${rowsPerBin}u;
 let upper=(vec2<u32>(ranges.y,ranges.w)-vec2<u32>(1u))/${rowsPerBin}u;
 for(var gridRow=lower.y;gridRow<=upper.y;gridRow++){
  for(var gridColumn=lower.x;gridColumn<=upper.x;gridColumn++){
   let cell=gridRow*${dimension}u+gridColumn;
   let first=pointIndex[pointIndexOffset+${offsetBase}u+cell];
   let end=pointIndex[pointIndexOffset+${offsetBase + 1}u+cell];
   for(var slot=first;slot<end;slot++){
    ${beforeRow}
    let candidate=pointIndex[pointIndexOffset+${rowBase}u+slot];
    ${body}
   }
  }
 }`;
}
