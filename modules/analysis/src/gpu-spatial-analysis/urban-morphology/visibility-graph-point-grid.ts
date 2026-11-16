// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createPointNeighborGridIndexNodes} from '../../analysis-core/kernels/point-neighbor-grid-index';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateUrbanRecipeBindings,
  validateUrbanStorageByteLength
} from './building-tessellation-validation';
import {createRecipeKernelNode} from '../recipes/expansion-recipe-utils';

/** Bounds the cell table and the neighbourhood traversal, independently of input row count. */
const URBAN_POINT_GRID_MAXIMUM_DIMENSION = 256;

/** Four float32 bound words precede the cell offsets and sorted input rows in one binding. */
const URBAN_POINT_GRID_BOUND_WORD_COUNT = 4;

/** Packed point-neighbour index shared by urban visibility and Voronoi clipping. @internal */
export type UrbanPointGrid = {
  dimension: number;
  rowOffset: number;
  data: GraphDataView<'uint32'>;
};

/** Exact shape used by allocation and callers' preflight checks. @internal */
export function getUrbanPointGridShape(rowCount: number) {
  const dimension = Math.min(
    URBAN_POINT_GRID_MAXIMUM_DIMENSION,
    Math.max(1, Math.ceil(Math.sqrt(rowCount)))
  );
  const rowOffset = URBAN_POINT_GRID_BOUND_WORD_COUNT + dimension * dimension + 1;
  return {dimension, rowOffset, byteLength: (rowOffset + rowCount) * 4};
}

/**
 * Builds the shared published point grid with a GPU-derived per-frame extent. Packs its bounds,
 * offsets and row IDs into one binding so ray kernels still fit eight storage bindings with a BVH.
 * Non-finite points and the unused tail of a sampled boundary are excluded from the index.
 * @internal
 */
export function addUrbanPointGrid<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    positions: GraphDataView<'float32x2'>;
    /** Optional first word is the number of populated leading positions. */
    count?: GraphDataView<'uint32x4'>;
  }
): UrbanPointGrid {
  const {id, positions} = props;
  const rowCount = positions.length;
  const {dimension, rowOffset} = getUrbanPointGridShape(rowCount);
  const cellCount = dimension * dimension;
  validateUrbanRecipeBindings(graph, id, {positions, count: props.count}, []);
  // Validate before creating any scratch or command nodes.
  validateUrbanStorageByteLength(
    graph.device,
    id,
    'point grid',
    (rowOffset + rowCount) * 4,
    'reduce positions length'
  );
  validateUrbanStorageByteLength(
    graph.device,
    id,
    'positions',
    rowCount * 8,
    'reduce positions length'
  );
  const coordinates = ['minimum-x', 'minimum-y', 'maximum-x', 'maximum-y'].map(name =>
    createTransientView(graph, `${id}-${name}-values`, 'float32', rowCount)
  );
  const extrema = coordinates.map((_, index) =>
    createTransientView(graph, `${id}-extent-${index}`, 'float32', 1)
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-coordinates`,
      operation: id,
      invocationCount: rowCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        ...(props.count
          ? [{name: 'count', view: props.count, type: 'u32', access: 'read'} as const]
          : []),
        ...coordinates.map(
          (view, index) =>
            ({name: `coordinate${index}`, view, type: 'f32', access: 'read_write'}) as const
        )
      ],
      body: `let horizontal=positions[positionsOffset+2u*index];
let vertical=positions[positionsOffset+2u*index+1u];
let valid=isFiniteValue(horizontal)&&isFiniteValue(vertical)${props.count ? '&&index<count[countOffset]' : ''};
coordinate0[coordinate0Offset+index]=select(3.402823e38,horizontal,valid);
coordinate1[coordinate1Offset+index]=select(3.402823e38,vertical,valid);
coordinate2[coordinate2Offset+index]=select(-3.402823e38,horizontal,valid);
coordinate3[coordinate3Offset+index]=select(-3.402823e38,vertical,valid);`
    })
  );
  coordinates.forEach((input, index) =>
    graph.add(
      new GPUReduction({
        id: `${id}-extent-reduction-${index}`,
        input,
        output: extrema[index],
        operation: index < 2 ? 'min' : 'max'
      }).getCommandNodes(graph)
    )
  );
  const data = createTransientView(graph, `${id}-packed`, 'uint32', rowOffset + rowCount);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-bounds`,
      operation: id,
      invocationCount: 1,
      bindings: [
        ...extrema.map(
          (view, index) => ({name: `extent${index}`, view, type: 'f32', access: 'read'}) as const
        ),
        {name: 'pointGrid', view: data, type: 'u32', access: 'read_write'}
      ],
      // Half-extent subtraction keeps the span finite even for opposite float32 extremes.
      body: `let lower=vec2<f32>(extent0[extent0Offset],extent1[extent1Offset]);
let upper=vec2<f32>(extent2[extent2Offset],extent3[extent3Offset]);
let populated=all(lower<=upper);
let minimum=select(vec2<f32>(0.0),lower,populated);
let span=select(vec2<f32>(1.0),max(0.5*upper-0.5*lower,vec2<f32>(1.0e-20)),populated);
pointGrid[pointGridOffset]=bitcast<u32>(minimum.x);
pointGrid[pointGridOffset+1u]=bitcast<u32>(minimum.y);
pointGrid[pointGridOffset+2u]=bitcast<u32>(span.x);
pointGrid[pointGridOffset+3u]=bitcast<u32>(span.y);`
    })
  );
  const grid: UrbanPointGrid = {dimension, rowOffset, data};
  const cellKeys = createTransientView(graph, `${id}-cell-keys`, 'uint32', rowCount);
  const sourceRows = createTransientView(graph, `${id}-source-rows`, 'uint32', rowCount);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-keys`,
      operation: id,
      invocationCount: rowCount,
      declarations: IS_FINITE_VALUE_WGSL + getUrbanPointGridDeclarations(grid),
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        ...(props.count
          ? [{name: 'count', view: props.count, type: 'u32', access: 'read'} as const]
          : []),
        {name: 'pointGrid', view: data, type: 'u32', access: 'read'},
        {name: 'keys', view: cellKeys, type: 'u32', access: 'read_write'},
        {name: 'rows', view: sourceRows, type: 'u32', access: 'read_write'}
      ],
      body: `let point=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
let valid=isFiniteValue(point.x)&&isFiniteValue(point.y)${props.count ? '&&index<count[countOffset]' : ''};
let cell=getUrbanPointGridCell(point);
keys[keysOffset+index]=select(${cellCount}u,u32(cell.y)*${dimension}u+u32(cell.x),valid);
rows[rowsOffset+index]=index;`
    })
  );
  const index = createPointNeighborGridIndexNodes(graph, {
    id,
    rowCount,
    cellKeys,
    sourceRows,
    cellCount
  });
  graph.add(index.nodes);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-pack`,
      operation: id,
      invocationCount: Math.max(rowCount, cellCount + 1),
      bindings: [
        {name: 'offsets', view: index.cellOffsets, type: 'u32', access: 'read'},
        {name: 'rows', view: index.sortedRows, type: 'u32', access: 'read'},
        {name: 'pointGrid', view: data, type: 'u32', access: 'read_write'}
      ],
      body: `if(index<${cellCount + 1}u){pointGrid[pointGridOffset+${URBAN_POINT_GRID_BOUND_WORD_COUNT}u+index]=offsets[offsetsOffset+index];}
if(index<${rowCount}u){pointGrid[pointGridOffset+${rowOffset}u+index]=rows[rowsOffset+index];}`
    })
  );
  return grid;
}

/** Requires the packed index bound as `pointGrid`. @internal */
export function getUrbanPointGridDeclarations(grid: UrbanPointGrid): string {
  return `
fn getUrbanPointGridMinimum()->vec2<f32>{return vec2<f32>(bitcast<f32>(pointGrid[pointGridOffset]),bitcast<f32>(pointGrid[pointGridOffset+1u]));}
fn getUrbanPointGridHalfSpan()->vec2<f32>{return vec2<f32>(bitcast<f32>(pointGrid[pointGridOffset+2u]),bitcast<f32>(pointGrid[pointGridOffset+3u]));}
fn getUrbanPointGridCell(point:vec2<f32>)->vec2<i32>{
 let unit=(0.5*point-0.5*getUrbanPointGridMinimum())/getUrbanPointGridHalfSpan();
 return vec2<i32>(clamp(floor(unit*${grid.dimension}.0),vec2<f32>(0.0),vec2<f32>(${grid.dimension - 1}.0)));
}
fn getUrbanPointGridStart(cell:vec2<i32>)->u32{return pointGrid[pointGridOffset+${URBAN_POINT_GRID_BOUND_WORD_COUNT}u+u32(cell.y)*${grid.dimension}u+u32(cell.x)];}
fn getUrbanPointGridEnd(cell:vec2<i32>)->u32{return pointGrid[pointGridOffset+${URBAN_POINT_GRID_BOUND_WORD_COUNT + 1}u+u32(cell.y)*${grid.dimension}u+u32(cell.x)];}
fn getUrbanPointGridRow(slot:u32)->u32{return pointGrid[pointGridOffset+${grid.rowOffset}u+slot];}
`;
}

/** Radius candidates, with cell rounding slack. beforeRow may return from the enclosing function before reading another row. @internal */
export function getUrbanPointGridRadiusWalk(
  grid: UrbanPointGrid,
  point: string,
  radius: string,
  row: string,
  body: string,
  beforeRow = ''
): string {
  return `let gridLower=max(vec2<i32>(0),getUrbanPointGridCell(${point}-vec2<f32>(${radius}))-vec2<i32>(1));
let gridUpper=min(vec2<i32>(${grid.dimension - 1}),getUrbanPointGridCell(${point}+vec2<f32>(${radius}))+vec2<i32>(1));
for(var gridRow=gridLower.y;gridRow<=gridUpper.y;gridRow++){
 for(var gridColumn=gridLower.x;gridColumn<=gridUpper.x;gridColumn++){
  let gridCell=vec2<i32>(gridColumn,gridRow);
  for(var gridSlot=getUrbanPointGridStart(gridCell);gridSlot<getUrbanPointGridEnd(gridCell);gridSlot++){
   ${beforeRow}
   let ${row}=getUrbanPointGridRow(gridSlot);
   ${body}
  }
 }
}`;
}

/**
 * Visits nearby cells first. A point farther than twice every current polygon vertex cannot
 * clip that polygon (Cauchy-Schwarz), so whole cells outside that bound are skipped. The bound
 * shrinks after each clip; no fixed neighbour capacity or distance approximation is used.
 * Requires `site`, `polygon` and `size` from the Voronoi clipping kernel.
 * @internal
 */
export function getUrbanPointGridVoronoiWalk(grid: UrbanPointGrid, body: string): string {
  return `let gridCenter=getUrbanPointGridCell(site);
for(var gridRadius=0;gridRadius<${grid.dimension};gridRadius++){
 let gridLower=max(vec2<i32>(0),gridCenter-vec2<i32>(gridRadius));
 let gridUpper=min(vec2<i32>(${grid.dimension - 1}),gridCenter+vec2<i32>(gridRadius));
 for(var gridRow=gridLower.y;gridRow<=gridUpper.y;gridRow++){
  for(var gridColumn=gridLower.x;gridColumn<=gridUpper.x;gridColumn++){
   let gridCell=vec2<i32>(gridColumn,gridRow);
   if(max(abs(gridCell.x-gridCenter.x),abs(gridCell.y-gridCenter.y))!=gridRadius||size==0u){continue;}
   var maximumSquared=0.0;
   for(var vertex=0u;vertex<size;vertex++){let offset=polygon[vertex]-site;maximumSquared=max(maximumSquared,dot(offset,offset));}
   // Work in half-distances to avoid overflowing the extent subtraction.
   let halfSpan=getUrbanPointGridHalfSpan();
   let lower=(0.5*getUrbanPointGridMinimum()-0.5*site)+halfSpan*(vec2<f32>(gridCell)/${grid.dimension}.0);
   let upper=lower+halfSpan/${grid.dimension}.0;
   // Coordinate-scale rounding slack makes pruning conservative at grid boundaries.
   let slack=1.0e-5*(abs(site)+abs(getUrbanPointGridMinimum())+halfSpan)+vec2<f32>(1.0e-10);
   let separation=max(vec2<f32>(0.0),max(lower,-upper)-slack);
   if(dot(separation,separation)>maximumSquared*(1.0+1.0e-5)){continue;}
   for(var gridSlot=getUrbanPointGridStart(gridCell);gridSlot<getUrbanPointGridEnd(gridCell);gridSlot++){
    let other=getUrbanPointGridRow(gridSlot);
    ${body}
   }
  }
 }
}`;
}
