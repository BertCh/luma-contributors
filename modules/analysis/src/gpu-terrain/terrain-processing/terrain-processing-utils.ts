// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GraphDataView,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateGraphViewsBelongToGraph,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getRasterGridGeoreference,
  getRasterGroundScaleWGSL,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {
  rasterBinding,
  validateRasterViews
} from '../../raster-analysis/remote-sensing/raster-kernel-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Shares the raster validator: packed views, distinct output buffers, outputs disjoint from inputs and each other. @internal */
export function checkTerrainViews(
  id: string,
  inputs: GraphDataView[],
  outputs: GraphDataView[]
): void {
  validateRasterViews(id, inputs, outputs);
  validateRequirement(
    id,
    new Set(outputs.map(view => view.buffer)).size === outputs.length,
    'outputs must use distinct buffers'
  );
}
/** @internal */
export function checkTerrainGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: GraphDataView[]
): void {
  validateGraphViewsBelongToGraph(id, graph, views);
}
/** Maps uint32 to u32, sint32 to i32 and float32 to f32. @internal */
export function terrainBinding(
  name: string,
  view: GraphDataView,
  access: 'read' | 'read_write' = 'read'
): WGSLKernelBinding {
  return rasterBinding(name, view, access === 'read_write');
}
/** @internal */
export const TERRAIN_FINITE_WGSL = `fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn missing() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }
fn infinity() -> f32 { var bits = 0x7f800000u; return bitcast<f32>(bits); }`;

/**
 * Returns WGSL `getGroundEdgeLength(cell, other)` and `getGroundCellArea(cell)` for a grid. The
 * kernel must declare `GRID_WIDTH` and `GRID_HEIGHT` (see `getTerrainFlowTracingWGSL`).
 *
 * `'uniform'` (the default of every consumer) measures in the affine's own units, so geographic
 * degrees are not ground meters. `'web-mercator'` and `'geographic'` measure ground meters with
 * the WGS84 row scale of `getRasterGroundScaleWGSL` at the midpoint row of the move, the same
 * midpoint rule as `getD8Distance` of the older terrain classes. Those two modes need an
 * axis-aligned grid (`validateRasterGridAxisAligned`) and bake the georeference into the shader,
 * so unlike the older classes they recompile when the grid origin changes.
 *
 * @internal
 */
export function getTerrainGroundMeasureWGSL(
  grid: GPURasterMetadata,
  cellSizeMode: GPURasterCellSizeMode = 'uniform'
): string {
  if (cellSizeMode === 'uniform') {
    const [a, b, , d, e] = grid.affine.map(getWGSLFloatLiteral);
    const area = getWGSLFloatLiteral(
      Math.abs(grid.affine[0] * grid.affine[4] - grid.affine[1] * grid.affine[3])
    );
    return `fn getGroundEdgeLength(cell: u32, other: u32) -> f32 {
 let delta = vec2<f32>(f32(other % GRID_WIDTH) - f32(cell % GRID_WIDTH), f32(other / GRID_WIDTH) - f32(cell / GRID_WIDTH));
 return length(vec2<f32>(${a} * delta.x + ${b} * delta.y, ${d} * delta.x + ${e} * delta.y));
}
fn getGroundCellArea(cell: u32) -> f32 { return ${area}; }`;
  }
  const georeference = getRasterGridGeoreference(grid);
  return `fn getGroundCellSizeAtRow(rowCoordinate: f32) -> vec2<f32> {
 let edge = mix(${getWGSLFloatLiteral(georeference.northEdge)}, ${getWGSLFloatLiteral(georeference.southEdge)}, rowCoordinate / f32(GRID_HEIGHT));
 ${getRasterGroundScaleWGSL(cellSizeMode)}
 return vec2<f32>(${getWGSLFloatLiteral(georeference.cellSize[0])}, ${getWGSLFloatLiteral(georeference.cellSize[1])}) * groundScale;
}
fn getGroundEdgeLength(cell: u32, other: u32) -> f32 {
 let cellRow = f32(cell / GRID_WIDTH); let otherRow = f32(other / GRID_WIDTH);
 let ground = getGroundCellSizeAtRow((cellRow + otherRow + 1.0) * 0.5);
 let columns = abs(f32(other % GRID_WIDTH) - f32(cell % GRID_WIDTH));
 return length(vec2<f32>(ground.x * columns, ground.y * abs(otherRow - cellRow)));
}
fn getGroundCellArea(cell: u32) -> f32 {
 let ground = getGroundCellSizeAtRow(f32(cell / GRID_WIDTH) + 0.5);
 return ground.x * ground.y;
}`;
}

/** Throws unless a per-cell validity view holds one uint32 word per cell. @internal */
export function validateTerrainValidityView(
  id: string,
  validity: GraphDataView<'uint32'> | undefined,
  cellCount: number
): void {
  validateRequirement(
    id,
    !validity || validity.length === cellCount,
    'validity must hold one uint32 word per grid cell (nonzero is valid)'
  );
}

/**
 * Returns `values` unchanged without a validity view. With one, returns a transient copy whose
 * cells with a zero validity word are NaN, so every downstream NaN test also honours validity.
 *
 * @internal
 */
export function createTerrainValidityMask<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    values: GraphDataView<'float32'>;
    validity?: GraphDataView<'uint32'>;
  }
): {values: GraphDataView<'float32'>; nodes: GPUCommandNode<Parameters>[]} {
  const {validity} = props;
  if (!validity) {
    return {values: props.values, nodes: []};
  }
  const masked = createTransientView(graph, `${props.id}-masked`, 'float32', props.values.length);
  return {
    values: masked,
    nodes: [
      createWGSLKernelNode(graph, {
        id: `${props.id}-mask`,
        operation: props.operation,
        variant: 'validity-mask',
        invocationCount: props.values.length,
        declarations: TERRAIN_FINITE_WGSL,
        bindings: [
          terrainBinding('source', props.values),
          terrainBinding('validity', validity),
          terrainBinding('masked', masked, 'read_write')
        ],
        body: 'masked[maskedOffset + index] = select(missing(), source[sourceOffset + index], validity[validityOffset + index] != 0u);'
      })
    ]
  };
}

/**
 * Number of gated pointer-jumping rounds that resolve every path of at most `pathBound` cells:
 * each round doubles the distance a cell's pointer spans, so `ceil(log2(pathBound))` rounds reach
 * the end of the longest path (at least one round, which is harmless once converged).
 *
 * @internal
 */
export function getTerrainJumpRoundCount(pathBound: number): number {
  return Math.max(1, Math.ceil(Math.log2(Math.max(pathBound, 1))));
}
