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
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateSinglePackedView} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateTerrainOutputDispatches,
  validateTerrainReadWriteRanges,
  type TerrainViewDispatch
} from './terrain-view-validation';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  getRasterGridGeoreference,
  getRasterGroundScaleWGSL,
  type GPURasterCellSizeMode
} from '../../analysis-core/raster-grid';
import {rasterBinding} from '../../raster-analysis/remote-sensing/raster-kernel-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Packed metadata, physical input/output separation and dispatch-aware output checks. @internal */
export function checkTerrainViews(
  id: string,
  inputs: GraphDataView[],
  outputs: GraphDataView[],
  dispatches: readonly TerrainViewDispatch[] = [{reads: inputs, writes: outputs}]
): void {
  for (const view of [...inputs, ...outputs]) validateSinglePackedView(view, [view.format], id);
  validateTerrainReadWriteRanges(id, inputs, outputs, false);
  validateTerrainOutputDispatches(id, outputs, dispatches);
}
/** @internal */
export function checkTerrainGraph<Parameters>(
  id: string,
  graph: GPUCommandGraph<Parameters>,
  views: GraphDataView[]
): void {
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(views.map((view, index) => [`view[${index}]`, view]))
  );
}
/** Maps uint32 to u32, sint32 to i32 and float32 to f32. @internal */
export function terrainBinding(
  name: string,
  view: GraphDataView,
  access: 'read' | 'read_write' = 'read'
): WGSLKernelBinding {
  return rasterBinding(name, view, access === 'read_write');
}
/**
 * `isFiniteValue(value)` and `getNaN()` from `analysis-core` plus `infinity()`. A kernel that
 * includes this string must not include `IS_FINITE_VALUE_WGSL` or `GET_NAN_WGSL` again. @internal
 */
export const TERRAIN_FINITE_WGSL = `${IS_FINITE_VALUE_WGSL}${GET_NAN_WGSL}
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
  cellSizeMode: GPURasterCellSizeMode = 'uniform',
  includeArea = true
): string {
  // Subtract integer coordinates before conversion: neighbouring cells beyond 2^24 must not
  // collapse onto the same float32 coordinate. Both selected differences fit uint32.
  const coordinateDifference = `fn getTerrainCoordinateDifference(first: u32, second: u32) -> f32 {
 return select(-f32(second - first), f32(first - second), first >= second);
}`;
  if (cellSizeMode === 'uniform') {
    // A uniform edge/area never uses the world origin, which can exceed float32's range.
    const [a, b, d, e] = [grid.affine[0], grid.affine[1], grid.affine[3], grid.affine[4]].map(
      getWGSLFloatLiteral
    );
    const area = includeArea
      ? getWGSLFloatLiteral(
          Math.abs(grid.affine[0] * grid.affine[4] - grid.affine[1] * grid.affine[3])
        )
      : '';
    return `${coordinateDifference}
fn getGroundEdgeLength(cell: u32, other: u32) -> f32 {
 let delta = vec2<f32>(getTerrainCoordinateDifference(other % GRID_WIDTH, cell % GRID_WIDTH), getTerrainCoordinateDifference(other / GRID_WIDTH, cell / GRID_WIDTH));
 return length(vec2<f32>(${a} * delta.x + ${b} * delta.y, ${d} * delta.x + ${e} * delta.y));
}
${includeArea ? `fn getGroundCellArea(cell: u32) -> f32 { return ${area}; }` : ''}`;
  }
  const georeference = getRasterGridGeoreference(grid);
  return `${coordinateDifference}
fn getGroundCellSizeAtRow(rowCoordinate: f32) -> vec2<f32> {
 let edge = mix(${getWGSLFloatLiteral(georeference.northEdge)}, ${getWGSLFloatLiteral(georeference.southEdge)}, rowCoordinate / f32(GRID_HEIGHT));
 ${getRasterGroundScaleWGSL(cellSizeMode)}
 return vec2<f32>(${getWGSLFloatLiteral(georeference.cellSize[0])}, ${getWGSLFloatLiteral(georeference.cellSize[1])}) * groundScale;
}
fn getGroundEdgeLength(cell: u32, other: u32) -> f32 {
 let cellRow = f32(cell / GRID_WIDTH); let otherRow = f32(other / GRID_WIDTH);
 let ground = getGroundCellSizeAtRow((cellRow + otherRow + 1.0) * 0.5);
 let columns = abs(getTerrainCoordinateDifference(other % GRID_WIDTH, cell % GRID_WIDTH));
 let rows = abs(getTerrainCoordinateDifference(other / GRID_WIDTH, cell / GRID_WIDTH));
 return length(vec2<f32>(ground.x * columns, ground.y * rows));
}
${
  includeArea
    ? `fn getGroundCellArea(cell: u32) -> f32 {
 let ground = getGroundCellSizeAtRow(f32(cell / GRID_WIDTH) + 0.5);
 return ground.x * ground.y;
}`
    : ''
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
        body: 'masked[maskedOffset + index] = select(getNaN(), source[sourceOffset + index], validity[validityOffset + index] != 0u);'
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

/**
 * Publishes named one-row status ports (`requiredCount`, `overflow`, `converged`,
 * `iterationLimitReached`) from words of an existing loose status view, so the loose view stays as
 * it was and the named ports sit beside it. One invocation; returns no node when no port is given.
 * Each `expression` is WGSL over `source[sourceOffset + word]`.
 *
 * @internal
 */
export function createTerrainStatusPortNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    source: GraphDataView<'uint32'>;
    ports: readonly {
      name: string;
      view: GraphDataView<'uint32'> | undefined;
      expression: string;
    }[];
  }
): GPUCommandNode<Parameters>[] {
  const ports = props.ports.filter(port => port.view);
  if (ports.length === 0) {
    return [];
  }
  return [
    createWGSLKernelNode(graph, {
      id: `${props.id}-status-ports`,
      operation: props.operation,
      variant: 'status-ports',
      invocationCount: 1,
      bindings: [
        terrainBinding('source', props.source),
        ...ports.map(port =>
          terrainBinding(port.name, port.view as GraphDataView<'uint32'>, 'read_write')
        )
      ],
      body: ports.map(port => `${port.name}[${port.name}Offset] = ${port.expression};`).join('\n  ')
    })
  ];
}

/** Throws unless every present named status port is a packed one-row `uint32` view. @internal */
export function validateTerrainStatusPorts(
  id: string,
  ports: Record<string, GraphDataView<'uint32'> | undefined>,
  prefix = 'output.status'
): void {
  for (const [name, view] of Object.entries(ports)) {
    if (!view) {
      continue;
    }
    validateRequirement(
      id,
      view.length === 1 && view.format === 'uint32',
      `${prefix}.${name} must be one uint32 row`
    );
  }
}
