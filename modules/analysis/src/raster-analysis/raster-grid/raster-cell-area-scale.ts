// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getRasterGridGeoreference,
  getRasterGroundScale,
  getRasterGroundScaleWGSL,
  type GPURasterCellSizeMode
} from '../../analysis-core/contracts/raster-grid';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

/** Slot of the origin y (`affine[5]`) in the layout of `getGPURasterFractionalCoverageParameterValues`. */
const GEOREFERENCE_ORIGIN_Y_SLOT = 1;
/** Slot of the row step y (`affine[4]`) in the same layout. */
const GEOREFERENCE_ROW_Y_SLOT = 5;

/** Whether the effective frame supports one ground-area scale per row. @internal */
export function isRasterEffectiveAxisAligned(
  grid: GPURasterMetadata,
  mode: GPURasterCellSizeMode,
  parameters?: ArrayLike<number>
): boolean {
  return (
    mode === 'uniform' ||
    (parameters
      ? parameters[3] === 0 && parameters[4] === 0
      : grid.affine[1] === 0 && grid.affine[3] === 0)
  );
}

/** WGSL counterpart of isRasterEffectiveAxisAligned; no extra binding is introduced. @internal */
export function getRasterEffectiveAxisAlignmentWGSL(
  grid: GPURasterMetadata,
  mode: GPURasterCellSizeMode,
  binding?: string
): string {
  if (mode === 'uniform') return 'true';
  return binding
    ? `${binding}[${binding}Offset+3u]==0.0&&${binding}[${binding}Offset+4u]==0.0`
    : String(isRasterEffectiveAxisAligned(grid, mode));
}

/**
 * Ground area of one cell of `row`, relative to its area in world units, in float64: the product
 * of the two WGS84 ground scales at the row center (see {@link GPURasterCellSizeMode}). It is 1
 * for `'uniform'`. Row edges come from the grid affine, or, when `parameters` (the layout of
 * `getGPURasterFractionalCoverageParameterValues`) is given, from its origin y and row step y,
 * like the contributors that read their `parameters` view. CPU twin of the WGSL function from
 * {@link getRasterRowAreaScaleWGSL}.
 */
export function getRasterRowAreaScale(
  grid: GPURasterMetadata,
  cellSizeMode: GPURasterCellSizeMode,
  row: number,
  parameters?: ArrayLike<number>
): number {
  if (cellSizeMode === 'uniform') {
    return 1;
  }
  const {northEdge, southEdge} = getRasterGridGeoreference(
    parameters
      ? {
          ...grid,
          affine: [
            grid.affine[0],
            grid.affine[1],
            grid.affine[2],
            grid.affine[3],
            parameters[GEOREFERENCE_ROW_Y_SLOT],
            parameters[GEOREFERENCE_ORIGIN_Y_SLOT]
          ]
        }
      : grid
  );
  const edge = northEdge + (southEdge - northEdge) * ((row + 0.5) / grid.height);
  const [scaleX, scaleY] = getRasterGroundScale(cellSizeMode, edge);
  return scaleX * scaleY;
}

/**
 * Returns WGSL statements that define `let edge: f32`, the world y of the center of `row`
 * (a `u32` in scope): the north and south edges of `grid` interpolated at the row center. With a
 * `georeferenceName` the edges are computed from the origin y and the row step y of that `f32`
 * binding (layout of `getGPURasterFractionalCoverageParameterValues`, with `grid.height` and
 * `grid.pixelInterpretation` compile-time), so they follow a repacked georeference on every
 * encoding. Without it they are baked from `grid.affine`, as the planar cell area is when no
 * `parameters` view is given. Float32, like the terrain `getGroundCellSize`.
 */
export function getRasterRowEdgeWGSL(grid: GPURasterMetadata, georeferenceName?: string): string {
  const rowShift = grid.pixelInterpretation === 'point' ? -0.5 : 0;
  const rowFraction = `let rowFraction = (f32(row) + 0.5) / ${getWGSLFloatLiteral(grid.height)};`;
  if (georeferenceName === undefined) {
    const {northEdge, southEdge} = getRasterGridGeoreference(grid);
    return `${rowFraction}
  let edge = mix(${getWGSLFloatLiteral(northEdge)}, ${getWGSLFloatLiteral(southEdge)}, rowFraction);`;
  }
  const offset = `${georeferenceName}Offset`;
  return `${rowFraction}
  let originY = ${georeferenceName}[${offset} + ${GEOREFERENCE_ORIGIN_Y_SLOT}u];
  let rowStepY = ${georeferenceName}[${offset} + ${GEOREFERENCE_ROW_Y_SLOT}u];
  let northEdge = originY + rowStepY * ${getWGSLFloatLiteral(rowShift)};
  let southEdge = originY + rowStepY * ${getWGSLFloatLiteral(grid.height + rowShift)};
  let edge = mix(northEdge, southEdge, rowFraction);`;
}

/**
 * Returns WGSL `fn getRowAreaScale(row: u32) -> f32` for a non-uniform `cellSizeMode`: ground
 * area per world area of one cell in `row`. The row edges are read from the `georeferenceName`
 * binding when given (see {@link getRasterRowEdgeWGSL}), else baked from `grid.affine`.
 */
export function getRasterRowAreaScaleWGSL(
  grid: GPURasterMetadata,
  cellSizeMode: Exclude<GPURasterCellSizeMode, 'uniform'>,
  georeferenceName?: string
): string {
  return `
fn getRowAreaScale(row: u32) -> f32 {
  ${getRasterRowEdgeWGSL(grid, georeferenceName)}
  ${getRasterGroundScaleWGSL(cellSizeMode)}
  return groundScale.x * groundScale.y;
}`;
}
