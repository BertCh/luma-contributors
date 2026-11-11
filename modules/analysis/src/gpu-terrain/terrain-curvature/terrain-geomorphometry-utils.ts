// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  type GPURasterRowDirection,
  validateRasterRowDirection,
  writeRasterGridParameterValues
} from '../../analysis-core/raster-grid';

/**
 * Resolves the row direction of a terrain contributor from its grid, the single source of truth.
 *
 * The row term of the grid affine (`affine[4]`, world y per row) states where rows go: negative
 * means rows advance south (row 0 is the northernmost row, a north-up raster), positive means
 * they advance north. An omitted `rowDirection` takes that direction. An explicit one must agree
 * with it. When the row term is 0 (a grid rotated a quarter turn) the affine states no direction
 * and the explicit value, or `'south'`, is used.
 *
 * Only the row term is read. Columns are taken to advance east (`affine[0] > 0`), and on a
 * rotated `'uniform'` grid azimuths are measured clockwise from the grid's up axis, not from
 * true north.
 *
 * @param id Contributor ID, for the message.
 * @param grid Raster grid of the contributor.
 * @param rowDirection The `rowDirection` prop as passed.
 * @returns The direction the kernels use.
 * @throws If `rowDirection` is not `'south'` or `'north'`, or contradicts the sign of `affine[4]`.
 * @internal
 */
export function resolveTerrainRowDirection(
  id: string,
  grid: GPURasterMetadata,
  rowDirection: GPURasterRowDirection | undefined
): GPURasterRowDirection {
  validateRasterRowDirection(id, rowDirection);
  const rowStep = grid.affine[4];
  const gridRowDirection: GPURasterRowDirection | undefined =
    rowStep > 0 ? 'north' : rowStep < 0 ? 'south' : undefined;
  validateRequirement(
    id,
    rowDirection === undefined ||
      gridRowDirection === undefined ||
      rowDirection === gridRowDirection,
    `rowDirection '${rowDirection}' must match the grid: grid.affine[4] is ${rowStep}, so rows advance ${gridRowDirection}; omit rowDirection or correct the grid`
  );
  return rowDirection ?? gridRowDirection ?? 'south';
}

/**
 * Number of leading float32 values shared by every geomorphometry settings layout:
 * `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge]`.
 *
 * Contributor-specific values follow from index 5.
 *
 * @internal
 */
export const TERRAIN_GEOMORPHOMETRY_SHARED_PARAMETER_LENGTH = 5;

/** Shared elevation-scale part of every geomorphometry settings object. @internal */
export type TerrainGeomorphometryCellSettings = {
  /** Elevation multiplier converting elevation units into ground units. Defaults to 1. */
  zFactor?: number;
};

/**
 * Writes the shared `[cellSizeX, cellSizeY, zFactor, northEdge, southEdge]` prefix. The cell size
 * and row edges come from `grid`.
 *
 * @internal
 */
export function writeTerrainGeomorphometryCellSettings(
  settings: TerrainGeomorphometryCellSettings,
  grid: GPURasterMetadata,
  target: Float32Array
): void {
  target[2] = settings.zFactor ?? 1;
  writeRasterGridParameterValues(target, grid, TERRAIN_GEOMORPHOMETRY_CELL_SLOTS);
}

/**
 * Settings slots of the shared prefix, for
 * {@link getRasterGroundCellSizeWGSL}: cell size at index 0 and the row edges at 3 and 4.
 *
 * @internal
 */
export const TERRAIN_GEOMORPHOMETRY_CELL_SLOTS = {cellSizeIndex: 0, northEdgeIndex: 3} as const;

/**
 * WGSL `fn readExtendedSample(column: i32, row: i32) -> vec2<f32>`, the `'one-sided'` border rule
 * of the terrain derivative stencils, for samples at any distance outside the grid: `.x` is the
 * value and `.y` is 1 when it is valid, else 0.
 *
 * A stencil sample outside the grid is the quadratic through the three nearest in-grid samples
 * along that axis (the line through two when the third is missing or invalid, the edge sample on
 * a grid one cell wide). A centred first difference over that sample is
 * `(-3 f0 + 4 f1 - f2) / (2 h)`, the second-order one-sided stencil of luma.gl's
 * `GPUFiniteDifference2D` (`boundary: 'one-sided'`), so slope is exact on planes and quadratic
 * surfaces up to the border. A centred second difference over it is `(f0 - 2 f1 + f2) / h^2`,
 * the second difference one cell inward: zero on a plane and exact on a quadratic surface, but
 * first-order where `GPUFiniteDifference2D` uses a four-point second-order stencil. The
 * alternative `'clamp'` rule repeats the edge sample, which halves the first derivative across
 * the edge and reports curvature on a plane (`r = p / h` in the border column).
 *
 * The kernel must declare `WIDTH_SIGNED: i32`, `HEIGHT_SIGNED: i32` and
 * `fn readGridSample(column: i32, row: i32) -> vec2<f32>` for in-grid samples with the same
 * return convention. Extrapolation is linear in the samples with weights summing to 1, so
 * `readGridSample` may return values relative to any constant (the window centre). Differences of
 * neighbours are formed before they are scaled, so the result rounds once at the scale of the
 * edge sample. An extrapolated sample is valid when the edge sample and its inward neighbour are:
 * the same pixels are valid as under `'clamp'`.
 *
 * @internal
 */
export const TERRAIN_ONE_SIDED_BORDER_WGSL = /* wgsl */ `
// Value 'overshoot' cells outside the edge sample a, from a and its inward neighbours b and c.
fn extrapolateOutside(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, overshoot: f32, extent: i32) -> vec2<f32> {
  if (a.y == 0.0 || extent < 2) { return a; }
  if (b.y == 0.0) { return vec2<f32>(0.0, 0.0); }
  let firstDifference = a.x - b.x;
  if (extent < 3 || c.y == 0.0) { return vec2<f32>(a.x + overshoot * firstDifference, 1.0); }
  let secondDifference = b.x - c.x;
  return vec2<f32>(
    a.x + 0.5 * overshoot * (3.0 * firstDifference - secondDifference) +
      0.5 * overshoot * overshoot * (firstDifference - secondDifference),
    1.0
  );
}
// 'column' is inside the grid; 'row' may be outside.
fn readRowExtendedSample(column: i32, row: i32) -> vec2<f32> {
  if (row >= 0 && row < HEIGHT_SIGNED) { return readGridSample(column, row); }
  let isBefore = row < 0;
  let edge = select(HEIGHT_SIGNED - 1, 0, isBefore);
  let inward = select(-1, 1, isBefore);
  let overshoot = f32(select(row - (HEIGHT_SIGNED - 1), -row, isBefore));
  return extrapolateOutside(
    readGridSample(column, edge),
    readGridSample(column, clamp(edge + inward, 0, HEIGHT_SIGNED - 1)),
    readGridSample(column, clamp(edge + 2 * inward, 0, HEIGHT_SIGNED - 1)),
    overshoot,
    HEIGHT_SIGNED
  );
}
fn readExtendedSample(column: i32, row: i32) -> vec2<f32> {
  if (column >= 0 && column < WIDTH_SIGNED) { return readRowExtendedSample(column, row); }
  let isBefore = column < 0;
  let edge = select(WIDTH_SIGNED - 1, 0, isBefore);
  let inward = select(-1, 1, isBefore);
  let overshoot = f32(select(column - (WIDTH_SIGNED - 1), -column, isBefore));
  return extrapolateOutside(
    readRowExtendedSample(edge, row),
    readRowExtendedSample(clamp(edge + inward, 0, WIDTH_SIGNED - 1), row),
    readRowExtendedSample(clamp(edge + 2 * inward, 0, WIDTH_SIGNED - 1), row),
    overshoot,
    WIDTH_SIGNED
  );
}`;

/**
 * Pads a canonical elevation band (calibrated float32 values with validity) by one cell on every
 * side, filling the ring with {@link TERRAIN_ONE_SIDED_BORDER_WGSL}.
 *
 * A 3x3 stencil of the official `GPURasterGradient` run on the padded `(width + 2) x (height + 2)`
 * grid then evaluates the border pixels of the original grid with one-sided differences. Read
 * its result for original pixel `index` at {@link getTerrainPaddedIndexWGSL}.
 *
 * @returns The padded band, its extent, and the node that fills it.
 * @internal
 */
export function getTerrainOneSidedPaddedBandNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    /** Prefix for the node and transient IDs. */
    id: string;
    /** Class name of the calling contributor, for kernel diagnostics. */
    operation: string;
    /** Extent of the source grid. */
    width: number;
    height: number;
    /** Canonical source band: float32 buffer values and a validity view. */
    band: GPURasterBufferBand;
  }
): {
  band: GPURasterBufferBand;
  width: number;
  height: number;
  nodes: GPUCommandNode<Parameters>[];
} {
  const paddedWidth = props.width + 2;
  const paddedHeight = props.height + 2;
  const paddedCount = paddedWidth * paddedHeight;
  const values = createTransientView(graph, `${props.id}-padded-values`, 'float32', paddedCount);
  const validity = createTransientView(graph, `${props.id}-padded-validity`, 'uint32', paddedCount);
  const node = createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-one-sided-padding`,
    operation: props.operation,
    variant: 'one-sided-padding',
    bindings: [
      {name: 'sourceValues', view: props.band.storage.values, type: 'f32', access: 'read'},
      {
        name: 'sourceValidity',
        view: props.band.validity as GraphDataView<'uint32'>,
        type: 'u32',
        access: 'read'
      },
      {name: 'paddedValues', view: values, type: 'f32', access: 'read_write'},
      {name: 'paddedValidity', view: validity, type: 'u32', access: 'read_write'}
    ],
    invocationCount: paddedCount,
    declarations: `const WIDTH_SIGNED: i32 = ${props.width};
const HEIGHT_SIGNED: i32 = ${props.height};
const PADDED_WIDTH: u32 = ${paddedWidth}u;
${IS_FINITE_VALUE_WGSL}
fn readGridSample(column: i32, row: i32) -> vec2<f32> {
  let sampleIndex = u32(row * WIDTH_SIGNED + column);
  let value = sourceValues[sourceValuesOffset + sampleIndex];
  let isValid = sourceValidity[sourceValidityOffset + sampleIndex] != 0u && isFiniteValue(value);
  return vec2<f32>(select(0.0, value, isValid), select(0.0, 1.0, isValid));
}
${TERRAIN_ONE_SIDED_BORDER_WGSL}`,
    body: `let paddedRow = i32(index / PADDED_WIDTH);
  let paddedColumn = i32(index % PADDED_WIDTH);
  let extended = readExtendedSample(paddedColumn - 1, paddedRow - 1);
  let isValid = extended.y != 0.0 && isFiniteValue(extended.x);
  paddedValues[paddedValuesOffset + index] =
    select(bitcast<f32>(0x7fc00000u | (index & 0u)), extended.x, isValid);
  paddedValidity[paddedValidityOffset + index] = select(0u, 1u, isValid);`
  });
  return {
    band: {
      id: `${props.id}-padded`,
      format: 'float32',
      storage: {kind: 'buffer', values},
      validity
    },
    width: paddedWidth,
    height: paddedHeight,
    nodes: [node]
  };
}

/**
 * WGSL statement declaring `let gradientIndex: u32`, the element of original pixel `index` in a
 * raster produced on the grid of {@link getTerrainOneSidedPaddedBandNodes} (`isPadded`), or
 * `index` itself. The kernel must declare `WIDTH: u32` of the original grid.
 *
 * @internal
 */
export function getTerrainPaddedIndexWGSL(isPadded: boolean): string {
  return isPadded
    ? 'let gradientIndex = (index / WIDTH + 1u) * (WIDTH + 2u) + index % WIDTH + 1u;'
    : 'let gradientIndex = index;';
}
