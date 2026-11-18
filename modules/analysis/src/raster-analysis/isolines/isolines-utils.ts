// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphVectorView, validatePackedView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateRasterAlgebraAliasing} from '../raster-algebra/raster-algebra-utils';

/**
 * Throws unless `view` is one packed view of `format` with at least `rowCount` rows.
 *
 * @internal
 */
export function validateIsolinesView(
  id: string,
  name: string,
  view: GraphDataView | GraphVectorView | undefined,
  format: 'float32' | 'uint32' | 'float32x2' | 'float32x4' | 'uint32x2',
  rowCount: number
): void {
  if (!view) {
    return;
  }
  if (view instanceof GraphVectorView) {
    throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
  }
  validatePackedView(view, [format], `${id} ${name}`);
  if (view.length < rowCount) {
    throw new Error(`${id} ${name} must hold at least ${rowCount} rows`);
  }
}

/**
 * Throws when two views of `outputs` share a buffer, or an output shares a buffer with `inputs`.
 *
 * @internal
 */
export function validateIsolinesAliasing(
  id: string,
  outputs: Parameters<typeof validateRasterAlgebraAliasing>[1],
  inputs: Parameters<typeof validateRasterAlgebraAliasing>[2]
): void {
  validateRasterAlgebraAliasing(id, outputs, inputs);
}

/**
 * WGSL `isSaddleJoined(v, level)`: the saddle rule shared by {@link GPUIsolines} and
 * {@link GPUIsobands}, the bilinear (asymptotic) decider of the published `GPURasterContours`.
 *
 * `v` holds the corners in boundary order, so `(v0, v2)` and `(v1, v3)` are the two diagonals, and
 * exactly one diagonal is high (`value >= level`). The bilinear surface through the four corners
 * has its saddle point at the value `(v0 * v2 - v1 * v3) / (v0 + v2 - v1 - v3)`. The two high
 * corners are joined through the cell when that value is at or above the level and separated
 * when it is below. With `d = v - level` the saddle value minus the level is
 * `(d0 * d2 - d1 * d3) / (d0 + d2 - d1 - d3)`, and the denominator is positive when `(v0, v2)` is
 * the high diagonal and negative otherwise, so only the sign of the determinant
 * `d0 * d2 - d1 * d3` is needed and nothing is divided. A saddle value exactly at the level joins,
 * in keeping with `value >= level` being high and with the published rule ("an exact determinant
 * tie connects the high-valued diagonal").
 *
 * The differences are taken unscaled, so on integer rasters with integer levels (elevations in
 * whole metres) the determinant is exact and a tie is a tie. Only when a magnitude is above 1e18,
 * where the products could overflow, are the corners divided by the largest magnitude first, as
 * the published kernel always does. The published kernel's unconditional scaling rounds the
 * differences, so on an exact tie its choice is decided by rounding; the two kernels can
 * therefore differ on exact ties and only there.
 *
 * The result is meaningless for a cell that is not a saddle; callers use it only for masks 5 and
 * 10.
 *
 * @internal
 */
export const ISOLINES_SADDLE_WGSL = /* wgsl */ `
fn isSaddleJoined(v: vec4<f32>, level: f32) -> bool {
  let magnitude = max(abs(level), max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w))));
  var delta = v - vec4<f32>(level);
  if (magnitude > 1.0e18) {
    delta = v / magnitude - vec4<f32>(level / magnitude);
  }
  let determinant = delta.x * delta.z - delta.y * delta.w;
  return select(determinant <= 0.0, determinant >= 0.0, v.x >= level);
}`;

/**
 * WGSL `getNaN`, `Corners` and `loadCorners(cx, cy)` for a raster binding `raster` with optional
 * `validity`. A cell is nodata when it is non-finite (NaN or an infinity, which a crossing cannot
 * interpolate), equals the optional sentinel, or has a zero validity word. Cell `(cx, cy)` reads samples v0 `(cx, cy)`, v1 `(cx + 1,
 * cy)`, v2 `(cx + 1, cy + 1)`, v3 `(cx, cy + 1)`; `valid` is false when any corner is nodata.
 *
 * @internal
 */
export function getIsolinesCornersWGSL(
  hasValidity: boolean,
  noDataValue: number | undefined
): string {
  return /* wgsl */ `
${IS_FINITE_VALUE_WGSL}
fn readCell(cell: u32, value: ptr<function, f32>) -> bool {
  let sample = raster[rasterOffset + cell];
  *value = sample;
  // Compare bits: NaN and infinities are nodata, since a crossing cannot interpolate them.
  var isValid = isFiniteValue(sample);
  ${noDataValue !== undefined ? `isValid = isValid && sample != ${getWGSLFloatLiteral(noDataValue)};` : ''}
  ${hasValidity ? 'isValid = isValid && validity[validityOffset + cell] != 0u;' : ''}
  return isValid;
}

struct Corners {
  valid: bool,
  v: vec4<f32>
}

fn loadCorners(cx: u32, cy: u32) -> Corners {
  var v0 = 0.0;
  var v1 = 0.0;
  var v2 = 0.0;
  var v3 = 0.0;
  let ok0 = readCell(cy * WIDTH + cx, &v0);
  let ok1 = readCell(cy * WIDTH + cx + 1u, &v1);
  let ok2 = readCell((cy + 1u) * WIDTH + cx + 1u, &v2);
  let ok3 = readCell((cy + 1u) * WIDTH + cx, &v3);
  var corners: Corners;
  corners.valid = ok0 && ok1 && ok2 && ok3;
  corners.v = vec4<f32>(v0, v1, v2, v3);
  return corners;
}

// Bit i is set when corner i (v0..v3) is high, i.e. value >= level.
fn getHighMask(v: vec4<f32>, level: f32) -> u32 {
  return select(0u, 1u, v.x >= level) | select(0u, 2u, v.y >= level) |
    select(0u, 4u, v.z >= level) | select(0u, 8u, v.w >= level);
}

// Counter-clockwise boundary walk v0 -> v1 -> v2 -> v3 -> v0: edge k runs from corner k to corner
// (k + 1) % 4, i.e. edges e0 bottom, e1 right, e2 top, e3 left. An exit is high -> low (the
// segment starts there, keeping high on its left); an entry is low -> high (the segment ends).
fn getRotatedMask(mask: u32) -> u32 {
  return ((mask >> 1u) | (mask << 3u)) & 15u;
}
fn getExitMask(mask: u32) -> u32 { return mask & ~getRotatedMask(mask) & 15u; }
fn getEntryMask(mask: u32) -> u32 { return ~mask & getRotatedMask(mask) & 15u; }

fn getEdgeId(k: u32, cx: u32, cy: u32) -> u32 {
  let horizontalCount = HEIGHT * (WIDTH - 1u);
  if (k == 0u) { return cy * (WIDTH - 1u) + cx; }
  if (k == 2u) { return (cy + 1u) * (WIDTH - 1u) + cx; }
  if (k == 3u) { return horizontalCount + cy * WIDTH + cx; }
  return horizontalCount + cy * WIDTH + cx + 1u;
}`;
}
