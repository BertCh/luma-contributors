// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Shared WGSL point-read and finite/NaN helpers for the spatial-analysis contributors.
 *
 * Each constant or builder declares exactly one function with a name that does not collide with
 * the helpers in `utils/wgsl-common.ts`, so kernels can mix both families. Declaring a function
 * twice in one shader fails compilation, so include each declaration once.
 */

/**
 * Declares `fn <functionName>(row: u32) -> vec2f` reading an interleaved `x, y` pair from a
 * `f32` storage binding that was declared through `createWGSLKernelNode` (so that
 * `<bindingName>Offset` exists).
 * @internal
 */
export function getReadPointWGSL(functionName: string, bindingName: string): string {
  return `fn ${functionName}(row: u32) -> vec2f { return vec2f(${bindingName}[${bindingName}Offset + row * 2u], ${bindingName}[${bindingName}Offset + row * 2u + 1u]); }`;
}

/**
 * Declares `fn <functionName>(row: u32) -> vec3f` reading an interleaved `x, y, z` triple.
 * @internal
 */
export function getReadPoint3WGSL(functionName: string, bindingName: string): string {
  return `fn ${functionName}(row: u32) -> vec3f { let base = ${bindingName}Offset + row * 3u; return vec3f(${bindingName}[base], ${bindingName}[base + 1u], ${bindingName}[base + 2u]); }`;
}

/** `isSpatialFinite(value)`: finite test on the exponent bits, immune to constant folding. @internal */
export const IS_SPATIAL_FINITE_WGSL = /* wgsl */ `
fn isSpatialFinite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
`;

/** `isSpatialFinitePoint(point)`: both components finite. Requires {@link IS_SPATIAL_FINITE_WGSL}. @internal */
export const IS_SPATIAL_FINITE_POINT_WGSL = /* wgsl */ `
fn isSpatialFinitePoint(point: vec2f) -> bool { return isSpatialFinite(point.x) && isSpatialFinite(point.y); }
`;

/** `getSpatialNaN()`: a quiet NaN built from a runtime bit pattern because WGSL rejects NaN constants. @internal */
export const GET_SPATIAL_NAN_WGSL = /* wgsl */ `
fn getSpatialNaN() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }
`;

/** `isSameSpatialPoint(first, second)`: exact component equality of two points. @internal */
export const IS_SAME_SPATIAL_POINT_WGSL = /* wgsl */ `
fn isSameSpatialPoint(first: vec2f, second: vec2f) -> bool { return first.x == second.x && first.y == second.y; }
`;
