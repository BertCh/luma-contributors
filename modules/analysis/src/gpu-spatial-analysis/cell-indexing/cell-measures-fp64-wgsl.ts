// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {fp64ify} from '@luma.gl/shaders';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Split a binary64 geometry constant once, before shader compilation. @internal */
export function getCellMeasuresFP64Literal(value: number): string {
  return `vec2f(${Array.from(fp64ify(value)).map(getWGSLFloatLiteral).join(', ')})`;
}

/** Cell geometry on the published fp64 arithmetic module; no local arithmetic implementation. @internal */
export const CELL_MEASURES_FP64_WGSL = /* wgsl */ `
struct CellMeasuresPlane64 { x: vec2f, y: vec2f };
struct CellMeasuresVector64 { x: vec2f, y: vec2f, z: vec2f };

// H3 substrate and S2 IJ coordinates fit signed i32, but exceed the f32 significand.
fn cellMeasuresInteger64(value: i32) -> vec2f {
  let high = f32(value);
  return vec2f(high, f32(value - i32(high)));
}

fn cellMeasuresVectorAdd64(first: CellMeasuresVector64, second: CellMeasuresVector64) -> CellMeasuresVector64 {
  return CellMeasuresVector64(sum_fp64(first.x, second.x), sum_fp64(first.y, second.y), sum_fp64(first.z, second.z));
}
fn cellMeasuresVectorSubtract64(first: CellMeasuresVector64, second: CellMeasuresVector64) -> CellMeasuresVector64 {
  return CellMeasuresVector64(sub_fp64(first.x, second.x), sub_fp64(first.y, second.y), sub_fp64(first.z, second.z));
}
fn cellMeasuresVectorScale64(value: CellMeasuresVector64, scale: vec2f) -> CellMeasuresVector64 {
  return CellMeasuresVector64(mul_fp64(value.x, scale), mul_fp64(value.y, scale), mul_fp64(value.z, scale));
}
fn cellMeasuresVectorDot64(first: CellMeasuresVector64, second: CellMeasuresVector64) -> vec2f {
  return sum_fp64(sum_fp64(mul_fp64(first.x, second.x), mul_fp64(first.y, second.y)), mul_fp64(first.z, second.z));
}
fn cellMeasuresVectorCross64(first: CellMeasuresVector64, second: CellMeasuresVector64) -> CellMeasuresVector64 {
  return CellMeasuresVector64(
    sub_fp64(mul_fp64(first.y, second.z), mul_fp64(first.z, second.y)),
    sub_fp64(mul_fp64(first.z, second.x), mul_fp64(first.x, second.z)),
    sub_fp64(mul_fp64(first.x, second.y), mul_fp64(first.y, second.x))
  );
}
fn cellMeasuresVectorLength64(value: CellMeasuresVector64) -> vec2f {
  return sqrt_fp64(cellMeasuresVectorDot64(value, value));
}
fn cellMeasuresVectorNormalize64(value: CellMeasuresVector64) -> CellMeasuresVector64 {
  return cellMeasuresVectorScale64(value, div_fp64(vec2f(1.0, 0.0), cellMeasuresVectorLength64(value)));
}
fn cellMeasuresVectorHigh64(value: CellMeasuresVector64) -> vec3f {
  return vec3f(value.x.x, value.y.x, value.z.x);
}
`;
