// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCellMeasuresFP64Literal} from './cell-measures-fp64-wgsl';

/** S2 quadratic projection with exact IJ conversion and stable local increments. @internal */
export const S2_CELL_MEASURES_FP64_WGSL = /* wgsl */ `
struct S2CellMeasures { area: f32, perimeter: f32, edges: vec4f };
struct CellMeasuresS2Interval64 { lower: vec2f, delta: vec2f };
const CELL_MEASURES_FOUR_THIRDS = ${getCellMeasuresFP64Literal(4 / 3)};
const CELL_MEASURES_ONE_THIRD = ${getCellMeasuresFP64Literal(1 / 3)};

fn cellMeasuresS2Interval64(coordinate: u32, step: vec2f, level: u32) -> CellMeasuresS2Interval64 {
  let surface = mul_fp64(cellMeasuresInteger64(i32(coordinate)), step);
  let belowMiddle = surface.x < 0.5 || (surface.x == 0.5 && surface.y < 0.0);
  var distance = surface;
  if (belowMiddle) { distance = sub_fp64(vec2f(1.0, 0.0), surface); }
  var lower = sub_fp64(mul_fp64(CELL_MEASURES_FOUR_THIRDS, mul_fp64(distance, distance)), CELL_MEASURES_ONE_THIRD);
  var increment = sum_fp64(mul_fp64(vec2f(2.0, 0.0), distance), step);
  if (belowMiddle) {
    lower = -lower;
    increment = sub_fp64(mul_fp64(vec2f(2.0, 0.0), distance), step);
  }
  var delta = mul_fp64(CELL_MEASURES_FOUR_THIRDS, mul_fp64(step, increment));
  if (level == 0u) { delta = vec2f(2.0, 0.0); }
  return CellMeasuresS2Interval64(lower, delta);
}
fn cellMeasuresGetS2(cell: vec2u) -> S2CellMeasures {
  let level = dggs_s2_get_level(cell);
  let step = vec2f(1.0 / f32(1u << level), 0.0);
  let coordinates = dggs_s2_get_ij(cell);
  let horizontal = cellMeasuresS2Interval64(coordinates.x, step, level);
  let vertical = cellMeasuresS2Interval64(coordinates.y, step, level);
  let upperHorizontal = sum_fp64(horizontal.lower, horizontal.delta);
  let upperVertical = sum_fp64(vertical.lower, vertical.delta);
  let first = CellMeasuresVector64(vec2f(1.0, 0.0), horizontal.lower, vertical.lower);
  let second = CellMeasuresVector64(vec2f(1.0, 0.0), upperHorizontal, vertical.lower);
  let third = CellMeasuresVector64(vec2f(1.0, 0.0), upperHorizontal, upperVertical);
  let fourth = CellMeasuresVector64(vec2f(1.0, 0.0), horizontal.lower, upperVertical);
  let firstLength = cellMeasuresVectorLength64(first);
  let secondLength = cellMeasuresVectorLength64(second);
  let thirdLength = cellMeasuresVectorLength64(third);
  let fourthLength = cellMeasuresVectorLength64(fourth);
  let firstSecond = cellMeasuresVectorDot64(first, second);
  let secondThird = cellMeasuresVectorDot64(second, third);
  let thirdFourth = cellMeasuresVectorDot64(third, fourth);
  let fourthFirst = cellMeasuresVectorDot64(fourth, first);
  let diagonal = cellMeasuresVectorDot64(first, third);
  let determinant = mul_fp64(horizontal.delta, vertical.delta);
  let firstDenominator = sum_fp64(sum_fp64(mul_fp64(mul_fp64(firstLength, secondLength), thirdLength), mul_fp64(firstSecond, thirdLength)), sum_fp64(mul_fp64(secondThird, firstLength), mul_fp64(diagonal, secondLength)));
  let secondDenominator = sum_fp64(sum_fp64(mul_fp64(mul_fp64(firstLength, thirdLength), fourthLength), mul_fp64(diagonal, fourthLength)), sum_fp64(mul_fp64(thirdFourth, firstLength), mul_fp64(fourthFirst, thirdLength)));
  let area = 2.0 * (atan2(determinant.x, firstDenominator.x) + atan2(determinant.x, secondDenominator.x));
  let edges = vec4f(
    atan2(mul_fp64(horizontal.delta, sqrt_fp64(sum_fp64(vec2f(1.0, 0.0), mul_fp64(vertical.lower, vertical.lower)))).x, firstSecond.x),
    atan2(mul_fp64(vertical.delta, sqrt_fp64(sum_fp64(vec2f(1.0, 0.0), mul_fp64(upperHorizontal, upperHorizontal)))).x, secondThird.x),
    atan2(mul_fp64(horizontal.delta, sqrt_fp64(sum_fp64(vec2f(1.0, 0.0), mul_fp64(upperVertical, upperVertical)))).x, thirdFourth.x),
    atan2(mul_fp64(vertical.delta, sqrt_fp64(sum_fp64(vec2f(1.0, 0.0), mul_fp64(horizontal.lower, horizontal.lower)))).x, fourthFirst.x)
  );
  return S2CellMeasures(area, edges.x + edges.y + edges.z + edges.w, edges.wzyx);
}
`;
