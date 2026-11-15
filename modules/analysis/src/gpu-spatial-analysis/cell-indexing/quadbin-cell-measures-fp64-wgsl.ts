// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCellMeasuresFP64Literal} from './cell-measures-fp64-wgsl';

/** Authalic Mercator rectangles using published fp64 exponential functions. @internal */
export const QUADBIN_CELL_MEASURES_FP64_WGSL = /* wgsl */ `
struct QuadbinCellMeasures { area: f32, perimeter: f32, edges: vec4f };
const CELL_MEASURES_PI = ${getCellMeasuresFP64Literal(Math.PI)};
const CELL_MEASURES_TWO_PI = ${getCellMeasuresFP64Literal(2 * Math.PI)};

fn cellMeasuresCosh64(value: vec2f) -> vec2f {
  return mul_fp64(vec2f(0.5, 0.0), sum_fp64(exp_fp64(value), exp_fp64(-value)));
}
fn cellMeasuresSinh64(value: vec2f) -> vec2f {
  // Retain tiny zoom-26 widths without differencing exponentials close to one.
  if (value.x < 0.01) {
    let square = mul_fp64(value, value);
    let correction = sum_fp64(${getCellMeasuresFP64Literal(1 / 6)}, mul_fp64(square, ${getCellMeasuresFP64Literal(1 / 120)}));
    return mul_fp64(value, sum_fp64(vec2f(1.0, 0.0), mul_fp64(square, correction)));
  }
  return mul_fp64(vec2f(0.5, 0.0), sub_fp64(exp_fp64(value), exp_fp64(-value)));
}
fn cellMeasuresGetQuadbin(cell: vec2u) -> QuadbinCellMeasures {
  let resolution = (cell.x >> 20u) & 31u;
  let morton = cellGetCompactKey(cell, 52u - 2u * resolution, 2u * resolution);
  var row = 0u;
  for (var bit = 0u; bit < resolution; bit++) {
    row = row | ((cellShiftRight(morton, 2u * bit + 1u).y & 1u) << bit);
  }
  let step = mul_fp64(CELL_MEASURES_TWO_PI, vec2f(1.0 / f32(1u << resolution), 0.0));
  let north = sub_fp64(CELL_MEASURES_PI, mul_fp64(cellMeasuresInteger64(i32(row)), step));
  let south = sub_fp64(north, step);
  let halfStep = mul_fp64(vec2f(0.5, 0.0), step);
  let middle = sub_fp64(north, halfStep);
  let northCosine = cellMeasuresCosh64(north);
  let southCosine = cellMeasuresCosh64(south);
  let meridian = 2.0 * atan2(cellMeasuresSinh64(halfStep).x, cellMeasuresCosh64(middle).x);
  let edges = vec4f(div_fp64(step, northCosine).x, meridian, div_fp64(step, southCosine).x, meridian);
  let area = div_fp64(mul_fp64(step, cellMeasuresSinh64(step)), mul_fp64(northCosine, southCosine));
  return QuadbinCellMeasures(area.x, edges.x + edges.y + edges.z + edges.w, edges);
}
`;
