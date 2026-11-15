// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Mercator tiles on the authalic sphere. Requires CELL_KEY_WGSL. @internal */
export const QUADBIN_CELL_MEASURES_WGSL = /* wgsl */ `
struct QuadbinCellMeasures {
  area: f32,
  perimeter: f32,
  edges: vec4f,
};

fn cellMeasuresGetQuadbin(cell: vec2u) -> QuadbinCellMeasures {
  let resolution = (cell.x >> 20u) & 31u;
  let morton = cellGetCompactKey(cell, 52u - 2u * resolution, 2u * resolution);
  var row = 0u;
  for (var bit = 0u; bit < resolution; bit++) {
    let digit = cellShiftRight(morton, 2u * bit + 1u).y & 1u;
    row = row | (digit << bit);
  }
  let step = 6.283185307179586 / f32(1u << resolution);
  let north = 3.141592653589793 - f32(row) * step;
  let south = north - step;
  let middle = north - 0.5 * step;
  // Half-angle and hyperbolic difference identities retain sub-ulp tile widths at zoom 26.
  // Builtin sinh loses its small argument to exp(x)-exp(-x) on some devices; use its series.
  var halfSine = 0.5 * step;
  var sine = step;
  if (step < 0.01) {
    halfSine *= 1.0 + step * step / 24.0;
    sine *= 1.0 + step * step / 6.0;
  } else {
    halfSine = sinh(0.5 * step);
    sine = sinh(step);
  }
  let northCosine = cosh(north);
  let southCosine = cosh(south);
  let meridian = 2.0 * atan2(halfSine, cosh(middle));
  let edges = vec4f(step / northCosine, meridian, step / southCosine, meridian);
  return QuadbinCellMeasures(step * sine / (northCosine * southCosine), edges.x + edges.y + edges.z + edges.w, edges);
}
`;
