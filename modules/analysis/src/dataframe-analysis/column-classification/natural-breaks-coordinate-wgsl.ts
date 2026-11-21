// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Local coordinates for each coarse range and each refinement break. @internal */
export const NATURAL_BREAKS_COORDINATE_WGSL = /* wgsl */ `
const NATURAL_SCALE_DOWN: f32 = 5.421010862427522e-20;
const NATURAL_SCALE_UP: f32 = 1.8446744073709552e19;

fn getNaturalCoordinateScale(minimum: f32, maximum: f32) -> f32 {
  let magnitude = max(abs(minimum), abs(maximum));
  if (magnitude > 1e19) { return NATURAL_SCALE_DOWN; }
  if (magnitude > 0.0 && magnitude < 1e-19) { return NATURAL_SCALE_UP; }
  return 1.0;
}

fn increaseNaturalDistance(value: f32, limit: f32) -> f32 {
  if (value >= limit * NATURAL_SCALE_DOWN) { return limit; }
  return value * NATURAL_SCALE_UP;
}

// Saved widths and neighbor movements carry their own coordinate scale. Convert in at most
// two exact power-of-two steps, capping upward conversions before a second step can overflow.
fn convertNaturalDistance(value: f32, previousScale: f32, scale: f32, limit: f32) -> f32 {
  if (value == 0.0) { return 0.0; }
  if (previousScale == 0.0) { return limit; }
  var converted = value;
  if (scale > previousScale) {
    converted = increaseNaturalDistance(converted, limit);
    if (previousScale == NATURAL_SCALE_DOWN && scale == NATURAL_SCALE_UP) {
      converted = increaseNaturalDistance(converted, limit);
    }
  } else if (scale < previousScale) {
    converted = converted * NATURAL_SCALE_DOWN;
    if (previousScale == NATURAL_SCALE_UP && scale == NATURAL_SCALE_DOWN) {
      converted = converted * NATURAL_SCALE_DOWN;
    }
  }
  return min(converted, limit);
}
`;
