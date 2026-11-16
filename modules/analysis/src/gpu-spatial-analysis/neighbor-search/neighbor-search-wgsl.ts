// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** WGSL finite-float test shared by the neighbor-search kernels. @internal */
export const NEIGHBOR_SEARCH_FLOAT_WGSL = /* wgsl */ IS_FINITE_FLOAT_WGSL;

/** Exact stored-f32 longitude reduction shared by search and spacing preparation. @internal */
export const NEIGHBOR_SEARCH_CANONICAL_LONGITUDE_WGSL = /* wgsl */ `
// Reduce the stored f32 angle exactly modulo a turn. Floating quotient/product reduction can
// erase the remainder for large angles (the stored f32 near 1e20 is -88 degrees, not zero).
// For fractional angles, use integer significand units. For integer angles, modular powers of
// two need at most seven iterations even at the largest finite f32 exponent.
fn getNeighborSearchCanonicalLongitude(longitude: f32) -> f32 {
  if (abs(longitude) <= 180.0) { return longitude; }
  let bits = bitcast<u32>(longitude);
  let exponent = (bits >> 23u) & 255u;
  let significand = (bits & 0x7fffffu) | 0x800000u;
  var wrapped: f32;
  if (exponent < 150u) {
    let fractionalBits = 150u - exponent;
    let turnUnits = 360u << fractionalBits;
    let remainder = significand % turnUnits;
    let centered = select(i32(remainder), i32(remainder) - i32(turnUnits), remainder > turnUnits / 2u);
    wrapped = f32(centered) * bitcast<f32>((127u - fractionalBits) << 23u);
  } else {
    var remainingExponent = exponent - 150u;
    var power = 2u;
    var remainder = significand % 360u;
    while (remainingExponent > 0u) {
      if ((remainingExponent & 1u) != 0u) { remainder = (remainder * power) % 360u; }
      power = (power * power) % 360u;
      remainingExponent >>= 1u;
    }
    wrapped = f32(select(i32(remainder), i32(remainder) - 360, remainder > 180u));
  }
  return select(wrapped, -wrapped, (bits & 0x80000000u) != 0u);
}
`;
