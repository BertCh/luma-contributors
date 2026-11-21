// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {FIXED_POINT_WGSL} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  DECODE_ORDERED_KEY_WGSL,
  GET_ORDERED_KEY_WGSL
} from '../../analysis-core/kernels/wgsl-common';

/**
 * Shared WGSL for the 64-bit signed fixed-point accumulators. Quantizing, the unit and decoding
 * are those of the shared fixed-point accumulator (`FIXED_POINT_WGSL`: integer bit manipulation,
 * round to nearest with ties to even, correctly rounded decode). A sum is stored in two words
 * here instead of three, so the unit is 2^32 times the shared one: every term is below
 * 2^(63 - headroomBits) units and no sum of at most 2^headroomBits - 1 terms can overflow.
 */
export function getFixedPointSource(saturates: boolean): string {
  // With a data-derived unit no term can leave the two stored words. With a fixed unit
  // (`fixedPointScale`) a large term can: it saturates at the float32 nearest 4e18 units, with
  // its sign, as this contributor's fixed scale always did, instead of wrapping into the sign
  // bit or vanishing. An infinite term saturates too; NaN adds nothing.
  const saturationUnits = BigInt(Math.fround(4.0e18));
  const quantizeSource = saturates
    ? `const SATURATION_LOW: u32 = ${saturationUnits & 0xffffffffn}u;
const SATURATION_HIGH: u32 = ${saturationUnits >> 32n}u;
fn quantize(value: f32, unitExponent: i32) -> vec2<u32> {
  let wide = quantizeFixedPoint(abs(value), unitExponent);
  var magnitude = wide.xy;
  let isInfinite = (bitcast<u32>(value) & 0x7fffffffu) == 0x7f800000u;
  if (isInfinite || wide.z != 0u || wide.y > SATURATION_HIGH ||
      (wide.y == SATURATION_HIGH && wide.x > SATURATION_LOW)) {
    magnitude = vec2<u32>(SATURATION_LOW, SATURATION_HIGH);
  }
  if (value < 0.0) {
    let low = ~magnitude.x + 1u;
    magnitude = vec2<u32>(low, ~magnitude.y + select(0u, 1u, low == 0u));
  }
  return magnitude;
}`
    : `fn quantize(value: f32, unitExponent: i32) -> vec2<u32> {
  return quantizeFixedPoint(value, unitExponent).xy;
}`;
  return `${FIXED_POINT_WGSL}
fn getSumUnitExponent(maximumExponentBits: u32, headroomBits: u32) -> i32 {
  return getFixedPointUnitExponent(maximumExponentBits, headroomBits) + 32;
}
${quantizeSource}
fn dequantize(low: u32, high: u32, unitExponent: i32) -> f32 {
  let signWord = select(0u, 0xffffffffu, (high & 0x80000000u) != 0u);
  return getFixedPointFloat(vec3<u32>(low, high, signWord), unitExponent);
}
${GET_ORDERED_KEY_WGSL}${DECODE_ORDERED_KEY_WGSL}`;
}

/** WGSL statements that add `vec2<u32>` `quantized` into the atomic array words `index`, `index + 1`. */
export function getFixedPointAddSource(array: string, index: string, quantized: string): string {
  return `{
    let addendOld = atomicAdd(&${array}[${index}], ${quantized}.x);
    let addendCarry = select(0u, 1u, addendOld + ${quantized}.x < addendOld);
    atomicAdd(&${array}[${index} + 1u], ${quantized}.y + addendCarry);
  }`;
}
