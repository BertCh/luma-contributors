// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL `load<Name>(entry)` and `store<Name>(entry, words)` for a fixed-point state binding with
 * three `uint32` words per entry. Requires `FIXED_POINT_WGSL`.
 *
 * @internal
 */
export function getRasterFixedPointStateWGSL(bindingName: string, writable = true): string {
  const name = `${bindingName[0].toUpperCase()}${bindingName.slice(1)}`;
  return `
fn load${name}(entry: u32) -> vec3<u32> {
  return vec3<u32>(${bindingName}[${bindingName}Offset + 3u * entry], ${bindingName}[${bindingName}Offset + 3u * entry + 1u], ${bindingName}[${bindingName}Offset + 3u * entry + 2u]);
}
${
  writable
    ? `fn store${name}(entry: u32, words: vec3<u32>) {
  ${bindingName}[${bindingName}Offset + 3u * entry] = words.x;
  ${bindingName}[${bindingName}Offset + 3u * entry + 1u] = words.y;
  ${bindingName}[${bindingName}Offset + 3u * entry + 2u] = words.z;
}`
    : ''
}`;
}

/**
 * Calibration observations retain value - location in fixed-point units before rounding the shift.
 * Nearby same-sign f32 operands subtract exactly (Sterbenz); the bit gate prevents an optimizer from
 * reassociating a compensation expression. Otherwise each operand is at most twice the difference,
 * so quantizing the operands separately fits the existing sum scale and preserves the low bits.
 * Positive calibration observations give a signed-96 centered residual, bounded by the sum range.
 * Requires FIXED_POINT_WGSL and FIXED_POINT_MOMENTS_WGSL. No additional retained state or units.
 * @internal
 */
export const RASTER_CALIBRATION_MOMENTS_WGSL = /* wgsl */ `
fn getRasterCalibrationObservation(sampleValue: f32, modelLocation: f32, encodingExponent: i32) -> vec3<u32> {
  let sampleBits = bitcast<u32>(sampleValue);
  let locationBits = bitcast<u32>(modelLocation);
  let sampleMagnitude = sampleBits & 0x7fffffffu;
  let locationMagnitude = locationBits & 0x7fffffffu;
  let sameSign = ((sampleBits ^ locationBits) & 0x80000000u) == 0u;
  if (sameSign && max(sampleMagnitude, locationMagnitude) - min(sampleMagnitude, locationMagnitude) <= 0x00800000u) {
    return quantizeFixedPoint(sampleValue - modelLocation, encodingExponent);
  }
  return addFixedPoint(quantizeFixedPoint(sampleValue, encodingExponent),
    negateFixedPoint(quantizeFixedPoint(modelLocation, encodingExponent)));
}
fn getRasterCalibrationMomentIncrement(previousSum: vec3<u32>, sampleCount: u32,
  observationWords: vec3<u32>, encodingExponent: i32, squaredExponent: i32) -> vec3<u32> {
  if (sampleCount == 0u) { return vec3<u32>(0u); }
  var multiplied = vec3<u32>(0u);
  var shiftedWords = observationWords;
  var remaining = sampleCount;
  // A uint32 observation count has at most 32 bits; partial products are bounded by n*x.
  for (var position = 0u; position < 32u; position++) {
    if ((remaining & 1u) != 0u) { multiplied = addFixedPoint(multiplied, shiftedWords); }
    remaining = remaining >> 1u;
    if (remaining == 0u) { break; }
    shiftedWords = vec3<u32>(shiftedWords.x << 1u,
      (shiftedWords.y << 1u) | (shiftedWords.x >> 31u),
      (shiftedWords.z << 1u) | (shiftedWords.y >> 31u));
  }
  let centeredSum = addFixedPoint(previousSum, negateFixedPoint(multiplied));
  return getFixedPointMomentIncrement(centeredSum, sampleCount, 0.0, encodingExponent, squaredExponent);
}`;
