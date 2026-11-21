// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Preserve the anchor subtraction for nearby same-sign normal float32 components. Integer
 * significands prevent Metal fast math from folding `(value - anchor) - meanOffset` into
 * `value - (anchor + meanOffset)`, which rounds away a low-contrast mean offset.
 * Larger exponent gaps, zero and opposite signs use ordinary subtraction; those cases do not
 * cancel a shared large offset. The integer branch also covers adjacent binades and negatives.
 * @internal
 */
export const RASTER_EMBEDDING_CENTERING_WGSL = /* wgsl */ `
fn getEmbeddingAnchorDifference(value: f32, anchor: f32) -> f32 {
  let valueBits = bitcast<u32>(value);
  let anchorBits = bitcast<u32>(anchor);
  let valueMagnitude = valueBits & 0x7fffffffu;
  let anchorMagnitude = anchorBits & 0x7fffffffu;
  let valueExponent = valueMagnitude >> 23u;
  let anchorExponent = anchorMagnitude >> 23u;
  let commonExponent = min(valueExponent, anchorExponent);
  if (commonExponent > 0u && max(valueExponent, anchorExponent) < 255u &&
      max(valueExponent, anchorExponent) - commonExponent <= 1u &&
      ((valueBits ^ anchorBits) & 0x80000000u) == 0u) {
    // Each shifted significand is below 2^25; signed subtraction cannot overflow i32.
    let valueUnits = ((valueMagnitude & 0x7fffffu) | 0x800000u) << (valueExponent - commonExponent);
    let anchorUnits = ((anchorMagnitude & 0x7fffffu) | 0x800000u) << (anchorExponent - commonExponent);
    let differenceUnits = i32(valueUnits) - i32(anchorUnits);
    let signedUnits = select(differenceUnits, -differenceUnits, (valueBits & 0x80000000u) != 0u);
    return ldexp(f32(signedUnits), i32(commonExponent) - 150);
  }
  return value - anchor;
}
`;
