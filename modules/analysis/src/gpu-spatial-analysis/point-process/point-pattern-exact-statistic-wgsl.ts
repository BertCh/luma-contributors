// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Exact comparison of weighted squared deviations of finite f32 inputs. Each dyadic term
 * is accumulated in units of 2^-447 (three minimum-subnormal input factors). A term is
 * below 2^386 and 4096 ordinates need fewer than 846 bits; 28 words provide 896 bits.
 * Expanding weight*(value-center)^2 also preserves subtraction/product residuals. All
 * significand products and carries use integers, so Metal cannot reassociate them away.
 * Requires observed/simulations/reference/weights bindings and CURVE_ROWS/SIMULATION_COUNT.
 * @internal
 */
export const POINT_PATTERN_EXACT_STATISTIC_WGSL = /* wgsl */ `
fn addEnvelopeProduct(accumulator: ptr<function, array<u32, 28>>, firstBits: u32, secondBits: u32, weightBits: u32, negative: bool, extraExponent: i32) {
  let magnitudes = vec3u(firstBits, secondBits, weightBits) & vec3u(0x7fffffffu);
  if (any(magnitudes == vec3u(0u)) || any(magnitudes >= vec3u(0x7f800000u))) { return; }
  let biased = magnitudes >> vec3u(23u);
  let significands = (magnitudes & vec3u(0x7fffffu)) | select(vec3u(0u), vec3u(0x800000u), biased != vec3u(0u));
  let exponents = select(vec3i(-149), vec3i(biased) - vec3i(150), biased != vec3u(0u));
  // Base-256 convolution: at most seven products of three bytes occupy a digit,
  // so each unnormalized digit is below 7*255^3 < 2^27, safely within u32.
  var digits = array<u32, 9>();
  for (var first = 0u; first < 3u; first++) {
    for (var second = 0u; second < 3u; second++) {
      for (var third = 0u; third < 3u; third++) {
        digits[first + second + third] += ((significands.x >> (8u * first)) & 255u) * ((significands.y >> (8u * second)) & 255u) * ((significands.z >> (8u * third)) & 255u);
      }
    }
  }
  for (var digit = 0u; digit < 8u; digit++) { digits[digit + 1u] += digits[digit] >> 8u; digits[digit] &= 255u; }
  let product = array<u32, 3>(digits[0] | (digits[1] << 8u) | (digits[2] << 16u) | (digits[3] << 24u), digits[4] | (digits[5] << 8u) | (digits[6] << 16u) | (digits[7] << 24u), digits[8]);
  let shift = u32(exponents.x + exponents.y + exponents.z + extraExponent + 447);
  let firstWord = shift / 32u;
  let bit = shift % 32u;
  var carry = 0u;
  for (var word = firstWord; word < 28u; word++) {
    let part = word - firstWord;
    var term = 0u;
    if (part < 3u) { term = product[part] << bit; }
    if (bit > 0u && part > 0u && part <= 3u) { term |= product[part - 1u] >> (32u - bit); }
    let previous = (*accumulator)[word];
    if (negative) {
      let partial = previous - term;
      (*accumulator)[word] = partial - carry;
      carry = select(0u, 1u, previous < term || partial < carry);
    } else {
      let partial = previous + term;
      let next = partial + carry;
      (*accumulator)[word] = next;
      carry = select(0u, 1u, partial < previous || next < partial);
    }
    if (part >= 3u && carry == 0u) { break; }
  }
}
fn getEnvelopeExactStatistic(curve: u32) -> array<u32, 28> {
  var accumulator = array<u32, 28>();
  for (var row = 0u; row < CURVE_ROWS; row++) {
    var value = observed[observedOffset + row];
    if (curve < SIMULATION_COUNT) { value = simulations[simulationsOffset + curve * CURVE_ROWS + row]; }
    let valueBits = bitcast<u32>(value);
    let centerBits = bitcast<u32>(reference[referenceOffset + row]);
    let weightBits = bitcast<u32>(weights[weightsOffset + row]);
    addEnvelopeProduct(&accumulator, valueBits, valueBits, weightBits, false, 0);
    addEnvelopeProduct(&accumulator, centerBits, centerBits, weightBits, false, 0);
    addEnvelopeProduct(&accumulator, valueBits, centerBits, weightBits, ((valueBits ^ centerBits) >> 31u) == 0u, 1);
  }
  return accumulator;
}
fn envelopeExactAtLeast(first: array<u32, 28>, second: array<u32, 28>) -> bool {
  for (var remaining = 28u; remaining > 0u; remaining--) {
    let word = remaining - 1u;
    if (first[word] != second[word]) { return first[word] > second[word]; }
  }
  return true;
}
`;

/** @internal Exact maximum absolute deviation, compared through its monotone square. */
export const POINT_PATTERN_EXACT_MAXIMUM_WGSL = POINT_PATTERN_EXACT_STATISTIC_WGSL.replace(
  /fn getEnvelopeExactStatistic\(curve: u32\) -> array<u32, 28> \{[\s\S]*?(?=fn envelopeExactAtLeast)/,
  `fn getEnvelopeExactMaximum(curve: u32) -> array<u32, 28> {
  var maximum = array<u32, 28>();
  for (var row = 0u; row < CURVE_ROWS; row++) {
    var value = observed[observedOffset + row];
    if (curve < SIMULATION_COUNT) { value = simulations[simulationsOffset + curve * CURVE_ROWS + row]; }
    let valueBits = bitcast<u32>(value);
    let centerBits = bitcast<u32>(reference[referenceOffset + row]);
    var squared = array<u32, 28>();
    addEnvelopeProduct(&squared, valueBits, valueBits, 0x3f800000u, false, 0);
    addEnvelopeProduct(&squared, centerBits, centerBits, 0x3f800000u, false, 0);
    addEnvelopeProduct(&squared, valueBits, centerBits, 0x3f800000u, ((valueBits ^ centerBits) >> 31u) == 0u, 1);
    if (envelopeExactAtLeast(squared, maximum)) { maximum = squared; }
  }
  return maximum;
}
`
);
