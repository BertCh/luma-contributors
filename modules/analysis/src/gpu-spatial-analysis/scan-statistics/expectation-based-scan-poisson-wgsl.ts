// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL Poisson draws for the expectation-based scan over `PhiloxStream` (needs
 * `PERMUTATION_RANDOM_WGSL`, which carries `philoxOpenUnitFloat`; every uniform is in the open
 * interval `(0, 1)`): `drawPhiloxPoisson(stream, mean)` is exact in distribution below 30
 * (Knuth's product of uniforms) and PTRS (Hormann 1993) above it, in f32 arithmetic. Above 1e7 a
 * rounded normal is used, where f32 cannot resolve the PTRS acceptance test.
 *
 * @internal
 */
export const EXPECTATION_BASED_SCAN_POISSON_WGSL = /* wgsl */ `
// Uniform in the open interval (0, 1): the shared 23-bit form of PHILOX_WGSL, which cannot
// return 0 or 1. (The 24-bit form (word >> 8) + 0.5 rounds to exactly 1.0 for the largest word:
// 16777215.5 is not an f32. A draw of 1.0 gave the PTRS step a zero headroom to divide by.)
fn drawPhiloxUniform(stream: ptr<function, PhiloxStream>) -> f32 {
  return philoxOpenUnitFloat(nextPhiloxUint32(stream));
}
fn getLogFactorial(value: f32) -> f32 {
  // Stirling series of ln(value!): exact to f32 for value >= 10; smaller values are summed directly.
  if (value < 10.0) {
    var total = 0.0;
    var term = 2.0;
    loop {
      if (term > value) { break; }
      total += log(term);
      term += 1.0;
    }
    return total;
  }
  let shifted = value + 1.0;
  let inverse = 1.0 / shifted;
  return (shifted - 0.5) * log(shifted) - shifted + 0.9189385332046727 + inverse / 12.0 - inverse * inverse * inverse / 360.0;
}
fn drawPhiloxPoisson(stream: ptr<function, PhiloxStream>, mean: f32) -> u32 {
  if (!(mean > 0.0) || (bitcast<u32>(mean) & 0x7f800000u) == 0x7f800000u) {
    return 0u;
  }
  if (mean < 30.0) {
    let limit = exp(-mean);
    var product = 1.0;
    var count = 0u;
    loop {
      product *= drawPhiloxUniform(stream);
      if (product <= limit) {
        break;
      }
      count++;
      if (count >= 1024u) {
        break;
      }
    }
    return count;
  }
  if (mean >= 1e7) {
    let radius = sqrt(-2.0 * log(drawPhiloxUniform(stream)));
    let angle = 6.283185307179586 * drawPhiloxUniform(stream);
    return u32(max(floor(mean + sqrt(mean) * radius * cos(angle) + 0.5), 0.0));
  }
  let logMean = log(mean);
  let scale = 0.931 + 2.53 * sqrt(mean);
  let offset = -0.059 + 0.02483 * scale;
  let inverseAlpha = 1.1239 + 1.1328 / (scale - 3.4);
  let squeeze = 0.9277 - 3.6224 / (scale - 2.0);
  for (var attempt = 0u; attempt < 64u; attempt++) {
    let offsetUniform = drawPhiloxUniform(stream) - 0.5;
    let acceptance = drawPhiloxUniform(stream);
    let headroom = 0.5 - abs(offsetUniform);
    let candidate = floor((2.0 * offset / headroom + scale) * offsetUniform + mean + 0.43);
    if (headroom >= 0.07 && acceptance <= squeeze) {
      return u32(candidate);
    }
    if (candidate < 0.0 || (headroom < 0.013 && acceptance > headroom)) {
      continue;
    }
    if (log(acceptance) + log(inverseAlpha) - log(offset / (headroom * headroom) + scale) <= -mean + candidate * logMean - getLogFactorial(candidate)) {
      return u32(candidate);
    }
  }
  return u32(mean + 0.5);
}`;
