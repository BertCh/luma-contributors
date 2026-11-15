// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {IS_FINITE_VALUE_WGSL} from './wgsl-common';

/**
 * Computes C(n) = 1 - beta^n by composing positive gaps from one. WGSL pow inherits exp2/log2
 * error, amplified by subtracting its result from one when beta is near one (WGSL 15.7.4).
 * C(1) = 1 - beta is exact for float32 betas near one by Sterbenz's lemma. Binary doubling uses
 * C(2n) = C(n) * (2 - C(n)); selected blocks use C(n+m) = C(n) + (1 - C(n)) * C(m).
 * Both recurrences keep the small gap directly, without subtracting an approximate power from
 * one or accumulating rounding in powers near one. All added terms are nonnegative. No
 * data-sized walk or persistent beta-power state is needed.
 * @internal
 */
export function getAdamBiasCorrectionWGSL(maximumIterations: number): string {
  const bitCount = Math.ceil(Math.log2(maximumIterations + 1));
  return /* wgsl */ `
fn getAdamBiasCorrection(beta: f32, iteration: u32) -> f32 {
  var remaining = iteration;
  var correction = 0.0;
  var blockCorrection = 1.0 - beta;
  for (var bit = 0u; bit < ${bitCount}u; bit++) {
    if (remaining == 0u) { break; }
    if ((remaining & 1u) != 0u) {
      correction += (1.0 - correction) * blockCorrection;
    }
    remaining >>= 1u;
    blockCorrection *= 2.0 - blockCorrection;
  }
  return correction;
}
`;
}

/**
 * Range-controlled Adam moment, update and loss-comparison arithmetic. Float reassociation can square the gradient
 * before its small multiplier or combine sqrt divisions into an overflowing corrected moment.
 * Split magnitudes with integer bit operations so these rearrangements cannot cross the split.
 * Large gradient contributions multiply only significands, then restore the final contribution's
 * exponent. The corrected square root divides significands in [1, 2), then restores half the
 * exponent difference. Zero and subnormal magnitudes have explicit decompositions.
 * @internal
 */
export const ADAM_MOMENT_RANGE_WGSL = /* wgsl */ `
${IS_FINITE_VALUE_WGSL}
// Squaring at most 2^63 stays below 2^126, with headroom for finite float32 products.
const ADAM_MAXIMUM_DIRECT_GRADIENT: f32 = 9223372036854775808.0;

struct AdamMagnitudeParts {
  significand: f32,
  exponent: i32
}

fn getAdamMagnitudeParts(value: f32) -> AdamMagnitudeParts {
  let magnitudeBits = bitcast<u32>(value) & 0x7fffffffu;
  let exponentBits = magnitudeBits >> 23u;
  let fractionBits = magnitudeBits & 0x7fffffu;
  if (exponentBits != 0u) {
    return AdamMagnitudeParts(bitcast<f32>(fractionBits | 0x3f800000u), i32(exponentBits) - 127);
  }
  if (fractionBits == 0u) { return AdamMagnitudeParts(0.0, 0); }
  let shift = countLeadingZeros(fractionBits) - 8u;
  let significandBits = ((fractionBits << shift) & 0x7fffffu) | 0x3f800000u;
  return AdamMagnitudeParts(bitcast<f32>(significandBits), -126 - i32(shift));
}

// Restore only the final scale, with one float32 rounding even at subnormal boundaries.
// Integer encoding keeps a normal result from depending on an underflowing intermediate.
fn getAdamScaledMagnitude(significand: f32, exponent: i32) -> f32 {
  let parts = getAdamMagnitudeParts(significand);
  if (parts.significand == 0.0) { return 0.0; }
  let combinedExponent = exponent + parts.exponent;
  let fractionBits = bitcast<u32>(parts.significand) & 0x7fffffu;
  if (combinedExponent > 127) {
    var infinityBits = 0x7f800000u;
    return bitcast<f32>(infinityBits);
  }
  if (combinedExponent >= -126) {
    return bitcast<f32>((u32(combinedExponent + 127) << 23u) | fractionBits);
  }
  if (combinedExponent < -150) { return 0.0; }
  let shift = u32(-126 - combinedExponent);
  let fullSignificand = fractionBits | 0x800000u;
  let retained = fullSignificand >> shift;
  let remainder = fullSignificand & ((1u << shift) - 1u);
  let halfway = 1u << (shift - 1u);
  let roundUp = remainder > halfway || (remainder == halfway && (retained & 1u) != 0u);
  return bitcast<f32>(retained + select(0u, 1u, roundUp));
}

fn getAdamSecondMomentContribution(gradient: f32, beta: f32) -> f32 {
  if (abs(gradient) <= ADAM_MAXIMUM_DIRECT_GRADIENT) {
    return (1.0 - beta) * gradient * gradient;
  }
  let gradientParts = getAdamMagnitudeParts(gradient);
  let scaledSquare = (1.0 - beta) * gradientParts.significand * gradientParts.significand;
  return getAdamScaledMagnitude(scaledSquare, 2 * gradientParts.exponent);
}

fn getAdamCorrectedSecondMomentRoot(second: f32, correction: f32) -> f32 {
  if (!isFiniteValue(second)) { return second; }
  let secondParts = getAdamMagnitudeParts(second);
  let correctionParts = getAdamMagnitudeParts(correction);
  let exponentDifference = secondParts.exponent - correctionParts.exponent;
  let halfExponent = exponentDifference >> 1;
  let oddExponent = exponentDifference - 2 * halfExponent;
  let normalizedRatio = (secondParts.significand / correctionParts.significand)
    * select(1.0, 2.0, oddExponent != 0);
  return getAdamScaledMagnitude(sqrt(normalizedRatio), halfExponent);
}

// Multiply/divide bounded significands before restoring the final update's exponent. This
// includes firstCorrection in the scale so no corrected first moment or rate*first is formed.
fn getAdamWeightDelta(learningRate: f32, first: f32, firstCorrection: f32, denominator: f32) -> f32 {
  if (first == 0.0) { return 0.0; }
  if (!isFiniteValue(first) || !isFiniteValue(denominator)) {
    return learningRate * (first / firstCorrection) / denominator;
  }
  let rateParts = getAdamMagnitudeParts(learningRate);
  let firstParts = getAdamMagnitudeParts(first);
  let correctionParts = getAdamMagnitudeParts(firstCorrection);
  let denominatorParts = getAdamMagnitudeParts(denominator);
  let significand = (rateParts.significand * firstParts.significand)
    / (correctionParts.significand * denominatorParts.significand);
  let exponent = rateParts.exponent + firstParts.exponent
    - correctionParts.exponent - denominatorParts.exponent;
  let magnitude = getAdamScaledMagnitude(significand, exponent);
  return bitcast<f32>(bitcast<u32>(magnitude) | (bitcast<u32>(first) & 0x80000000u));
}

fn isAdamLossPlateau(current: f32, previous: f32, tolerance: f32) -> bool {
  let previousMagnitude = abs(previous);
  // Subtracting a value smaller than one cannot overflow a finite current loss.
  if (previousMagnitude < 1.0) { return abs(current - previous) <= tolerance; }
  let previousParts = getAdamMagnitudeParts(previous);
  let oppositeSigns = ((bitcast<u32>(current) ^ bitcast<u32>(previous)) & 0x80000000u) != 0u;
  if (!oppositeSigns) {
    // Same-sign subtraction stays finite and preserves small changes between adjacent losses.
    let differenceParts = getAdamMagnitudeParts(current - previous);
    let relativeChange = getAdamScaledMagnitude(differenceParts.significand / previousParts.significand,
      differenceParts.exponent - previousParts.exponent);
    return relativeChange <= tolerance;
  }
  let currentParts = getAdamMagnitudeParts(current);
  let ratio = getAdamScaledMagnitude(currentParts.significand / previousParts.significand,
    currentParts.exponent - previousParts.exponent);
  // ratio <= abs(current), since abs(previous) >= 1. Adding one cannot overflow float32 here.
  // Integer normalization prevents a compiler from reassociating this into (current-previous).
  return ratio + 1.0 <= tolerance;
}
`;
