// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getDenseLUWGSL} from '../../analysis-core/kernels/dense-solve-wgsl';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {ADDITION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/addition-rounding-error-wgsl';
import {MULTIPLICATION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/multiplication-rounding-error-wgsl';
import {ROUNDED_DIFFERENCE_WGSL} from './rounded-difference-wgsl';

/** Bounded interpolation candidates, never a sample-based decision: every row verifies the result. */
export const SPATIAL_ERROR_GM_AFFINE_CANDIDATE_ROW_COUNT = 256;

/**
 * Proposes dyadic and common-denominator affine vectors from anchored, row-pivoted interpolation.
 * A failed/singular candidate falls back to the original OLS coefficients. No approximate
 * fit is accepted here: the subsequent integer certificate checks the ORIGINAL rows.
 * Bindings: design, response, olsCoefficients, workspace, candidateOut. Constants: K, Q,
 * ROWS, W_RUNTIME_ZERO. candidateOut has two blocks of Q numerators plus one denominator.
 * The caller supplies isFiniteValue.
 * @internal
 */
export function getSpatialErrorGMAffineCandidateWGSL(predictorCount: number): string {
  return `${getDenseLUWGSL(predictorCount)}
${ROUNDED_DIFFERENCE_WGSL}
${ADDITION_ROUNDING_ERROR_WGSL}
${MULTIPLICATION_ROUNDING_ERROR_WGSL}

// A proposal only: the exact integer check below decides whether this rationalization is valid.
fn proposeRationalCoefficient(value: f32) -> vec2<f32> {
  if (value == 0.0) { return vec2<f32>(0.0, 1.0); }
  var remainder = abs(value);
  var previousNumerator = 0.0; var numerator = 1.0;
  var previousDenominator = 1.0; var denominator = 0.0;
  for (var iteration = 0u; iteration < 24u; iteration++) {
    let integer = floor(remainder);
    let nextNumerator = integer * numerator + previousNumerator;
    let nextDenominator = integer * denominator + previousDenominator;
    if (!(nextDenominator > 0.0) || nextDenominator > 65536.0 || !isFiniteValue(nextNumerator)) { break; }
    previousNumerator = numerator; numerator = nextNumerator;
    previousDenominator = denominator; denominator = nextDenominator;
    if (abs(numerator / denominator - abs(value)) <= 2.384185791015625e-7 * abs(value)) {
      return vec2<f32>(sign(value) * numerator, denominator);
    }
    let fraction = remainder - integer;
    if (!(fraction > 0.0)) { break; }
    remainder = 1.0 / fraction;
  }
  return vec2<f32>(value, 1.0);
}

fn proposeCommonDenominator() {
  var fractions: array<vec2<f32>, ${predictorCount + 1}>;
  var denominator = 1u;
  for (var column = 1u; column < Q; column++) {
    fractions[column] = proposeRationalCoefficient(candidateOut[candidateOutOffset + column]);
    let next = u32(fractions[column].y);
    var first = denominator; var second = next;
    while (second > 0u) { let remainder = first % second; first = second; second = remainder; }
    let multiplier = next / max(first, 1u);
    if (denominator > 16777216u / max(multiplier, 1u)) { return; }
    denominator *= multiplier;
  }
  var numerators: array<f32, ${predictorCount + 1}>;
  for (var column = 1u; column < Q; column++) {
    numerators[column] = fractions[column].x * f32(denominator / u32(fractions[column].y));
  }
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  let product = bitcast<f32>(bitcast<u32>(response[responseOffset] * f32(denominator)) ^ runtimeZero);
  var intercept = vec2<f32>(product, getMultiplicationRoundingError(response[responseOffset], f32(denominator), product));
  for (var column = 1u; column < Q; column++) {
    intercept = subtractAffineProduct(intercept, design[designOffset + column - 1u], numerators[column], runtimeZero);
  }
  numerators[0] = intercept.x + intercept.y;
  for (var column = 0u; column < Q; column++) {
    candidateOut[candidateOutOffset + Q + 1u + column] = numerators[column];
  }
  candidateOut[candidateOutOffset + 2u * Q + 1u] = f32(denominator);
}

// Preserve both errors of value - first*second; integer helpers prevent reassociation.
fn subtractAffineProduct(value: vec2<f32>, first: f32, second: f32, runtimeZero: u32) -> vec2<f32> {
  let product = bitcast<f32>(bitcast<u32>(first * second) ^ runtimeZero);
  let next = bitcast<f32>(bitcast<u32>(value.x - product) ^ runtimeZero);
  return vec2<f32>(next, value.y + getAdditionRoundingError(value.x, -product, next)
    - getMultiplicationRoundingError(first, second, product));
}

fn proposeAffineCandidate() {
  for (var column = 0u; column < Q; column++) {
    candidateOut[candidateOutOffset + column] = olsCoefficients[olsCoefficientsOffset + column];
    candidateOut[candidateOutOffset + Q + 1u + column] = olsCoefficients[olsCoefficientsOffset + column];
  }
  candidateOut[candidateOutOffset + Q] = 1.0;
  candidateOut[candidateOutOffset + 2u * Q + 1u] = 1.0;
  proposeCommonDenominator();
  if (ROWS <= K) { return; }
  let runtimeZero = bitcast<u32>(workspace[workspaceOffset + W_RUNTIME_ZERO]);
  let sampleCount = min(ROWS, ${SPATIAL_ERROR_GM_AFFINE_CANDIDATE_ROW_COUNT}u);
  let sampleDenominator = max(sampleCount - 1u, 1u);
  var basis: array<f32, ${predictorCount * predictorCount}>;
  var selected: array<u32, ${predictorCount}>;
  var interpolation: array<f32, ${predictorCount * predictorCount}>;
  var targets: array<f32, ${predictorCount}>;
  for (var pivot = 0u; pivot < K; pivot++) {
    var largest = 0.0;
    var selectedRow = 0u;
    var best: array<f32, ${predictorCount}>;
    for (var sampleIndex = 0u; sampleIndex < sampleCount; sampleIndex++) {
      // Quotient/remainder form avoids overflowing sampleIndex * (ROWS-1).
      let row = ((ROWS - 1u) / sampleDenominator) * sampleIndex
        + (((ROWS - 1u) % sampleDenominator) * sampleIndex) / sampleDenominator;
      var used = row == 0u;
      for (var previous = 0u; previous < pivot; previous++) { used = used || selected[previous] == row; }
      if (used) { continue; }
      var residual: array<f32, ${predictorCount}>;
      for (var column = 0u; column < K; column++) {
        residual[column] = getRoundedDifference(design[designOffset + row*K + column], design[designOffset + column], runtimeZero);
      }
      // Reorthogonalization only chooses interpolation rows; it never certifies a fit.
      for (var sweepIndex = 0u; sweepIndex < 2u; sweepIndex++) {
        for (var previous = 0u; previous < pivot; previous++) {
          var product = 0.0;
          for (var column = 0u; column < K; column++) { product += residual[column] * basis[previous*K + column]; }
          for (var column = 0u; column < K; column++) { residual[column] -= product * basis[previous*K + column]; }
        }
      }
      var norm = 0.0;
      for (var column = 0u; column < K; column++) { norm += residual[column] * residual[column]; }
      if (isFiniteValue(norm) && norm > largest) { largest = norm; selectedRow = row; best = residual; }
    }
    if (!(largest > 0.0)) { return; }
    selected[pivot] = selectedRow;
    for (var column = 0u; column < K; column++) {
      basis[pivot*K + column] = best[column] / sqrt(largest);
      interpolation[pivot*K + column] = getRoundedDifference(design[designOffset + selectedRow*K + column], design[designOffset + column], runtimeZero);
    }
    targets[pivot] = getRoundedDifference(response[responseOffset + selectedRow], response[responseOffset], runtimeZero);
  }
  var original = interpolation;
  var factor = factorDenseLU_${predictorCount}(&interpolation, K, 0u, 0.0, 0.0);
  if (factor.status != 0u) { return; }
  var slopes = targets;
  if (!solveDenseLU_${predictorCount}(&interpolation, &factor, &slopes)) { return; }
  // Two bounded refinement steps improve the proposal; exact row verification decides.
  for (var iteration = 0u; iteration < 2u; iteration++) {
    var correction: array<f32, ${predictorCount}>;
    for (var row = 0u; row < K; row++) {
      var residual = vec2<f32>(targets[row], 0.0);
      for (var column = 0u; column < K; column++) { residual = subtractAffineProduct(residual, original[row*K + column], slopes[column], runtimeZero); }
      correction[row] = residual.x + residual.y;
    }
    if (!solveDenseLU_${predictorCount}(&interpolation, &factor, &correction)) { return; }
    for (var column = 0u; column < K; column++) { slopes[column] += correction[column]; }
  }
  var intercept = vec2<f32>(response[responseOffset], 0.0);
  for (var column = 0u; column < K; column++) {
    if (!isFiniteValue(slopes[column])) { return; }
    intercept = subtractAffineProduct(intercept, design[designOffset + column], slopes[column], runtimeZero);
  }
  let interceptValue = intercept.x + intercept.y;
  if (!isFiniteValue(interceptValue)) { return; }
  candidateOut[candidateOutOffset] = interceptValue;
  for (var column = 0u; column < K; column++) { candidateOut[candidateOutOffset + 1u + column] = slopes[column]; }
  proposeCommonDenominator();
}
`;
}

/**
 * Exact anchored dyadic equality sum numerator_j*(x_j-x_0j) == denominator*(y-y_0),
 * using the shared 24-bit multiply
 * and 256-bit unsigned accumulators. At most 32 terms span <=200 exponent positions:
 * 48 product bits + 200 positions + 5 carry bits fit 256 bits. Unsupported nonfinite,
 * subnormal or wider-span terms return false (not a certificate). No residual tolerance.
 * Bindings: design, response, candidate as u32 views of their float32 bits, avoiding device
 * subnormal flushing before the unsupported-input check. Constants: K, Q. All rows must verify.
 * @internal
 */
export function getSpatialErrorGMAffineEqualityWGSL(): string {
  return `${EXACT_ORIENTATION_WGSL}
fn isExactAffineRow(row: u32, candidateBase: u32) -> bool {
  var lows: array<u32, 2u * Q>;
  var highs: array<u32, 2u * Q>;
  var exponents: array<i32, 2u * Q>;
  var negative: array<u32, 2u * Q>;
  var present: array<u32, 2u * Q>;
  var minimum = 100000;
  var maximum = -100000;
  for (var term = 0u; term < 2u * Q; term++) {
    var first = 0x3f800000u;
    var second = 0u;
    // Anchored exact identity: sum b_j*x_row - sum b_j*x_0 - d*y_row + d*y_0=0.
    // The expanded intercept need not be representable as one float32 (e.g. 16777217).
    if (term < K) {
      first = design[designOffset + row*K + term];
      second = candidate[candidateOffset + candidateBase + 1u + term];
    } else if (term < 2u * K) {
      first = design[designOffset + term - K];
      second = candidate[candidateOffset + candidateBase + 1u + term - K] ^ 0x80000000u;
    } else {
      first = response[responseOffset + select(row, 0u, term == 2u * K + 1u)];
      second = candidate[candidateOffset + candidateBase + Q];
      if (term == 2u * K) { second ^= 0x80000000u; }
    }
    let firstExponent = (first >> 23u) & 255u;
    let secondExponent = (second >> 23u) & 255u;
    if (firstExponent == 255u || secondExponent == 255u) { return false; }
    if ((first & 0x7fffffffu) == 0u || (second & 0x7fffffffu) == 0u) { continue; }
    if (firstExponent == 0u || secondExponent == 0u) { return false; }
    let product = multiply24((first & 0x7fffffu) | 0x800000u, (second & 0x7fffffu) | 0x800000u);
    lows[term] = product.x; highs[term] = product.y;
    exponents[term] = i32(firstExponent + secondExponent) - 300;
    negative[term] = (first ^ second) >> 31u;
    present[term] = 1u;
    minimum = min(minimum, exponents[term]); maximum = max(maximum, exponents[term]);
  }
  if (maximum < minimum) { return true; }
  if (maximum - minimum > 200) { return false; }
  var positive: array<u32, 8>;
  var negativeSum: array<u32, 8>;
  for (var term = 0u; term < 2u * Q; term++) {
    if (present[term] == 0u) { continue; }
    let shift = u32(exponents[term] - minimum);
    if (negative[term] != 0u) { accumulateShifted(&negativeSum, lows[term], highs[term], shift); }
    else { accumulateShifted(&positive, lows[term], highs[term], shift); }
  }
  for (var word = 0u; word < 8u; word++) { if (positive[word] != negativeSum[word]) { return false; } }
  return true;
}
`;
}
