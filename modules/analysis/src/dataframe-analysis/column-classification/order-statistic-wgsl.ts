// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL of the one order-statistic policy shared by `GPUColumnQuantiles`, `GPUClassBreaks`
 * (`'quantile'` and `'box-plot'`) and the `GPUGroupStatistics` medians and percentiles.
 *
 * Policy (finding B-14; the tables in `test/gpu-dataframe/dataframe-edge-policy.ts` pin it):
 * - The sample is the finite values of the selected rows. NaN and +-Infinity are missing (rule 8:
 *   a kernel that interpolates treats non-finite as missing), so the result equals
 *   `numpy.quantile(x[numpy.isfinite(x)], p)`. numpy and pandas keep infinities; this is the one
 *   documented deviation.
 * - `-0` and `+0` are one value (numpy orders them as equal); a result that is zero is `+0`.
 * - A probability that is NaN or outside `[0, 1]` gives NaN (numpy raises; a GPU kernel cannot).
 * - An empty sample gives NaN.
 * - `'linear'` is R-7 (numpy, pandas and d3 default): `h = fround((n - 1) * p)`, then
 *   `x[floor h] + (x[floor h + 1] - x[floor h]) * (h - floor h)`, every step rounded to f32 with
 *   no fused multiply-add, so a CPU `Math.fround` mirror is bit exact. When the difference
 *   overflows (values of opposite sign near the f32 limit) the convex form
 *   `x[lo] * (1 - t) + x[hi] * t` is used instead, which stays finite.
 * - Resolution of the rank: `h` is a float32, so its fraction has a step of `2^(e - 23)` for `h`
 *   in `[2^e, 2^(e + 1))`: 1/8 at a rank near one million, 1/2 near four million, and no
 *   fraction at all from `2^23` (8,388,608). The result is then off by at most that step times
 *   the gap between two adjacent order statistics (numpy computes `h` in float64). The float32
 *   product is kept on purpose: `fround(1000 * fround(0.7))` is 700, the rank numpy gets from a
 *   float64 0.7, while the exact product of the float32 0.7 is 699.99999; over common decimal
 *   fractions and up to 5,000 rows the exact product gives another `floor(h)` than numpy 17
 *   times as often. An exact rank needs the probability as a rational:
 *   {@link RATIONAL_ORDER_STATISTIC_RANK_WGSL}, used by `GPUColumnQuantiles` when
 *   `rationalProbabilities` is given (math review 2026-11-10, dataframe-08). `GPUClassBreaks` and
 *   `GPUGroupStatistics` use the float32 rank only.
 *
 * Functions: `isOrderStatisticProbability(p)`, `getOrderStatisticKey(value)` (canonical `-0`,
 * order-preserving u32; only for finite values), `interpolateOrderStatistic(low, high, fraction,
 * zero)` and `getOrderStatisticMidpoint(low, high, zero)`. `zero` must be a runtime `0u` the
 * compiler cannot see (for example `count >> 31u` with `count < 2^31`); it blocks multiply-add
 * fusion.
 *
 * @internal
 */
export const ORDER_STATISTIC_WGSL = /* wgsl */ `
fn isOrderStatisticProbability(p: f32) -> bool {
  // Ordered comparisons with NaN are false, but a compiler may fold them; test the bits.
  let bits = bitcast<u32>(p);
  if ((bits & 0x7f800000u) == 0x7f800000u) {
    return false;
  }
  return p >= 0.0 && p <= 1.0;
}

fn getOrderStatisticKey(value: f32) -> u32 {
  var bits = bitcast<u32>(value);
  if ((bits & 0x7fffffffu) == 0u) {
    bits = 0u;
  }
  return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

// Identity the compiler cannot fold: the bit operation on a runtime zero stops multiply-add
// fusion, so every f32 step rounds exactly like a Math.fround mirror.
fn roundOrderStatisticStep(value: f32, zero: u32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) | zero);
}

fn interpolateOrderStatistic(low: f32, high: f32, fraction: f32, zero: u32) -> f32 {
  if (fraction == 0.0 || low == high) {
    return low;
  }
  let difference = roundOrderStatisticStep(high - low, zero);
  if ((bitcast<u32>(difference) & 0x7f800000u) != 0x7f800000u) {
    return low + roundOrderStatisticStep(difference * fraction, zero);
  }
  // high - low overflowed; a convex combination of two finite values cannot.
  return roundOrderStatisticStep(low * (1.0 - fraction), zero) +
    roundOrderStatisticStep(high * fraction, zero);
}

fn getOrderStatisticMidpoint(low: f32, high: f32, zero: u32) -> f32 {
  if (low == high) {
    return low;
  }
  let sum = roundOrderStatisticStep(low + high, zero);
  if ((bitcast<u32>(sum) & 0x7f800000u) != 0x7f800000u) {
    return sum * 0.5;
  }
  return roundOrderStatisticStep(low * 0.5, zero) + roundOrderStatisticStep(high * 0.5, zero);
}
`;

/**
 * WGSL of the exact order-statistic rank for a probability given as a ratio of two `u32`,
 * `p = numerator / denominator` (math review 2026-11-10, dataframe-08). Requires `UINT64_WGSL`
 * (`analysis-core/kernels/uint64-wgsl.ts`).
 *
 * With `count` sample values, `(count - 1) * numerator` is formed as an exact 64-bit product and
 * divided by `denominator` with integers, so the lower rank `floor((count - 1) p)` and the
 * remainder are exact for every `count` below `2^32`: no float product is rounded, at any row
 * count. The interpolation fraction `remainder / denominator` is rounded once, to the nearest
 * float32 (ties to even), by an integer division and a bit pattern, because float division is
 * not correctly rounded on every GPU (contributor contract rule 11).
 *
 * Functions:
 * - `isRationalOrderStatisticProbability(numerator, denominator)`: `denominator != 0` and
 *   `numerator <= denominator`, the counterpart of `isOrderStatisticProbability`.
 * - `getRationalOrderStatisticRank(count, numerator, denominator) -> RationalOrderStatisticRank`
 *   with `lower` and `remainder`; `count >= 1` and a valid probability are the caller's duty.
 * - `getRationalOrderStatisticUpperRank(rank)`: `ceil((count - 1) p)`.
 * - `getRationalOrderStatisticNearestRank(rank, denominator)`: round half to even (numpy
 *   `nearest`), comparing `remainder` with `denominator - remainder`.
 * - `getRationalOrderStatisticFraction(remainder, denominator) -> f32` for
 *   `remainder < denominator`.
 *
 * Checked against Python integer and `fractions.Fraction` arithmetic for counts up to `2^32 - 1`
 * in `test/gpu-dataframe/column-classification/gpu-column-quantiles-rational.spec.ts`.
 *
 * @internal
 */
export const RATIONAL_ORDER_STATISTIC_RANK_WGSL = /* wgsl */ `
struct RationalOrderStatisticRank {
  lower: u32,
  remainder: u32,
}

fn isRationalOrderStatisticProbability(numerator: u32, denominator: u32) -> bool {
  return denominator != 0u && numerator <= denominator;
}

fn getRationalOrderStatisticRank(count: u32, numerator: u32, denominator: u32) -> RationalOrderStatisticRank {
  // numerator <= denominator, so the quotient is at most count - 1 and fits the low word.
  let division = divideUint64ByUint32(multiplyUint32Wide(count - 1u, numerator), denominator);
  return RationalOrderStatisticRank(division.quotient.y, division.remainder);
}

fn getRationalOrderStatisticUpperRank(rank: RationalOrderStatisticRank) -> u32 {
  // A nonzero remainder means (count - 1) p is below count - 1, so lower + 1 is a valid rank.
  return rank.lower + select(0u, 1u, rank.remainder != 0u);
}

fn getRationalOrderStatisticNearestRank(rank: RationalOrderStatisticRank, denominator: u32) -> u32 {
  // remainder / denominator against 1 / 2 without doubling, which could overflow.
  let complement = denominator - rank.remainder;
  if (rank.remainder < complement) {
    return rank.lower;
  }
  if (rank.remainder > complement) {
    return rank.lower + 1u;
  }
  return rank.lower + (rank.lower & 1u);
}

fn getRationalOrderStatisticFraction(remainder: u32, denominator: u32) -> f32 {
  if (remainder == 0u || remainder >= denominator) {
    return 0.0;
  }
  // Scale so the integer quotient has 25 or 26 bits: 24 for the significand, then a guard bit.
  // remainder * 2^shift is below 2^58, so it fits 64 bits.
  var shift = 25u + firstLeadingBit(denominator) - firstLeadingBit(remainder);
  let division = divideUint64ByUint32(shiftLeftUint64(vec2<u32>(0u, remainder), shift), denominator);
  var quotient = division.quotient.y;
  var sticky = division.remainder != 0u;
  if (quotient >= 0x2000000u) {
    sticky = sticky || (quotient & 1u) != 0u;
    quotient = quotient >> 1u;
    shift = shift - 1u;
  }
  // quotient is in [2^24, 2^25): 24 significand bits and one guard bit. Round to nearest even.
  var significand = quotient >> 1u;
  if ((quotient & 1u) != 0u && (sticky || (significand & 1u) != 0u)) {
    significand = significand + 1u;
  }
  // The value is significand * 2^(1 - shift), with significand in [2^23, 2^24].
  var exponent = 151u - shift;
  if (significand == 0x1000000u) {
    significand = 0x800000u;
    exponent = exponent + 1u;
  }
  return bitcast<f32>((exponent << 23u) | (significand & 0x7fffffu));
}
`;
