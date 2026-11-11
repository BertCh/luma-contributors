// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * WGSL `getAdditionRoundingError(first, second, sum) -> f32`: the exact rounding error
 * `(first + second) - sum` of the f32 addition `sum = first + second`, for finite operands of
 * either sign. It is the correction term of a compensated (Neumaier) sum:
 *
 * ```wgsl
 * let next = sum + term;
 * compensation = compensation + getAdditionRoundingError(sum, term, next);
 * sum = next;
 * // ... and the result is sum + compensation
 * ```
 *
 * Why this exists (contributor contract section 11). The usual float forms of the term, Kahan's
 * `(next - sum) - term`, Neumaier's `(sum - next) + term` and the branch-free two-sum, are
 * algebraically zero. Shader compilers that reassociate floating-point expressions (Metal fast
 * math) remove them: on the test GPU a "compensated" sum written that way is bit for bit the plain
 * float32 sum (`test/utils/addition-rounding-error-wgsl.spec.ts` pins this). Here the term is read
 * off the integer significands with wrapping `u32` arithmetic, and one `ldexp` scales the result:
 * integer bit operations give the compiler no identity to fold.
 *
 * How. With `|larger| >= |smaller|`, all three values are integers in units of the smaller
 * operand's last place, and their combination `larger + smaller - sum` is at most `2^25` of those
 * units in magnitude whenever the exponents are at most 25 apart, so it can be formed modulo
 * `2^32`. The rounding error of a float addition is itself a float, so the conversion is exact.
 * When the exponents are more than 25 apart the sum is the larger operand and the error is the
 * smaller one.
 *
 * Returns 0 when the addition is exact (an operand is zero, or the operands cancel), when an
 * operand or the sum is not finite, and when the smaller operand is below `2^-103` (its error
 * would be subnormal). `sum` must be the float32 sum of the two operands computed by the caller;
 * any other value gives a meaningless result.
 *
 * A sum over a data-sized range still belongs to the fixed-point accumulator
 * (`fixed-point-accumulator.ts`) or to `sorted-segment-sums.ts`: this helper is for a bounded
 * serial walk (a prefix along one line, a running total inside one cell).
 *
 * Checked on the GPU against float64 in `test/utils/addition-rounding-error-wgsl.spec.ts`.
 *
 * @internal
 */
export const ADDITION_ROUNDING_ERROR_WGSL = /* wgsl */ `
fn getAdditionRoundingError(first: f32, second: f32, sum: f32) -> f32 {
  let firstBits = bitcast<u32>(first);
  let secondBits = bitcast<u32>(second);
  let sumBits = bitcast<u32>(sum);
  let firstMagnitude = firstBits & 0x7fffffffu;
  let secondMagnitude = secondBits & 0x7fffffffu;
  let sumMagnitude = sumBits & 0x7fffffffu;
  // A zero operand makes the addition exact; an infinity or a NaN has no rounding error to report.
  if (firstMagnitude == 0u || secondMagnitude == 0u ||
      max(max(firstMagnitude, secondMagnitude), sumMagnitude) >= 0x7f800000u) {
    return 0.0;
  }
  let firstIsLarger = firstMagnitude >= secondMagnitude;
  let largerBits = select(secondBits, firstBits, firstIsLarger);
  let smallerBits = select(firstBits, secondBits, firstIsLarger);
  let largerMagnitude = largerBits & 0x7fffffffu;
  let smallerMagnitude = smallerBits & 0x7fffffffu;
  // Biased exponents (1 for subnormals) and 24-bit significands: |value| = significand * 2^(exponent - 150).
  let largerExponent = max(largerMagnitude >> 23u, 1u);
  let smallerExponent = max(smallerMagnitude >> 23u, 1u);
  let sumExponent = max(sumMagnitude >> 23u, 1u);
  // Below 2^-103 the error would be subnormal, where ldexp is not dependable: report none.
  if (smallerExponent < 24u) {
    return 0.0;
  }
  let gap = largerExponent - smallerExponent;
  if (gap > 25u) {
    // The smaller operand is below a quarter of the larger one's last place: the sum is the larger
    // operand and the whole smaller operand is the error.
    if (sumBits == largerBits) {
      return bitcast<f32>(smallerBits);
    }
    return 0.0;
  }
  // The sum's last place, as a left shift from the smaller operand's last place. A rounded sum has
  // the larger operand's exponent, the next one (a carry) or the one below (a borrow). Anything
  // lower is a cancellation, which is exact, and so is a sum whose last place is finer than the
  // smaller operand's.
  if (sumMagnitude == 0u || sumExponent + 1u < largerExponent || sumExponent > largerExponent + 1u ||
      sumExponent < smallerExponent) {
    return 0.0;
  }
  let sumShift = sumExponent - smallerExponent;
  let largerSignificand = (largerMagnitude & 0x7fffffu) | select(0u, 0x800000u, largerMagnitude >= 0x800000u);
  let smallerSignificand = (smallerMagnitude & 0x7fffffu) | select(0u, 0x800000u, smallerMagnitude >= 0x800000u);
  let sumSignificand = (sumMagnitude & 0x7fffffu) | select(0u, 0x800000u, sumMagnitude >= 0x800000u);
  // Signed integers in units of the smaller operand's last place, modulo 2^32. Each term can pass
  // 2^32, but their combination is at most 2^25 in magnitude, so the wrapped result is exact.
  let largerUnits = largerSignificand << gap;
  let sumUnits = sumSignificand << sumShift;
  let signedLarger = select(largerUnits, 0u - largerUnits, (largerBits & 0x80000000u) != 0u);
  let signedSmaller = select(smallerSignificand, 0u - smallerSignificand, (smallerBits & 0x80000000u) != 0u);
  let signedSum = select(sumUnits, 0u - sumUnits, (sumBits & 0x80000000u) != 0u);
  let errorUnits = bitcast<i32>(signedLarger + signedSmaller - signedSum);
  // The rounding error of an addition is an f32, so the conversion and the scaling are exact.
  return ldexp(f32(errorUnits), i32(smallerExponent) - 150);
}
`;
