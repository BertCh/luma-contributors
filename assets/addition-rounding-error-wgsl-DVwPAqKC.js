var e=`
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
`;export{e as t};