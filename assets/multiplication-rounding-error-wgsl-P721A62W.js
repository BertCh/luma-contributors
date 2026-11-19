var e=`
fn getMultiplicationRoundingError(first: f32, second: f32, product: f32) -> f32 {
  let firstBits = bitcast<u32>(first);
  let secondBits = bitcast<u32>(second);
  let productBits = bitcast<u32>(product);
  // Biased exponents: |value| = significand * 2^(exponent - 150) for a normal value.
  let firstExponent = (firstBits >> 23u) & 0xffu;
  let secondExponent = (secondBits >> 23u) & 0xffu;
  let productExponent = (productBits >> 23u) & 0xffu;
  if (firstExponent == 0u || secondExponent == 0u || productExponent == 0u ||
      firstExponent == 255u || secondExponent == 255u || productExponent == 255u) {
    return 0.0;
  }
  // The exact product is exactSignificand * 2^(firstExponent + secondExponent - 300), with a
  // significand of 47 or 48 bits. The error is scaled by one ldexp whose exponent is kept at
  // -126 or above, where ldexp is dependable: operands whose binary exponents sum to less than
  // -80 report no error (every product below 2^-80, and those below 2^-79 whose significands
  // multiply to 2 or more).
  let exponentSum = firstExponent + secondExponent;
  if (exponentSum < 174u) {
    return 0.0;
  }
  let firstSignificand = (firstBits & 0x7fffffu) | 0x800000u;
  let secondSignificand = (secondBits & 0x7fffffu) | 0x800000u;
  // 48-bit product of the two 24-bit significands, as two 24-bit limbs.
  let firstHigh = firstSignificand >> 12u;
  let firstLow = firstSignificand & 0xfffu;
  let secondHigh = secondSignificand >> 12u;
  let secondLow = secondSignificand & 0xfffu;
  let middle = firstHigh * secondLow + firstLow * secondHigh;
  let lowSum = firstLow * secondLow + ((middle & 0xfffu) << 12u);
  let exactLow = lowSum & 0xffffffu;
  let exactHigh = firstHigh * secondHigh + (middle >> 12u) + (lowSum >> 24u);
  // The significand of the rounded product sits 23, 24 or 25 bits above the last place of the
  // exact product (25 when the rounding carried into the next binade).
  let shift = productExponent + 150u - exponentSum;
  if (shift < 23u || shift > 25u) {
    return 0.0;
  }
  let productSignificand = (productBits & 0x7fffffu) | 0x800000u;
  var roundedHigh = productSignificand;
  var roundedLow = 0u;
  if (shift == 23u) {
    roundedHigh = productSignificand >> 1u;
    roundedLow = (productSignificand & 1u) << 23u;
  } else if (shift == 25u) {
    roundedHigh = productSignificand << 1u;
  }
  // Exact as an integer. Its conversion to f32 is exact for a correctly rounded product (at
  // most 2^24 units); for a product one or two steps off it can need 25 bits and is rounded.
  let errorUnits = (i32(exactHigh) - i32(roundedHigh)) * 16777216 + (i32(exactLow) - i32(roundedLow));
  let magnitude = ldexp(f32(errorUnits), i32(exponentSum) - 300);
  // The exact product has the sign of the operands' signs combined; so has its error's scale.
  return select(magnitude, -magnitude, ((firstBits ^ secondBits) & 0x80000000u) != 0u);
}
`;export{e as t};