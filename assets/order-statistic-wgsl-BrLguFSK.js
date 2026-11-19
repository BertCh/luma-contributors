var e=`
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
`,t=`
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
`;export{t as n,e as t};