var e=`
fn getScaledQuotient(value: f32, first: f32, second: f32, power: i32) -> f32 {
  if (value == 0.0) { return value; }
  let numerator = frexp(value);
  let firstDivisor = frexp(first);
  let secondDivisor = frexp(second);
  let quotient = frexp(numerator.fract / (firstDivisor.fract * secondDivisor.fract));
  return ldexp(quotient.fract,
    numerator.exp - firstDivisor.exp - secondDivisor.exp + power + quotient.exp);
}
`;export{e as t};