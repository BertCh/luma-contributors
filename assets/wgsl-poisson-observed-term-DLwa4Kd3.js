var e=`
// (1 + d) ln(1 + d) - d for d >= -1. Below |d| = 1/4 the Taylor series
// d^2 * sum_k>=2 (-d)^(k - 2) / (k (k - 1)), terms to k = 14 (truncation below 1e-9 relative),
// avoids the cancellation of the direct form.
fn getEntropyExcess(delta: f32) -> f32 {
  if (abs(delta) < 0.25) {
    var series = 1.0 / 182.0;
    series = -1.0 / 156.0 + delta * series;
    series = 1.0 / 132.0 + delta * series;
    series = -1.0 / 110.0 + delta * series;
    series = 1.0 / 90.0 + delta * series;
    series = -1.0 / 72.0 + delta * series;
    series = 1.0 / 56.0 + delta * series;
    series = -1.0 / 42.0 + delta * series;
    series = 1.0 / 30.0 + delta * series;
    series = -1.0 / 20.0 + delta * series;
    series = 1.0 / 12.0 + delta * series;
    series = -1.0 / 6.0 + delta * series;
    series = 1.0 / 2.0 + delta * series;
    return delta * delta * series;
  }
  // At d = -1 the product is 0 * log(0), which is NaN in float32; its limit is 0, so the value is
  // -d = 1. Reached when the observed count is positive but below 2^-25 of the expected count (one
  // case against an expectation above 3.4e7), where (o - e) / e rounds to exactly -1.
  let scaled = 1.0 + delta;
  if (scaled <= 0.0) {
    return -delta;
  }
  return scaled * log(scaled) - delta;
}
// Kulldorff term o ln(o/e) minus its linear part o - e, which sums to 0 over the cells of a full
// partition of the cases, as e * ((1 + d) ln(1 + d) - d) with d = (o - e) / e: near o = e the
// LLR no longer comes from the difference of two large logarithm terms.
fn getObservedTerm(observed: f32, expected: f32) -> f32 {
  if (!(expected > 0.0)) {
    return 0.0;
  }
  if (!(observed > 0.0)) {
    return expected;
  }
  return expected * getEntropyExcess((observed - expected) / expected);
}`;export{e as t};