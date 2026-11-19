var e=`
fn getLogGammaValue(z: f32) -> f32 {
  var shifted = z;
  var product = 1.0;
  for (var step = 0u; step < 8u; step++) {
    if (shifted >= 8.0) {
      break;
    }
    product = product * shifted;
    shifted = shifted + 1.0;
  }
  let inverse = 1.0 / shifted;
  let inverseSquared = inverse * inverse;
  let series = inverse * (1.0 / 12.0 - inverseSquared * (1.0 / 360.0 - inverseSquared * (1.0 / 1260.0)));
  return (shifted - 0.5) * log(shifted) - shifted + 0.9189385332 + series - log(product);
}

// Standard normal survival function via the Numerical Recipes erfc fit (relative error under 1.2e-7).
fn getNormalSurvivalValue(z: f32) -> f32 {
  let magnitude = abs(z) * 0.7071067811865476;
  let t = 1.0 / (1.0 + 0.5 * magnitude);
  let r = t * exp(-magnitude * magnitude - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return 0.5 * select(2.0 - r, r, z >= 0.0);
}

fn getChiSquareSurvivalValue(value: f32, degreesOfFreedom: f32) -> f32 {
  if (isNanBits(value) || !(degreesOfFreedom > 0.0)) {
    return getNaN();
  }
  if (!(value > 0.0)) {
    return 1.0;
  }
  let a = degreesOfFreedom * 0.5;
  let x = value * 0.5;
  if (a > 1000.0) {
    // Wilson-Hilferty: (X / dof)^(1/3) is approximately normal with mean 1 - 2 / (9 dof).
    let variance = 2.0 / (9.0 * degreesOfFreedom);
    let z = (pow(value / degreesOfFreedom, 1.0 / 3.0) - (1.0 - variance)) / sqrt(variance);
    return getNormalSurvivalValue(z);
  }
  let logPrefactor = -x + a * log(x) - getLogGammaValue(a);
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    for (var iteration = 1u; iteration <= 500u; iteration++) {
      term = term * x / (a + f32(iteration));
      sum = sum + term;
      if (term < sum * 1.0e-7) {
        break;
      }
    }
    return clamp(1.0 - sum * exp(logPrefactor), 0.0, 1.0);
  }
  let tiny = 1.0e-30;
  var b = x + 1.0 - a;
  var c = 1.0 / tiny;
  var d = 1.0 / b;
  var h = d;
  for (var iteration = 1u; iteration <= 500u; iteration++) {
    let stepIndex = f32(iteration);
    let an = -stepIndex * (stepIndex - a);
    b = b + 2.0;
    d = an * d + b;
    if (abs(d) < tiny) {
      d = tiny;
    }
    c = b + an / c;
    if (abs(c) < tiny) {
      c = tiny;
    }
    d = 1.0 / d;
    let delta = d * c;
    h = h * delta;
    if (abs(delta - 1.0) < 2.0e-7) {
      break;
    }
  }
  return clamp(exp(logPrefactor) * h, 0.0, 1.0);
}
`;export{e as t};