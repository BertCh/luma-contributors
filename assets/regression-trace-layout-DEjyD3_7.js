import{S as e}from"./gpu-contributor-utils-BIwLQON5.js";import{t}from"./chunked-weights-kernels-BYYFtXmd.js";var n=4096;function r(e,t){if(e<3)throw Error(`tail extrapolation needs maximumPower >= 3`);let r=e=>`${t}[${t}Offset + ${e-1}u]`;return`
fn getSeriesTailPower(t: f32, exponent: u32) -> f32 {
  if (t == 0.0) {
    return 0.0;
  }
  let magnitude = pow(abs(t), f32(exponent));
  return select(magnitude, -magnitude, t < 0.0 && (exponent & 1u) == 1u);
}

// (level, ratio, alternatingLevel, alternatingRatio) of the tail c_(Q+k) = level * ratio^k + alternatingLevel * alternatingRatio^k.
fn getSeriesTailModel() -> vec4f {
  let last = ${r(e)};
  let before = ${r(e-1)};
  let first = ${r(e-2)};
  let pairNow = last + before;
  let pairBefore = before + first;
  var ratio = 0.0;
  if (pairBefore > 1.0e-30 && pairNow > 0.0) {
    ratio = min(pairNow / pairBefore, 1.0);
  }
  let differenceNow = last - before;
  let differenceBefore = before - first;
  var alternatingRatio = 0.0;
  if (abs(differenceBefore) > 1.0e-30 && differenceNow * differenceBefore < 0.0) {
    alternatingRatio = max(differenceNow / differenceBefore, -1.0);
  }
  return vec4f(0.5 * pairNow, ratio, select(0.5 * differenceNow, 0.0, alternatingRatio == 0.0), alternatingRatio);
}

fn getSeriesTailRatio(t: f32, ratio: f32) -> f32 {
  return clamp(t * ratio, -0.9999, 0.9999);
}

fn getSeriesTailDerivativeComponent(scale: f32, t: f32, level: f32, ratio: f32) -> f32 {
  let u = getSeriesTailRatio(t, ratio);
  return level * scale * u / (1.0 - u);
}

fn getSeriesTailDerivative(t: f32) -> f32 {
  let model = getSeriesTailModel();
  let scale = getSeriesTailPower(t, ${e-1}u);
  return getSeriesTailDerivativeComponent(scale, t, model.x, model.y)
    + getSeriesTailDerivativeComponent(scale, t, model.z, model.w);
}

fn getSeriesTailSecondComponent(scale: f32, t: f32, level: f32, ratio: f32) -> f32 {
  let u = getSeriesTailRatio(t, ratio);
  let geometric = u / (1.0 - u);
  return level * scale * (${e-1}.0 * geometric + geometric / (1.0 - u));
}

fn getSeriesTailSecond(t: f32) -> f32 {
  let model = getSeriesTailModel();
  let scale = getSeriesTailPower(t, ${e-2}u);
  return getSeriesTailSecondComponent(scale, t, model.x, model.y)
    + getSeriesTailSecondComponent(scale, t, model.z, model.w);
}

fn getSeriesTailLogDeterminantComponent(scale: f32, t: f32, level: f32, ratio: f32) -> f32 {
  let u = getSeriesTailRatio(t, ratio);
  var power = u;
  var sum = 0.0;
  var step = 1u;
  for (; step <= ${n}u; step++) {
    sum += power / f32(${e}u + step);
    power *= u;
    if (abs(power) < 1.0e-9) {
      step++;
      break;
    }
  }
  sum += power / ((1.0 - u) * f32(${e}u + step));
  return level * scale * sum;
}

fn getSeriesTailLogDeterminant(t: f32) -> f32 {
  let model = getSeriesTailModel();
  let scale = getSeriesTailPower(t, ${e}u);
  return -(getSeriesTailLogDeterminantComponent(scale, t, model.x, model.y)
    + getSeriesTailLogDeterminantComponent(scale, t, model.z, model.w));
}
`}var i=4096;function a(n,r,a){let o=Math.max(32,Math.ceil(a/1024)),s=t(r,o).tileCount;for(;s>i&&o<a;)o=Math.min(a,o*2),s=t(r,o).tileCount;return e(n,`trace tileCount`,s,i),{tileRowCount:o,tileCount:s}}export{r as n,a as t};