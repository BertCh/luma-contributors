import{t as e}from"./dense-solve-wgsl-0xzjbd8V.js";var t=.001;function n(e={},t=new Float32Array(1)){if(t.length<1)throw Error(`getGPUOrdinaryLeastSquaresParameterValues target must hold 1 element`);return t[0]=e.ridgeLambda??0,t}function r(t,n=16){s(t,n);let r=t,i=r*r,a=`array<f32, ${r}>(${Array.from({length:r},()=>`1.0`).join(`, `)})`;return`${e(r)}
fn choleskyFactor_${r}(a: ptr<function, array<f32, ${i}>>) -> bool {
  return factorDenseCholesky_${r}(a, ${r}u, false, 0.0).status == 0u;
}

// The factorization record of a successful unequilibrated factor: unit scale, full size.
fn getCholeskyFactorization_${r}() -> DenseCholesky_${r} {
  return DenseCholesky_${r}(${a}, ${r}u, 0u, 1.0);
}

fn choleskySolve_${r}(a: ptr<function, array<f32, ${i}>>, b: ptr<function, array<f32, ${r}>>) {
  var factorization = getCholeskyFactorization_${r}();
  solveDenseCholesky_${r}(a, &factorization, b);
}

fn choleskyInverseDiagonal_${r}(a: ptr<function, array<f32, ${i}>>, column: u32) -> f32 {
  var factorization = getCholeskyFactorization_${r}();
  return getDenseCholeskyInverseDiagonal_${r}(a, &factorization, column);
}
`}function i(e,n=16){s(e,n);let r=e;return`
fn choleskyFactorLimited_${r}(a: ptr<function, array<f32, ${r*r}>>) -> bool {
  var diagonal: array<f32, ${r}>;
  for (var column = 0u; column < ${r}u; column++) {
    diagonal[column] = (*a)[column * ${r}u + column];
  }
  if (!choleskyFactor_${r}(a)) {
    return false;
  }
  for (var column = 0u; column < ${r}u; column++) {
    if (!((*a)[column * ${r}u + column] >= ${t} * sqrt(diagonal[column]))) {
      return false;
    }
  }
  return true;
}
`}function a(e){s(e);let t=e;return`
fn getSymmetricEigenvalueRange_${t}(a: ptr<function, array<f32, ${t*t}>>) -> vec2<f32> {
  for (var sweep = 0u; sweep < 16u; sweep++) {
    var offDiagonal = 0.0;
    var diagonal = 0.0;
    for (var row = 0u; row < ${t}u; row++) {
      diagonal = diagonal + (*a)[row * ${t}u + row] * (*a)[row * ${t}u + row];
      for (var column = row + 1u; column < ${t}u; column++) {
        offDiagonal = offDiagonal + (*a)[row * ${t}u + column] * (*a)[row * ${t}u + column];
      }
    }
    // Both masses are sums of squares, so the f32 noise floor is about eps^2 (1.4e-14): a 1e-14
    // threshold is unreachable. 1e-12 on squared sums is about 1e-6 relative on the entries.
    if (offDiagonal <= 1.0e-12 * diagonal) {
      break;
    }
    for (var first = 0u; first + 1u < ${t}u; first++) {
      for (var second = first + 1u; second < ${t}u; second++) {
        let pivot = (*a)[first * ${t}u + second];
        if (abs(pivot) < 1.0e-30) {
          continue;
        }
        let theta = ((*a)[second * ${t}u + second] - (*a)[first * ${t}u + first]) / (2.0 * pivot);
        let tangent = select(-1.0, 1.0, theta >= 0.0) / (abs(theta) + sqrt(theta * theta + 1.0));
        let cosine = 1.0 / sqrt(tangent * tangent + 1.0);
        let sine = tangent * cosine;
        for (var k = 0u; k < ${t}u; k++) {
          let left = (*a)[k * ${t}u + first];
          let right = (*a)[k * ${t}u + second];
          (*a)[k * ${t}u + first] = cosine * left - sine * right;
          (*a)[k * ${t}u + second] = sine * left + cosine * right;
        }
        for (var k = 0u; k < ${t}u; k++) {
          let top = (*a)[first * ${t}u + k];
          let bottom = (*a)[second * ${t}u + k];
          (*a)[first * ${t}u + k] = cosine * top - sine * bottom;
          (*a)[second * ${t}u + k] = sine * top + cosine * bottom;
        }
      }
    }
  }
  var smallest = (*a)[0];
  var largest = (*a)[0];
  for (var row = 1u; row < ${t}u; row++) {
    smallest = min(smallest, (*a)[row * ${t}u + row]);
    largest = max(largest, (*a)[row * ${t}u + row]);
  }
  return vec2<f32>(smallest, largest);
}
`}function o(e){s(e+1);let t=e,n=t+1;return`
fn getBelsleyConditionNumber_${t}(
  factor: ptr<function, array<f32, ${t*t}>>,
  alpha: ptr<function, array<f32, ${t}>>,
  beta: ptr<function, array<f32, ${t}>>
) -> f32 {
  // Column-major B: b[column * ${n}u + row].
  var b: array<f32, ${n*n}>;
  b[0] = 1.0;
  for (var column = 1u; column < ${n}u; column++) {
    let j = column - 1u;
    b[column * ${n}u] = (*alpha)[j];
    for (var i = 0u; i <= j; i++) {
      b[column * ${n}u + 1u + i] = (*beta)[j] * (*factor)[j * ${t}u + i];
    }
  }
  for (var sweep = 0u; sweep < 16u; sweep++) {
    var rotated = false;
    for (var first = 0u; first + 1u < ${n}u; first++) {
      for (var second = first + 1u; second < ${n}u; second++) {
        var normFirst = 0.0;
        var normSecond = 0.0;
        var product = 0.0;
        for (var row = 0u; row < ${n}u; row++) {
          let x = b[first * ${n}u + row];
          let y = b[second * ${n}u + row];
          normFirst = normFirst + x * x;
          normSecond = normSecond + y * y;
          product = product + x * y;
        }
        if (!(abs(product) > 1.0e-7 * sqrt(normFirst * normSecond))) {
          continue;
        }
        rotated = true;
        let zeta = (normSecond - normFirst) / (2.0 * product);
        let tangent = select(-1.0, 1.0, zeta >= 0.0) / (abs(zeta) + sqrt(1.0 + zeta * zeta));
        let cosine = 1.0 / sqrt(1.0 + tangent * tangent);
        let sine = cosine * tangent;
        for (var row = 0u; row < ${n}u; row++) {
          let x = b[first * ${n}u + row];
          let y = b[second * ${n}u + row];
          b[first * ${n}u + row] = cosine * x - sine * y;
          b[second * ${n}u + row] = sine * x + cosine * y;
        }
      }
    }
    if (!rotated) {
      break;
    }
  }
  var largest = 0.0;
  var smallest = 3.0e38;
  for (var column = 0u; column < ${n}u; column++) {
    var norm = 0.0;
    for (var row = 0u; row < ${n}u; row++) {
      norm = norm + b[column * ${n}u + row] * b[column * ${n}u + row];
    }
    largest = max(largest, sqrt(norm));
    smallest = min(smallest, sqrt(norm));
  }
  return largest / smallest;
}
`}function s(e,t=16){if(!Number.isInteger(e)||e<1||e>t)throw Error(`Spatial regression predictor count must be an integer in [1, ${t}]`)}export{t as a,a as i,r as n,n as o,i as r,o as t};