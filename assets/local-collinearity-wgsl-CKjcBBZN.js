var e=3e38,t=1e-7;function n(e){let t=e*e;return`
fn updateDesignFactor_${e}(factor: ptr<function, array<f32, ${t}>>, row: ptr<function, array<f32, ${e}>>) {
  for (var column = 0u; column < ${e}u; column++) {
    let value = (*row)[column];
    if (value == 0.0) {
      continue;
    }
    let diagonal = (*factor)[column * ${e}u + column];
    let radius = sqrt(diagonal * diagonal + value * value);
    let cosine = diagonal / radius;
    let sine = value / radius;
    for (var entry = column; entry < ${e}u; entry++) {
      let held = (*factor)[column * ${e}u + entry];
      let incoming = (*row)[entry];
      (*factor)[column * ${e}u + entry] = cosine * held + sine * incoming;
      (*row)[entry] = cosine * incoming - sine * held;
    }
  }
}

// Diagonal entry of (R^T R)^-1 = R^-1 R^-T, the variance inflation factor when the design columns
// have unit norm. R is upper triangular and must be nonsingular.
fn getDesignFactorInverseDiagonal_${e}(factor: ptr<function, array<f32, ${t}>>, column: u32) -> f32 {
  var total = 0.0;
  for (var last = column; last < ${e}u; last++) {
    // Solve R x = e_target by back substitution; only x[column] is accumulated.
    var solution: array<f32, ${e}>;
    solution[last] = 1.0 / (*factor)[last * ${e}u + last];
    for (var offset = 1u; offset <= last; offset++) {
      let row = last - offset;
      var value = 0.0;
      for (var other = row + 1u; other <= last; other++) {
        value += (*factor)[row * ${e}u + other] * solution[other];
      }
      solution[row] = -value / (*factor)[row * ${e}u + row];
    }
    total += solution[column] * solution[column];
  }
  return total;
}

// One-sided Jacobi on the factor columns. vectors is column-major (right singular vectors);
// returns false when the sweeps are exhausted.
fn getDesignSingularValues_${e}(
  factor: ptr<function, array<f32, ${t}>>,
  vectors: ptr<function, array<f32, ${t}>>,
  values: ptr<function, array<f32, ${e}>>
) -> bool {
  var columns: array<f32, ${t}>;
  for (var column = 0u; column < ${e}u; column++) {
    for (var row = 0u; row < ${e}u; row++) {
      columns[column * ${e}u + row] = (*factor)[row * ${e}u + column];
      (*vectors)[column * ${e}u + row] = select(0.0, 1.0, row == column);
    }
  }
  var converged = false;
  for (var sweep = 0u; sweep < 64u; sweep++) {
    var rotated = false;
    for (var first = 0u; first + 1u < ${e}u; first++) {
      for (var second = first + 1u; second < ${e}u; second++) {
        var normFirst = 0.0;
        var normSecond = 0.0;
        var product = 0.0;
        for (var row = 0u; row < ${e}u; row++) {
          let x = columns[first * ${e}u + row];
          let y = columns[second * ${e}u + row];
          normFirst += x * x;
          normSecond += y * y;
          product += x * y;
        }
        if (!(abs(product) > 1.0e-6 * sqrt(normFirst * normSecond))) {
          continue;
        }
        rotated = true;
        let zeta = (normSecond - normFirst) / (2.0 * product);
        let tangent = select(-1.0, 1.0, zeta >= 0.0) / (abs(zeta) + sqrt(1.0 + zeta * zeta));
        let cosine = inverseSqrt(1.0 + tangent * tangent);
        let sine = cosine * tangent;
        for (var row = 0u; row < ${e}u; row++) {
          let x = columns[first * ${e}u + row];
          let y = columns[second * ${e}u + row];
          columns[first * ${e}u + row] = cosine * x - sine * y;
          columns[second * ${e}u + row] = sine * x + cosine * y;
          let vectorFirst = (*vectors)[first * ${e}u + row];
          let vectorSecond = (*vectors)[second * ${e}u + row];
          (*vectors)[first * ${e}u + row] = cosine * vectorFirst - sine * vectorSecond;
          (*vectors)[second * ${e}u + row] = sine * vectorFirst + cosine * vectorSecond;
        }
      }
    }
    if (!rotated) {
      converged = true;
      break;
    }
  }
  for (var column = 0u; column < ${e}u; column++) {
    var norm = 0.0;
    for (var row = 0u; row < ${e}u; row++) {
      norm += columns[column * ${e}u + row] * columns[column * ${e}u + row];
    }
    (*values)[column] = sqrt(norm);
  }
  return converged;
}
`}export{t as n,n as r,e as t};