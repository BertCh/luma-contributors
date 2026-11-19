function e(e){if(!Number.isInteger(e)||e<1||e>64)throw Error(`getSymmetricEigenWGSL capacity must be an integer in [1, 64], got ${e}`);let t=e;return`
struct SymmetricEigen_${t} {
  values: array<f32, ${t}>,
  size: u32,
  status: u32,
  sweeps: u32,
}

fn getSymmetricEigen_${t}(
  matrix: ptr<function, array<f32, ${t*t}>>,
  vectors: ptr<function, array<f32, ${t*t}>>,
  size: u32,
  maximumSweeps: u32,
  tolerance: f32
) -> SymmetricEigen_${t} {
  var result: SymmetricEigen_${t};
  result.size = min(size, ${t}u);
  result.status = 0u;
  result.sweeps = 0u;
  let count = result.size;
  // Validate first, so a NONFINITE return leaves matrix and vectors untouched.
  var largest = 0.0;
  for (var row = 0u; row < count; row++) {
    for (var column = 0u; column <= row; column++) {
      let value = (*matrix)[row * ${t}u + column];
      if (!isFiniteValue(value)) {
        result.status = 1u;
        return result;
      }
      largest = max(largest, abs(value));
    }
  }
  for (var row = 0u; row < count; row++) {
    for (var column = 0u; column <= row; column++) {
      (*matrix)[column * ${t}u + row] = (*matrix)[row * ${t}u + column];
      (*vectors)[row * ${t}u + column] = select(0.0, 1.0, row == column);
      (*vectors)[column * ${t}u + row] = select(0.0, 1.0, row == column);
    }
  }
  // The stopping test runs on entries scaled by the power of two that brings the largest input
  // entry into [0.5, 1): exact, so well-scaled input stops exactly as unscaled, and the squares
  // neither underflow (entries below about 1e-19) nor overflow (above about 1.8e19). Rotations
  // keep the Frobenius norm, so scaled entries stay at most the size.
  let scaleExponent = -frexp(largest).exp;
  for (var sweep = 0u; sweep <= maximumSweeps; sweep++) {
    var offDiagonal = 0.0;
    var diagonal = 0.0;
    for (var row = 0u; row < count; row++) {
      let diagonalValue = ldexp((*matrix)[row * ${t}u + row], scaleExponent);
      diagonal = diagonal + diagonalValue * diagonalValue;
      for (var column = row + 1u; column < count; column++) {
        let offDiagonalValue = ldexp((*matrix)[row * ${t}u + column], scaleExponent);
        offDiagonal = offDiagonal + offDiagonalValue * offDiagonalValue;
      }
    }
    if (offDiagonal == 0.0 || offDiagonal <= tolerance * tolerance * diagonal) {
      break;
    }
    if (sweep == maximumSweeps) {
      result.status = 4u;
      break;
    }
    result.sweeps = sweep + 1u;
    for (var lower = 0u; lower + 1u < count; lower++) {
      for (var upper = lower + 1u; upper < count; upper++) {
        let coupling = (*matrix)[lower * ${t}u + upper];
        if (coupling == 0.0) {
          continue;
        }
        let theta = ((*matrix)[upper * ${t}u + upper] - (*matrix)[lower * ${t}u + lower]) / (2.0 * coupling);
        let direction = select(-1.0, 1.0, theta >= 0.0);
        let tangent = select(direction / (abs(theta) + sqrt(theta * theta + 1.0)), 0.5 / theta, abs(theta) > 1.0e18);
        let cosine = 1.0 / sqrt(tangent * tangent + 1.0);
        let sine = tangent * cosine;
        for (var k = 0u; k < count; k++) {
          let lowerValue = (*matrix)[k * ${t}u + lower];
          let upperValue = (*matrix)[k * ${t}u + upper];
          (*matrix)[k * ${t}u + lower] = cosine * lowerValue - sine * upperValue;
          (*matrix)[k * ${t}u + upper] = sine * lowerValue + cosine * upperValue;
        }
        for (var k = 0u; k < count; k++) {
          let lowerValue = (*matrix)[lower * ${t}u + k];
          let upperValue = (*matrix)[upper * ${t}u + k];
          (*matrix)[lower * ${t}u + k] = cosine * lowerValue - sine * upperValue;
          (*matrix)[upper * ${t}u + k] = sine * lowerValue + cosine * upperValue;
        }
        (*matrix)[lower * ${t}u + upper] = 0.0;
        (*matrix)[upper * ${t}u + lower] = 0.0;
        for (var k = 0u; k < count; k++) {
          let lowerValue = (*vectors)[k * ${t}u + lower];
          let upperValue = (*vectors)[k * ${t}u + upper];
          (*vectors)[k * ${t}u + lower] = cosine * lowerValue - sine * upperValue;
          (*vectors)[k * ${t}u + upper] = sine * lowerValue + cosine * upperValue;
        }
      }
    }
  }
  for (var index = 0u; index < count; index++) {
    result.values[index] = (*matrix)[index * ${t}u + index];
  }
  // Selection sort, ascending, moving the vector columns with their values.
  for (var index = 0u; index + 1u < count; index++) {
    var smallest = index;
    for (var other = index + 1u; other < count; other++) {
      if (result.values[other] < result.values[smallest]) {
        smallest = other;
      }
    }
    if (smallest != index) {
      let held = result.values[index];
      result.values[index] = result.values[smallest];
      result.values[smallest] = held;
      for (var k = 0u; k < count; k++) {
        let component = (*vectors)[k * ${t}u + index];
        (*vectors)[k * ${t}u + index] = (*vectors)[k * ${t}u + smallest];
        (*vectors)[k * ${t}u + smallest] = component;
      }
    }
  }
  return result;
}
`}export{e as t};