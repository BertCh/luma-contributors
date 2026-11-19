function e(e,t){if(!Number.isInteger(t)||t<1||t>64)throw Error(`${e} capacity must be an integer in [1, 64], got ${t}`)}function t(t){e(`getDenseCholeskyWGSL`,t);let n=t;return`
struct DenseCholesky_${n} {
  scale: array<f32, ${n}>,
  size: u32,
  status: u32,
  smallestPivotRatio: f32,
}

fn factorDenseCholesky_${n}(
  matrix: ptr<function, array<f32, ${n*n}>>,
  size: u32,
  equilibrate: bool,
  minimumPivotRatio: f32
) -> DenseCholesky_${n} {
  var result: DenseCholesky_${n};
  result.size = min(size, ${n}u);
  result.status = 0u;
  result.smallestPivotRatio = 1.0;
  let count = result.size;
  for (var row = 0u; row < count; row++) {
    for (var column = 0u; column <= row; column++) {
      if (!isFiniteValue((*matrix)[row * ${n}u + column])) {
        result.status = 1u;
        return result;
      }
    }
  }
  for (var column = 0u; column < count; column++) {
    let diagonal = (*matrix)[column * ${n}u + column];
    if (!(diagonal > 0.0)) {
      result.status = 2u;
      return result;
    }
    result.scale[column] = select(1.0, 1.0 / sqrt(diagonal), equilibrate);
  }
  if (equilibrate) {
    for (var row = 0u; row < count; row++) {
      for (var column = 0u; column <= row; column++) {
        (*matrix)[row * ${n}u + column] = (*matrix)[row * ${n}u + column] * result.scale[row] * result.scale[column];
      }
    }
  }
  for (var column = 0u; column < count; column++) {
    let original = (*matrix)[column * ${n}u + column];
    var diagonal = original;
    for (var k = 0u; k < column; k++) {
      let value = (*matrix)[column * ${n}u + k];
      diagonal = diagonal - value * value;
    }
    if (!isFiniteValue(diagonal)) {
      result.status = 1u;
      return result;
    }
    if (!(diagonal > 0.0) || !(original > 0.0)) {
      result.status = 2u;
      result.smallestPivotRatio = 0.0;
      return result;
    }
    result.smallestPivotRatio = min(result.smallestPivotRatio, diagonal / original);
    let pivot = sqrt(diagonal);
    (*matrix)[column * ${n}u + column] = pivot;
    for (var row = column + 1u; row < count; row++) {
      var value = (*matrix)[row * ${n}u + column];
      for (var k = 0u; k < column; k++) {
        value = value - (*matrix)[row * ${n}u + k] * (*matrix)[column * ${n}u + k];
      }
      (*matrix)[row * ${n}u + column] = value / pivot;
    }
  }
  if (result.smallestPivotRatio < minimumPivotRatio) {
    result.status = 3u;
  }
  return result;
}

fn solveDenseCholesky_${n}(
  matrix: ptr<function, array<f32, ${n*n}>>,
  factorization: ptr<function, DenseCholesky_${n}>,
  vector: ptr<function, array<f32, ${n}>>
) -> bool {
  let count = (*factorization).size;
  for (var row = 0u; row < count; row++) {
    (*vector)[row] = (*vector)[row] * (*factorization).scale[row];
  }
  for (var row = 0u; row < count; row++) {
    var value = (*vector)[row];
    for (var k = 0u; k < row; k++) {
      value = value - (*matrix)[row * ${n}u + k] * (*vector)[k];
    }
    (*vector)[row] = value / (*matrix)[row * ${n}u + row];
  }
  for (var offset = 0u; offset < count; offset++) {
    let row = count - 1u - offset;
    var value = (*vector)[row];
    for (var k = row + 1u; k < count; k++) {
      value = value - (*matrix)[k * ${n}u + row] * (*vector)[k];
    }
    (*vector)[row] = value / (*matrix)[row * ${n}u + row];
  }
  var finite = true;
  for (var row = 0u; row < count; row++) {
    (*vector)[row] = (*vector)[row] * (*factorization).scale[row];
    finite = finite && isFiniteValue((*vector)[row]);
  }
  return finite;
}

fn getDenseCholeskyInverseDiagonal_${n}(
  matrix: ptr<function, array<f32, ${n*n}>>,
  factorization: ptr<function, DenseCholesky_${n}>,
  column: u32
) -> f32 {
  var unit: array<f32, ${n}>;
  unit[column] = 1.0;
  solveDenseCholesky_${n}(matrix, factorization, &unit);
  return unit[column];
}

fn invertDenseCholesky_${n}(
  matrix: ptr<function, array<f32, ${n*n}>>,
  factorization: ptr<function, DenseCholesky_${n}>,
  inverse: ptr<function, array<f32, ${n*n}>>
) -> bool {
  let count = (*factorization).size;
  var finite = true;
  for (var column = 0u; column < count; column++) {
    var unit: array<f32, ${n}>;
    unit[column] = 1.0;
    finite = solveDenseCholesky_${n}(matrix, factorization, &unit) && finite;
    for (var row = 0u; row < count; row++) {
      (*inverse)[row * ${n}u + column] = unit[row];
    }
  }
  return finite;
}

fn getDenseCholeskyLogDeterminant_${n}(
  matrix: ptr<function, array<f32, ${n*n}>>,
  factorization: ptr<function, DenseCholesky_${n}>
) -> f32 {
  var total = 0.0;
  for (var row = 0u; row < (*factorization).size; row++) {
    total = total + 2.0 * (log((*matrix)[row * ${n}u + row]) - log((*factorization).scale[row]));
  }
  return total;
}
`}function n(t){e(`getDenseLUWGSL`,t);let n=t;return`
struct DenseLU_${n} {
  rowScale: array<f32, ${n}>,
  columnScale: array<f32, ${n}>,
  pivots: array<u32, ${n}>,
  size: u32,
  status: u32,
  pivotRatio: f32,
  logAbsDeterminant: f32,
  determinantSign: f32,
}

fn factorDenseLU_${n}(
  matrix: ptr<function, array<f32, ${n*n}>>,
  size: u32,
  equilibrationPasses: u32,
  pivotTolerance: f32,
  minimumPivotRatio: f32
) -> DenseLU_${n} {
  var result: DenseLU_${n};
  result.size = min(size, ${n}u);
  result.status = 0u;
  result.pivotRatio = 0.0;
  result.logAbsDeterminant = 0.0;
  result.determinantSign = 1.0;
  let count = result.size;
  for (var row = 0u; row < count; row++) {
    result.rowScale[row] = 1.0;
    result.columnScale[row] = 1.0;
    result.pivots[row] = row;
    for (var column = 0u; column < count; column++) {
      if (!isFiniteValue((*matrix)[row * ${n}u + column])) {
        result.status = 1u;
        return result;
      }
    }
  }
  for (var equilibrationPass = 0u; equilibrationPass < equilibrationPasses; equilibrationPass++) {
    var rowMaximum: array<f32, ${n}>;
    var columnMaximum: array<f32, ${n}>;
    for (var row = 0u; row < count; row++) {
      for (var column = 0u; column < count; column++) {
        let magnitude = abs((*matrix)[row * ${n}u + column]) * result.rowScale[row] * result.columnScale[column];
        rowMaximum[row] = max(rowMaximum[row], magnitude);
        columnMaximum[column] = max(columnMaximum[column], magnitude);
      }
    }
    for (var index = 0u; index < count; index++) {
      if (!(rowMaximum[index] > 0.0) || !(columnMaximum[index] > 0.0)) {
        result.status = 2u;
        return result;
      }
      result.rowScale[index] = result.rowScale[index] / sqrt(rowMaximum[index]);
      result.columnScale[index] = result.columnScale[index] / sqrt(columnMaximum[index]);
    }
  }
  var largestEntry = 0.0;
  for (var row = 0u; row < count; row++) {
    for (var column = 0u; column < count; column++) {
      let value = (*matrix)[row * ${n}u + column] * result.rowScale[row] * result.columnScale[column];
      (*matrix)[row * ${n}u + column] = value;
      largestEntry = max(largestEntry, abs(value));
    }
  }
  var smallestPivot = 3.402823e38;
  var largestPivot = 0.0;
  for (var column = 0u; column < count; column++) {
    var pivotRow = column;
    for (var row = column + 1u; row < count; row++) {
      if (abs((*matrix)[row * ${n}u + column]) > abs((*matrix)[pivotRow * ${n}u + column])) {
        pivotRow = row;
      }
    }
    let pivotMagnitude = abs((*matrix)[pivotRow * ${n}u + column]);
    if (!isFiniteValue(pivotMagnitude)) {
      result.status = 1u;
      return result;
    }
    if (!(pivotMagnitude > pivotTolerance * largestEntry) || !(pivotMagnitude > 0.0)) {
      result.status = 2u;
      return result;
    }
    result.pivots[column] = pivotRow;
    if (pivotRow != column) {
      result.determinantSign = -result.determinantSign;
      for (var entry = 0u; entry < count; entry++) {
        let held = (*matrix)[column * ${n}u + entry];
        (*matrix)[column * ${n}u + entry] = (*matrix)[pivotRow * ${n}u + entry];
        (*matrix)[pivotRow * ${n}u + entry] = held;
      }
    }
    let pivot = (*matrix)[column * ${n}u + column];
    smallestPivot = min(smallestPivot, pivotMagnitude);
    largestPivot = max(largestPivot, pivotMagnitude);
    result.logAbsDeterminant = result.logAbsDeterminant + log(pivotMagnitude);
    if (pivot < 0.0) {
      result.determinantSign = -result.determinantSign;
    }
    for (var row = column + 1u; row < count; row++) {
      let factor = (*matrix)[row * ${n}u + column] / pivot;
      (*matrix)[row * ${n}u + column] = factor;
      for (var entry = column + 1u; entry < count; entry++) {
        (*matrix)[row * ${n}u + entry] = (*matrix)[row * ${n}u + entry] - factor * (*matrix)[column * ${n}u + entry];
      }
    }
  }
  for (var index = 0u; index < count; index++) {
    result.logAbsDeterminant = result.logAbsDeterminant - log(result.rowScale[index]) - log(result.columnScale[index]);
  }
  result.pivotRatio = select(1.0, smallestPivot / largestPivot, count > 0u);
  if (result.pivotRatio < minimumPivotRatio) {
    result.status = 3u;
  }
  return result;
}

fn solveDenseLU_${n}(
  matrix: ptr<function, array<f32, ${n*n}>>,
  factorization: ptr<function, DenseLU_${n}>,
  vector: ptr<function, array<f32, ${n}>>
) -> bool {
  let count = (*factorization).size;
  for (var row = 0u; row < count; row++) {
    (*vector)[row] = (*vector)[row] * (*factorization).rowScale[row];
  }
  for (var column = 0u; column < count; column++) {
    let pivotRow = (*factorization).pivots[column];
    if (pivotRow != column) {
      let held = (*vector)[column];
      (*vector)[column] = (*vector)[pivotRow];
      (*vector)[pivotRow] = held;
    }
  }
  for (var row = 0u; row < count; row++) {
    var value = (*vector)[row];
    for (var k = 0u; k < row; k++) {
      value = value - (*matrix)[row * ${n}u + k] * (*vector)[k];
    }
    (*vector)[row] = value;
  }
  var finite = true;
  for (var offset = 0u; offset < count; offset++) {
    let row = count - 1u - offset;
    var value = (*vector)[row];
    for (var k = row + 1u; k < count; k++) {
      value = value - (*matrix)[row * ${n}u + k] * (*vector)[k];
    }
    (*vector)[row] = value / (*matrix)[row * ${n}u + row];
  }
  for (var row = 0u; row < count; row++) {
    (*vector)[row] = (*vector)[row] * (*factorization).columnScale[row];
    finite = finite && isFiniteValue((*vector)[row]);
  }
  return finite;
}

fn invertDenseLU_${n}(
  matrix: ptr<function, array<f32, ${n*n}>>,
  factorization: ptr<function, DenseLU_${n}>,
  inverse: ptr<function, array<f32, ${n*n}>>
) -> bool {
  let count = (*factorization).size;
  var finite = true;
  for (var column = 0u; column < count; column++) {
    var unit: array<f32, ${n}>;
    unit[column] = 1.0;
    finite = solveDenseLU_${n}(matrix, factorization, &unit) && finite;
    for (var row = 0u; row < count; row++) {
      (*inverse)[row * ${n}u + column] = unit[row];
    }
  }
  return finite;
}
`}export{n,t};