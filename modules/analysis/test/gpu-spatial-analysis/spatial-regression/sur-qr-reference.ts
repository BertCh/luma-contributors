// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Test-only f64 QR with reorthogonalization; does not form normal equations. */
function solveLeastSquares(design: Float64Array[], response: Float64Array) {
  const columns = design.length;
  const orthogonal = design.map(column => column.slice());
  const triangular = Array.from({length: columns}, () => new Float64Array(columns));
  const right = new Float64Array(columns);
  for (let column = 0; column < columns; column++) {
    const values = orthogonal[column];
    let originalSquare = 0;
    for (const value of values) originalSquare += value * value;
    for (let pass = 0; pass < 2; pass++) {
      for (let previous = 0; previous < column; previous++) {
        let projection = 0;
        for (let row = 0; row < values.length; row++) {
          projection += orthogonal[previous][row] * values[row];
        }
        triangular[previous][column] += projection;
        for (let row = 0; row < values.length; row++) {
          values[row] -= projection * orthogonal[previous][row];
        }
      }
    }
    let square = 0;
    for (const value of values) square += value * value;
    const norm = Math.sqrt(square);
    if (!(norm > Number.EPSILON * values.length * Math.sqrt(originalSquare))) {
      throw new Error('Rank-deficient reference design');
    }
    triangular[column][column] = norm;
    for (let row = 0; row < values.length; row++) {
      values[row] /= norm;
      right[column] += values[row] * response[row];
    }
  }
  const solve = (values: Float64Array) => {
    const result = values.slice();
    for (let row = columns - 1; row >= 0; row--) {
      for (let column = row + 1; column < columns; column++) {
        result[row] -= triangular[row][column] * result[column];
      }
      result[row] /= triangular[row][row];
    }
    return result;
  };
  const inverseColumns = Array.from({length: columns}, (_, column) => {
    const unit = new Float64Array(columns);
    unit[column] = 1;
    return solve(unit);
  });
  const covariance = Array.from({length: columns * columns}, (_, index) => {
    let value = 0;
    for (const inverse of inverseColumns) {
      value += inverse[Math.floor(index / columns)] * inverse[index % columns];
    }
    return value;
  });
  return {coefficients: Array.from(solve(right)), covariance};
}

/** Independent whitened block GLS; inputs must already contain the common complete-case rows. */
export function calculateSURReferenceWithQR(
  predictors: ArrayLike<number>,
  responses: ArrayLike<number>,
  equationCount = 2,
  predictorCount = 1,
  suppliedResiduals?: ArrayLike<number>
) {
  const rows = responses.length / equationCount;
  const width = predictorCount + 1;
  const designs = Array.from({length: equationCount}, (_, equation) =>
    Array.from({length: width}, (_, column) =>
      Float64Array.from({length: rows}, (_, row) =>
        column === 0 ? 1 : predictors[(equation * rows + row) * predictorCount + column - 1]
      )
    )
  );
  const initialResiduals = designs.map((design, equation) => {
    const response = Float64Array.from(
      {length: rows},
      (_, row) => responses[equation * rows + row]
    );
    const {coefficients} = solveLeastSquares(design, response);
    return Float64Array.from(response, (value, row) =>
      suppliedResiduals
        ? suppliedResiduals[equation * rows + row]
        : value -
          coefficients.reduce(
            (total, coefficient, column) => total + coefficient * design[column][row],
            0
          )
    );
  });
  const residualCovariance = Array.from({length: equationCount ** 2}, (_, index) => {
    let value = 0;
    for (let row = 0; row < rows; row++) {
      value +=
        initialResiduals[Math.floor(index / equationCount)][row] *
        initialResiduals[index % equationCount][row];
    }
    return value / rows;
  });
  const lower = Array.from({length: equationCount}, () => new Float64Array(equationCount));
  for (let row = 0; row < equationCount; row++) {
    for (let column = 0; column <= row; column++) {
      let value = residualCovariance[row * equationCount + column];
      for (let inner = 0; inner < column; inner++)
        value -= lower[row][inner] * lower[column][inner];
      if (row === column && !(value > 0)) throw new Error('Singular reference error covariance');
      lower[row][column] = row === column ? Math.sqrt(value) : value / lower[column][column];
    }
  }
  const size = equationCount * width;
  const whitened = Array.from({length: size}, () => new Float64Array(rows * equationCount));
  const whitenedResponse = new Float64Array(rows * equationCount);
  for (let row = 0; row < rows; row++) {
    for (let equation = 0; equation < equationCount; equation++) {
      const index = equation * rows + row;
      let value = responses[index];
      for (let previous = 0; previous < equation; previous++) {
        value -= lower[equation][previous] * whitenedResponse[previous * rows + row];
      }
      whitenedResponse[index] = value / lower[equation][equation];
      for (let column = 0; column < size; column++) {
        value =
          Math.floor(column / width) === equation ? designs[equation][column % width][row] : 0;
        for (let previous = 0; previous < equation; previous++) {
          value -= lower[equation][previous] * whitened[column][previous * rows + row];
        }
        whitened[column][index] = value / lower[equation][equation];
      }
    }
  }
  const {coefficients, covariance} = solveLeastSquares(whitened, whitenedResponse);
  const residuals = Array.from({length: responses.length}, (_, index) => {
    const equation = Math.floor(index / rows);
    let value = responses[index];
    for (let column = 0; column < width; column++) {
      value -= coefficients[equation * width + column] * designs[equation][column][index % rows];
    }
    return value;
  });
  return {coefficients, covariance, residualCovariance, residuals};
}
