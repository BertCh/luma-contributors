// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUPredictionCovariance} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';

/** Test-only f64 Schur complement, independently conditioning observations before query factoring. */
export function calculateConditionalSimulationReference(
  positions: ArrayLike<number>,
  observations: ArrayLike<number>,
  queries: ArrayLike<number>,
  model: GPUPredictionCovariance,
  knownMean: number
) {
  const observed = Array.from({length: observations.length / 4}, (_, index) => index).filter(
    index => observations[index * 4 + 3] > 0
  );
  const count = observed.length;
  const queryCount = queries.length / 2;
  const covariance = (
    first: ArrayLike<number>,
    firstIndex: number,
    second: ArrayLike<number>,
    secondIndex: number
  ) => {
    const distance = Math.hypot(
      first[firstIndex * 2] - second[secondIndex * 2],
      first[firstIndex * 2 + 1] - second[secondIndex * 2 + 1]
    );
    const scaled = distance / model.range;
    const correlation =
      model.model === 'spherical'
        ? scaled >= 1
          ? 0
          : 1 - 1.5 * scaled + 0.5 * scaled ** 3
        : Math.exp(-3 * (model.model === 'exponential' ? scaled : scaled * scaled));
    return model.sill * correlation + (distance === 0 ? (model.nugget ?? 0) : 0);
  };
  const lower = Array.from({length: count}, () => new Float64Array(count));
  for (let row = 0; row < count; row++) {
    for (let column = 0; column <= row; column++) {
      let value = covariance(positions, observed[row], positions, observed[column]);
      if (row === column) value += observations[observed[row] * 4 + 1];
      for (let component = 0; component < column; component++)
        value -= lower[row][component] * lower[column][component];
      if (row === column && !(value > 0)) throw new Error('Singular reference observations');
      lower[row][column] = row === column ? Math.sqrt(value) : value / lower[column][column];
    }
  }
  const solveLower = (right: Float64Array) => {
    const result = right.slice();
    for (let row = 0; row < count; row++) {
      for (let column = 0; column < row; column++)
        result[row] -= lower[row][column] * result[column];
      result[row] /= lower[row][row];
    }
    return result;
  };
  const values = solveLower(
    Float64Array.from(observed, index => observations[index * 4] - knownMean)
  );
  const cross = Array.from({length: queryCount}, (_, query) =>
    solveLower(Float64Array.from(observed, index => covariance(queries, query, positions, index)))
  );
  const means = cross.map(row =>
    row.reduce((sum, value, column) => sum + value * values[column], knownMean)
  );
  const conditionalCovariance = Array.from({length: queryCount}, (_, row) =>
    Array.from({length: queryCount}, (_, column) => {
      let value = covariance(queries, row, queries, column);
      for (let component = 0; component < count; component++)
        value -= cross[row][component] * cross[column][component];
      return value;
    })
  );
  return {means, covariance: conditionalCovariance};
}
