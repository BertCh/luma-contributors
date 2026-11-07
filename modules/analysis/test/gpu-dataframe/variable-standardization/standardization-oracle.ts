// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUVariableStandardizationMethod} from '../../../src/dataframe-analysis/variable-standardization/index';

/** Independent float64 definition oracle, deliberately not a mirror of GPU reduction trees. */
export function standardizeReference(
  values: readonly number[],
  method: GPUVariableStandardizationMethod,
  mask?: readonly number[],
  variance: 'sample' | 'population' = 'sample'
): number[] {
  const population = values.filter(
    (value, row) => Number.isFinite(value) && (!mask || mask[row] !== 0)
  );
  const mean = population.reduce((sum, value) => sum + value, 0) / population.length;
  const minimum = Math.min(...population);
  const range = Math.max(...population) - minimum;
  const absoluteDeviation =
    population.reduce((sum, value) => sum + Math.abs(value - mean), 0) / population.length;
  const standardDeviation = Math.sqrt(
    population.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
      (population.length - (variance === 'sample' ? 1 : 0))
  );
  return values.map((value, row) => {
    if (!Number.isFinite(value) || (mask && !mask[row])) return NaN;
    switch (method) {
      case 'deviation-from-mean':
        return value - mean;
      case 'z-score':
        if (population.length <= (variance === 'sample' ? 1 : 0)) return NaN;
        return standardDeviation === 0 ? 0 : (value - mean) / standardDeviation;
      case 'mad':
        return absoluteDeviation === 0 ? value : (value - mean) / absoluteDeviation;
      case 'range-adjust':
        return range === 0 ? value : value / range;
      case 'range-standardize':
        return range === 0 ? value : (value - minimum) / range;
    }
  });
}
