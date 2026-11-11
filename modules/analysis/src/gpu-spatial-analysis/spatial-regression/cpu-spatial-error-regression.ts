// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  fitCPURegressionWithQR,
  getCPURegressionSingularValueRange
} from './cpu-regression-linear-algebra';
import type {CPURegressionWeights} from './cpu-regression-weights';

/** Float64 sparse multiplication; invalid neighbor sentinels are skipped as in the kernels. @internal */
export function multiplyCPUErrorWeights(
  weights: CPURegressionWeights,
  values: readonly number[]
): number[] {
  return Array.from({length: weights.offsets.length - 1}, (_, row) => {
    let total = 0;
    for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
      const neighbor = weights.neighbors[slot];
      if (neighbor < values.length) total += weights.weights[slot] * values[neighbor];
    }
    return total;
  });
}

/** Shifted means keep small deviations from large float32 offsets. @internal */
export function getCPUErrorMean(values: readonly number[]): number {
  return values[0] + values.reduce((sum, value) => sum + (value - values[0]), 0) / values.length;
}

/** Dot product of float64 rows. @internal */
export function getCPUErrorDot(first: readonly number[], second: readonly number[]): number {
  return first.reduce((sum, value, row) => sum + value * second[row], 0);
}

/** Centered filtered design diagnostics, with raw-column Belsley conditioning. @internal */
export function getCPUErrorConditioning(design: number[][], means: readonly number[]): number[] {
  const width = means.length + 1;
  const norms = Array.from({length: width}, (_, column) => {
    let norm = 0;
    for (const row of design) norm = Math.hypot(norm, row[column]);
    return norm;
  });
  const normalized = design.map(row => row.map((value, column) => value / norms[column]));
  const [minimum, maximum] = getCPURegressionSingularValueRange(normalized);
  const raw = design.map(row =>
    row.map((value, column) => (column === 0 ? value : value + means[column - 1] * row[0]))
  );
  const rawNorms = norms.map((_, column) => {
    let norm = 0;
    for (const row of raw) norm = Math.hypot(norm, row[column]);
    return norm;
  });
  const [rawMinimum, rawMaximum] = getCPURegressionSingularValueRange(
    raw.map(row => row.map((value, column) => value / rawNorms[column]))
  );
  const fit = fitCPURegressionWithQR(normalized, new Array<number>(design.length).fill(0))!;
  let inflation = 0;
  for (let column = 1; column < width; column++) {
    inflation = Math.max(
      inflation,
      getCPUErrorDot(fit.inverseRoot[column], fit.inverseRoot[column])
    );
  }
  return [rawMaximum / rawMinimum, (maximum / minimum) ** 2, inflation];
}

function getQuadraticRoots(constant: number, linear: number, quadratic: number): number[] {
  if (quadratic === 0) return linear === 0 ? [] : [-constant / linear];
  const discriminant = linear * linear - 4 * quadratic * constant;
  if (discriminant < 0) return [];
  const numerator = -0.5 * (linear + (linear >= 0 ? 1 : -1) * Math.sqrt(discriminant));
  return numerator === 0 ? [0] : [numerator / quadratic, constant / numerator];
}

/**
 * All roots of a degree-at-most-three polynomial on [-1,1], plus interval endpoints.
 * Quadratic derivative roots split a cubic into monotone pieces; bisection stops only when the
 * endpoints are adjacent float64 numbers. Repeated roots are included among the split points.
 * @internal
 */
export function getCPUErrorPolynomialCandidates(coefficients: readonly number[]): number[] {
  const scale = Math.max(...coefficients.map(Math.abs));
  if (!(scale > 0) || !Number.isFinite(scale)) return [-1, 1];
  const polynomial = coefficients.map(value => value / scale);
  const [constant = 0, linear = 0, quadratic = 0, cubic = 0] = polynomial;
  if (cubic === 0)
    return [
      -1,
      1,
      ...getQuadraticRoots(constant, linear, quadratic).filter(value => value > -1 && value < 1)
    ];
  const partitions = [
    -1,
    ...getQuadraticRoots(linear, 2 * quadratic, 3 * cubic).filter(value => value > -1 && value < 1),
    1
  ].sort((first, second) => first - second);
  const evaluate = (value: number) =>
    constant + value * (linear + value * (quadratic + value * cubic));
  const candidates = partitions.slice();
  for (let interval = 1; interval < partitions.length; interval++) {
    let lower = partitions[interval - 1];
    let upper = partitions[interval];
    const lowerSign = Math.sign(evaluate(lower));
    if (lowerSign === 0 || Math.sign(evaluate(upper)) === lowerSign || evaluate(upper) === 0)
      continue;
    while (true) {
      const middle = lower + (upper - lower) / 2;
      if (middle === lower || middle === upper) {
        candidates.push(Math.abs(evaluate(lower)) < Math.abs(evaluate(upper)) ? lower : upper);
        break;
      }
      const value = evaluate(middle);
      if (value === 0) {
        candidates.push(middle);
        break;
      }
      if (Math.sign(value) === lowerSign) lower = middle;
      else upper = middle;
    }
  }
  return candidates;
}
