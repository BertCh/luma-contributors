// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSinglePackedView} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getPairwiseSumWGSL} from './pairwise-sum-wgsl';
import {MULTIPLICATION_ROUNDING_ERROR_WGSL} from '../../analysis-core/kernels/multiplication-rounding-error-wgsl';

/** Magnitude as an integer significand times a power of two, including subnormals. */
function getSpatialFloat64Parts(value: number): {significand: number; exponent: number} {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, Math.abs(value));
  const high = view.getUint32(0);
  const low = view.getUint32(4);
  const exponent = (high >>> 20) & 0x7ff;
  return {
    significand: (high & 0xfffff) * 2 ** 32 + low + (exponent === 0 ? 0 : 2 ** 52),
    exponent: exponent === 0 ? -1074 : exponent - 1075
  };
}

/** Spatial similarity products whose final result may survive an overflowing intermediate. */
function getSpatialScaledProductQuotient(first: number, second: number, divisor: number): number {
  const product = first * second;
  const quotient = product / divisor;
  if (Number.isFinite(product) && product !== 0 && Number.isFinite(quotient) && quotient !== 0) {
    return quotient;
  }
  if (
    !Number.isFinite(first) ||
    !Number.isFinite(second) ||
    !Number.isFinite(divisor) ||
    first === 0 ||
    second === 0 ||
    divisor === 0
  )
    return quotient;
  const firstParts = getSpatialFloat64Parts(first);
  const secondParts = getSpatialFloat64Parts(second);
  const divisorParts = getSpatialFloat64Parts(divisor);
  const bounded =
    ((firstParts.significand / 2 ** 52) * (secondParts.significand / 2 ** 52)) /
    (divisorParts.significand / 2 ** 52);
  const boundedParts = getSpatialFloat64Parts(bounded);
  const exponent =
    firstParts.exponent +
    secondParts.exponent -
    divisorParts.exponent +
    boundedParts.exponent +
    104;
  const normalized = boundedParts.significand / 2 ** 52;
  const magnitude =
    exponent < -1022
      ? normalized * 2 ** -1022 * 2 ** (exponent + 1022)
      : normalized * 2 ** exponent;
  return Math.sign(first) * Math.sign(second) * Math.sign(divisor) * magnitude;
}

/** Recover `log(abs(1 - first * second))` when the rounded product is one. */
function getSpatialRoundedUnitProductLog(first: number, second: number): number {
  const firstParts = getSpatialFloat64Parts(first);
  const secondParts = getSpatialFloat64Parts(second);
  const exponent = firstParts.exponent + secondParts.exponent;
  const difference =
    (1n << BigInt(-exponent)) - BigInt(firstParts.significand) * BigInt(secondParts.significand);
  if (difference === 0n) return -Infinity;
  return Math.log(Number(difference < 0n ? -difference : difference)) + exponent * Math.LN2;
}

/**
 * Largest row count of `logDeterminant: 'exact'`. The eigenvalues of `W` come from a dense
 * symmetric eigen-decomposition on the CPU (`getSpatialWeightsEigenvalues`, about `(4/3) n^3`
 * operations and `n^2` float64 values of memory), which takes well under a second at 1024 rows
 * and grows cubically; above this threshold use the trace series.
 */
export const GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT = 1024;

/** How the maximum-likelihood estimators evaluate `ln|I - rho W|` and its trace derivatives. */
export type GPUSpatialLogDeterminantMode = 'series' | 'exact';

/**
 * Eigenvalues of a square CSR weights matrix `W` whose sparsity pattern is symmetric and that is
 * similar to a symmetric matrix, which holds for any `W = D^-1 C` with `C` symmetric (for example
 * row-standardized symmetric contiguity or distance weights; Ord 1975). A positive scaling `d`
 * with `d_i w_ij = d_j w_ji` is recovered from the pattern, `S = D^(1/2) W D^(-1/2)` is symmetric
 * with the same real eigenvalues, and they are returned ascending (float64).
 *
 * Throws when a link has no reverse link or the matrix is not similar to a symmetric one (relative
 * tolerance 1e-4 relative to each transformed pair's magnitude, with no absolute floor), or
 * when the recovered similarity scales are not positive finite numbers. Upload the result as a
 * float32 view for `eigenvalues`.
 */
export function getSpatialWeightsEigenvalues(
  offsets: ArrayLike<number>,
  neighbors: ArrayLike<number>,
  weights: ArrayLike<number>,
  rowCount: number
): Float64Array {
  const n = rowCount;
  const find = (row: number, column: number): number => {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      if (neighbors[slot] === column && weights[slot] !== 0) return weights[slot];
    }
    return 0;
  };
  const scale = new Float64Array(n);
  const visited = new Uint8Array(n);
  for (let root = 0; root < n; root++) {
    if (visited[root]) continue;
    visited[root] = 1;
    scale[root] = 1;
    const queue = [root];
    for (let head = 0; head < queue.length; head++) {
      const row = queue[head];
      for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
        const column = neighbors[slot];
        if (column >= n || column === row || weights[slot] === 0) continue;
        const reverse = find(column, row);
        // Pattern must be symmetric: a link without a reverse link has no real symmetric similar form.
        if (reverse === 0) throw new Error('getSpatialWeightsEigenvalues: asymmetric pattern');
        if (!visited[column]) {
          visited[column] = 1;
          scale[column] = getSpatialScaledProductQuotient(scale[row], weights[slot], reverse);
          if (!(scale[column] > 0 && Number.isFinite(scale[column]))) {
            throw new Error(
              'getSpatialWeightsEigenvalues: similarity scales must be positive finite numbers'
            );
          }
          queue.push(column);
        }
      }
    }
  }
  const matrix = Array.from({length: n}, () => new Float64Array(n));
  for (let row = 0; row < n; row++) {
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const column = neighbors[slot];
      if (column >= n) continue;
      matrix[row][column] += getSpatialScaledProductQuotient(
        Math.sqrt(scale[row]),
        weights[slot],
        Math.sqrt(scale[column])
      );
    }
  }
  for (let row = 0; row < n; row++) {
    for (let column = 0; column < row; column++) {
      const forward = matrix[row][column];
      const reverse = matrix[column][row];
      const magnitude = Math.max(Math.abs(forward), Math.abs(reverse));
      // Not reversible (d_i w_ij != d_j w_ji): the eigenvalues need not be real.
      if (
        !Number.isFinite(magnitude) ||
        (magnitude > 0 && Math.abs(forward / magnitude - reverse / magnitude) > 1e-4)
      ) {
        throw new Error('getSpatialWeightsEigenvalues: not similar to a symmetric matrix');
      }
      matrix[row][column] = matrix[column][row] = 0.5 * forward + 0.5 * reverse;
    }
  }
  return getSymmetricEigenvalues(matrix);
}

/**
 * `ln|I - rho W| = sum ln|1 - rho lambda_i|`. Negative factors are permitted; a zero
 * factor returns `-Infinity`. Uses `log1p` for positive factors to retain small terms,
 * exact significands when a product rounds to one, and operand logs when a finite product
 * overflows. A true zero factor takes priority over every other factor.
 */
export function getSpatialExactLogDeterminant(rho: number, eigenvalues: ArrayLike<number>): number {
  let sum = 0;
  for (let index = 0; index < eigenvalues.length; index++) {
    const eigenvalue = eigenvalues[index];
    const product = rho * eigenvalue;
    if (product === 1) {
      const logarithm = getSpatialRoundedUnitProductLog(rho, eigenvalue);
      if (logarithm === -Infinity) return -Infinity;
      sum += logarithm;
    } else if (!Number.isFinite(product) && Number.isFinite(rho) && Number.isFinite(eigenvalue)) {
      sum += Math.log(Math.abs(rho)) + Math.log(Math.abs(eigenvalue));
    } else {
      sum += product < 1 ? Math.log1p(-product) : Math.log(Math.abs(1 - product));
    }
  }
  return sum;
}

/**
 * Eigenvalues (ascending) of a dense symmetric matrix by Householder tridiagonalization and the
 * implicit QL algorithm (values only). The input rows are overwritten.
 */
export function getSymmetricEigenvalues(a: Float64Array[]): Float64Array {
  const n = a.length;
  const d = new Float64Array(n);
  const e = new Float64Array(n);
  for (let i = n - 1; i > 0; i--) {
    const l = i - 1;
    let h = 0;
    if (l > 0) {
      let scale = 0;
      for (let k = 0; k <= l; k++) scale += Math.abs(a[i][k]);
      if (scale === 0) {
        e[i] = a[i][l];
      } else {
        for (let k = 0; k <= l; k++) {
          a[i][k] /= scale;
          h += a[i][k] * a[i][k];
        }
        let f = a[i][l];
        let g = f >= 0 ? -Math.sqrt(h) : Math.sqrt(h);
        e[i] = scale * g;
        h -= f * g;
        a[i][l] = f - g;
        f = 0;
        for (let j = 0; j <= l; j++) {
          g = 0;
          for (let k = 0; k <= j; k++) g += a[j][k] * a[i][k];
          for (let k = j + 1; k <= l; k++) g += a[k][j] * a[i][k];
          e[j] = g / h;
          f += e[j] * a[i][j];
        }
        const hh = f / (h + h);
        for (let j = 0; j <= l; j++) {
          f = a[i][j];
          g = e[j] - hh * f;
          e[j] = g;
          for (let k = 0; k <= j; k++) a[j][k] -= f * e[k] + g * a[i][k];
        }
      }
    } else {
      e[i] = a[i][l];
    }
    d[i] = h;
  }
  for (let i = 0; i < n; i++) d[i] = a[i][i];
  for (let i = 1; i < n; i++) e[i - 1] = e[i];
  e[n - 1] = 0;
  for (let l = 0; l < n; l++) {
    let iteration = 0;
    let m: number;
    do {
      for (m = l; m < n - 1; m++) {
        const scale = Math.abs(d[m]) + Math.abs(d[m + 1]);
        if (Math.abs(e[m]) <= Number.EPSILON * scale) break;
      }
      if (m !== l) {
        // QL did not converge: the matrix has non-finite entries.
        if (iteration++ === 200) throw new Error('getSymmetricEigenvalues: no convergence');
        let g = (d[l + 1] - d[l]) / (2 * e[l]);
        let r = Math.hypot(g, 1);
        g = d[m] - d[l] + e[l] / (g + (g >= 0 ? Math.abs(r) : -Math.abs(r)));
        let s = 1;
        let c = 1;
        let p = 0;
        let i: number;
        for (i = m - 1; i >= l; i--) {
          const f = s * e[i];
          const b = c * e[i];
          r = Math.hypot(f, g);
          e[i + 1] = r;
          if (r === 0) {
            d[i + 1] -= p;
            e[m] = 0;
            break;
          }
          s = f / r;
          c = g / r;
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2 * c * b;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - b;
        }
        if (r === 0 && i >= l) continue;
        d[l] -= p;
        e[l] = g;
        e[m] = 0;
      }
    } while (m !== l);
  }
  return d.sort();
}

/** Validation shared by the lag and error estimators. @internal */
export function validateExactLogDeterminant(
  id: string,
  props: {logDeterminant?: GPUSpatialLogDeterminantMode; eigenvalues?: GraphDataView<'float32'>},
  rows: number
): boolean {
  if (props.logDeterminant !== undefined && !['series', 'exact'].includes(props.logDeterminant)) {
    throw new Error(`${id} logDeterminant must be 'series' or 'exact'`);
  }
  if (props.logDeterminant !== 'exact') {
    if (props.eigenvalues) throw new Error(`${id} eigenvalues need logDeterminant 'exact'`);
    return false;
  }
  if (rows > GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT) {
    throw new Error(
      `${id} logDeterminant 'exact' supports at most ${GPU_SPATIAL_EXACT_LOG_DETERMINANT_MAXIMUM_ROW_COUNT} rows`
    );
  }
  if (!props.eigenvalues) {
    throw new Error(
      `${id} logDeterminant 'exact' needs eigenvalues (getSpatialWeightsEigenvalues)`
    );
  }
  validateSinglePackedView(props.eigenvalues, ['float32'], `${id} eigenvalues`);
  if (props.eigenvalues.length !== rows) {
    throw new Error(`${id} eigenvalues must hold one value per row`);
  }
  return true;
}

/**
 * WGSL defining the same functions as the trace series, exactly, from the f32 binding
 * `bindingName` of the `rowCount` eigenvalues of `W`: `getSpatialLogDeterminant(rho) = sum
 * ln|1 - rho lambda|`, `getTraceOne(rho) = sum lambda / (1 - rho lambda)`,
 * `getLogDeterminantDerivative(rho) = -getTraceOne(rho)` and `getTraceTwo(rho) = sum (lambda /
 * (1 - rho lambda))^2`. Each is a fixed pairwise sum over the eigenvalues (`getPairwiseSumWGSL`).
 */
export function getExactLogDeterminantWGSL(bindingName: string, rowCount: number): string {
  const sum = (term: string) => `${getPairwiseSumWGSL({
    count: `${rowCount}u`,
    name: 'total',
    setup: `let lambda = ${bindingName}[${bindingName}Offset + totalItem];
    let ratio = lambda / (1.0 - rho * lambda);`,
    term: `(${term})`
  })}
  return total;`;
  return /* wgsl */ `
${MULTIPLICATION_ROUNDING_ERROR_WGSL}
fn getSpatialLogAbsoluteFactor(rho: f32, eigenvalue: f32) -> f32 {
  let product = rho * eigenvalue;
  if (product == 1.0) {
    // Rebalance exact powers of two so even very unequal operands are normal and bounded.
    let first = frexp(rho);
    let second = frexp(eigenvalue);
    let residual = getMultiplicationRoundingError(
      ldexp(first.fract, first.exp + second.exp), second.fract, product);
    if (residual != 0.0) { return log(abs(residual)); }
    // Nonfinite constant expressions are shader-creation errors in WGSL.
    var negativeInfinityBits = 0xff800000u;
    return bitcast<f32>(negativeInfinityBits);
  }
  // The log magnitude is representable even when finite operands overflow on multiplication.
  if ((bitcast<u32>(product) & 0x7fffffffu) == 0x7f800000u) {
    return log(abs(rho)) + log(abs(eigenvalue));
  }
  // WGSL has no log1p. Near zero this series retains terms lost by subtracting from 1.
  // At |product| < 0.01 the omitted term is below 2.1e-11.
  if (abs(product) < 0.01) {
    return -product * (1.0 + product * (0.5 + product * (1.0 / 3.0 + product * 0.25)));
  }
  return log(abs(1.0 - product));
}

fn getSpatialLogDeterminant(rho: f32) -> f32 {
  ${sum('getSpatialLogAbsoluteFactor(rho, lambda)')}
}

fn getTraceOne(rho: f32) -> f32 {
  ${sum('ratio')}
}

fn getLogDeterminantDerivative(rho: f32) -> f32 {
  return -getTraceOne(rho);
}

fn getTraceTwo(rho: f32) -> f32 {
  ${sum('ratio * ratio')}
}
`;
}
