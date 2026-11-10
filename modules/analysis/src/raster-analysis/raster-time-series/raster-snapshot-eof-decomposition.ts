// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getGPURasterFixedPointValues} from './gpu-raster-streaming-accumulator';

/** Cyclic Jacobi sweeps before giving up; binary64 converges well within this for 256 x 256. */
const MAXIMUM_JACOBI_SWEEP_COUNT = 64;

/**
 * Temporal eigendecomposition of a snapshot covariance (the CPU eigenproblem of the method of
 * snapshots), after the caller has read `GPURasterSnapshotCovariance`'s words back from the GPU.
 */
export type GPURasterSnapshotEOFDecomposition = {
  /** Singular values `sqrt(eigenvalue)` of the weighted anomaly matrix, largest first; upload these. */
  singularValues: Float32Array;
  /** Temporal eigenvalues `s^2 / (sliceCount - 1)`, the explained variance of each mode. */
  explainedVariance: Float64Array;
  /** Explained variance over the trace of the covariance. */
  explainedVarianceRatio: Float64Array;
  /** Mode-major `mode * sliceCount + slice`: orthonormal right singular vectors; upload these. */
  temporalModes: Float32Array;
};

/** Decodes the packed upper triangle of `covarianceWords` into a symmetric binary64 matrix. */
export function getGPURasterSnapshotCovarianceMatrix(
  covarianceWords: ArrayLike<number>,
  sliceCount: number,
  unitExponent: number
): Float64Array {
  const packed = getGPURasterFixedPointValues(covarianceWords, unitExponent);
  const matrix = new Float64Array(sliceCount * sliceCount);
  let entry = 0;
  for (let row = 0; row < sliceCount; row++) {
    for (let column = row; column < sliceCount; column++) {
      matrix[row * sliceCount + column] = packed[entry];
      matrix[column * sliceCount + row] = packed[entry];
      entry++;
    }
  }
  return matrix;
}

/**
 * The CPU step of the method of snapshots: a cyclic Jacobi eigendecomposition of the symmetric
 * `sliceCount x sliceCount` Gram matrix `X_w' X_w`, in binary64. It is explicit because it needs a
 * readback, which no contributor performs: read `covarianceWords` back, call
 * {@link getGPURasterSnapshotCovarianceMatrix}, call this, upload `temporalModes` and
 * `singularValues`, then run `GPURasterEOFProjection`.
 *
 * Sign convention: a temporal mode's sign is fixed so that its largest-magnitude entry (the first
 * such entry on a tie) is positive. Singular vectors have no intrinsic sign, so the convention is
 * what makes results reproducible; a spatial EOF inherits its sign from its temporal mode.
 *
 * @param covariance Row-major `sliceCount x sliceCount` symmetric matrix (not modified).
 * @param modeCount Number of leading modes to return.
 */
export function getGPURasterSnapshotEOFDecomposition(
  covariance: ArrayLike<number>,
  sliceCount: number,
  modeCount: number
): GPURasterSnapshotEOFDecomposition {
  validateCapacity('getGPURasterSnapshotEOFDecomposition', 'modeCount', modeCount, 1, sliceCount);
  const size = sliceCount;
  const matrix = Float64Array.from(covariance);
  const vectors = new Float64Array(size * size);
  for (let diagonal = 0; diagonal < size; diagonal++) vectors[diagonal * size + diagonal] = 1;
  const scale = Math.max(
    ...Array.from({length: size}, (_, i) => Math.abs(matrix[i * size + i])),
    1
  );
  for (let sweep = 0; sweep < MAXIMUM_JACOBI_SWEEP_COUNT; sweep++) {
    let offDiagonalTotal = 0;
    for (let row = 0; row < size; row++) {
      for (let column = row + 1; column < size; column++) {
        offDiagonalTotal += matrix[row * size + column] ** 2;
      }
    }
    if (offDiagonalTotal <= 1e-28 * scale * scale) break;
    for (let row = 0; row < size - 1; row++) {
      for (let column = row + 1; column < size; column++) {
        rotateJacobi(matrix, vectors, size, row, column);
      }
    }
  }
  const order = Array.from({length: size}, (_, index) => index).sort(
    (left, right) => matrix[right * size + right] - matrix[left * size + left]
  );
  const singularValues = new Float32Array(modeCount);
  const explainedVariance = new Float64Array(modeCount);
  const explainedVarianceRatio = new Float64Array(modeCount);
  const temporalModes = new Float32Array(modeCount * size);
  let trace = 0;
  for (let diagonal = 0; diagonal < size; diagonal++)
    trace += Math.max(matrix[diagonal * size + diagonal], 0);
  for (let mode = 0; mode < modeCount; mode++) {
    const source = order[mode];
    const eigenvalue = Math.max(matrix[source * size + source], 0);
    singularValues[mode] = Math.sqrt(eigenvalue);
    explainedVariance[mode] = eigenvalue / Math.max(sliceCount - 1, 1);
    explainedVarianceRatio[mode] = trace > 0 ? eigenvalue / trace : 0;
    let largest = 0;
    for (let slice = 0; slice < size; slice++) {
      if (Math.abs(vectors[slice * size + source]) > Math.abs(vectors[largest * size + source])) {
        largest = slice;
      }
    }
    const sign = vectors[largest * size + source] < 0 ? -1 : 1;
    for (let slice = 0; slice < size; slice++) {
      temporalModes[mode * size + slice] = sign * vectors[slice * size + source];
    }
  }
  return {singularValues, explainedVariance, explainedVarianceRatio, temporalModes};
}

/** One Jacobi rotation annihilating `matrix[row, column]`; columns of `vectors` accumulate them. */
function rotateJacobi(
  matrix: Float64Array,
  vectors: Float64Array,
  size: number,
  row: number,
  column: number
): void {
  const off = matrix[row * size + column];
  if (off === 0) return;
  const diagonalRow = matrix[row * size + row];
  const diagonalColumn = matrix[column * size + column];
  const theta = (diagonalColumn - diagonalRow) / (2 * off);
  const tangent = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
  const cosine = 1 / Math.sqrt(1 + tangent * tangent);
  const sine = tangent * cosine;
  matrix[row * size + row] = diagonalRow - tangent * off;
  matrix[column * size + column] = diagonalColumn + tangent * off;
  matrix[row * size + column] = 0;
  matrix[column * size + row] = 0;
  for (let index = 0; index < size; index++) {
    if (index !== row && index !== column) {
      const left = matrix[index * size + row];
      const right = matrix[index * size + column];
      const newLeft = cosine * left - sine * right;
      const newRight = sine * left + cosine * right;
      matrix[index * size + row] = newLeft;
      matrix[row * size + index] = newLeft;
      matrix[index * size + column] = newRight;
      matrix[column * size + index] = newRight;
    }
    const vectorLeft = vectors[index * size + row];
    const vectorRight = vectors[index * size + column];
    vectors[index * size + row] = cosine * vectorLeft - sine * vectorRight;
    vectors[index * size + column] = sine * vectorLeft + cosine * vectorRight;
  }
}
