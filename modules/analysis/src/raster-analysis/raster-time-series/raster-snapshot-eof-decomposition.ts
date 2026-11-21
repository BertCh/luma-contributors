// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getPackedUint32Values,
  type GPUPackedReadbackValues
} from '../../analysis-core/kernels/packed-readback';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getGPURasterFixedPointValues} from './gpu-raster-streaming-accumulator';

/** Cyclic Jacobi sweeps before giving up; binary64 converges well within this for 256 x 256. */
const MAXIMUM_JACOBI_SWEEP_COUNT = 64;
/** Leaves binary64 headroom for the sum of squares and the relative convergence threshold. */
const MAXIMUM_JACOBI_NORM_EXPONENT = 450;

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
  covarianceWords: GPUPackedReadbackValues,
  sliceCount: number,
  unitExponent: number
): Float64Array {
  validateCapacity('getGPURasterSnapshotCovarianceMatrix', 'sliceCount', sliceCount, 1);
  const entryCount = (sliceCount * (sliceCount + 1)) / 2;
  const words = getPackedUint32Values(covarianceWords, entryCount * 3);
  const packed = getGPURasterFixedPointValues(words.subarray(0, entryCount * 3), unitExponent);
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
  let maximumMagnitude = 0;
  for (const entry of matrix) maximumMagnitude = Math.max(maximumMagnitude, Math.abs(entry));
  const matrixExponent = Math.floor(Math.log2(maximumMagnitude || 1));
  const normExponent = Math.max(
    -MAXIMUM_JACOBI_NORM_EXPONENT,
    Math.min(matrixExponent, MAXIMUM_JACOBI_NORM_EXPONENT)
  );
  const normScale = Number.isFinite(matrixExponent) ? 2 ** (matrixExponent - normExponent) : 1;
  // Scale tiny matrices up before rotating. For large matrices, scale only the norm unless
  // eigenvalues or rotation intermediates could overflow.
  let matrixScale = Math.min(normScale, 1);
  const rotationHeadroom = 2 * size;
  if (Number.isFinite(maximumMagnitude) && maximumMagnitude > Number.MAX_VALUE / rotationHeadroom) {
    matrixScale = 2 ** Math.ceil(Math.log2(rotationHeadroom));
  }
  // An isolated diagonal is already an eigenvalue. Keep its exact value if downscaling rounds it.
  const preservedDiagonals = new Map<number, number>();
  if (matrixScale > 1) {
    for (let row = 0; row < size; row++) {
      const rowOffset = row * size;
      const diagonal = matrix[rowOffset + row];
      if ((diagonal / matrixScale) * matrixScale === diagonal) continue;
      const rowEntries = matrix.subarray(rowOffset, rowOffset + size);
      if (rowEntries.every((entry, column) => column === row || entry === 0)) {
        preservedDiagonals.set(row, diagonal);
      }
    }
  }
  if (matrixScale !== 1) {
    for (let index = 0; index < matrix.length; index++) matrix[index] /= matrixScale;
  }
  const vectors = new Float64Array(size * size);
  for (let diagonal = 0; diagonal < size; diagonal++) vectors[diagonal * size + diagonal] = 1;
  // Use the same power-of-two shift for both squared norms in the relative convergence test.
  const normDivisor = normScale / matrixScale;
  let squaredNorm = 0;
  for (const entry of matrix) {
    const scaledEntry = entry / normDivisor;
    squaredNorm += scaledEntry * scaledEntry;
  }
  for (let sweep = 0; sweep < MAXIMUM_JACOBI_SWEEP_COUNT; sweep++) {
    let offDiagonalTotal = 0;
    for (let row = 0; row < size; row++) {
      for (let column = row + 1; column < size; column++) {
        offDiagonalTotal += (matrix[row * size + column] / normDivisor) ** 2;
      }
    }
    if (offDiagonalTotal <= 1e-28 * squaredNorm) break;
    for (let row = 0; row < size - 1; row++) {
      for (let column = row + 1; column < size; column++) {
        rotateJacobi(matrix, vectors, size, row, column);
      }
    }
  }
  const order = Array.from({length: size}, (_, index) => index).sort((left, right) => {
    const leftValue = matrix[left * size + left];
    const rightValue = matrix[right * size + right];
    const difference = rightValue - leftValue;
    if (difference !== 0 || preservedDiagonals.size === 0) return difference;
    // Downscaling can round distinct subnormal eigenvalues to the same working value.
    const originalLeft = preservedDiagonals.get(left) ?? leftValue * matrixScale;
    const originalRight = preservedDiagonals.get(right) ?? rightValue * matrixScale;
    return originalRight - originalLeft;
  });
  const singularValues = new Float32Array(modeCount);
  const explainedVariance = new Float64Array(modeCount);
  const explainedVarianceRatio = new Float64Array(modeCount);
  const temporalModes = new Float32Array(modeCount * size);
  const singularValueScale = Math.sqrt(matrixScale);
  const sampleDivisor = Math.max(sliceCount - 1, 1);
  const varianceDivisor = sampleDivisor / matrixScale;
  let trace = 0;
  for (let diagonal = 0; diagonal < size; diagonal++)
    trace += Math.max(matrix[diagonal * size + diagonal], 0);
  for (let mode = 0; mode < modeCount; mode++) {
    const source = order[mode];
    const eigenvalue = Math.max(matrix[source * size + source], 0);
    const preservedDiagonal = preservedDiagonals.get(source);
    if (preservedDiagonal === undefined) {
      singularValues[mode] = Math.sqrt(eigenvalue) * singularValueScale;
      explainedVariance[mode] = eigenvalue / varianceDivisor;
    } else {
      const originalEigenvalue = Math.max(preservedDiagonal, 0);
      singularValues[mode] = Math.sqrt(originalEigenvalue);
      explainedVariance[mode] = originalEigenvalue / sampleDivisor;
    }
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
