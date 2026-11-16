// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {MAXIMUM_REFERENCE_RASTER_EMBEDDING_DIMENSIONS} from './raster-embedding-common';

/** Jacobi sweeps; the cyclic method converges quadratically and a 128 x 128 matrix needs under 15. */
const MAXIMUM_JACOBI_SWEEPS = 60;

/** Principal components of a covariance matrix, see {@link getRasterEmbeddingPrincipalComponents}. */
export type RasterEmbeddingPrincipalComponents = {
  /** Eigenvalues (component variances), largest first, `componentCount` entries. */
  eigenvalues: Float64Array;
  /** Unit eigenvectors, `component * dimensions + band`, in the sign convention of scikit-learn. */
  components: Float64Array;
  /** Eigenvalue of each component over the trace of the covariance (explained variance ratio). */
  explainedVarianceRatio: Float64Array;
};

/**
 * Eigendecomposition of the covariance from {@link GPURasterEmbeddingCovariance}, on the CPU in
 * binary64 (cyclic Jacobi; the matrix is at most 128 x 128). Returns the `componentCount` largest
 * components. The sign of each component makes its largest-magnitude loading positive, which is
 * scikit-learn's `svd_flip(u_based_decision=False)` convention for `PCA`, so the maps agree with
 * a NumPy or scikit-learn reference.
 *
 * Upload `components` (converted to float32) and the mean to {@link GPURasterEmbeddingProjection}.
 *
 * @param covariance Symmetric `dimensions x dimensions` matrix, row-major.
 */
export function getRasterEmbeddingPrincipalComponents(
  covariance: ArrayLike<number>,
  dimensions: number,
  componentCount: number
): RasterEmbeddingPrincipalComponents {
  const id = 'getRasterEmbeddingPrincipalComponents';
  validateCapacity(id, 'dimensions', dimensions, 1, MAXIMUM_REFERENCE_RASTER_EMBEDDING_DIMENSIONS);
  validateCapacity(id, 'componentCount', componentCount, 1, dimensions);
  validateRequirement(
    id,
    covariance.length === dimensions * dimensions,
    'covariance must hold dimensions squared entries'
  );
  const matrix = Float64Array.from(covariance);
  for (const value of matrix) {
    validateFiniteNumber(id, 'covariance entry', value);
  }
  const eigenvectors = new Float64Array(dimensions * dimensions);
  for (let index = 0; index < dimensions; index++) {
    eigenvectors[index * dimensions + index] = 1;
  }
  for (let sweep = 0; sweep < MAXIMUM_JACOBI_SWEEPS; sweep++) {
    let offDiagonal = 0;
    let diagonal = 0;
    for (let row = 0; row < dimensions; row++) {
      diagonal += matrix[row * dimensions + row] ** 2;
      for (let column = row + 1; column < dimensions; column++) {
        offDiagonal += matrix[row * dimensions + column] ** 2;
      }
    }
    if (offDiagonal <= 1e-30 * Math.max(diagonal, Number.MIN_VALUE)) {
      break;
    }
    for (let p = 0; p < dimensions - 1; p++) {
      for (let q = p + 1; q < dimensions; q++) {
        const apq = matrix[p * dimensions + q];
        if (apq === 0) continue;
        const theta = (matrix[q * dimensions + q] - matrix[p * dimensions + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < dimensions; k++) {
          const akp = matrix[k * dimensions + p];
          const akq = matrix[k * dimensions + q];
          matrix[k * dimensions + p] = c * akp - s * akq;
          matrix[k * dimensions + q] = s * akp + c * akq;
        }
        for (let k = 0; k < dimensions; k++) {
          const apk = matrix[p * dimensions + k];
          const aqk = matrix[q * dimensions + k];
          matrix[p * dimensions + k] = c * apk - s * aqk;
          matrix[q * dimensions + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < dimensions; k++) {
          const vkp = eigenvectors[k * dimensions + p];
          const vkq = eigenvectors[k * dimensions + q];
          eigenvectors[k * dimensions + p] = c * vkp - s * vkq;
          eigenvectors[k * dimensions + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const order = Array.from({length: dimensions}, (_, index) => index).sort(
    (left, right) => matrix[right * dimensions + right] - matrix[left * dimensions + left]
  );
  let trace = 0;
  for (let index = 0; index < dimensions; index++) trace += matrix[index * dimensions + index];
  const eigenvalues = new Float64Array(componentCount);
  const components = new Float64Array(componentCount * dimensions);
  for (let component = 0; component < componentCount; component++) {
    const source = order[component];
    eigenvalues[component] = matrix[source * dimensions + source];
    let largestBand = 0;
    for (let band = 0; band < dimensions; band++) {
      components[component * dimensions + band] = eigenvectors[band * dimensions + source];
      if (
        Math.abs(components[component * dimensions + band]) >
        Math.abs(components[component * dimensions + largestBand])
      ) {
        largestBand = band;
      }
    }
    if (components[component * dimensions + largestBand] < 0) {
      for (let band = 0; band < dimensions; band++) components[component * dimensions + band] *= -1;
    }
  }
  return {
    eigenvalues,
    components,
    explainedVarianceRatio: eigenvalues.map(value => (trace > 0 ? value / trace : 0))
  };
}

/**
 * Stretch ranges for the RGB view of {@link GPURasterEmbeddingProjection}: `+-sigmaCount`
 * standard deviations of each component, from {@link getRasterEmbeddingPrincipalComponents}.
 */
export function getRasterEmbeddingStretchRanges(
  eigenvalues: ArrayLike<number>,
  sigmaCount: number = 2
): [number, number][] {
  validateFiniteNumber('getRasterEmbeddingStretchRanges', 'sigmaCount', sigmaCount, 1e-6, 100);
  return Array.from(eigenvalues, value => {
    const spread = sigmaCount * Math.sqrt(Math.max(value, 0));
    return [-spread, spread] as [number, number];
  });
}
