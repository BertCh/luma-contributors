// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  computeGeographicallyWeightedRegressionOnCPU,
  getGeographicallyWeightedRegressionWeight
} from './geographically-weighted-regression-oracle';
import {getStudentTCriticalValue} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-student-t';

// scipy.stats.t.isf(tail / 2, min(df, 200000)), SciPy 1.13.1; independent of WGSL.
export const LARGE_SAMPLE_QUANTILE_CASES = [
  [4096, 0.05, 1.9605433205921143],
  [4096, 1.846629945e-6, 4.776444738411734],
  [4096, 1e-10, 6.483891077629365],
  [200000, 0.05, 1.9599758459667684],
  [200000, 1.846629945e-6, 4.769664223435063],
  [200000, 1e-10, 6.46729725843051],
  [200000, 1.8474139169110982e-6, 4.76957870690452],
  [1048583, 1.8474139169110982e-6, 4.76957870690452],
  [20, 0.05, 2.085963447265837]
];

export const REVIEW_FOCUS_ROWS = 262144;
export const REVIEW_MOTIF_ROWS = 15;
export const REVIEW_BANDWIDTHS = [0.25, 1.2];
const NOISE = [0.25, -0.5, 0.5, -0.5, 0.25];

/** Three nearby sites, five observations each. Type 0 benefits from pooling; type 1 has slope drift. */
function createMotif(kind: number, phase: number, length: number) {
  const positions = new Float32Array(length * 2);
  const predictors = new Float32Array(length);
  const response = new Float32Array(length);
  for (let row = 0; row < length; row++) {
    const site = Math.floor(row / 5);
    const observation = row % 5;
    const predictor = observation - 2;
    const slope = 2 + phase * 0.5 + (kind === 1 ? (site - 1) * 4 : 0);
    positions[row * 2] = site * 0.5;
    predictors[row] = predictor;
    response[row] = 1 + phase * 0.25 + slope * predictor + NOISE[observation];
  }
  return {positions, predictors, response, predictorCount: 1};
}

/** Cache tiny independent CPU fits; isolated motifs make the full-data oracle O(motifs), not O(n²). */
export function createBoundaryReference(rowCount: number) {
  const positions = new Float32Array(rowCount * 2);
  const predictors = new Float32Array(rowCount);
  const response = new Float32Array(rowCount);
  const firstCohortMotifs = Math.floor(REVIEW_FOCUS_ROWS / REVIEW_MOTIF_ROWS);
  const motifCount = Math.ceil(rowCount / REVIEW_MOTIF_ROWS);
  const templates = new Map<
    string,
    {kind: number; phase: number; length: number; multiplicity: number}
  >();
  for (let motif = 0; motif < motifCount; motif++) {
    const kind = motif < firstCohortMotifs ? 0 : 1;
    const phase = motif % 4;
    const length = Math.min(REVIEW_MOTIF_ROWS, rowCount - motif * REVIEW_MOTIF_ROWS);
    const template = createMotif(kind, phase, length);
    const key = `${kind}:${phase}:${length}`;
    const existing = templates.get(key);
    if (existing) existing.multiplicity++;
    else templates.set(key, {kind, phase, length, multiplicity: 1});
    for (let localRow = 0; localRow < length; localRow++) {
      const row = motif * REVIEW_MOTIF_ROWS + localRow;
      positions[row * 2] = (motif % 512) * 10 + template.positions[localRow * 2];
      positions[row * 2 + 1] = Math.floor(motif / 512) * 10;
      predictors[row] = template.predictors[localRow];
      response[row] = template.response[localRow];
    }
  }
  const cached = new Map<string, ReturnType<typeof computeGeographicallyWeightedRegressionOnCPU>>();
  const fit = (kind: number, phase: number, length: number, bandwidth: number) => {
    const key = `${kind}:${phase}:${length}:${bandwidth}`;
    let result = cached.get(key);
    if (!result) {
      result = computeGeographicallyWeightedRegressionOnCPU({
        ...createMotif(kind, phase, length),
        settings: {kernel: 'bisquare', bandwidths: [bandwidth]}
      });
      cached.set(key, result);
    }
    return result;
  };
  const score = (bandwidth: number, cohort?: number) => {
    let residualSum = 0;
    let trace = 0;
    let count = 0;
    let singular = 0;
    let responseSum = 0;
    let responseSquares = 0;
    for (const {kind, phase, length, multiplicity} of templates.values()) {
      if (cohort !== undefined && kind !== cohort) continue;
      const result = fit(kind, phase, length, bandwidth);
      // A 3/4-row tail has valid local fits but no standalone AICc denominator. Aggregate
      // its raw candidate statistics; selecting a model here would incorrectly fit per motif.
      const candidate = result.candidates[0];
      residualSum += candidate.residualSumOfSquares * multiplicity;
      trace += candidate.traceOfHat * multiplicity;
      count += candidate.observationCount * multiplicity;
      singular += candidate.singularRowCount * multiplicity;
      const template = createMotif(kind, phase, length);
      for (let row = 0; row < length; row++)
        if (result.localStatus[row] === 0) {
          responseSum += template.response[row] * multiplicity;
          responseSquares += template.response[row] ** 2 * multiplicity;
        }
    }
    const variance = residualSum / count;
    const aicc =
      count * Math.log(Math.max(variance, 1e-30)) +
      count * Math.log(2 * Math.PI) +
      (count * (count + trace)) / (count - trace - 2);
    const sigmaSquared = residualSum / (count - trace);
    const adjustedAlpha = (0.05 * 2) / trace;
    return {
      bandwidth,
      residualSum,
      trace,
      count,
      singular,
      aicc,
      sigmaSquared,
      adjustedAlpha,
      criticalT: getStudentTCriticalValue(adjustedAlpha, count - 1),
      logLikelihood:
        -0.5 * count * (Math.log(2 * Math.PI) + Math.log(Math.max(variance, 1e-30)) + 1),
      rSquared: 1 - residualSum / (responseSquares - (responseSum * responseSum) / count)
    };
  };
  const local = (row: number, bandwidth: number) => {
    const motif = Math.floor(row / REVIEW_MOTIF_ROWS);
    const kind = motif < firstCohortMotifs ? 0 : 1;
    const phase = motif % 4;
    const length = Math.min(REVIEW_MOTIF_ROWS, rowCount - motif * REVIEW_MOTIF_ROWS);
    const result = fit(kind, phase, length, bandwidth);
    const localRow = row % REVIEW_MOTIF_ROWS;
    const global = score(bandwidth);
    // Compute diag((X'WX)^-1 X'W²X (X'WX)^-1) directly. Tiny motifs with invalid
    // standalone AICc have NaN inference outputs, even though the global fit is valid.
    const template = createMotif(kind, phase, length);
    let weightSum = 0;
    let weightedPredictor = 0;
    let weightedSquare = 0;
    const weights = Array.from({length}, (_, other) => {
      const weight = getGeographicallyWeightedRegressionWeight(
        'bisquare',
        Math.abs(template.positions[2 * other] - template.positions[2 * localRow]),
        bandwidth
      );
      weightSum += weight;
      weightedPredictor += weight * template.predictors[other];
      weightedSquare += weight * template.predictors[other] ** 2;
      return weight;
    });
    const determinant = weightSum * weightedSquare - weightedPredictor ** 2;
    const variances = [0, 0];
    let squaredWeights = 0;
    let squaredWeightedPredictor = 0;
    let squaredWeightedSquare = 0;
    weights.forEach((weight, other) => {
      const predictor = template.predictors[other];
      squaredWeights += weight ** 2;
      squaredWeightedPredictor += weight ** 2 * predictor;
      squaredWeightedSquare += weight ** 2 * predictor ** 2;
      variances[0] +=
        ((weight * (weightedSquare - weightedPredictor * predictor)) / determinant) ** 2;
      variances[1] += ((weight * (weightSum * predictor - weightedPredictor)) / determinant) ** 2;
    });
    const standardErrors = variances.map(value => Math.sqrt(value * global.sigmaSquared));
    const coefficients = result.coefficients.slice(2 * localRow, 2 * localRow + 2);
    const tValues = coefficients.map((value, column) => value / standardErrors[column]);
    // The normalized two-column weighted design has Gram eigenvalues 1 ± |cosine|.
    const cosine = Math.abs(
      squaredWeightedPredictor / Math.sqrt(squaredWeights * squaredWeightedSquare)
    );
    return {
      coefficients,
      standardErrors,
      tValues,
      significant: tValues.map(value => (Math.abs(value) > global.criticalT ? 1 : 0)),
      fitted: result.fitted[localRow],
      residual: result.residuals[localRow],
      hat: result.hatDiagonal[localRow],
      localR2: result.localR2[localRow],
      conditionNumber: Math.sqrt((1 + cosine) / (1 - cosine))
    };
  };
  return {scene: {positions, predictors, response, predictorCount: 1}, score, local};
}
