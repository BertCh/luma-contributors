// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUGeographicallyWeightedRegressionModelResult,
  getGPUGeographicallyTemporallyWeightedRegressionModelResult
} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-model-result';

// Both dense and indexed GTWR publish these same slots; slot 5 is deliberately unpublished.
function getGTWRSummary(observations = 10, trace = 2) {
  const residualSumOfSquares = observations > 0 ? 8 : 0;
  const logLikelihood =
    observations > 0
      ? (-observations / 2) * (Math.log((2 * Math.PI * residualSumOfSquares) / observations) + 1)
      : NaN;
  const denominator = observations - trace - 2;
  return Float32Array.of(
    residualSumOfSquares,
    trace,
    observations > 0 && denominator > 0
      ? -2 * logLikelihood + (2 * observations * (trace + 1)) / denominator
      : NaN,
    0.75,
    observations,
    NaN,
    1,
    observations > trace ? residualSumOfSquares / (observations - trace) : NaN,
    NaN,
    NaN,
    logLikelihood,
    observations + 1
  );
}

it('reads dense and indexed GTWR summaries without a bandwidth-selection flag', () => {
  const summary = getGTWRSummary();
  // Preserve the existing GWR reader: its slot 5 really does describe candidate selection.
  expect(getGPUGeographicallyWeightedRegressionModelResult({summary}).status.ok).toBe(false);
  const result = getGPUGeographicallyTemporallyWeightedRegressionModelResult({summary});
  expect(result.status.ok).toBe(true);
  expect(result.rowCount).toBe(10);
  expect(result.singularRowCount).toBe(1);
  expect(result.includedRowCount).toBe(11);
  expect(result.logLikelihood).toBeCloseTo(-13.073667575475678, 5);
  // ll=-5*(log(2*pi*0.8)+1); AIC=-2*ll+6; RSS/(n-tr(S))=1.
  expect(result.akaikeInformationCriterion).toBeCloseTo(32.147335150951356, 5);
  expect(result.sigmaSquared).toBe(1);
  expect(result.correctedAkaikeInformationCriterion).toBeCloseTo(36.147335150951356, 5);
});

it('reports no usable GTWR rows independently of the singular-row count', () => {
  const summary = getGTWRSummary(0);
  summary[10] = NaN;
  const result = getGPUGeographicallyTemporallyWeightedRegressionModelResult({summary});
  expect(result.status.ok).toBe(false);
  expect(result.logLikelihood).toBeNull();
  expect(result.akaikeInformationCriterion).toBeNull();
  expect(result.correctedAkaikeInformationCriterion).toBeNull();
});

it('retains a valid GTWR likelihood when n-tr(S)-2 makes AICc undefined', () => {
  const summary = getGTWRSummary(4, 2);
  summary[2] = NaN;
  const result = getGPUGeographicallyTemporallyWeightedRegressionModelResult({summary});
  expect(result.status.ok).toBe(true);
  expect(result.logLikelihood).toBeCloseTo(-7.0620484939385815, 5);
  expect(result.akaikeInformationCriterion).toBeCloseTo(20.124096987877163, 5);
  expect(result.correctedAkaikeInformationCriterion).toBeNull();
});

it('preserves panel-error likelihood and unavailable criteria, counting absorbed entity effects', async () => {
  const {getGPUSpatialPanelErrorMaximumLikelihoodModelResult} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/spatial-panel-error-maximum-likelihood-model-result'
  );
  const summary = new Float32Array(22).fill(NaN);
  summary[0] = 48;
  summary[2] = 0.5;
  summary[3] = -56;
  summary[15] = 1;
  const result = getGPUSpatialPanelErrorMaximumLikelihoodModelResult(
    {
      table: Float32Array.of(NaN, NaN, NaN, NaN, 1.25, 0.125, 10, 0, 0.125, 0.25, 0.5, 0.6),
      summary,
      status: Uint32Array.of(0)
    },
    {entityCount: 6}
  );
  expect(result.kind).toBe('spatial-panel-error-maximum-likelihood');
  expect(result.coefficients[0].estimate).toBeNaN();
  expect(result.coefficients[0].standardError).toBeNaN();
  expect(result.coefficients[1].standardError).toBe(0.125);
  expect(result.absorbedEntityCount).toBe(6);
  // k = 6 entity intercepts + 1 slope + lambda; df = 48 - 8 = 40.
  expect(result.parameterCount).toBe(8);
  expect(result.degreesOfFreedom).toBe(40);
  expect(result.residualSumOfSquares).toBe(24);
  expect(result.sigmaSquared).toBe(0.6);
  expect(result.sigmaSquaredMaximumLikelihood).toBe(0.5);
  expect(result.logLikelihood).toBe(-56);
  expect(result.akaikeInformationCriterion).toBeNull();
  expect(result.schwarzCriterion).toBeNull();
  expect(result.correctedAkaikeInformationCriterion).toBeNull();
});

it('preserves the published minus-infinite GTWR AICc of an exact fit', () => {
  // n=8, trace=2, RSS=0: ll=+Infinity and AICc=-2*ll+2*8*3/(8-2-2)=-Infinity.
  const summary = Float32Array.of(0, 2, -Infinity, 1, 8, NaN, 0, 0, NaN, NaN, Infinity, 8);
  const result = getGPUGeographicallyTemporallyWeightedRegressionModelResult({summary});
  expect(result.status.ok).toBe(true);
  expect(result.logLikelihood).toBe(Infinity);
  expect(result.correctedAkaikeInformationCriterion).toBe(-Infinity);
});

it('reads only the published GTWR summary prefix from a capacity-sized input', () => {
  const published = getGTWRSummary();
  const summary = new Proxy(
    {length: 1000000},
    {
      get(target, property) {
        if (property === 'length') return target.length;
        if (typeof property === 'string' && /^\d+$/.test(property)) {
          const index = Number(property);
          if (index >= 12) throw new Error('unpublished capacity tail read');
          return published[index];
        }
        return undefined;
      }
    }
  ) as ArrayLike<number>;
  expect(() =>
    getGPUGeographicallyTemporallyWeightedRegressionModelResult({summary})
  ).not.toThrow();
});

it('uses a successful GTWR summary consistent with the published likelihood and AICc formulas', () => {
  const summary = getGTWRSummary();
  // RSS=8, n=10, trace=2: ll=-5*(log(2*pi*0.8)+1); AICc=-2*ll+60/6.
  expect(summary[10]).toBeCloseTo(-13.073667575475678, 5);
  expect(summary[2]).toBeCloseTo(36.147335150951356, 5);
});
