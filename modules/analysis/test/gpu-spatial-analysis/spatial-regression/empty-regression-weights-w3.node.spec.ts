// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {validateCPURegressionWeights} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-regression-weights';
import {computeSpatialRegressionDiagnosticsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation';
import {computeSpatialWeightsTracesOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-weights-traces-operation';
import {computeSpatialTwoStageLeastSquaresOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-two-stage-least-squares-operation';
import {computeSpatialErrorGMOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-gm-operation';
import {computeSpatialErrorMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-maximum-likelihood-operation';
import {computeSpatialLagMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-maximum-likelihood-operation';
it('DGN-10 distinguishes empty CSR from malformed offsets in all six regression consumers', () => {
  const weights = {
    offsets: Uint32Array.of(0),
    neighbors: new Uint32Array(),
    weights: new Float32Array()
  };
  expect(validateCPURegressionWeights('empty', weights)).toBe(0);
  const input = {
    weights,
    predictors: new Float32Array(),
    response: new Float32Array(),
    residuals: new Float32Array()
  };
  const fits = [
    computeSpatialRegressionDiagnosticsOnCPU(input, {predictorCount: 1}),
    computeSpatialTwoStageLeastSquaresOnCPU(input, {predictorCount: 1}),
    computeSpatialErrorGMOnCPU(input, {predictorCount: 1}),
    computeSpatialErrorMaximumLikelihoodOnCPU(input, {predictorCount: 1, maximumPower: 2}),
    computeSpatialLagMaximumLikelihoodOnCPU(input, {predictorCount: 1, maximumPower: 2})
  ];
  // Every existing fit uses status 2 for too few observations, rather than malformed topology.
  for (const fit of fits) {
    expect([...fit.status]).toEqual([2]);
    expect(fit.summary[0]).toBe(0);
  }
  const traces = computeSpatialWeightsTracesOnCPU({weights}, {maximumPower: 2, mode: 'exact'});
  // The trace and sum over a zero-dimensional matrix are empty sums, hence exactly zero.
  expect([...traces.powerTraces]).toEqual([0, 0]);
  expect([...traces.powerSums]).toEqual([0, 0]);
  for (const offsets of [new Uint32Array(), Uint32Array.of(1)])
    expect(() => validateCPURegressionWeights('invalid', {...weights, offsets})).toThrow(
      /weights.offsets/
    );
  expect(() =>
    validateCPURegressionWeights('decreasing', {
      offsets: Uint32Array.of(0, 1, 0),
      neighbors: Uint32Array.of(0),
      weights: Float32Array.of(1)
    })
  ).toThrow(/nondecreasing/);
});
