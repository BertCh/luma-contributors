// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslExec, WgslParser} from 'wgsl_reflect';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {getGWRLargeSampleInferenceWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-large-sample-inference';
import {createBoundaryReference, LARGE_SAMPLE_QUANTILE_CASES} from './w2-gwr-review-reference';
import {computeGeographicallyWeightedRegressionOnCPU} from './geographically-weighted-regression-oracle';

it('evaluates large-sample critical values without the beta argument rounding plateau', () => {
  const input = Float32Array.from(LARGE_SAMPLE_QUANTILE_CASES.flatMap(([df, tail]) => [df, tail]));
  const output = new Float32Array(LARGE_SAMPLE_QUANTILE_CASES.length * 2);
  const source = `${GET_NAN_WGSL}
${getGWRLargeSampleInferenceWGSL()}
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) invocation: vec3u) {
  let offset = invocation.x * 2u;
  outputs[offset] = gwrLargeSampleCriticalValue(inputs[offset + 1u], inputs[offset]);
  outputs[offset + 1u] = gwrStudentCriticalValue(inputs[offset + 1u], inputs[offset]);
}`;
  new WgslExec(new WgslParser().parse(source)).dispatchWorkgroups(
    'main',
    [LARGE_SAMPLE_QUANTILE_CASES.length, 1, 1],
    {0: {0: input, 1: output}}
  );
  LARGE_SAMPLE_QUANTILE_CASES.forEach(([, , expected], index) => {
    expect(Math.abs(output[index * 2] - expected), `case ${index}`).toBeLessThan(2e-5);
  });
  // Demonstrates the float32 defect in the historical evaluator, rather than comparing routes.
  expect(Math.abs(output[9] - LARGE_SAMPLE_QUANTILE_CASES[4][2])).toBeGreaterThan(2e-5);
});

it('aggregates raw tail statistics even when a motif has no standalone AICc', () => {
  const reference = createBoundaryReference(18);
  const tail = computeGeographicallyWeightedRegressionOnCPU({
    positions: reference.scene.positions.slice(30),
    predictors: reference.scene.predictors.slice(15),
    response: reference.scene.response.slice(15),
    predictorCount: 1,
    settings: {kernel: 'bisquare', bandwidths: [0.25]}
  });
  expect(tail.hasValidCandidate).toBe(false);
  expect(tail.singularRowCount).toBeNaN();
  expect(tail.localStatus).toEqual([0, 0, 0]);
  expect(tail.candidates[0].singularRowCount).toBe(0);
  const score = reference.score(0.25);
  expect(score.singular).toBe(0);
  expect(score.count).toBe(18);
  expect(Number.isFinite(score.aicc)).toBe(true);
  // Closed-form OLS for x=[-2,-1,0], y=[-3.5,-1.75,1.75].
  const local = reference.local(17, 0.25);
  expect(local.coefficients[0]).toBeCloseTo(35 / 24, 12);
  expect(local.coefficients[1]).toBeCloseTo(21 / 8, 12);
  expect(local.hat).toBeCloseTo(5 / 6, 12);
  expect(local.standardErrors[0] ** 2).toBeCloseTo((score.sigmaSquared * 5) / 6, 12);
  expect(local.standardErrors[1] ** 2).toBeCloseTo(score.sigmaSquared / 2, 12);
});

it('keeps every candidate and tail inference finite at all reviewed focus boundaries', () => {
  for (const count of [262143, 262144, 262145, 1048584]) {
    const reference = createBoundaryReference(count);
    for (const bandwidth of [0.25, 1.2]) {
      const score = reference.score(bandwidth);
      expect(score.count).toBe(count);
      expect(score.singular).toBe(0);
      for (const value of Object.values(score)) expect(Number.isFinite(value)).toBe(true);
      const local = reference.local(count - 1, bandwidth);
      expect(
        local.coefficients.concat(local.standardErrors, local.tValues).every(Number.isFinite)
      ).toBe(true);
    }
  }
});
