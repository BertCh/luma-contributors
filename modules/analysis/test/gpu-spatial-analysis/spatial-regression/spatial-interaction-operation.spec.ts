// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpatialInteractionOperation} from '../../../src/gpu-spatial-analysis/spatial-interaction/cpu-spatial-interaction-operation';
import {GPUSpatialInteractionOperation} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

for (const constraint of ['unconstrained', 'origin', 'destination', 'doubly'] as const) {
  for (const scene of ['ordinary', 'no-mass'] as const) {
    it(`gravity operations agree on ${constraint}/${scene}, relative tolerance 5e-4`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const names = [
        'predictions',
        'summary',
        'status',
        'candidates',
        'originResiduals',
        'destinationResiduals'
      ];
      const results = [];
      for (const Operation of [GPUSpatialInteractionOperation, CPUSpatialInteractionOperation])
        results.push(
          await runOperationGraph(device, {
            inputs: {
              origins: {format: 'uint32', values: Uint32Array.of(0, 0, 1, 1, 0, 99)},
              destinations: {format: 'uint32', values: Uint32Array.of(0, 1, 0, 1, 2, 99)},
              flows: {
                format: 'float32',
                values:
                  scene === 'no-mass' ? new Float32Array(6) : Float32Array.of(10, 20, 30, 40, 0, 0)
              },
              covariate: {format: 'float32', values: Float32Array.of(0, 1, 2, 0, 0, 0)},
              pairState: {format: 'uint32', values: Uint32Array.of(1, 1, 1, 1, 0, 2)}
            },
            operations: [
              new Operation({
                pairCount: 6,
                originCount: 2,
                destinationCount: 3,
                origins: 'origins',
                destinations: 'destinations',
                flows: 'flows',
                covariate: 'covariate',
                pairState: 'pairState',
                constraint,
                coefficients: [-2, -Math.log(2), 0, 2],
                tolerance: 1e-4,
                output: {
                  predictions: 'predictions',
                  summary: 'summary',
                  candidates: 'candidates',
                  status: 'status',
                  originResiduals: 'originResiduals',
                  destinationResiduals: 'destinationResiduals'
                }
              })
            ],
            outputs: names
          })
        );
      for (const result of results) expect(result['status'][0]).toBe(scene === 'no-mass' ? 1 : 0);
      for (const name of names) {
        for (let index = 0; index < results[1][name].length; index++) {
          const expected = results[1][name][index];
          const actual = results[0][name][index];
          const diagnosticError =
            (name === 'summary' && index === 2) || (name === 'candidates' && index % 4 === 1);
          const marginal = name === 'originResiduals' || name === 'destinationResiduals';
          if (Number.isFinite(expected) && (diagnosticError || marginal))
            expect(Math.abs(actual - expected)).toBeLessThanOrEqual(marginal ? 5e-5 : 5e-7);
          else
            expectRegressionRowsClose(
              [actual],
              [expected],
              name === 'status' ||
                (name === 'candidates' && index % 4 >= 2) ||
                (name === 'summary' && index === 3)
                ? 0
                : 5e-4
            );
        }
      }
      if (scene === 'ordinary') {
        expect(results[0]['predictions'][4] === 0).toBe(true);
        expect(results[0]['predictions'][5]).toBeNaN();
      }
    });
  }
}

it('bounded gravity profiles agree within the coefficient bracket tolerance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const names = ['predictions', 'summary', 'status', 'candidates', 'calibration'];
  const results = [];
  for (const Operation of [GPUSpatialInteractionOperation, CPUSpatialInteractionOperation])
    results.push(
      await runOperationGraph(device, {
        inputs: {
          origins: {format: 'uint32', values: Uint32Array.of(0, 0, 1, 1)},
          destinations: {format: 'uint32', values: Uint32Array.of(0, 1, 0, 1)},
          flows: {format: 'float32', values: Float32Array.of(10, 20, 30, 40)},
          covariate: {format: 'float32', values: Float32Array.of(0, 1, 2, 0)}
        },
        operations: [
          new Operation({
            pairCount: 4,
            originCount: 2,
            destinationCount: 2,
            origins: 'origins',
            destinations: 'destinations',
            flows: 'flows',
            covariate: 'covariate',
            constraint: 'doubly',
            coefficients: [-1, 1],
            calibration: 'bounded',
            coefficientTolerance: 1e-3,
            maximumCalibrationIterations: 64,
            tolerance: 1e-5,
            output: {
              predictions: 'predictions',
              summary: 'summary',
              candidates: 'candidates',
              status: 'status',
              calibration: 'calibration'
            }
          })
        ],
        outputs: names
      })
    );
  for (const result of results) {
    expect(result['status'][0]).toBe(0);
    expect(Math.abs(result['summary'][0] - Math.log(1.5) / 3)).toBeLessThan(2e-3);
    const [lower, upper, width, converged] = result['calibration'];
    expect(width).toBeGreaterThan(0);
    expect(width).toBeLessThanOrEqual(1e-3);
    expect(upper).toBeGreaterThan(lower);
    const endpointRounding = 2 * 2 ** -23 * Math.max(Math.abs(lower), Math.abs(upper));
    expect(Math.abs(width - (upper - lower))).toBeLessThanOrEqual(endpointRounding);
    expect(converged).toBe(1);
  }
  // The GPU deviance near a saturated optimum cancels in float32 and can change the final
  // bracket by several coefficient steps. These diagnostics use explicit absolute tolerances.
  for (const name of names)
    for (let index = 0; index < results[1][name].length; index++) {
      const actual = results[0][name][index];
      const expected = results[1][name][index];
      if (name === 'predictions') expectRegressionRowsClose([actual], [expected], 2e-3);
      else if (name === 'calibration' && index === 2) {
        // Compare widths at the scale of endpoint float32 rounding, not coefficientTolerance.
        const endpointScale = Math.max(
          Math.abs(results[0]['calibration'][0]),
          Math.abs(results[0]['calibration'][1]),
          Math.abs(results[1]['calibration'][0]),
          Math.abs(results[1]['calibration'][1])
        );
        expect(Math.abs(actual - expected)).toBeLessThanOrEqual(8 * 2 ** -23 * endpointScale);
      } else if (
        name === 'status' ||
        (name === 'calibration' && index === 3) ||
        (name === 'candidates' && index % 4 >= 2) ||
        (name === 'summary' && index === 3)
      )
        expect(actual === expected).toBe(true);
      else
        expect(Math.abs(actual - expected)).toBeLessThan(
          name === 'calibration' || (name === 'summary' && index === 0) ? 2e-3 : 1e-4
        );
    }
});

it('checks the GPU finite-deviance gate when only the float32 arithmetic overflows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const names = [
    'predictions',
    'summary',
    'status',
    'candidates',
    'originResiduals',
    'destinationResiduals'
  ];
  const results = [];
  const flows = Float32Array.of(1e38, 0);
  for (const Operation of [GPUSpatialInteractionOperation, CPUSpatialInteractionOperation])
    results.push(
      await runOperationGraph(device, {
        inputs: {
          origins: {format: 'uint32', values: Uint32Array.of(0, 0)},
          destinations: {format: 'uint32', values: Uint32Array.of(0, 1)},
          flows: {format: 'float32', values: flows},
          covariate: {format: 'float32', values: Float32Array.of(0, 10)}
        },
        operations: [
          new Operation({
            pairCount: 2,
            originCount: 1,
            destinationCount: 2,
            origins: 'origins',
            destinations: 'destinations',
            flows: 'flows',
            covariate: 'covariate',
            constraint: 'unconstrained',
            coefficients: [1],
            output: {
              predictions: 'predictions',
              summary: 'summary',
              status: 'status',
              candidates: 'candidates',
              originResiduals: 'originResiduals',
              destinationResiduals: 'destinationResiduals'
            }
          })
        ],
        outputs: names
      })
    );
  // Hypothesis to verify on the target GPU: overflowing shader deviance rejects the candidate.
  // The CPU's float64 deviance is finite (2*F*log(1+exp(10))), so final rounding alone gives Inf.
  expect(results[0]['status'][0]).toBe(3);
  expect(results[0]['candidates'][3]).toBe(1);
  expect(results[0]['candidates'][1] === 0).toBe(true);
  expect(results[0]['candidates'][2] === 0).toBe(true);
  for (const port of ['predictions', 'summary', 'originResiduals', 'destinationResiduals'])
    expect(Array.from(results[0][port]).every(Number.isNaN)).toBe(true);
  expect(results[1]['status'][0]).toBe(0);
  const denominator = 1 + Math.exp(10);
  expectRegressionRowsClose(
    results[1]['predictions'],
    Float32Array.of(flows[0] / denominator, (flows[0] * Math.exp(10)) / denominator),
    5e-4
  );
  expectRegressionRowsClose(results[1]['summary'], [1, Infinity, 0, 0], 0);
  expectRegressionRowsClose(results[1]['candidates'], [Infinity, 0, 0, 0], 0);
  expect(Math.abs(results[1]['originResiduals'][0]) / flows[0]).toBeLessThan(1e-12);
  expectRegressionRowsClose(
    results[1]['destinationResiduals'],
    [flows[0] - flows[0] / denominator, (-flows[0] * Math.exp(10)) / denominator],
    5e-4
  );
});
