// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {GPUSpatialPanelErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood-operation';
import {CPUSpatialPanelErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';
import {
  getSmallSlopeInput,
  getReviewProps,
  getReviewInputs,
  reviewOutputNames
} from './panel-error-review.test-utils';
import {getPanelErrorInput, panelErrorFixture} from './panel-error-effects.test-utils';

for (const Operation of [
  GPUSpatialPanelErrorMaximumLikelihoodOperation,
  CPUSpatialPanelErrorMaximumLikelihoodOperation
]) {
  it(`${Operation.name} R3: scalar identity has one boundary tie and no lambda inference`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = {
      predictors: Float32Array.of(0, 1, 3, 2, 4, 1, 6, 3),
      response: Float32Array.of(3, 0, 7, 1, 4, 2, 0, 6),
      weights: {
        offsets: Uint32Array.of(0, 1, 2),
        neighbors: Uint32Array.of(0, 1),
        weights: Float32Array.of(1, 1)
      }
    };
    const actual = await runOperationGraph(device, {
      operations: [
        new Operation({
          ...getReviewProps(2, 4),
          logDeterminant: 'exact',
          eigenvalues: 'eigenvalues'
        })
      ],
      inputs: {
        ...getReviewInputs(input),
        eigenvalues: {format: 'float32', values: new Float32Array(8).fill(1)}
      },
      outputs: reviewOutputNames
    });
    expect(actual['summary'][1]).toBeCloseTo(-0.99, 6);
    expect(actual['summary'][20]).toBe(1);
    expect(actual['summary'][16]).toBe(1);
    expect(actual['summary'][19]).toBe(0);
    expect(actual['table'][9]).toBeNaN();
    expect(Array.from(actual['converged'])).toEqual([0]);
  });
  for (const missing of [true, false])
    it(`${Operation.name} R4: ${missing ? 'missing' : 'singular'} retains seeded trace uncertainty`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = getPanelErrorInput();
      if (missing) input.response[0] = NaN;
      else input.predictors.fill(1);
      const actual = await runOperationGraph(device, {
        operations: [
          new Operation({
            ...getReviewProps(12, 6, 2),
            traceMode: 'stochastic',
            traceProbeCount: 4,
            traceParameters: 'traceParameters'
          })
        ],
        inputs: {
          ...getReviewInputs(input),
          traceParameters: {format: 'uint32', values: Uint32Array.of(11)}
        },
        outputs: reviewOutputNames
      });
      expect(Array.from(actual['code'])).toEqual([missing ? 3 : 1]);
      for (const [index, value] of panelErrorFixture.stochasticTraceErrors.entries())
        expect(actual['traceErrors'][index]).toBeCloseTo(value, 4);
    });
  it(`${Operation.name} R5: orthogonal data have zero slope and undefined pseudo R squared`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = {
      predictors: Float32Array.of(1, -1, 2, -2, -1, 1, -2, 2),
      response: Float32Array.of(1, 1, -1, -1, -1, -1, 1, 1),
      weights: {
        offsets: Uint32Array.of(0, 3, 6, 9, 12),
        neighbors: Uint32Array.of(1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2),
        weights: new Float32Array(12).fill(1)
      }
    };
    const actual = await runOperationGraph(device, {
      operations: [
        new Operation({
          ...getReviewProps(4, 2),
          logDeterminant: 'exact',
          eigenvalues: 'eigenvalues'
        })
      ],
      inputs: {
        ...getReviewInputs(input),
        eigenvalues: {format: 'float32', values: Float32Array.of(-1, -1, -1, 3, -1, -1, -1, 3)}
      },
      outputs: reviewOutputNames
    });
    expect(Array.from(actual['code'])).toEqual([0]);
    expect(actual['table'][4]).toBe(0);
    expect(actual['summary'][6]).toBeNaN();
  });
  it(`${Operation.name} R6: additive entity and period predictor is singular`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = getSmallSlopeInput(1);
    input.predictors = Float32Array.from(
      [0, 1, 3].flatMap(period => [0, 1, 3].map(entity => entity + period))
    );
    input.response = Float32Array.of(0, 1, 4, 9, 5, 3, 3, 5, 9);
    const actual = await runOperationGraph(device, {
      operations: [new Operation({...getReviewProps(3, 3), effects: 'twoWay'})],
      inputs: getReviewInputs(input),
      outputs: reviewOutputNames
    });
    expect(Array.from(actual['code'])).toEqual([1]);
    expect(Array.from(actual['converged'])).toEqual([0]);
  });
  it(`${Operation.name} R2: absorbed unit shifts reject ML without rejecting pooled tests`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = getPanelErrorInput();
    for (let row = 0; row < 72; row++)
      input.response[row] = 3 + 2 * input.predictors[row * 2] + Number(row % 12 === 1);
    const actual = await runOperationGraph(device, {
      operations: [new Operation({...getReviewProps(12, 6, 2), maximumPower: 128})],
      inputs: getReviewInputs(input),
      outputs: reviewOutputNames
    });
    expect(Array.from(actual['code'])).toEqual([3]);
    expect(Array.from(actual['converged'])).toEqual([0]);
    expect(Array.from(actual['table']).every(Number.isNaN)).toBe(true);
    expect(Array.from(actual['residuals']).every(Number.isNaN)).toBe(true);
    expect(Array.from(actual['diagnosticStatus'])).toEqual([0]);
    for (const [index, value] of [2.35135119, 0.15276008, 2.22570536].entries())
      expect(actual['tests'][index * 2]).toBeCloseTo(value, 3);
  });
  it(`${Operation.name} R1: small-slope robust pooled tests retain their information`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const actual = await runOperationGraph(device, {
      operations: [new Operation(getReviewProps(3, 4))],
      inputs: getReviewInputs(getSmallSlopeInput(2 ** -12)),
      outputs: reviewOutputNames
    });
    expect(Array.from(actual['diagnosticStatus'])).toEqual([0]);
    // Closed form in the Node case: trace=12, q=5 delta^2, score contrast=5 delta.
    // Float32 projections may round; 2e-3 absolute still distinguishes the old status 2.
    for (const [index, expected] of [25 / 12, 5, 5.001017180142704].entries()) {
      expect(Number.isFinite(actual['tests'][index * 2])).toBe(true);
      expect(Math.abs(actual['tests'][index * 2] - expected)).toBeLessThan(2e-3);
    }
  });
}
