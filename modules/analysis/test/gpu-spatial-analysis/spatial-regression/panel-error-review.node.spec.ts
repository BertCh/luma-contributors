// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeSpatialPanelErrorMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';

import {getSmallSlopeInput} from './panel-error-review.test-utils';
import {getPanelErrorInput, panelErrorFixture} from './panel-error-effects.test-utils';

it('R2: an absorbed unit shift leaves exact ML residuals but valid pooled tests', () => {
  const input = getPanelErrorInput();
  for (let row = 0; row < 72; row++)
    input.response[row] = 3 + 2 * input.predictors[row * 2] + Number(row % 12 === 1);
  const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(input, panelErrorFixture);
  expect(result.status[0]).toBe(3);
  expect(result.diagnosticStatus[0]).toBe(0);
  for (const [index, value] of [2.35135119, 0.15276008, 2.22570536].entries())
    expect(result.diagnosticTests[index * 2]).toBeCloseTo(value, 6);
});

it('R1: pooled robust scores retain small projected information without subtracting the trace', () => {
  for (const delta of [2 ** -25, 2 ** -27]) {
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(getSmallSlopeInput(delta), {
      periodCount: 4,
      predictorCount: 1,
      maximumPower: 8
    });
    // Dense equations: variance=4/5, trace=12, q=5 delta^2,
    // error score=-5, lag contrast=5 delta. Robust lag=5 exactly.
    const robustError = (5 * (12 + 5 * delta) ** 2) / (12 * (12 + 5 * delta ** 2));
    expect(result.diagnosticStatus[0]).toBe(0);
    expect(result.diagnosticTests[0]).toBeCloseTo(25 / 12, 6);
    expect(result.diagnosticTests[2]).toBeCloseTo(5, 6);
    expect(result.diagnosticTests[4]).toBeCloseTo(robustError, 6);
  }
});

it('R3: identity weights publish the single negative-boundary flat-profile tie', () => {
  const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
    {
      predictors: Float32Array.of(0, 1, 3, 2, 4, 1, 6, 3),
      response: Float32Array.of(3, 0, 7, 1, 4, 2, 0, 6),
      weights: {
        offsets: Uint32Array.of(0, 1, 2),
        neighbors: Uint32Array.of(0, 1),
        weights: Float32Array.of(1, 1)
      },
      eigenvalues: new Float32Array(8).fill(1)
    },
    {periodCount: 4, predictorCount: 1, logDeterminant: 'exact'}
  );
  expect(result.summary[1]).toBeCloseTo(-0.99, 6);
  expect(result.summary[20]).toBe(1);
  expect(result.summary[16]).toBe(1);
  expect(result.converged[0]).toBe(0);
  expect(result.summary[19]).toBe(0);
  expect(result.table[9]).toBeNaN();
});

it('R4: missing and singular fits retain every independent stochastic trace error', () => {
  for (const missing of [true, false]) {
    const input = getPanelErrorInput();
    if (missing) input.response[0] = NaN;
    else input.predictors.fill(1);
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
      {...input, traceParameters: Uint32Array.of(11)},
      {...panelErrorFixture, maximumPower: 8, traceMode: 'stochastic', traceProbeCount: 4}
    );
    expect(result.status[0]).toBe(missing ? 3 : 1);
    for (const [index, value] of panelErrorFixture.stochasticTraceErrors.entries())
      expect(result.powerTraceStandardErrors[index]).toBeCloseTo(value, 6);
  }
});

it('R5: exactly orthogonal centered predictions have zero slope and undefined correlation', () => {
  const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
    {
      predictors: Float32Array.of(1, -1, 2, -2, -1, 1, -2, 2),
      response: Float32Array.of(1, 1, -1, -1, -1, -1, 1, 1),
      weights: {
        offsets: Uint32Array.of(0, 3, 6, 9, 12),
        neighbors: Uint32Array.of(1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2),
        weights: new Float32Array(12).fill(1)
      },
      eigenvalues: Float32Array.of(-1, -1, -1, 3, -1, -1, -1, 3)
    },
    {periodCount: 2, predictorCount: 1, logDeterminant: 'exact'}
  );
  expect(result.status[0]).toBe(0);
  expect(result.table[4]).toBe(0);
  expect(result.summary[6]).toBeNaN();
});

it('R6: a predictor in the sum of both effect spaces is singular before exact-response testing', () => {
  const input = getSmallSlopeInput(1);
  input.predictors = Float32Array.from(
    [0, 1, 3].flatMap(period => [0, 1, 3].map(entity => entity + period))
  );
  input.response = Float32Array.of(0, 1, 4, 9, 5, 3, 3, 5, 9);
  const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(input, {
    periodCount: 3,
    predictorCount: 1,
    effects: 'twoWay'
  });
  expect(result.status[0]).toBe(1);
});
