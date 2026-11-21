// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {CPUSpatialPanelErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';
import {
  GPUSpatialPanelErrorMaximumLikelihoodOperation,
  type SpatialPanelErrorMaximumLikelihoodOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood-operation';
import {
  panelErrorFixture,
  getPanelErrorInput,
  getSmallVariancePanelInput,
  getTailPerturbedPanelInput
} from './panel-error-effects.test-utils';

function compare(actual: ArrayLike<number>, expected: readonly number[]) {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) expect(actual[index], `slot ${index}`).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), `slot ${index}`).toBe(true);
      // Float32 tiled moments, solves, search and condition estimates; no stochastic traces.
      expect(
        Math.abs(actual[index] - expected[index]),
        `slot ${index}: ${actual[index]} versus ${expected[index]}`
      ).toBeLessThanOrEqual(3e-3 * Math.max(1, Math.abs(expected[index])));
    }
  }
}

function getProps(
  effects: 'unit' | 'time' | 'twoWay'
): SpatialPanelErrorMaximumLikelihoodOperationProps {
  return {
    entityCount: 12,
    periodCount: 6,
    predictorCount: 2,
    effects,
    predictors: 'predictors',
    response: 'response',
    traceMode: 'exact',
    weights: {
      offsets: 'offsets',
      neighbors: 'neighbors',
      weights: 'weights',
      status: {candidateOverflow: 'candidateOverflow'}
    },
    output: {
      table: 'table',
      summary: 'summary',
      residuals: 'residuals',
      powerTraceStandardErrors: 'traceErrors',
      status: {code: 'code', converged: 'converged'}
    },
    diagnostics: {tests: 'tests', status: 'diagnosticStatus'}
  };
}

const outputNames = [
  'table',
  'summary',
  'residuals',
  'traceErrors',
  'code',
  'converged',
  'tests',
  'diagnosticStatus'
];

it('panel contributor forwards nonzero stochastic trace errors from the independent dense oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const input = getPanelErrorInput();
  const actual = await runOperationGraph(device, {
    operations: [
      new GPUSpatialPanelErrorMaximumLikelihoodOperation({
        ...getProps('unit'),
        maximumPower: 8,
        traceMode: 'stochastic',
        traceProbeCount: 4,
        traceParameters: 'traceParameters'
      })
    ],
    outputs: outputNames,
    inputs: {
      predictors: {format: 'float32', values: input.predictors},
      response: {format: 'float32', values: input.response},
      offsets: {format: 'uint32', values: input.weights.offsets},
      neighbors: {format: 'uint32', values: input.weights.neighbors},
      weights: {format: 'float32', values: input.weights.weights},
      candidateOverflow: {format: 'uint32', values: Uint32Array.of(0)},
      traceParameters: {format: 'uint32', values: Uint32Array.of(11)}
    }
  });
  compare(actual['traceErrors'], panelErrorFixture.stochasticTraceErrors);
  expect(actual['summary'][14]).toBe(1);
  expect(Array.from(actual['code'])).toEqual([0]);
});

for (const Operation of [
  GPUSpatialPanelErrorMaximumLikelihoodOperation,
  CPUSpatialPanelErrorMaximumLikelihoodOperation
]) {
  it(`${Operation.name}: a final-tile residual prevents an affine-fit rejection`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = getTailPerturbedPanelInput();
    const actual = await runOperationGraph(device, {
      operations: [
        new Operation({
          ...getProps('unit'),
          entityCount: 2,
          periodCount: 34,
          predictorCount: 1,
          diagnostics: undefined
        })
      ],
      outputs: ['table', 'summary', 'residuals', 'traceErrors', 'code', 'converged'],
      inputs: {
        predictors: {format: 'float32', values: input.predictors},
        response: {format: 'float32', values: input.response},
        offsets: {format: 'uint32', values: input.weights.offsets},
        neighbors: {format: 'uint32', values: input.weights.neighbors},
        weights: {format: 'float32', values: input.weights.weights},
        candidateOverflow: {format: 'uint32', values: Uint32Array.of(0)}
      }
    });
    // Independent closed form in the matching node case; zero variance and a zero tail fail.
    expect(Array.from(actual['code'])).toEqual([0]);
    expect(Array.from(actual['converged'])).toEqual([0]);
    expect(Number.isFinite(actual['summary'][2])).toBe(true);
    expect(Math.abs(actual['summary'][2] / 2.7236057275859883e-10 - 1)).toBeLessThan(0.02);
    compare([actual['table'][4], actual['table'][8]], [2, 0.930321861972383]);
    const expected = [
      ...new Array<number>(64).fill(0),
      0.0009765625,
      0.0009765625,
      -0.0009765625,
      -0.0009765625
    ];
    for (let row = 0; row < 68; row++) {
      expect(Number.isFinite(actual['residuals'][row])).toBe(true);
      expect(Math.abs(actual['residuals'][row] - expected[row])).toBeLessThan(2e-6);
    }
    expect(Array.from(actual['traceErrors'])).toEqual(new Array(128).fill(0));
  });
  it(`${Operation.name}: exact affine rejection retains independent stochastic trace errors`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = getPanelErrorInput();
    for (let row = 0; row < input.response.length; row++)
      input.response[row] = 3 + 2 * input.predictors[row * 2];
    const actual = await runOperationGraph(device, {
      operations: [
        new Operation({
          ...getProps('unit'),
          maximumPower: 8,
          traceMode: 'stochastic',
          traceProbeCount: 4,
          traceParameters: 'traceParameters'
        })
      ],
      outputs: outputNames,
      inputs: {
        predictors: {format: 'float32', values: input.predictors},
        response: {format: 'float32', values: input.response},
        offsets: {format: 'uint32', values: input.weights.offsets},
        neighbors: {format: 'uint32', values: input.weights.neighbors},
        weights: {format: 'float32', values: input.weights.weights},
        candidateOverflow: {format: 'uint32', values: Uint32Array.of(0)},
        traceParameters: {format: 'uint32', values: Uint32Array.of(11)}
      }
    });
    expect(Array.from(actual['code'])).toEqual([3]);
    expect(Array.from(actual['converged'])).toEqual([0]);
    expect(Array.from(actual['diagnosticStatus'])).toEqual([2]);
    expect(Array.from(actual['table']).every(Number.isNaN)).toBe(true);
    expect(Array.from(actual['tests']).every(Number.isNaN)).toBe(true);
    compare(actual['traceErrors'], panelErrorFixture.stochasticTraceErrors);
  });
}

for (const effects of ['unit', 'time', 'twoWay'] as const) {
  for (const Operation of [
    GPUSpatialPanelErrorMaximumLikelihoodOperation,
    CPUSpatialPanelErrorMaximumLikelihoodOperation
  ]) {
    it(`${Operation.name} ${effects}: small positive variance is retained without an affine-fit cutoff`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = getSmallVariancePanelInput();
      const actual = await runOperationGraph(device, {
        operations: [
          new Operation({...getProps(effects), entityCount: 2, periodCount: 4, predictorCount: 1})
        ],
        outputs: outputNames,
        inputs: {
          predictors: {format: 'float32', values: input.predictors},
          response: {format: 'float32', values: input.response},
          offsets: {format: 'uint32', values: input.weights.offsets},
          neighbors: {format: 'uint32', values: input.weights.neighbors},
          weights: {format: 'float32', values: input.weights.weights},
          candidateOverflow: {format: 'uint32', values: Uint32Array.of(0)}
        }
      });
      // The closed-form derivation is recorded in the matching node spec. The residual is
      // 2^-10 against a response of size 7; allow float32 solve/near-bound cancellation but
      // require a strictly positive variance close to its independent value (zero fails).
      const expectedVariance = effects === 'unit' ? 9.5367431640625e-7 : 2.31506486844809e-9;
      expect(Array.from(actual['code'])).toEqual([0]);
      expect(Number.isFinite(actual['summary'][2])).toBe(true);
      expect(Math.abs(actual['summary'][2] / expectedVariance - 1)).toBeLessThan(0.02);
      compare([actual['table'][4]], [2]);
      expect(Array.from(actual['diagnosticStatus'])).toEqual([0]);
      compare(actual['tests'], [0, 1, 0, 1, 0, 1]);
      expect(Array.from(actual['traceErrors'])).toEqual(new Array(128).fill(0));
    });
    it(`${Operation.name} ${effects}: every output matches independent dense Gaussian ML`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = getPanelErrorInput();
      const actual = await runOperationGraph(device, {
        operations: [new Operation(getProps(effects))],
        outputs: outputNames,
        inputs: {
          predictors: {format: 'float32', values: input.predictors},
          response: {format: 'float32', values: input.response},
          offsets: {format: 'uint32', values: input.weights.offsets},
          neighbors: {format: 'uint32', values: input.weights.neighbors},
          weights: {format: 'float32', values: input.weights.weights},
          candidateOverflow: {format: 'uint32', values: Uint32Array.of(0)}
        }
      });
      const expected = panelErrorFixture.models[effects];
      compare(actual['table'], expected.table);
      compare(actual['summary'], expected.summary);
      compare(actual['residuals'], expected.residuals);
      compare(actual['tests'], panelErrorFixture.diagnostics);
      expect(Array.from(actual['traceErrors'])).toEqual(new Array(128).fill(0));
      expect(Array.from(actual['code'])).toEqual([0]);
      expect(Array.from(actual['converged'])).toEqual([1]);
      expect(Array.from(actual['diagnosticStatus'])).toEqual([0]);
    });

    for (const failure of [
      'candidateOverflow',
      'missingPeriod',
      'singular',
      'boundary',
      'perfectFit'
    ] as const) {
      it(`${Operation.name} ${effects}: ${failure} publishes the expected status and dependent results`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const input = getPanelErrorInput();
        if (failure === 'missingPeriod') input.response.fill(NaN, 12, 24);
        if (failure === 'perfectFit')
          for (let row = 0; row < 72; row++)
            input.response[row] = 3 + 2 * input.predictors[row * 2];
        if (failure === 'singular')
          for (let row = 0; row < 72; row++)
            input.predictors[row * 2] = effects === 'time' ? Math.floor(row / 12) : row % 12;
        const actual = await runOperationGraph(device, {
          operations: [
            new Operation({...getProps(effects), maximumPower: failure === 'boundary' ? 2 : 128})
          ],
          outputs: outputNames,
          inputs: {
            predictors: {format: 'float32', values: input.predictors},
            response: {format: 'float32', values: input.response},
            offsets: {format: 'uint32', values: input.weights.offsets},
            neighbors: {format: 'uint32', values: input.weights.neighbors},
            weights: {format: 'float32', values: input.weights.weights},
            candidateOverflow: {
              format: 'uint32',
              values: Uint32Array.of(Number(failure === 'candidateOverflow'))
            }
          }
        });
        if (failure === 'boundary') {
          // The Q=2 geometric bound solves s^3 / (3(1-s)) = 1e-5. The independent
          // dense objective is monotone toward the stated end over this tiny interval.
          expect(Array.from(actual['code'])).toEqual([0]);
          expect(Array.from(actual['converged'])).toEqual([0]);
          compare([actual['table'][12]], [(effects === 'unit' ? 1 : -1) * 0.030750506892117536]);
          expect(actual['summary'][16]).toBe(1);
          return;
        }
        expect(Array.from(actual['code'])).toEqual([failure === 'singular' ? 1 : 3]);
        expect(Array.from(actual['converged'])).toEqual([0]);
        expect(Array.from(actual['table']).every(Number.isNaN)).toBe(true);
        expect(Array.from(actual['residuals']).every(Number.isNaN)).toBe(true);
        if (failure !== 'singular' && failure !== 'perfectFit') {
          expect(Array.from(actual['diagnosticStatus'])).toEqual([1]);
          expect(Array.from(actual['tests']).every(Number.isNaN)).toBe(true);
        }
        if (failure === 'perfectFit') {
          // Exact affine pooled response: RSS=0 makes the LM variance denominator zero.
          expect(Array.from(actual['diagnosticStatus'])).toEqual([2]);
          expect(Array.from(actual['tests']).every(Number.isNaN)).toBe(true);
        }
        if (failure === 'candidateOverflow') {
          expect(Array.from(actual['summary']).every(Number.isNaN)).toBe(true);
          expect(Array.from(actual['traceErrors']).every(Number.isNaN)).toBe(true);
        } else {
          expect(actual['summary'][0]).toBe(72);
          expect(Array.from(actual['summary']).slice(1).every(Number.isNaN)).toBe(true);
          expect(Array.from(actual['traceErrors'])).toEqual(new Array(128).fill(0));
        }
      });
    }
  }
}
