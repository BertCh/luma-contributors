// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {GPUSpatialPanelErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood';
import {getPanelWithinScratchLengths} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-within-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {computeSpatialPanelErrorMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';
import {
  panelErrorFixture,
  getPanelErrorInput,
  getSmallVariancePanelInput,
  getTailPerturbedPanelInput
} from './panel-error-effects.test-utils';
import {getPanelAffineFitAllocationLengths} from '../../../src/gpu-spatial-analysis/spatial-regression/panel-affine-fit-nodes';

function compare(actual: ArrayLike<number>, expected: readonly number[], tolerance = 2e-6) {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) expect(actual[index], `slot ${index}`).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), `slot ${index}`).toBe(true);
      expect(
        Math.abs(actual[index] - expected[index]),
        `slot ${index}: ${actual[index]} versus ${expected[index]}`
      ).toBeLessThanOrEqual(tolerance * Math.max(1, Math.abs(expected[index])));
    }
  }
}

for (const effects of ['unit', 'time', 'twoWay'] as const) {
  it(`${effects} error ML and every report match dense nuisance-intercept Gaussian ML`, () => {
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(getPanelErrorInput(), {
      ...panelErrorFixture,
      effects
    });
    const expected = panelErrorFixture.models[effects];
    compare(result.table, expected.table);
    compare(result.summary, expected.summary);
    compare(result.residuals, expected.residuals);
    compare(result.diagnosticTests, panelErrorFixture.diagnostics);
    expect(Array.from(result.status)).toEqual([0]);
    expect(Array.from(result.converged)).toEqual([1]);
    expect(Array.from(result.diagnosticStatus)).toEqual([0]);
    expect(Array.from(result.powerTraceStandardErrors)).toEqual(new Array(128).fill(0));
  });

  it(`${effects} error ML absorbs shifts on its selected axes and rescales variance`, () => {
    const input = getPanelErrorInput();
    for (let row = 0; row < input.response.length; row++) {
      const shift =
        (effects !== 'time' ? (row % 12) * 32 : 0) +
        (effects !== 'unit' ? Math.floor(row / 12) * 16 : 0);
      input.response[row] = 2 * input.response[row] + shift;
    }
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(input, {
      ...panelErrorFixture,
      effects
    });
    const expected = panelErrorFixture.models[effects];
    compare(
      result.table.subarray(4, 6),
      expected.table.slice(4, 6).map(value => 2 * value)
    );
    compare(
      result.table.subarray(8, 10),
      expected.table.slice(8, 10).map(value => 2 * value)
    );
    compare(result.table.subarray(12, 14), expected.table.slice(12, 14));
    compare(
      result.residuals,
      expected.residuals.map(value => 2 * value)
    );
    compare([result.summary[2]], [expected.summary[2] * 4]);
    compare([result.summary[3]], [expected.summary[3] - 72 * Math.log(2)]);
  });
}

it('incomplete weights and a whole missing period suppress dependent outputs', () => {
  for (const effects of ['unit', 'time', 'twoWay'] as const) {
    const input = getPanelErrorInput();
    input.response.fill(NaN, 12, 24);
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(input, {
      ...panelErrorFixture,
      effects
    });
    expect(Array.from(result.status)).toEqual([3]);
    expect(Array.from(result.diagnosticStatus)).toEqual([1]);
    expect(result.table.every(Number.isNaN)).toBe(true);
    expect(result.residuals.every(Number.isNaN)).toBe(true);
    const overflow = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
      {
        ...getPanelErrorInput(),
        weights: {...getPanelErrorInput().weights, status: {overflow: Uint32Array.of(1)}}
      },
      {...panelErrorFixture, effects}
    );
    expect(Array.from(overflow.status)).toEqual([3]);
    expect(Array.from(overflow.converged)).toEqual([0]);
    expect(Array.from(overflow.diagnosticStatus)).toEqual([1]);
    expect(overflow.table.every(Number.isNaN)).toBe(true);
    expect(overflow.diagnosticTests.every(Number.isNaN)).toBe(true);
  }
});

it('a predictor in the absorbed effect space reports singularity', () => {
  for (const effects of ['unit', 'time', 'twoWay'] as const) {
    const input = getPanelErrorInput();
    for (let row = 0; row < input.response.length; row++)
      input.predictors[row * 2] = effects === 'time' ? Math.floor(row / 12) : row % 12;
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(input, {
      ...panelErrorFixture,
      effects
    });
    expect(Array.from(result.status)).toEqual([1]);
    expect(Array.from(result.converged)).toEqual([0]);
    expect(result.table.every(Number.isNaN)).toBe(true);
  }
});

it('refuses an independent-unit random covariance instead of fitting a lag-model covariance', () => {
  expect(() =>
    computeSpatialPanelErrorMaximumLikelihoodOnCPU(getPanelErrorInput(), {
      ...panelErrorFixture,
      effects: 'random'
    })
  ).toThrow(/effects 'random'.*second covariance determinant.*dense/);
});

it('candidate loss in supplied weights suppresses pooled diagnostics as well as the fit', () => {
  const input = getPanelErrorInput();
  const weights = {...input.weights, status: {candidateOverflow: Uint32Array.of(1)}};
  const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
    {...input, weights},
    panelErrorFixture
  );
  expect(Array.from(result.status)).toEqual([3]);
  expect(Array.from(result.diagnosticStatus)).toEqual([1]);
  expect(result.table.every(Number.isNaN)).toBe(true);
  expect(result.diagnosticTests.every(Number.isNaN)).toBe(true);
  expect(result.summary.every(Number.isNaN)).toBe(true);
  expect(result.powerTraceStandardErrors.every(Number.isNaN)).toBe(true);
});

it('both mean axes have the independently counted scratch shapes', () => {
  // 12 entities times 3 components, two words each, and one runtime zero.
  expect(getPanelWithinScratchLengths(12, 6, 2)).toEqual({
    partialLength: 72,
    meanLength: 73,
    tileCount: 1
  });
  // 6 periods times 3 components; the time-axis means sum 12 entities each.
  expect(getPanelWithinScratchLengths(12, 6, 2, 'time')).toEqual({
    partialLength: 36,
    meanLength: 37,
    tileCount: 1
  });
});

it('an exactly fitted panel has no finite Gaussian variance maximum', () => {
  const input = getPanelErrorInput();
  for (let row = 0; row < 72; row++) input.response[row] = 3 + 2 * input.predictors[row * 2];
  for (const effects of ['unit', 'time', 'twoWay'] as const) {
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(input, {
      ...panelErrorFixture,
      effects
    });
    // The exact residual is zero for every admissible lambda. LL is unbounded as sigma2 -> 0.
    expect(Array.from(result.status)).toEqual([3]);
    expect(result.table.every(Number.isNaN)).toBe(true);
    expect(Array.from(result.diagnosticStatus)).toEqual([2]);
    expect(result.diagnosticTests.every(Number.isNaN)).toBe(true);
  }
});

it('an exact pooled affine fit reports degenerate LM information rather than QR rounding noise', () => {
  const predictors = Float32Array.of(0.25, -1, 0.5, 2, -0.75, 0.125, 1.5, -0.5);
  const response = Float32Array.from(predictors, value => 3 + 2 * value);
  for (const effects of ['unit', 'time', 'twoWay'] as const) {
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
      {
        predictors,
        response,
        weights: {
          offsets: Uint32Array.of(0, 1, 2),
          neighbors: Uint32Array.of(1, 0),
          weights: Float32Array.of(1, 1)
        }
      },
      {periodCount: 4, predictorCount: 1, effects}
    );
    // Exact arithmetic: beta=(3,2), RSS=0, so all three LM statistics divide by zero.
    expect(Array.from(result.diagnosticStatus)).toEqual([2]);
    expect(result.diagnosticTests.every(Number.isNaN)).toBe(true);
    expect(Array.from(result.status)).toEqual([3]);
  }
});

it('small positive residual variance remains a fit instead of being classified as exact affine', () => {
  for (const effects of ['unit', 'time', 'twoWay'] as const) {
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(getSmallVariancePanelInput(), {
      periodCount: 4,
      predictorCount: 1,
      effects
    });
    // x'e=x'We=e'We=0. Unit RSS(lambda)=8 delta^2(1+lambda^2), giving lambda=0.
    // After time demeaning W e=-e and RSS(lambda)=4 delta^2(1+lambda)^2; the N*T
    // likelihood is monotone toward the negative series bound. No variance floor is used.
    const expectedVariance = effects === 'unit' ? 9.5367431640625e-7 : 2.31506486844809e-9;
    expect(result.status).toEqual(Uint32Array.of(0));
    expect(Number.isFinite(result.summary[2])).toBe(true);
    expect(Math.abs(result.summary[2] / expectedVariance - 1)).toBeLessThan(2e-6);
    expect(result.table[4]).toBeCloseTo(2, 6);
    expect(result.diagnosticStatus).toEqual(Uint32Array.of(0));
    compare(result.diagnosticTests, [0, 1, 0, 1, 0, 1]);
  }
});

it('exact-affine certificate scratch grows with bounded tiles and predictor columns', () => {
  expect(getPanelAffineFitAllocationLengths(8, 1)).toEqual({
    'affine certificate partials': 1,
    'affine certificate anchors': 1,
    'affine certificate columns': 1,
    'affine certificate flags': 2
  });
  expect(getPanelAffineFitAllocationLengths(2 ** 24, 15)).toEqual({
    'affine certificate partials': 61440,
    'affine certificate anchors': 15,
    'affine certificate columns': 15,
    'affine certificate flags': 2
  });
});

it('a final-tile residual cannot be mistaken for an affine response', () => {
  const actual = computeSpatialPanelErrorMaximumLikelihoodOnCPU(getTailPerturbedPanelInput(), {
    periodCount: 34,
    predictorCount: 1
  });
  // x'e=x'We=0 and We=e. RSS(lambda)=4 delta^2(1-lambda)^2; the N*T likelihood
  // increases up to the positive series bound. First 64 residuals are zero, last four +/-delta.
  expect(actual.status).toEqual(Uint32Array.of(0));
  expect(actual.converged).toEqual(Uint32Array.of(0));
  expect(Math.abs(actual.summary[2] / 2.7236057275859883e-10 - 1)).toBeLessThan(2e-6);
  compare(actual.residuals, [
    ...new Array<number>(64).fill(0),
    0.0009765625,
    0.0009765625,
    -0.0009765625,
    -0.0009765625
  ]);
});

it('a binding series interval reports a boundary fit and no interior convergence', () => {
  for (const effects of ['unit', 'time', 'twoWay'] as const) {
    const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(getPanelErrorInput(), {
      ...panelErrorFixture,
      effects,
      maximumPower: 2
    });
    expect(Array.from(result.status)).toEqual([0]);
    expect(Array.from(result.converged)).toEqual([0]);
    // Closed-form tail criterion s^3/(3(1-s))=1e-5, solved independently in Python.
    compare([result.summary[13]], [0.030750506892117536]);
    compare([result.table[12]], [(effects === 'unit' ? 1 : -1) * 0.030750506892117536]);
    expect(result.summary[15]).toBe(0);
    expect(result.summary[16]).toBe(1);
  }
});

it('publishes nonzero trace sampling errors from independent dense Philox quadratic forms', () => {
  const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
    {...getPanelErrorInput(), traceParameters: Uint32Array.of(11)},
    {...panelErrorFixture, maximumPower: 8, traceMode: 'stochastic', traceProbeCount: 4}
  );
  compare(result.powerTraceStandardErrors, panelErrorFixture.stochasticTraceErrors);
  expect(result.summary[14]).toBe(1);
  expect(result.status[0]).toBe(0);
});

it('exact affine rejection preserves independently valid trace standard errors', () => {
  const input = getPanelErrorInput();
  for (let row = 0; row < input.response.length; row++)
    input.response[row] = 3 + 2 * input.predictors[row * 2];
  const result = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
    {...input, traceParameters: Uint32Array.of(11)},
    {...panelErrorFixture, maximumPower: 8, traceMode: 'stochastic', traceProbeCount: 4}
  );
  expect(result.status).toEqual(Uint32Array.of(3));
  expect(result.diagnosticStatus).toEqual(Uint32Array.of(2));
  // These are diagnostics of W and its seeded probes, independent of whether y admits a fit.
  compare(result.powerTraceStandardErrors, panelErrorFixture.stochasticTraceErrors);
});

it('captures effect options and audits all new kernels within eight bindings', () => {
  for (const effects of ['unit', 'time', 'twoWay'] as const) {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
        createTransientView(graph, `panel-effects-${serial++}`, format, length);
      const props = {
        periodCount: 6,
        predictorCount: 2,
        effects,
        maximumPower: 2,
        traceMode: 'stochastic' as const,
        traceProbeCount: 2,
        weights: {
          offsets: view('uint32', 13),
          neighbors: view('uint32', 22),
          weights: view('float32', 22),
          status: {candidateOverflow: view('uint32', 1)}
        },
        predictors: view('float32', 144),
        response: view('float32', 72),
        output: {
          table: view('float32', 16),
          summary: view('float32', 22),
          residuals: view('float32', 72),
          status: {code: view('uint32', 1), converged: view('uint32', 1)}
        },
        diagnostics: {tests: view('float32', 6), status: view('uint32', 1)}
      };
      const contributor = new GPUSpatialPanelErrorMaximumLikelihood(props);
      const diagnosticTests = props.diagnostics.tests;
      props.periodCount = 3;
      props.diagnostics.tests = view('float32', 6);
      expect(contributor.props.periodCount).toBe(6);
      expect(contributor.props.diagnostics?.tests).toBe(diagnosticTests);
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      for (const source of getGPUCommandNodeKernelSources(nodes)) {
        expect(new WgslReflect(source.source).storage.length, source.id).toBeLessThanOrEqual(8);
        expect(auditWGSLModule(source.source, source.constants), source.id).toEqual([]);
      }
    } finally {
      device.destroy();
    }
  }
});
