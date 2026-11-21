// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpatialLagMaximumLikelihoodOnCPU,
  CPUSpatialLagMaximumLikelihoodOperation
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-lag-maximum-likelihood-operation';
import {computeSpatialErrorMaximumLikelihoodOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-maximum-likelihood-operation';
import {getSpatialExactLogDeterminant} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-exact-log-determinant';
import {
  getSpatialMultiplierTotalFromPowerSums,
  getSpatialMultiplierTruncationBound
} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {getCPUSpatialErrorLikelihoodCandidates} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-likelihood-polynomial';
import {getScalarLagExpected} from './likelihood-review-regressions-test-utils';
import {getReviewPanelInput} from './likelihood-review-regressions-test-utils';
import {
  CPUSpatialPanelErrorMaximumLikelihoodOperation,
  computeSpatialPanelErrorMaximumLikelihoodOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-panel-error-maximum-likelihood-operation';
import {getParameterSearchWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-wgsl';
import {searchSpatialMaximumLikelihoodParameter} from '../../reference/spatial-maximum-likelihood-cpu';

for (const sign of [1, -1]) {
  it(`review 2 R5 stationary endpoint minima are excluded, sign=${sign}`, () => {
    const rows = [-1, 1, -1, 1].map(value => [1, value]);
    // Constant SSE; traces give logdet=t^2+sign*t/2. Only sign/4 is a maximum.
    expect(
      getCPUSpatialErrorLikelihoodCandidates(
        rows,
        rows.map(() => [0, 0]),
        [-sign / 2, -2],
        undefined,
        undefined,
        4,
        0.25
      )
    ).toEqual([{parameter: sign / 4, atBoundary: true}]);
    const actual = searchSpatialMaximumLikelihoodParameter(
      parameter => parameter ** 2 + (sign * parameter) / 2,
      parameter => 2 * parameter + sign / 2,
      0.25
    );
    expect(actual).toEqual({parameter: sign / 4, localMaximumCount: 1, atBoundary: true});
    expect(
      searchSpatialMaximumLikelihoodParameter(
        parameter => -((parameter - sign / 4) ** 2),
        parameter => -2 * (parameter - sign / 4),
        0.25
      )
    ).toEqual({parameter: sign / 4, localMaximumCount: 1, atBoundary: true});
  });
}

it('review 2 R5 panel scan retains one lower candidate on a flat profile', () => {
  // Nilpotent W has logdet=0; Wy=WX=0 and beta=0, so SSE=4 for every parameter.
  const actual = computeSpatialPanelErrorMaximumLikelihoodOnCPU(
    {
      weights: {
        offsets: Uint32Array.of(0, 1, 1, 1, 1),
        neighbors: Uint32Array.of(1),
        weights: Float32Array.of(1)
      },
      predictors: Float32Array.of(0, 0, 1, -1, 0, 0, -1, 1),
      response: Float32Array.of(0, 0, 1, 1, 0, 0, -1, -1)
    },
    {periodCount: 2, predictorCount: 1, maximumPower: 3}
  );
  expect(actual.status[0]).toBe(0);
  expect(actual.summary[20]).toBe(1);
  expect(actual.summary[1]).toBe(-actual.summary[13]);
  expect(actual.summary[15]).toBe(0);
  expect(actual.summary[16]).toBe(1);
});

it('review 2 R4 emitted search retains the interior cubic maximum', () => {
  const difference = 1 / 1024;
  let lower = Math.fround(1 - 2 / 1024);
  let upper = 1;
  for (let iteration = 0; iteration < 32; iteration++) {
    const middle = Math.fround(0.5 * Math.fround(lower + upper));
    const offset = Math.fround(1 - middle);
    if (Math.fround(offset * Math.fround(offset - difference)) > 0) lower = middle;
    else upper = middle;
  }
  const expression = getParameterSearchWGSL()
    .match(/let root = ([^;]+);/)![1]
    .replace(/(\d+)u\b/g, '$1');
  // Evaluate the emitted selection after f32 bisection, not a rewritten selection rule.
  const selectRoot = new Function(
    'low',
    'high',
    'bound',
    'index',
    'current',
    'select',
    `return ${expression};`
  ) as (...arguments_: unknown[]) => number;
  const actual = Math.fround(
    selectRoot(lower, upper, 1, 1024, 0, (first: number, second: number, condition: boolean) =>
      condition ? second : first
    )
  );
  // f=u^2(d/2-u/3): unique maximum at u=d, f=d^3/6 > f(1)=0.
  expect(actual).toBe(1 - difference);
});

for (const reflection of [1, -1]) {
  it(`v3-1 reflected cubic scan finds the first/last-cell interior maximum, reflection=${reflection}`, () => {
    const difference = 1 / 1024;
    const actual = searchSpatialMaximumLikelihoodParameter(
      parameter => {
        const offset = 1 - reflection * parameter;
        return offset ** 2 * (difference / 2 - offset / 3);
      },
      parameter => {
        const offset = 1 - reflection * parameter;
        return reflection * offset * (offset - difference);
      },
      1
    );
    // f'=s*u*(u-d): a stationary endpoint minimum and a unique interior maximum at u=d.
    expect(Number.isFinite(actual.parameter)).toBe(true);
    expect(Math.abs(actual.parameter - reflection * (1 - difference))).toBeLessThan(2e-12);
    expect(actual.localMaximumCount).toBe(1);
    expect(actual.atBoundary).toBe(false);
  });
}

it('review 2 R7 retained stochastic trace error is separate from truncation', () => {
  const actual = computeSpatialLagMaximumLikelihoodOnCPU(
    {
      weights: {
        offsets: Uint32Array.from({length: 9}, (_, row) => row),
        neighbors: Uint32Array.from({length: 8}, (_, row) => row ^ 1),
        weights: new Float32Array(8).fill(1)
      },
      predictors: Float32Array.from({length: 8}, (_, row) => row),
      response: Float32Array.of(-4, -2, -1, 0, 1, 1, 4, 4),
      traceParameters: Uint32Array.of(0)
    },
    {
      predictorCount: 1,
      maximumPower: 3,
      traceMode: 'stochastic',
      traceProbeCount: 2,
      tailExtrapolation: true
    }
  );
  expect(actual.status[0]).toBe(0);
  const parameter = actual.summary[1];
  // Four exchange blocks: det(I-tW)=(1-t^2)^4 independently of stochastic probes.
  const error = Math.abs(actual.summary[21] - 4 * Math.log1p(-(parameter ** 2)));
  expect(Number.isFinite(error)).toBe(true);
  expect(error).toBeGreaterThan(0.5);
  expect(actual.summary[18]).toBeLessThan(0.01);
  expect(actual.summary[18]).toBeCloseTo(0.00387176, 7);
  expect(error).toBeCloseTo(0.6992161, 6);
});

it('review 2 R1 panel CPU operation bounds the tail-corrected determinant', async () => {
  const actual = await new CPUSpatialPanelErrorMaximumLikelihoodOperation({
    entityCount: 14,
    periodCount: 2,
    predictorCount: 1,
    maximumPower: 3,
    tailExtrapolation: true,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    predictors: 'predictors',
    response: 'response',
    output: {table: 'table', summary: 'summary', status: {code: 'status'}}
  }).execute({
    inputs: Object.fromEntries(
      Object.entries(getReviewPanelInput()).map(([port, value]) => [
        port,
        {
          format: value.format,
          length: value.values.length,
          chunks: [{values: value.values, length: value.values.length}],
          pack: () => value.values
        }
      ])
    )
  });
  const summary = actual['summary'] as Float32Array;
  expect(Array.from(actual['status'] as Uint32Array)).toEqual([0]);
  const parameter = summary[1];
  // Two copies of four rotation blocks and three symmetric exchanges.
  const exact = 8 * Math.log1p(parameter ** 2) + 6 * Math.log1p(-(parameter ** 2));
  expect(Number.isFinite(summary[21])).toBe(true);
  expect(Math.abs(summary[21] - exact)).toBeLessThanOrEqual(summary[18]);
  // tr(W^2)=-4 on the expanded graph, so the Q=3 plain determinant is 2*t^2.
  const expectedBound =
    (28 * Math.abs(parameter) ** 4) / (4 * (1 - Math.abs(parameter))) +
    Math.abs(summary[21] - 2 * parameter ** 2);
  // Slot 21 is independently rounded to f32 before its small correction is subtracted.
  expect(Math.abs(summary[18] - expectedBound)).toBeLessThan(2e-10);
});
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import {
  fixtures,
  getFlatProfileInput,
  getFlatProfileProps
} from './likelihood-flat-profile-test-utils';

for (const diagonal of [false, true])
  it(`W5-39 constant lag fit diagonal=${diagonal}`, async () => {
    const input = {
      weights: {
        offsets: Uint32Array.of(0, 1, 2, 2, 2, 2, 2, 2),
        neighbors: Uint32Array.of(1, 0),
        weights: Float32Array.of(1, 1)
      },
      predictors: Float32Array.of(0, 0, 1, -1, -1, 1, 0),
      response: Float32Array.of(0, 0, -2, -2, 2, 2, -2)
    };
    if (diagonal)
      input.weights = {
        offsets: Uint32Array.of(0, 1, 2, 2, 2, 2, 2, 2),
        neighbors: Uint32Array.of(0, 1),
        weights: Float32Array.of(5, -1)
      };
    const actual = computeSpatialLagMaximumLikelihoodOnCPU(input, {
      predictorCount: 1,
      maximumPower: 3
    });
    // n*sum(x*y)-sum(x)*sum(y)=0 exactly, Wy=0: beta=(-2/7,0), rho=0.
    expect(actual.status[0]).toBe(0);
    if (!diagonal) expect(actual.summary[1]).toBe(0);
    expect(actual.table[4]).toBe(0);
    expect(actual.summary[6]).toBeNaN();
    const named = {...input.weights, predictors: input.predictors, response: input.response};
    const operation = await new CPUSpatialLagMaximumLikelihoodOperation({
      ...getFlatProfileProps(fixtures[0]),
      rowCount: 7,
      logDeterminant: 'series',
      eigenvalues: undefined
    }).execute({
      inputs: Object.fromEntries(
        Object.entries(named).map(([port, values]) => [
          port,
          {
            format: values instanceof Uint32Array ? 'uint32' : 'float32',
            length: values.length,
            chunks: [{values, length: values.length}],
            pack: () => values
          }
        ])
      )
    });
    expect(Array.from(operation['status'] as Uint32Array)).toEqual([0]);
    expect((operation['table'] as Float32Array)[4]).toBe(0);
    expect((operation['summary'] as Float32Array)[6]).toBeNaN();
  });

for (const [kind, compute] of [
  ['lag', computeSpatialLagMaximumLikelihoodOnCPU],
  ['error', computeSpatialErrorMaximumLikelihoodOnCPU]
] as const)
  for (const shift of [0, 1]) {
    it(`review 2 R3 CPU ${kind} one-hot predictor is translation invariant, shift=${shift}`, () => {
      const actual = compute(
        {
          weights: {
            offsets: Uint32Array.of(0, 1, 2, 2, 2, 2, 2, 2, 2),
            neighbors: Uint32Array.of(1, 0),
            weights: Float32Array.of(1, 1)
          },
          eigenvalues: Float32Array.of(-1, 0, 0, 0, 0, 0, 0, 1),
          predictors: Float32Array.from(
            {length: 8},
            (_, row) => shift + (row === 2 ? 2 ** -23 : 0)
          ),
          response: Float32Array.of(0, 0, 1, -1, 1, -1, 1, -1)
        },
        {predictorCount: 1, maximumPower: 3, logDeterminant: 'exact'}
      );
      // y'y=6, centered indicator variance=7/8, covariance=1: corr^2=4/21.
      expect(actual.status[0]).toBe(0);
      expect(Number.isFinite(actual.summary[6])).toBe(true);
      expect(Math.abs(actual.summary[6] - 4 / 21)).toBeLessThan(1e-7);
    });
  }

for (const sign of [1, -1]) {
  it(`R6 reflected stationary endpoint is a boundary, sign=${sign}`, () => {
    const response = [-1, 1, -1, 1];
    const orthogonal = [-1, -1, 1, 1];
    // SSE/n=(1-2*sign*t)^2+4*t^2 has its minimum at sign/4.
    const actual = getCPUSpatialErrorLikelihoodCandidates(
      response.map(value => [1, value]),
      response.map((value, row) => [0, sign * (2 * value + 2 * orthogonal[row])]),
      [0, 0],
      undefined,
      undefined,
      4,
      0.25
    );
    expect(actual).toEqual([{parameter: sign * 0.25, atBoundary: true}]);
  });
}

it('R5 directed path requires a row-norm bound for total multipliers', () => {
  // Eight-row superdiagonal path: W^q has 8-q ones and W^8=0; all eigenvalues are zero.
  const truncated = getSpatialMultiplierTotalFromPowerSums(0.5, [7, 6, 5], 8, 1);
  const exact = 14.0078125; // sum_{q=0}^7 (8-q)/2^q, also 1'(I-W/2)^-1 1.
  expect(truncated).toBe(13.625);
  expect(exact - truncated).toBe(0.3828125);
  // Reproduction: a spectral-radius argument of zero falsely claims no omitted contribution.
  expect(getSpatialMultiplierTruncationBound(0.5, 3, 8, 0)).toBe(0);
  const bound = getSpatialMultiplierTruncationBound(0.5, 3, 8, 1);
  expect(bound).toBe(1);
  expect(exact - truncated).toBeLessThanOrEqual(bound);
});

it('R4 exact determinant recovers a near-pole factor whose product rounds below one', () => {
  const difference = 3 * 2 ** -28;
  // (1+d)(1-d)=1-d^2 exactly, so the factor is d^2, not 2^-53.
  expect(getSpatialExactLogDeterminant(1 + difference, [1 - difference])).toBeCloseTo(
    -36.61901753402072,
    10
  );
});

for (const [kind, compute] of [
  ['lag', computeSpatialLagMaximumLikelihoodOnCPU],
  ['error', computeSpatialErrorMaximumLikelihoodOnCPU]
] as const) {
  for (const exponent of [-28, -14]) {
    it(`R3 ${kind} retains weak fitted variance h=2^${exponent}`, () => {
      const height = 2 ** exponent;
      const actual = compute(
        {
          weights: {
            offsets: Uint32Array.of(0, 1, 2, 2, 2, 2, 2, 2, 2),
            neighbors: Uint32Array.of(1, 0),
            weights: Float32Array.of(1, 1)
          },
          eigenvalues: Float32Array.of(-1, 0, 0, 0, 0, 0, 0, 1),
          predictors: Float32Array.of(0, 0, 1, -1, 0, 0, 0, 0),
          response: Float32Array.of(0, 0, height, -height, 1, -1, 1, -1)
        },
        {predictorCount: 1, maximumPower: 3, logDeterminant: 'exact'}
      );
      // Orthogonal vectors: y=h*x+noise, x'x=2, noise'noise=4.
      const expected = height ** 2 / (2 + height ** 2);
      expect(actual.status[0]).toBe(0);
      expect(actual.summary[1]).toBe(0);
      expect(Number.isFinite(actual.summary[6])).toBe(true);
      expect(Math.abs(actual.summary[6] / expected - 1)).toBeLessThan(2e-6);
    });
  }
}

for (const [kind, compute] of [
  ['lag', computeSpatialLagMaximumLikelihoodOnCPU],
  ['error', computeSpatialErrorMaximumLikelihoodOnCPU]
] as const) {
  it(`R2 ${kind} corrected determinant bound covers signed-block error`, () => {
    const actual = compute(
      {
        weights: {
          offsets: Uint32Array.from({length: 9}, (_, row) => row),
          neighbors: Uint32Array.from({length: 8}, (_, row) => row ^ 1),
          weights: Float32Array.from({length: 8}, (_, row) => (row % 2 ? -1 : 1))
        },
        predictors: Float32Array.from({length: 8}, (_, row) => row),
        response: Float32Array.of(-4, -2, -1, 0, 1, 1, 4, 4)
      },
      {predictorCount: 1, maximumPower: 3, tailExtrapolation: true}
    );
    expect(actual.status[0]).toBe(0);
    const parameter = actual.summary[1];
    // Each 2x2 rotation block has determinant 1+t^2, independently of the trace model.
    const error = Math.abs(actual.summary[21] - 4 * Math.log1p(parameter ** 2));
    expect(Number.isFinite(error)).toBe(true);
    expect(
      error,
      `${kind}: actual error=${error}, bound=${actual.summary[18]}`
    ).toBeLessThanOrEqual(actual.summary[18]);
    const expectedBound =
      (8 * Math.abs(parameter) ** 4) / (4 * (1 - Math.abs(parameter))) +
      Math.abs(actual.summary[21] - 4 * parameter ** 2);
    expect(Math.abs(actual.summary[18] - expectedBound)).toBeLessThan(2e-7 + 2e-6 * expectedBound);
  });
}

for (const maximumPower of [3, 4, 17]) {
  it(`R1 CPU lag operation retains all closed-form ports, Q=${maximumPower}`, async () => {
    const fixture = {...fixtures[0], maximumPower};
    const input = getFlatProfileInput(fixture);
    const props = getFlatProfileProps(fixture);
    const actual = await new CPUSpatialLagMaximumLikelihoodOperation({
      ...props,
      output: {
        ...props.output,
        powerTraces: 'powerTraces',
        powerSums: 'powerSums',
        infinityNorm: 'infinityNorm'
      }
    }).execute({
      inputs: Object.fromEntries(
        Object.entries(input).map(([port, value]) => [
          port,
          {
            format: value.format,
            length: value.values.length,
            chunks: [{values: value.values, length: value.values.length}],
            pack: () => value.values
          }
        ])
      )
    });
    for (const [port, expected] of Object.entries(getScalarLagExpected(maximumPower)))
      expectRegressionRowsClose(actual[port] as Float32Array, expected, 3e-6);
  });
  it(`R1 scalar lag profile suppresses joint inference, Q=${maximumPower}`, () => {
    const actual = computeSpatialLagMaximumLikelihoodOnCPU(
      {
        weights: {
          offsets: Uint32Array.from({length: 9}, (_, row) => row),
          neighbors: Uint32Array.from({length: 8}, (_, row) => row),
          weights: new Float32Array(8).fill(3)
        },
        eigenvalues: new Float32Array(8).fill(3),
        predictors: Float32Array.from({length: 8}, (_, row) => row),
        response: Float32Array.from({length: 8}, (_, row) => row * row)
      },
      {predictorCount: 1, maximumPower, logDeterminant: 'exact'}
    );
    // W=3I: log|I-rho W| cancels n/2 log(sigma2), so all rho tie.
    expect(actual.summary[1]).toBeCloseTo(-0.33, 6);
    expect(actual.summary[3]).toBeCloseTo(-23.5295980165, 5);
    expect(Array.from(actual.summary.slice(15, 21))).toEqual([0, 1, 0, 0, 0, 1]);
    expect(Array.from(actual.converged)).toEqual([0]);
    expect(Array.from(actual.status)).toEqual([0]);
    for (let row = 0; row < 3; row++) {
      for (let column = 1; column < 4; column++) expect(actual.table[row * 4 + column]).toBeNaN();
    }
    for (const [port, expected] of Object.entries(getScalarLagExpected(maximumPower)))
      expectRegressionRowsClose(actual[port as keyof typeof actual], expected, 3e-6);
  });
}

// Execute the production search, independently of the reference scanner's endpoint logic.
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';

for (const reflection of [1, -1])
  it(`W5-40 negated cubic counts both boundary maxima reflection=${reflection}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device),
        memory = createCPUGraphMemory();
      const output = createTransientView(graph, 'search-result', 'float32', 3);
      const node = createWGSLKernelNode(graph, {
        id: 'search',
        operation: 'search',
        invocationCount: 1,
        bindings: [{name: 'output', view: output, type: 'f32', access: 'read_write'}],
        declarations: `${GET_NAN_WGSL}
fn getObjective(parameter:f32)->f32{let offset=1.0-(${reflection}.0)*parameter;return -offset*offset*(0.0009765625/2.0-offset/3.0);}
fn getDerivative(parameter:f32)->f32{let offset=1.0-(${reflection}.0)*parameter;return -(${reflection}.0)*offset*(offset-0.0009765625);}
${getParameterSearchWGSL()}`,
        body: 'let result=searchParameter(1.0);output[outputOffset]=result.parameter;output[outputOffset+1u]=f32(result.localMaximumCount);output[outputOffset+2u]=select(0.0,1.0,result.atBoundary);'
      });
      executeCommandNodesOnCPU([node], memory);
      // f=-u²(d/2-u/3). Both u=0 and u=2 are boundary maxima; u=2 wins.
      expect(Array.from(memory.readFloat32(output))).toEqual([-reflection, 2, 1]);
    } finally {
      device.destroy();
    }
  });
