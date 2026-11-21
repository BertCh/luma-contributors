// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createRegressionTestContext} from './math-review-regression-test-utils';
import {GPUSpatialLagMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood-operation';
import {GPUSpatialErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  fixtures,
  getFlatProfileInput,
  getFlatProfileProps
} from './likelihood-flat-profile-test-utils';
import {getExactLogDeterminantWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-exact-log-determinant';
import {getParameterSearchWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-wgsl';
import {GET_NAN_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {getScalarLagExpected} from './likelihood-review-regressions-test-utils';
import {getReviewPanelInput} from './likelihood-review-regressions-test-utils';
import {GPUSpatialPanelErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-panel-error-maximum-likelihood-operation';

it('review 2 R1 panel GPU operation bounds the tail-corrected determinant', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const actual = await runOperationGraph(device, {
    inputs: getReviewPanelInput(),
    operations: [
      new GPUSpatialPanelErrorMaximumLikelihoodOperation({
        entityCount: 14,
        periodCount: 2,
        predictorCount: 1,
        maximumPower: 3,
        tailExtrapolation: true,
        weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
        predictors: 'predictors',
        response: 'response',
        output: {table: 'table', summary: 'summary', status: {code: 'status'}}
      })
    ],
    outputs: ['summary', 'status']
  });
  expect(Array.from(actual['status'])).toEqual([0]);
  const summary = actual['summary'];
  const parameter = summary[1];
  // Closed-form determinant of eight rotation blocks and six symmetric exchanges.
  const exact = 8 * Math.log1p(parameter ** 2) + 6 * Math.log1p(-(parameter ** 2));
  expect(Number.isFinite(summary[21])).toBe(true);
  expect(Math.abs(summary[21] - exact)).toBeLessThanOrEqual(summary[18] + 1e-8);
  const expectedBound =
    (28 * Math.abs(parameter) ** 4) / (4 * (1 - Math.abs(parameter))) +
    Math.abs(summary[21] - 2 * parameter ** 2);
  expect(Math.abs(summary[18] - expectedBound)).toBeLessThan(1e-8 + 5e-5 * expectedBound);
});
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

for (const diagonal of [false, true])
  it(`W5-39 GPU lag constant fit diagonal=${diagonal}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = getFlatProfileInput(fixtures[0]);
    input.offsets.values = Uint32Array.of(0, 1, 2, 2, 2, 2, 2, 2);
    input.neighbors.values = Uint32Array.of(1, 0);
    input.weights.values = Float32Array.of(1, 1);
    input.eigenvalues.values = Float32Array.of(-1, 0, 0, 0, 0, 0, 1);
    input.predictors.values = Float32Array.of(0, 0, 1, -1, -1, 1, 0);
    input.response.values = Float32Array.of(0, 0, -2, -2, 2, 2, -2);
    if (diagonal) {
      input.neighbors.values = Uint32Array.of(0, 1);
      input.weights.values = Float32Array.of(5, -1);
      input.eigenvalues.values = Float32Array.of(-1, 0, 0, 0, 0, 0, 5);
    }
    const actual = await runOperationGraph(device, {
      inputs: input,
      operations: [
        new GPUSpatialLagMaximumLikelihoodOperation({
          ...getFlatProfileProps(fixtures[0]),
          rowCount: 7
        })
      ],
      outputs: ['status', 'table', 'summary']
    });
    expect(Array.from(actual['status'])).toEqual([0]);
    expect(actual['table'][4]).toBe(0);
    expect(actual['summary'][6]).toBeNaN();
  });

for (const [kind, Operation] of [
  ['lag', GPUSpatialLagMaximumLikelihoodOperation],
  ['error', GPUSpatialErrorMaximumLikelihoodOperation]
] as const)
  for (const shift of [0, 1]) {
    it(`review 2 R3 GPU ${kind} one-hot correlation is translation invariant, shift=${shift}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = getFlatProfileInput(fixtures[0]);
      input.offsets.values = Uint32Array.of(0, 1, 2, 2, 2, 2, 2, 2, 2);
      input.neighbors.values = Uint32Array.of(1, 0);
      input.weights.values = Float32Array.of(1, 1);
      input.eigenvalues.values = Float32Array.of(-1, 0, 0, 0, 0, 0, 0, 1);
      input.predictors.values = Float32Array.from(
        {length: 8},
        (_, row) => shift + (row === 2 ? 2 ** -23 : 0)
      );
      input.response.values = Float32Array.of(0, 0, 1, -1, 1, -1, 1, -1);
      const actual = await runOperationGraph(device, {
        inputs: input,
        operations: [new Operation(getFlatProfileProps(fixtures[0]))],
        outputs: ['status', 'summary']
      });
      expect(Array.from(actual['status'])).toEqual([0]);
      expect(Number.isFinite(actual['summary'][6])).toBe(true);
      // Exact centered indicator correlation=4/21; 0.1% allows f32 OLS/moment rounding.
      expect(Math.abs(actual['summary'][6] - 4 / 21)).toBeLessThan(2e-4);
    });
  }

it('review 2 R4 GPU search preserves interior cubic maxima beside stationary endpoints', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const parameterBuffer = createInputBuffer(device, Float32Array.of(1, -1));
  const outputBuffer = createOutputBuffer(device, 8);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'cubic-search',
      operation: 'cubic-search',
      variant: 'cubic-search',
      invocationCount: 2,
      bindings: [
        {
          name: 'parameters',
          view: importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', 2),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'output',
          view: importGraphBuffer(graph, 'output', outputBuffer, 'float32', 8),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `${GET_NAN_WGSL}
var<private> reflection: f32;
fn getObjective(parameter: f32) -> f32 {
  let offset = 1.0 - reflection * parameter;
  return offset * offset * (0.0009765625 / 2.0 - offset / 3.0);
}
fn getDerivative(parameter: f32) -> f32 {
  let offset = 1.0 - reflection * parameter;
  return reflection * offset * (offset - 0.0009765625);
}
${getParameterSearchWGSL()}`,
      body: `reflection = parameters[parametersOffset + index];
  let result = searchParameter(1.0);
  output[outputOffset + index * 4u] = result.parameter;
  output[outputOffset + index * 4u + 1u] = f32(result.localMaximumCount);
  output[outputOffset + index * 4u + 2u] = select(0.0, 1.0, result.atBoundary);
  output[outputOffset + index * 4u + 3u] = getObjective(result.parameter);`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(outputBuffer, 8);
    for (let index = 0; index < 2; index++) {
      expect(
        Number.isFinite(actual[index * 4]),
        `reflection=${index === 0 ? 1 : -1}, search=${Array.from(actual.slice(index * 4, index * 4 + 4))}`
      ).toBe(true);
      expect(Math.abs(actual[index * 4] - (index === 0 ? 1 : -1) * (1 - 1 / 1024))).toBeLessThan(
        2e-7
      );
      expect(actual[index * 4 + 1]).toBe(1);
      expect(actual[index * 4 + 2]).toBe(0);
      // f(maximum)=d^3/6, strictly above either stationary endpoint's value 0.
      expect(Math.abs(actual[index * 4 + 3] / (1 / (6 * 1024 ** 3)) - 1)).toBeLessThan(1e-4);
    }
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    outputBuffer.destroy();
  }
});

it('R6 GPU search classifies both reflected stationary endpoints as boundaries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const parameterBuffer = createInputBuffer(device, Float32Array.of(0.25, -0.25, 0.5, -0.5));
  const outputBuffer = createOutputBuffer(device, 16);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'endpoint-search',
      operation: 'endpoint-search',
      variant: 'endpoint-search',
      invocationCount: 4,
      bindings: [
        {
          name: 'parameters',
          view: importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', 4),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'output',
          view: importGraphBuffer(graph, 'output', outputBuffer, 'float32', 16),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `${GET_NAN_WGSL}
var<private> endpoint: f32;
fn getObjective(parameter: f32) -> f32 {
  if (abs(endpoint) > 0.25) { return parameter * parameter + endpoint * parameter; }
  let difference = parameter - endpoint; return -difference * difference;
}
fn getDerivative(parameter: f32) -> f32 {
  if (abs(endpoint) > 0.25) { return 2.0 * parameter + endpoint; }
  return -2.0 * (parameter - endpoint);
}
${getParameterSearchWGSL()}`,
      body: `endpoint = parameters[parametersOffset + index];
  let result = searchParameter(0.25);
  output[outputOffset + index * 4u] = result.parameter;
  output[outputOffset + index * 4u + 1u] = f32(result.localMaximumCount);
  output[outputOffset + index * 4u + 2u] = select(0.0, 1.0, result.atBoundary);
  output[outputOffset + index * 4u + 3u] = select(1.0, 0.0, result.atBoundary);`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(outputBuffer, 16);
    // Last two objectives have a stationary endpoint minimum: it must not add a candidate.
    expect(Array.from(actual)).toEqual([
      0.25, 1, 1, 0, -0.25, 1, 1, 0, 0.25, 1, 1, 0, -0.25, 1, 1, 0
    ]);
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    outputBuffer.destroy();
  }
});

it('R4 GPU determinant and derivative ratios recover the near-pole factor', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const difference = 3 * 2 ** -14;
  const parameterBuffer = createInputBuffer(device, Float32Array.of(1 + difference));
  const eigenvalueBuffer = createInputBuffer(device, Float32Array.of(1 - difference));
  const outputBuffer = createOutputBuffer(device, 4);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'near-pole',
      operation: 'near-pole',
      variant: 'near-pole',
      invocationCount: 1,
      bindings: [
        {
          name: 'parameters',
          view: importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', 1),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'eigenvalues',
          view: importGraphBuffer(graph, 'eigenvalues', eigenvalueBuffer, 'float32', 1),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'output',
          view: importGraphBuffer(graph, 'output', outputBuffer, 'float32', 4),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: getExactLogDeterminantWGSL('eigenvalues', 1),
      body: `let parameter = parameters[parametersOffset];
  output[outputOffset] = getSpatialLogDeterminant(parameter);
  output[outputOffset + 1u] = getTraceOne(parameter);
  output[outputOffset + 2u] = getLogDeterminantDerivative(parameter);
  output[outputOffset + 3u] = getTraceTwo(parameter);`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(outputBuffer, 4);
    // Factor=d^2 exactly. Closed forms; 3e-6 relative permits log and f32 quotient rounding.
    const ratio = (1 - difference) / difference ** 2;
    const expected = [-17.21089647834363, ratio, -ratio, ratio ** 2];
    for (let slot = 0; slot < expected.length; slot++) {
      expect(Number.isFinite(actual[slot])).toBe(true);
      expect(Math.abs(actual[slot] / expected[slot] - 1)).toBeLessThan(3e-6);
    }
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    eigenvalueBuffer.destroy();
    outputBuffer.destroy();
  }
});

for (const [kind, Operation] of [
  ['lag', GPUSpatialLagMaximumLikelihoodOperation],
  ['error', GPUSpatialErrorMaximumLikelihoodOperation]
] as const) {
  it(`R2 GPU ${kind} tail-corrected determinant retains a valid bound`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const input = getFlatProfileInput(fixtures[0]);
    input.neighbors.values = Uint32Array.from({length: 8}, (_, row) => row ^ 1);
    input.weights.values = Float32Array.from({length: 8}, (_, row) => (row % 2 ? -1 : 1));
    input.response.values = Float32Array.of(-4, -2, -1, 0, 1, 1, 4, 4);
    const actual = await runOperationGraph(device, {
      inputs: input,
      operations: [
        new Operation({
          ...getFlatProfileProps(fixtures[0]),
          logDeterminant: 'series',
          eigenvalues: undefined,
          tailExtrapolation: true
        })
      ],
      outputs: ['summary', 'status']
    });
    expect(Array.from(actual['status'])).toEqual([0]);
    const parameter = actual['summary'][1];
    const error = Math.abs(actual['summary'][21] - 4 * Math.log1p(parameter ** 2));
    expect(Number.isFinite(error)).toBe(true);
    expect(error).toBeLessThanOrEqual(actual['summary'][18]);
    // Q=3 plain logdet is 4*t^2. Float32 evaluation allowance: 2e-6 absolute.
    const expectedBound =
      (8 * Math.abs(parameter) ** 4) / (4 * (1 - Math.abs(parameter))) +
      Math.abs(actual['summary'][21] - 4 * parameter ** 2);
    expect(Math.abs(actual['summary'][18] - expectedBound)).toBeLessThan(
      2e-6 + 1e-5 * expectedBound
    );
  });
}

for (const [kind, Operation] of [
  ['lag', GPUSpatialLagMaximumLikelihoodOperation],
  ['error', GPUSpatialErrorMaximumLikelihoodOperation]
] as const) {
  for (const height of [0, 2 ** -14]) {
    it(`R3 GPU ${kind} preserves small prediction variance, h=${height}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const input = getFlatProfileInput(fixtures[0]);
      input.offsets.values = Uint32Array.of(0, 1, 2, 2, 2, 2, 2, 2, 2);
      input.neighbors.values = Uint32Array.of(1, 0);
      input.weights.values = Float32Array.of(1, 1);
      input.eigenvalues.values = Float32Array.of(-1, 0, 0, 0, 0, 0, 0, 1);
      input.predictors.values = Float32Array.of(0, 0, 1, -1, 0, 0, 0, 0);
      input.response.values = Float32Array.of(0, 0, height, -height, 1, -1, 1, -1);
      const actual = await runOperationGraph(device, {
        inputs: input,
        operations: [new Operation(getFlatProfileProps(fixtures[0]))],
        outputs: ['summary', 'table', 'status']
      });
      expect(Array.from(actual['status'])).toEqual([0]);
      expect(Math.abs(actual['summary'][1])).toBeLessThan(1e-6);
      if (height === 0) expect(actual['summary'][6]).toBeNaN();
      else {
        // h^2/(2+h^2), exact orthogonal-vector squared correlation, 0.1% float32 tolerance.
        expect(Number.isFinite(actual['summary'][6])).toBe(true);
        expect(Math.abs(actual['summary'][6] / 1.8626451457615105e-9 - 1)).toBeLessThan(1e-3);
      }
    });
  }
}

for (const maximumPower of [3, 4, 17]) {
  it(`R1 GPU scalar lag has one boundary candidate and no joint inference, Q=${maximumPower}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = {...fixtures[0], maximumPower};
    const props = getFlatProfileProps(fixture);
    const actual = await runOperationGraph(device, {
      inputs: getFlatProfileInput(fixture),
      operations: [
        new GPUSpatialLagMaximumLikelihoodOperation({
          ...props,
          output: {
            ...props.output,
            powerTraces: 'powerTraces',
            powerSums: 'powerSums',
            infinityNorm: 'infinityNorm'
          }
        })
      ],
      outputs: Object.keys(getScalarLagExpected(maximumPower))
    });
    // W=3I: determinant and innovation scale cancel exactly; lower endpoint wins the tie.
    expect(actual['summary'][1]).toBeCloseTo(-0.33, 5);
    expect(actual['summary'][3]).toBeCloseTo(-23.5295980165, 3);
    expect(Array.from(actual['summary'].slice(15, 21))).toEqual([0, 1, 0, 0, 0, 1]);
    expect(Array.from(actual['status'])).toEqual([0]);
    expect(Array.from(actual['converged'])).toEqual([0]);
    for (let row = 0; row < 3; row++) {
      for (let column = 1; column < 4; column++)
        expect(actual['table'][row * 4 + column]).toBeNaN();
    }
    for (const [port, expected] of Object.entries(getScalarLagExpected(maximumPower)))
      expectRegressionRowsClose(actual[port], expected, 5e-3);
  });
}

for (const reflection of [1, -1])
  it(`W5-40 GPU negated cubic counts both endpoint maxima reflection=${reflection}`, async () => {
    const context = await createRegressionTestContext();
    try {
      const output = context.output('float32', 3);
      context.graph.add(
        createWGSLKernelNode(context.graph, {
          id: 'negated-cubic',
          operation: 'SearchReview',
          invocationCount: 1,
          bindings: [{name: 'output', view: output.view, type: 'f32', access: 'read_write'}],
          declarations: `${GET_NAN_WGSL}
fn getObjective(parameter:f32)->f32{let offset=1.0-(${reflection}.0)*parameter;return -offset*offset*(0.0009765625/2.0-offset/3.0);}
fn getDerivative(parameter:f32)->f32{let offset=1.0-(${reflection}.0)*parameter;return -(${reflection}.0)*offset*(offset-0.0009765625);}
${getParameterSearchWGSL()}`,
          body: 'let result=searchParameter(1.0);output[outputOffset]=result.parameter;output[outputOffset+1u]=f32(result.localMaximumCount);output[outputOffset+2u]=select(0.0,1.0,result.atBoundary);'
        })
      );
      context.run();
      expect(Array.from(await readFloat32(output.buffer, 3))).toEqual([-reflection, 2, 1]);
    } finally {
      context.destroy();
    }
  });
