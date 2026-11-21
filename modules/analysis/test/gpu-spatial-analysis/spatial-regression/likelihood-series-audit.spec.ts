// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialWeightsTracesOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces-operation';
import {
  GPUSpatialLagImpactsOperation,
  type SpatialLagImpactsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-impacts-operation';
import {getSpatialSeriesTailWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-series-tail';
import {getTraceSeriesWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-maximum-likelihood-wgsl';
import {getSpatialLogDeterminantWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-weights-traces-parameters';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import fixture from './likelihood-series-audit-fixture.json';

for (const mode of ['exact', 'stochastic'] as const) {
  it(`every GPU ${mode} trace output matches dense NumPy, including Gram moments and sampling errors`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const expected = fixture.traces[mode];
    const impactProps: SpatialLagImpactsOperationProps = {
      coefficients: 'coefficients',
      coefficientStride: 1,
      rhoIndex: 3,
      predictorCount: 2,
      rowCount: 8,
      maximumPower: 8,
      traces: {powerTraces: 'powerTraces', powerSums: 'powerSums', infinityNorm: 'infinityNorm'},
      output: {impacts: 'impacts'}
    };
    const impactOperation =
      mode === 'exact' ? new GPUSpatialLagImpactsOperation(impactProps) : undefined;
    // The numerical oracle still requires two predictors and all eight powers after mutation.
    impactProps.maximumPower = 1;
    impactProps.predictorCount = 1;
    impactProps.output.impacts = 'changed';
    const result = await runOperationGraph(device, {
      inputs: {
        offsets: {format: 'uint32', values: Uint32Array.from(fixture.weights.offsets)},
        neighbors: {format: 'uint32', values: Uint32Array.from(fixture.weights.neighbors)},
        weights: {format: 'float32', values: Float32Array.from(fixture.weights.weights)},
        parameters: {format: 'uint32', values: Uint32Array.of(17)},
        coefficients: {format: 'float32', values: Float32Array.of(99, 2, -3, 0.3)}
      },
      operations: [
        new GPUSpatialWeightsTracesOperation({
          rowCount: 8,
          maximumPower: 8,
          mode,
          probeCount: 9,
          probeBatchSize: 4,
          tileRowCount: 3,
          includeGramTraces: true,
          weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
          parameters: 'parameters',
          output: {
            powerTraces: 'powerTraces',
            powerSums: 'powerSums',
            powerTraceStandardErrors: 'powerTraceStandardErrors',
            gramTraces: 'gramTraces',
            infinityNorm: 'infinityNorm',
            status: {approximation: 'approximation'}
          }
        }),
        ...(impactOperation ? [impactOperation] : [])
      ],
      outputs: [...Object.keys(expected), ...(mode === 'exact' ? ['impacts'] : [])]
    });
    // Eight float32 matvecs and short pairwise sums: 2e-4 relative accommodates rounding.
    for (const port of Object.keys(expected) as (keyof typeof expected)[]) {
      expectRegressionRowsClose(result[port], expected[port], port === 'approximation' ? 0 : 2e-4);
    }
    if (mode === 'exact') expectRegressionRowsClose(result['impacts'], fixture.impacts, 3e-6);
  });
}

it('GPU log determinant and both derivative tails match dense algebra near both interval ends', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const references = fixture.cases.filter(value => value.maximumPower === 128);
  const graph = new GPUCommandGraph(device);
  const parameterBuffer = createInputBuffer(
    device,
    Float32Array.from(references, value => value.parameter)
  );
  const traceBuffer = createInputBuffer(device, Float32Array.from(fixture.powerTraces));
  const resultBuffer = createOutputBuffer(device, references.length * 6);
  const parameters = importGraphBuffer(
    graph,
    'parameters',
    parameterBuffer,
    'float32',
    references.length
  );
  const traces = importGraphBuffer(graph, 'traces', traceBuffer, 'float32', 128);
  const output = importGraphBuffer(graph, 'output', resultBuffer, 'float32', references.length * 6);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'series-oracle',
      operation: 'series-oracle',
      variant: 'series-oracle',
      invocationCount: references.length,
      bindings: [
        {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
        {name: 'traces', view: traces, type: 'f32', access: 'read'},
        {name: 'output', view: output, type: 'f32', access: 'read_write'}
      ],
      declarations:
        getSpatialLogDeterminantWGSL(128, 'traces') +
        getTraceSeriesWGSL(128, 'traces') +
        getSpatialSeriesTailWGSL(128, 'traces'),
      body: `let parameter = parameters[parametersOffset + index];
  output[outputOffset + index * 6u] = getSpatialLogDeterminant(parameter);
  output[outputOffset + index * 6u + 1u] = getTraceOne(parameter);
  output[outputOffset + index * 6u + 2u] = getTraceTwo(parameter);
  output[outputOffset + index * 6u + 3u] = getSeriesTailLogDeterminant(parameter);
  output[outputOffset + index * 6u + 4u] = getSeriesTailDerivative(parameter);
  output[outputOffset + index * 6u + 5u] = getSeriesTailSecond(parameter);`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(resultBuffer, references.length * 6);
    const expected = references.flatMap(value => [
      value.truncated,
      value.traceOne,
      value.traceTwo,
      value.exact - value.truncated,
      value.derivativeTail,
      value.secondTail
    ]);
    for (let row = 0; row < expected.length; row++) {
      expect(Number.isFinite(actual[row])).toBe(true);
      // Float32 rho near +/-0.99 amplifies rounding by (1-|rho|)^-2 in the second trace.
      expect(Math.abs(actual[row] - expected[row])).toBeLessThanOrEqual(
        2e-4 * Math.max(1, Math.abs(expected[row]))
      );
    }
  } finally {
    compiled.destroy();
    parameterBuffer.destroy();
    traceBuffer.destroy();
    resultBuffer.destroy();
  }
});
