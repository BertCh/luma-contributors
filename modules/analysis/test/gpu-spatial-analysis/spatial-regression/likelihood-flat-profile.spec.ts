// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialErrorMaximumLikelihoodOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood-operation';
import {GPUSpatialErrorMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-maximum-likelihood';
import {GPUSpatialLagMaximumLikelihood} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-lag-maximum-likelihood';
import {getScalarLagExpected} from './likelihood-review-regressions-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../../../src/analysis-core/kernels/wgsl-common';
import {getSpatialErrorScalarCertificateWGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-scalar-profile';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import {
  fixtures,
  getFlatProfileInput,
  getFlatProfileProps
} from './likelihood-flat-profile-test-utils';

for (const fixture of fixtures) {
  it(`every GPU flat-profile port matches the closed form: Q=${fixture.maximumPower}, c=${fixture.scalarWeight}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const actual = await runOperationGraph(device, {
      inputs: getFlatProfileInput(fixture),
      operations: [new GPUSpatialErrorMaximumLikelihoodOperation(getFlatProfileProps(fixture))],
      outputs: Object.keys(fixture.expected)
    });
    for (const port of Object.keys(fixture.expected) as (keyof typeof fixture.expected)[]) {
      // Float32 filtered solve/conditioning and normal-tail evaluation: 0.5% relative.
      // NaN spatial inference and all zero flags compare exactly.
      expectRegressionRowsClose(
        actual[port],
        Float32Array.from(fixture.expected[port], value => value ?? NaN),
        5e-3
      );
    }
  });
}

for (const invalidEigenvalue of [NaN, Infinity]) {
  it(`a late non-finite eigenvalue cannot bypass rejection through the flat-profile certificate: ${invalidEigenvalue}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = fixtures[0];
    const input = getFlatProfileInput(fixture);
    input.eigenvalues.values[7] = invalidEigenvalue;
    const actual = await runOperationGraph(device, {
      inputs: input,
      operations: [new GPUSpatialErrorMaximumLikelihoodOperation(getFlatProfileProps(fixture))],
      outputs: Object.keys(fixture.expected)
    });
    expect(Array.from(actual['status'])).toEqual([3]);
    expect(Array.from(actual['converged'])).toEqual([0]);
    expect(actual['summary'][0]).toBe(8);
    for (const port of ['table', 'residuals']) expect(actual[port].every(Number.isNaN)).toBe(true);
    expect(actual['summary'].subarray(1).every(Number.isNaN)).toBe(true);
    expect(Array.from(actual['powerTraceStandardErrors'])).toEqual([0, 0, 0]);
  });
}

it('the GPU structural certificate retains both tiny and subnormal off-diagonal links', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const inputs = {
    offsets: Uint32Array.of(0, 2, 4, 5, 6, 7, 8, 9, 10),
    neighbors: Uint32Array.of(0, 1, 0, 1, 2, 3, 4, 5, 6, 7),
    weights: Float32Array.of(3, 2 ** -40, 2 ** -149, 3, 3, 3, 3, 3, 3, 3),
    eigenvalues: new Float32Array(8).fill(3)
  };
  const buffers = Object.fromEntries(
    Object.entries(inputs).map(([name, values]) => [name, createInputBuffer(device, values)])
  );
  const outputBuffer = createOutputBuffer(device, 8);
  const bindings = Object.entries(inputs).map(([name, values]) => ({
    name,
    view: importGraphBuffer(
      graph,
      name,
      buffers[name],
      values instanceof Uint32Array ? 'uint32' : 'float32',
      values.length
    ),
    type: values instanceof Uint32Array ? ('u32' as const) : ('f32' as const),
    access: 'read' as const
  }));
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'scalar-certificate',
      operation: 'scalar-certificate',
      variant: 'scalar-certificate',
      invocationCount: 8,
      bindings: [
        ...bindings,
        {
          name: 'scalarDifferences',
          view: importGraphBuffer(graph, 'differences', outputBuffer, 'float32', 8),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `const ROWS: u32 = 8u;\n${IS_FINITE_VALUE_WGSL}`,
      body: getSpatialErrorScalarCertificateWGSL()
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Both stored off-diagonal entries are nonzero, even though all eigenvalues round to 3.
    expectRegressionRowsClose(await readFloat32(outputBuffer, 8), [1, 1, 0, 0, 0, 0, 0, 0], 0);
  } finally {
    compiled.destroy();
    outputBuffer.destroy();
    for (const buffer of Object.values(buffers)) buffer.destroy();
  }
});

for (const [kind, Contributor] of [
  ['error', GPUSpatialErrorMaximumLikelihood],
  ['lag', GPUSpatialLagMaximumLikelihood]
] as const)
  it(`the standalone ${kind} contributor certifies a scalar identity across CSR chunk boundaries`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = fixtures[0];
    const expected = kind === 'lag' ? getScalarLagExpected(3) : fixture.expected;
    const graph = new GPUCommandGraph(device);
    const buffers: ReturnType<typeof createInputBuffer>[] = [];
    const uploadFloat = (name: string, values: Float32Array) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, 'float32', values.length);
    };
    const uploadUint = (name: string, values: Uint32Array) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, 'uint32', values.length);
    };
    const outputs = Object.fromEntries(
      Object.entries(expected).map(([name, values]) => {
        const buffer = createOutputBuffer(device, values.length);
        buffers.push(buffer);
        return [name, buffer];
      })
    );
    const chunks = [0, 3].map((rowStart, chunkIndex) => {
      const rowCount = chunkIndex === 0 ? 3 : 5;
      return {
        rowStart,
        rowCount,
        weights: {
          offsets: uploadUint(
            `offsets-${chunkIndex}`,
            Uint32Array.from({length: rowCount + 1}, (_, row) => row)
          ),
          neighbors: uploadUint(
            `neighbors-${chunkIndex}`,
            Uint32Array.from({length: rowCount}, (_, row) => rowStart + row)
          ),
          weights: uploadFloat(`weights-${chunkIndex}`, new Float32Array(rowCount).fill(3))
        }
      };
    });
    graph.add(
      new Contributor({
        weights: {rowCount: 8, chunks},
        predictorCount: 1,
        maximumPower: 3,
        logDeterminant: 'exact',
        tileRowCount: 2,
        predictors: uploadFloat('predictors', Float32Array.from(fixture.predictors)),
        response: uploadFloat('response', Float32Array.from(fixture.response)),
        eigenvalues: uploadFloat('eigenvalues', new Float32Array(8).fill(3)),
        output: {
          ...(kind === 'lag'
            ? {
                powerTraces: importGraphBuffer(
                  graph,
                  'powerTraces',
                  outputs['powerTraces'],
                  'float32',
                  3
                ),
                powerSums: importGraphBuffer(
                  graph,
                  'powerSums',
                  outputs['powerSums'],
                  'float32',
                  3
                ),
                infinityNorm: importGraphBuffer(
                  graph,
                  'infinityNorm',
                  outputs['infinityNorm'],
                  'float32',
                  1
                )
              }
            : {}),
          table: importGraphBuffer(graph, 'table', outputs['table'], 'float32', 12),
          summary: importGraphBuffer(graph, 'summary', outputs['summary'], 'float32', 22),
          residuals: importGraphBuffer(graph, 'residuals', outputs['residuals'], 'float32', 8),
          powerTraceStandardErrors: importGraphBuffer(
            graph,
            'errors',
            outputs['powerTraceStandardErrors'],
            'float32',
            3
          ),
          status: {
            code: importGraphBuffer(graph, 'status', outputs['status'], 'uint32', 1),
            converged: importGraphBuffer(graph, 'converged', outputs['converged'], 'uint32', 1)
          }
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      for (const [port, values] of Object.entries(expected)) {
        const actual =
          port === 'status' || port === 'converged'
            ? await readUint32(outputs[port], values.length)
            : await readFloat32(outputs[port], values.length);
        expectRegressionRowsClose(
          actual,
          Float32Array.from(values, value => value ?? NaN),
          5e-3
        );
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
