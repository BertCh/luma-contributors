// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUMantelTestOperation} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation';
import {CPUMantelTestOperation} from '../../../src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation';
import {constructNetworkOperationContributor} from '../../utils/network-operation-test-utils';
import {getPermutationRejectionFloor} from '../../../src/gpu-spatial-analysis/multiple-testing/permutation-p-value-floor';
import {AnalysisValidationError} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUNearRepeatKnox} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-near-repeat-knox';
import {DRAW_LOCAL_EXPECTED} from './review-round1-fixture';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {computeMantelTestOnCPU} from '../../../src/gpu-spatial-analysis/space-time-tests/cpu-mantel-test-operation';
import {GPUMantelTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test';
import {
  decodeGPUMantelTestSummary,
  decodeGPUSpaceTimeInferenceMetadata,
  getGPUSpaceTimeInferenceMetadata,
  getGPUSpaceTimeParameterValues
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

const input = {
  offsets: Uint32Array.of(0, 0, 0, 2),
  neighbors: Uint32Array.of(0, 1),
  weights: Float32Array.of(1, 1),
  distances: Float32Array.of(2, 4),
  times: Float32Array.of(3, 6, 0),
  parameters: getGPUSpaceTimeParameterValues({seed: 27, permutationCount: 19})
};

it('B1 records valid fixed and stopping WGSL, including runtime NaNs', () => {
  for (const stopping of [false, true]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `audit-${serial++}`, format, length);
    try {
      const contributor = new GPUMantelTest({
        pairs: {
          offsets: view('uint32', 4),
          neighbors: view('uint32', 3),
          weights: view('float32', 3),
          distances: view('float32', 3)
        },
        times: view('float32', 3),
        parameters: view('uint32', 4),
        maximumPermutations: 19,
        ...(stopping ? {stopAfterExceedances: 2} : {}),
        output: {
          statistics: view('float32', 20),
          summary: view('float32', 12),
          permutationCount: view('uint32', 1)
        }
      });
      const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      expect(kernels.length).toBeGreaterThan(0);
      for (const kernel of kernels)
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    } finally {
      device.destroy();
    }
  }
});

it('B3 preserves the attempted prefix and exposes a separate usable count', () => {
  // Hand Pearson signs for seed 27's first six independently generated permutations:
  // [-1,undefined,1,-1,undefined,1]. The second upper tie occurs at slot 6.
  const result = computeMantelTestOnCPU(input, {maximumPermutations: 19, stopAfterExceedances: 2});
  expect([...result.statistics.slice(1, 7)]).toEqual([-1, NaN, 1, -1, NaN, 1]);
  expect(result.permutationCount[0]).toBe(6);
  expect(result.usablePermutationCount[0]).toBe(4);
  expect(result.summary[8]).toBe(0.5); // h / four usable draws
});

it('B4 keeps data-dependent inference counts out of settings metadata', () => {
  const metadata = getGPUSpaceTimeInferenceMetadata(
    'mantel',
    {seed: 27, permutationCount: 19},
    19,
    undefined,
    'pseudoPTwoSided'
  );
  expect(metadata.permutationCount).toBeUndefined();
  expect(metadata).toMatchObject({
    requestedPermutationCount: 19,
    inferenceCountDefinition: 'usable-correlations'
  });
  const decoded = decodeGPUSpaceTimeInferenceMetadata(metadata, 19, 11);
  expect(decoded).toMatchObject({
    permutationCount: 19,
    usablePermutationCount: 11,
    inference: {permutationCount: 11}
  });
  expect(decoded.inference.pValueConstruction).toBe('doubled-minimum-tail');
  expect(2 / (decoded.inference.permutationCount + 1)).toBe(1 / 6);
  expect(
    getPermutationRejectionFloor({
      permutationCount: decoded.inference.permutationCount,
      alternative: decoded.inference.alternative,
      pValueConstruction: decoded.inference.pValueConstruction,
      rowCount: 1,
      alpha: 0.05
    }).pValueFloor
  ).toBe(1 / 6);
});

it('S1 rejects unsupported typed summary reads', () => {
  const words = Float32Array.of(1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  for (const unsupported of [
    new Int8Array(words.buffer),
    new Uint32Array(words.buffer),
    new Float64Array(12)
  ])
    expect(() => decodeGPUMantelTestSummary(unsupported as never)).toThrow(
      /Float32Array.*Uint8Array.*ArrayBuffer/
    );
});

it('UNFINISHED W2-14 checks temporal metadata in the standalone twin', () => {
  const options = {
    maximumPermutations: 19,
    temporalContext: {
      unit: 'second' as const,
      epoch: 0n,
      encoding: 'int64-words' as const,
      axis: {kind: 'irregular' as const}
    }
  };
  expect(() => computeMantelTestOnCPU(input, options)).toThrow(
    /mantel-test.*int64-words.*float32-relative/
  );
});

it('B2 pins every independently derived correlation and 19 usable null draws', () => {
  const result = computeMantelTestOnCPU(
    {
      offsets: Uint32Array.of(0, 3, 3, 3, 3),
      neighbors: Uint32Array.of(1, 2, 3),
      weights: Float32Array.of(1, 1, 1),
      distances: Float32Array.of(1, 2, 3),
      times: Float32Array.of(65536, 65537, 0, 65538),
      parameters: getGPUSpaceTimeParameterValues({seed: 3, permutationCount: 19})
    },
    {maximumPermutations: 19}
  );
  expect(result.statistics).toHaveLength(20);
  for (const [slot, expected] of DRAW_LOCAL_EXPECTED.entries()) {
    expect(Number.isFinite(result.statistics[slot])).toBe(true);
    expect(Math.abs(result.statistics[slot] - expected)).toBeLessThan(2e-6);
  }
  expect([...result.permutationCount]).toEqual([19]);
  expect([...result.usablePermutationCount]).toEqual([19]);
});

it('UNFINISHED W2-14 near-repeat structural refusals use named validation errors', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `named-knox-${serial++}`, format, length);
  const props = {
    id: 'named-knox',
    pairs: {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 3),
      weights: view('float32', 3),
      distances: view('float32', 3)
    },
    times: view('float32', 3),
    parameters: view('uint32', 4),
    maximumPermutations: 19,
    distanceEdges: [0, 3],
    timeEdges: [0, 3],
    output: {counts: view('uint32', 20), summary: view('float32', 8)}
  };
  try {
    for (const invalid of [
      {...props, times: view('float32', 2)},
      {...props, parameters: view('uint32', 3)},
      {...props, output: {...props.output, counts: view('uint32', 1)}}
    ]) {
      expect(() => new GPUNearRepeatKnox(invalid)).toThrow(AnalysisValidationError);
    }
  } finally {
    device.destroy();
  }
});

it('B3 operation forms capture and publish separate attempted and usable counts', async () => {
  const props = {
    rowCount: 3,
    pairs: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights', distances: 'distances'},
    times: 'times',
    parameters: 'parameters',
    maximumPermutations: 19,
    stopAfterExceedances: 2,
    output: {
      statistics: 'statistics',
      summary: 'summary',
      permutationCount: 'attempted',
      usablePermutationCount: 'usable'
    }
  };
  const gpu = new GPUMantelTestOperation(props);
  const cpu = new CPUMantelTestOperation(props);
  props.output.usablePermutationCount = 'replacement';
  expect(gpu.outputs['usablePermutationCount']).toMatchObject({
    value: 'usable',
    format: 'uint32',
    length: 1
  });
  const contributor = constructNetworkOperationContributor(gpu, {
    offsets: {format: 'uint32', length: 4},
    neighbors: {format: 'uint32', length: 2},
    weights: {format: 'float32', length: 2},
    distances: {format: 'float32', length: 2},
    times: {format: 'float32', length: 3},
    parameters: {format: 'uint32', length: 4}
  }) as GPUMantelTest;
  expect(contributor.props.output.usablePermutationCount?.format).toBe('uint32');
  const vector = (values: Float32Array | Uint32Array) => ({
    format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const result = await cpu.execute({
    inputs: Object.fromEntries(
      Object.entries(input).map(([name, values]) => [name, vector(values)])
    )
  });
  const attempted = result['attempted'];
  const usable = result['usable'];
  if (!(attempted instanceof Uint32Array) || !(usable instanceof Uint32Array))
    throw new Error('expected packed uint32 counters');
  expect([...attempted]).toEqual([6]);
  expect([...usable]).toEqual([4]);
  const fixed = computeMantelTestOnCPU(input, {maximumPermutations: 19});
  expect([...fixed.permutationCount]).toEqual([19]);
  expect([...fixed.usablePermutationCount]).toEqual([11]);
});
