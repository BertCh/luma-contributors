// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createBipartiteDiagnosticsInput,
  createWeightedConstantLagInput,
  createTranslatedAffineInput
} from './regression-numerics-w5-fixtures';
import {computeSpatialRegressionDiagnosticsOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation';
import {computeSpatialErrorGMOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-error-gm-operation';

it('W5-30 projected K8,8 operator has exactly zero variance and unavailable inference', () => {
  const result = computeSpatialRegressionDiagnosticsOnCPU(createBipartiteDiagnosticsInput(), {
    predictorCount: 1
  });
  expect(Array.from(result.status)).toEqual([0]);
  expect(result.summary[7]).toBe(0);
  for (const slot of [8, 9]) expect(result.summary[slot]).toBeNaN();
  for (const slot of [15, 17]) expect(result.tests[slot]).toBeNaN();
});

it('W5-31 weighted constant fitted lag has J=35 and unavailable robust LM', () => {
  const result = computeSpatialRegressionDiagnosticsOnCPU(createWeightedConstantLagInput(), {
    predictorCount: 1
  });
  expect(Array.from(result.status)).toEqual([0]);
  expect(result.summary[3]).toBe(35);
  for (const row of [2, 3, 4]) expect(result.tests[3 * row]).toBeNaN();
});

for (const estimator of ['homoskedastic', 'kelejian-prucha'] as const)
  it(`W5-32 ${estimator} distinguishes translated exact affine fit from one ULP`, () => {
    for (const departure of [false, true]) {
      const result = computeSpatialErrorGMOnCPU(createTranslatedAffineInput(departure), {
        predictorCount: 1,
        estimator
      });
      expect(Array.from(result.status)).toEqual([departure ? 0 : 3]);
    }
  });

it('W5-32 exact anchored shader identity retains the unrepresentable intercept numerator', async () => {
  const {GPUCommandGraph, createTransientView} = await import('@luma.gl/graph/gpu-core');
  const {createNullWebGPUDevice} = await import('../../utils/gpu-contributor-test-utils');
  const {createWGSLKernelNode} = await import(
    '../../../src/analysis-core/kernels/wgsl-kernel-nodes'
  );
  const {getSpatialErrorGMAffineEqualityWGSL} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/spatial-error-gm-affine-certificate'
  );
  const {createCPUGraphMemory, executeCommandNodesOnCPU} = await import(
    '../segment-intersection/wgsl-graph-executor'
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const design = createTransientView(graph, 'affine-design', 'float32', 12);
    const response = createTransientView(graph, 'affine-response', 'float32', 12);
    const candidate = createTransientView(graph, 'candidate', 'float32', 3);
    const output = createTransientView(graph, 'certificate', 'uint32', 12);
    const node = createWGSLKernelNode(graph, {
      id: 'affine-verify',
      operation: 'AffineTest',
      invocationCount: 12,
      bindings: [
        {name: 'design', view: design, type: 'u32', access: 'read'},
        {name: 'response', view: response, type: 'u32', access: 'read'},
        {name: 'candidate', view: candidate, type: 'u32', access: 'read'},
        {name: 'output', view: output, type: 'u32', access: 'read_write'}
      ],
      declarations: `const K: u32=1u; const Q: u32=2u;\n${getSpatialErrorGMAffineEqualityWGSL()}`,
      body: 'output[outputOffset + index] = select(0u, 1u, isExactAffineRow(index, 0u));'
    });
    const memory = createCPUGraphMemory();
    memory.write(candidate, Float32Array.of(16777216, 1, 7));
    for (const departure of [false, true]) {
      const input = createTranslatedAffineInput(departure);
      memory.write(design, input.predictors);
      memory.write(response, input.response);
      executeCommandNodesOnCPU([node], memory);
      expect(memory.readUint32(output).every(value => value === 1)).toBe(!departure);
    }
  } finally {
    device.destroy();
  }
});

it('W5-31/CCK-01 actual GPU reference and row residual kernels center contributions before sums', async () => {
  const {GPUCommandGraph, createTransientView} = await import('@luma.gl/graph/gpu-core');
  const {createNullWebGPUDevice} = await import('../../utils/gpu-contributor-test-utils');
  const {GPUSpatialRegressionDiagnostics} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics'
  );
  const {createCPUGraphMemory, executeCommandNodesOnCPU} = await import(
    '../segment-intersection/wgsl-graph-executor'
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `lag-${serial++}`, format, length);
    const input = createWeightedConstantLagInput();
    const weights = {
      offsets: view('uint32', 65),
      neighbors: view('uint32', input.weights.neighbors.length),
      weights: view('float32', input.weights.weights.length)
    };
    const contributor = new GPUSpatialRegressionDiagnostics({
      weights,
      predictorCount: 1,
      predictors: view('float32', 64),
      response: view('float32', 64),
      residuals: view('float32', 64),
      output: {
        tests: view('float32', 18),
        summary: view('float32', 13),
        status: {code: view('uint32', 1)}
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    const resources = nodes
      .flatMap(node => node.resources ?? [])
      .flatMap(resource =>
        'buffer' in resource
          ? [resource.buffer as import('@luma.gl/graph/gpu-core').GraphDataView]
          : []
      );
    const named = (suffix: string) =>
      resources.find(resource => resource.buffer.id.endsWith(suffix))!;
    const table = named('-row-table');
    const origin = named('-observed-origin');
    const memory = createCPUGraphMemory();
    memory.write(weights.offsets, input.weights.offsets);
    memory.write(weights.neighbors, input.weights.neighbors);
    memory.write(weights.weights, input.weights.weights);
    memory.write(
      table,
      Float32Array.from({length: 256}, (_, slot) => {
        const row = Math.floor(slot / 4);
        return [1, input.predictors[row], input.response[row], input.residuals[row]][slot % 4];
      })
    );
    memory.write(origin, Float32Array.of(0, 64, 0, 0, 0));
    memory.write(named('-fitted-lag-projection'), new Float32Array(3));
    const referenceNodes = nodes.filter(node => node.id.includes('-observed-origin'));
    executeCommandNodesOnCPU(referenceNodes, memory);
    expect(Array.from(memory.readFloat32(origin))).toEqual([0, 0, 33 / 64, 4096 + 1 / 256, 1]);
    executeCommandNodesOnCPU(
      nodes.filter(node => node.id.endsWith('-fitted-lag-residuals')),
      memory
    );
    expect(Array.from(memory.readFloat32(named('-fitted-lag-residual-squares')))).toEqual(
      new Array(64).fill(0)
    );
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    // The first selected row may have no usable neighbors; neither a stale link nor lag survives.
    const emptyFirst = input.weights.offsets.slice();
    emptyFirst[1] = 0;
    memory.write(weights.offsets, emptyFirst);
    memory.write(origin, Float32Array.of(0, 64, 9, 9, 9));
    executeCommandNodesOnCPU(referenceNodes, memory);
    expect(Array.from(memory.readFloat32(origin))).toEqual([0, 0, 0, 4096 + 1 / 256, 0]);
    memory.write(origin, Float32Array.of(0, 0, 9, 9, 9));
    executeCommandNodesOnCPU(referenceNodes, memory);
    expect(Array.from(memory.readFloat32(origin))).toEqual([0, 0, 0, 0, 0]);
  } finally {
    device.destroy();
  }
});

for (const [increment, variance] of [
  [2 ** -20, 8.7005227e-16],
  [1 / 32, 9.3212977e-7]
])
  it(`W5-30 retains resolved nearly-complete variation ${increment}`, async () => {
    const {createCompleteReviewInput} = await import('./spatial-regression-review-fixtures');
    const result = computeSpatialRegressionDiagnosticsOnCPU(
      createCompleteReviewInput(8, increment),
      {predictorCount: 1}
    );
    expect(Number.isFinite(result.summary[7])).toBe(true);
    expect(Math.abs(result.summary[7] / variance - 1)).toBeLessThan(1e-6);
    expect(result.summary[8]).toBeCloseTo(1.15470054, 6);
    expect(result.summary[9]).toBeCloseTo(0.24821308, 6);
  });

it('W5-33 actual moment kernels distinguish constant and one-ULP columns', async () => {
  const {GPUCommandGraph, createTransientView} = await import('@luma.gl/graph/gpu-core');
  const {createNullWebGPUDevice} = await import('../../utils/gpu-contributor-test-utils');
  const {GPUInstrumentalVariablesCovariance} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/gpu-instrumental-variables-covariance'
  );
  const {createCPUGraphMemory, executeCommandNodesOnCPU} = await import(
    '../segment-intersection/wgsl-graph-executor'
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `rank-${serial++}`, format, length);
    const column = view('float32', 512),
      residuals = view('float32', 512),
      status = view('uint32', 1);
    const contributor = new GPUInstrumentalVariablesCovariance({
      design: column,
      instruments: column,
      residuals,
      coefficientCount: 1,
      instrumentCount: 1,
      output: {covariance: view('float32', 4), status: {code: status}}
    });
    const nodes = contributor
      .getCommandNodes(graph)
      .filter(node =>
        /-(centering-zero|sums-tiles|sums-merge|means|moments-tiles|moments-merge)$/.test(node.id)
      );
    for (const departure of [false, true]) {
      const values = new Float32Array(512).fill(Math.fround(1.9793816));
      if (departure) values[511] += 2 ** -23;
      const memory = createCPUGraphMemory();
      memory.write(column, values);
      memory.write(residuals, new Float32Array(512));
      executeCommandNodesOnCPU(nodes, memory);
      const workspace = nodes
        .flatMap(node => node.resources ?? [])
        .flatMap(resource =>
          'buffer' in resource
            ? [resource.buffer as import('@luma.gl/graph/gpu-core').GraphDataView]
            : []
        )
        .find(view => view.buffer.id.endsWith('-workspace'))!;
      const moments = memory.readFloat32(workspace);
      const centeredSquares = moments[3] - moments[7] * (moments[7] / 512);
      expect(Number.isFinite(centeredSquares)).toBe(true);
      if (departure)
        expect(Math.abs(centeredSquares / ((511 / 512) * 2 ** -46) - 1)).toBeLessThan(1e-6);
      else expect(centeredSquares).toBe(0);
    }
  } finally {
    device.destroy();
  }
});
