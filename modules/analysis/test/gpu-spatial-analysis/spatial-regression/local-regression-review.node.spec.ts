// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {LOCAL_REGRESSION_INTERCEPT_WGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/local-regression-intercept-wgsl';
import {GTWR_SUMMARY_PUBLICATION_WGSL} from '../../../src/gpu-spatial-analysis/spatial-regression/gwr-summary-wgsl';

it('retains a representable 1/64 intercept after cancellation of million-scale origins', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const output = createTransientView(graph, 'intercept', 'float32', 1);
    const node = createWGSLKernelNode(graph, {
      id: 'intercept',
      operation: 'InterceptTest',
      invocationCount: 1,
      bindings: [{name: 'output', view: output, type: 'f32', access: 'read_write'}],
      declarations:
        LOCAL_REGRESSION_INTERCEPT_WGSL +
        '\nfn interpreterExponentProbe() { let unused = frexp(1.0); }',
      body: 'let state = subtractLocalRegressionInterceptProduct(vec2f(1000000.0, 0.015625), 1.0, 1000000.0); output[outputOffset] = state.x + state.y;'
    });
    const memory = createCPUGraphMemory();
    executeCommandNodesOnCPU([node], memory);
    expect(Array.from(memory.readFloat32(output))).toEqual([1 / 64]);
  } finally {
    device.destroy();
  }
});

it('publishes genuine exact-fit infinities explicitly rather than evaluating log zero', () => {
  expect(GTWR_SUMMARY_PUBLICATION_WGSL).toContain('0x7f800000u');
  expect(GTWR_SUMMARY_PUBLICATION_WGSL).toContain('0xff800000u');
});

it('GTWR does not mistake rounded fitted responses for an exact fit', async () => {
  const {GPUGeographicallyTemporallyWeightedRegression} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression'
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const output = {
      coefficients: view('coefficients', 'float32', 16),
      fitted: view('fitted', 'float32', 8),
      hatDiagonal: view('hat', 'float32', 8),
      localStatus: view('status', 'uint32', 8),
      residuals: view('residuals', 'float32', 8),
      summary: view('summary', 'float32', 12)
    };
    const response = view('response', 'float32', 8);
    const contributor = new GPUGeographicallyTemporallyWeightedRegression({
      positions: view('positions', 'float32x2', 8),
      times: view('times', 'float32', 8),
      temporalContext: {
        unit: 'second',
        epoch: 0n,
        encoding: 'float32-relative',
        axis: {kind: 'irregular'}
      },
      predictors: view('predictors', 'float32', 8),
      predictorCount: 1,
      response,
      parameters: view('parameters', 'float32', 4),
      kernel: 'gaussian',
      searchStructure: 'brute-force',
      output
    });
    const nodes = contributor.getCommandNodes(graph);
    const statistic =
      getGPUCommandNodeKernelSources(nodes).find(
        kernel => kernel.id === 'geographically-temporally-weighted-regression-statistics-partial'
      ) ??
      getGPUCommandNodeKernelSources(nodes).find(kernel =>
        kernel.id.endsWith('-statistics-partial')
      );
    expect(statistic?.source).toContain('let residual = residuals[residualsOffset + row]');
    // The integration fixture is also queued on device; publication is executed here with
    // independently reduced RSS so the sequential interpreter need not emulate barriers.
    const publication = createWGSLKernelNode(graph, {
      id: 'publication',
      operation: 'PrecisionTest',
      invocationCount: 1,
      bindings: [{name: 'summary', view: output.summary, type: 'f32', access: 'read_write'}],
      declarations: (await import('../../../src/analysis-core/kernels/wgsl-common')).GET_NAN_WGSL,
      body: `let count = 8.0; let included = 8.0; let singular = 0.0; let trace = 2.0;
      let residualSquares = ${5 / 11008}; let totalSquares = 0.0078125;
      ${GTWR_SUMMARY_PUBLICATION_WGSL}`
    });
    const memory = createCPUGraphMemory();
    executeCommandNodesOnCPU([publication], memory);
    const summary = memory.readFloat32(output.summary);
    expect(Number.isFinite(summary[0])).toBe(true);
    expect(summary[0]).toBeCloseTo(5 / 11008, 8);
    expect(summary[10]).toBeCloseTo(27.7540165, 4);
  } finally {
    device.destroy();
  }
});

it('shares every summary slot with the public model layout', async () => {
  const slots = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/gwr-summary-wgsl'
  );
  const {GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-parameters'
  );
  for (const [name, index] of Object.entries(GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY)) {
    expect((slots as unknown as Record<string, unknown>)[`GWR_SUMMARY_${name}`]).toBe(index);
  }
});

it('gates every GTWR search summary reduction with its own dispatch shape and budgets actual nodes', async () => {
  const {GPUGeographicallyTemporallyWeightedRegression} = await import(
    '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-temporally-weighted-regression'
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `search-shape-${serial++}`, format, length);
    const contributor = new GPUGeographicallyTemporallyWeightedRegression({
      positions: view('float32x2', 8),
      times: view('float32', 8),
      temporalContext: {
        unit: 'second',
        epoch: 0n,
        encoding: 'float32-relative',
        axis: {kind: 'irregular'}
      },
      predictors: view('float32', 8),
      predictorCount: 1,
      response: view('float32', 8),
      parameters: view('float32', 4),
      kernel: 'gaussian',
      bandwidthSearch: {parameters: view('float32', 6), maximumIterations: 1},
      output: {
        coefficients: view('float32', 16),
        fitted: view('float32', 8),
        hatDiagonal: view('float32', 8),
        localStatus: view('uint32', 8),
        summary: view('float32', 12)
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBe(contributor.estimatedCommandNodeCount);
    const summaries = nodes.filter(
      node => node.id.includes('evaluate-') && /-(bounds|partial|fold|summary)$/.test(node.id)
    );
    expect(summaries.length).toBeGreaterThan(0);
    for (const node of summaries)
      expect(node.condition, node.id).toMatchObject({source: 'gpu', mode: 'indirect'});
  } finally {
    device.destroy();
  }
});
