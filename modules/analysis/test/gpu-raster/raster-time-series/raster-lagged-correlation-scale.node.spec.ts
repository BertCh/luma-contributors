// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {WgslExec, WgslParser, WgslReflect} from 'wgsl_reflect';
import {
  GPURasterLaggedCorrelation,
  getGPURasterLaggedCorrelationParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-lagged-correlation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

/** Generates the kernel for one cell at lag 0 and executes its WGSL on the CPU. */
function correlate(indexSeries: readonly number[], series: readonly number[]) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const sliceCount = series.length;
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    const nodes = new GPURasterLaggedCorrelation({
      cellCount: 1,
      sliceCount,
      values: view('values', 'float32', sliceCount),
      referenceIndex: view('referenceIndex', 'float32', sliceCount),
      parameters: view('parameters', 'float32', 1),
      lags: [0],
      output: {
        correlations: view('correlations', 'float32', 1),
        status: view('status', 'uint32', 1),
        pairCount: view('pairCount', 'uint32', 1)
      }
    }).getCommandNodes(graph);
    const buffers: Record<string, Float32Array | Uint32Array> = {
      values: new Float32Array(series),
      referenceIndex: new Float32Array(indexSeries),
      parameters: getGPURasterLaggedCorrelationParameterValues(),
      correlations: new Float32Array([37]),
      status: new Uint32Array([37]),
      pairCount: new Uint32Array([37])
    };
    // Generated kernels assign to `_` to keep bindings live; the interpreter logs those, harmlessly.
    const errors = vi.spyOn(console, 'error').mockImplementation(message => {
      if (!String(message).startsWith('Variable _ not found.')) throw new Error(String(message));
    });
    try {
      const kernels = getGPUCommandNodeKernelSources(nodes);
      expect(kernels.length).toBe(1);
      for (const kernel of kernels) {
        const bindings = Object.fromEntries(
          new WgslReflect(kernel.source).storage.map(declaration => {
            const buffer = buffers[declaration.name] ?? new Uint32Array(4);
            return [declaration.binding, buffer];
          })
        );
        new WgslExec(new WgslParser().parse(kernel.source)).dispatchWorkgroups(
          'main',
          [1, 1, 1],
          {0: bindings},
          {constants: kernel.constants}
        );
      }
    } finally {
      errors.mockRestore();
    }
    return {
      correlation: buffers['correlations'][0],
      status: buffers['status'][0],
      pairCount: buffers['pairCount'][0]
    };
  } finally {
    device.destroy();
  }
}

const correlateWithItself = (series: readonly number[]) => correlate(series, series);

it.each([
  {name: 'RT-132 magnitude 1e-20', series: [1e-20, 2e-20, 3e-20]},
  {name: 'RT-132 magnitude 1', series: [1, 2, 3]},
  {name: 'RT-132 magnitude 1e10', series: [1e10, 2e10, 3e10]},
  {name: 'RT-132 magnitude 1e20', series: [1e20, 2e20, 3e20]},
  {name: 'RT-133 translated by 1e6', series: [1000000, 1000001, 1000002]}
])('$name: a series correlates 1 with itself', ({series}) => {
  // A series is perfectly correlated with itself (Pearson r = 1 by definition), with three pairs.
  const result = correlateWithItself(series);
  expect(result.status).toBe(0);
  expect(result.pairCount).toBe(3);
  expect(Number.isFinite(result.correlation)).toBe(true);
  expect(Math.abs(result.correlation - 1)).toBeLessThan(1e-6);
});

it('RT-133 a constant series still reports zero variance', () => {
  const result = correlateWithItself([1000000, 1000000, 1000000]);
  expect(result.status).toBe(2);
  expect(Number.isNaN(result.correlation)).toBe(true);
});

it('RT-133 a translated index keeps its NumPy correlation with a different series', () => {
  // /usr/bin/python3: np.corrcoef([0, 1, 2, 4], [1, 3, 2, 5])[0, 1] = 0.8857142857142857;
  // Pearson correlation is invariant to translating either series.
  const result = correlate([1000000, 1000001, 1000002, 1000004], [1, 3, 2, 5]);
  expect(result.status).toBe(0);
  expect(result.pairCount).toBe(4);
  expect(Number.isFinite(result.correlation)).toBe(true);
  expect(Math.abs(result.correlation - 0.8857142857142857)).toBeLessThan(1e-6);
});
