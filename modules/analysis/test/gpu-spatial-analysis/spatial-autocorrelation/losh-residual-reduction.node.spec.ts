// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPULocalSpatialHeteroscedasticityParameterValues,
  GPULocalSpatialHeteroscedasticity
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('LOSH reduces the global residual mean without a serial row loop', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const rows = 8193;
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  try {
    const nodes = new GPULocalSpatialHeteroscedasticity({
      id: 'losh',
      values: view('float32', rows),
      weights: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows),
        weights: view('float32', rows)
      },
      output: {
        statistic: view('float32', rows),
        status: view('uint32', rows),
        normalizedResiduals: view('float32', rows)
      }
    }).getCommandNodes(graph);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    const mean = kernels.find(kernel => kernel.id === 'losh-mean')!;
    expect(mean).toBeDefined();
    expect(mean.source).not.toMatch(/\b(for|while|loop)\b/);
    expect(kernels.some(kernel => kernel.id.startsWith('losh-maximum-residual'))).toBe(true);
    expect(mean.source).toContain('normalizedSum');
    expect(mean.source).toContain('frexp(maximumResidual[maximumResidualOffset])');
    expect(mean.source).toContain('ldexp(ldexp(fraction, firstExponent)');
    const scale = kernels.find(kernel => kernel.id === 'losh-scale-residuals')!;
    const finish = kernels.find(kernel => kernel.id === 'losh-finish')!;
    const normalized = kernels.find(kernel => kernel.id === 'losh-normalized-residuals')!;
    expect(scale.source).toContain('getScaledQuotient(finiteResiduals[');
    expect(finish.source).toContain('getScaledQuotient(weighted, total, mean[');
    expect(normalized.source).toContain('getScaledQuotient(residual, average, 1.0, 0)');
  } finally {
    device.destroy();
  }
});

it('LOSH parameter packing accepts only the two documented powers', () => {
  expect([...getGPULocalSpatialHeteroscedasticityParameterValues()]).toEqual([2]);
  expect([...getGPULocalSpatialHeteroscedasticityParameterValues({residualPower: 1})]).toEqual([1]);
  for (const invalid of [0, 1.5, 3, NaN]) {
    expect(() =>
      getGPULocalSpatialHeteroscedasticityParameterValues({residualPower: invalid as 1})
    ).toThrow(/residualPower/);
  }
});

it('LOSH numeric powers and parameter views use the same compute pipelines', () => {
  const sources = (power: 1 | 2 | 'runtime') => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `view-${serial++}`, format, length);
    try {
      const nodes = new GPULocalSpatialHeteroscedasticity({
        values: view('float32', 8),
        weights: {
          offsets: view('uint32', 9),
          neighbors: view('uint32', 16),
          weights: view('float32', 16)
        },
        residualPower: power === 'runtime' ? view('float32', 1) : power,
        output: {statistic: view('float32', 8), status: view('uint32', 8)}
      }).getCommandNodes(graph);
      return getGPUCommandNodeKernelSources(nodes).map(kernel => kernel.source);
    } finally {
      device.destroy();
    }
  };
  expect(sources(1)).toEqual(sources(2));
  expect(sources(1)).toEqual(sources('runtime'));
});
