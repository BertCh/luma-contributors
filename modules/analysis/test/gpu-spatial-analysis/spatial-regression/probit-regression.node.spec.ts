// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUProbitRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

function getProbitKernels(rows: number) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `probit-${rows}-${serial++}`, format, length);
    const nodes = new GPUProbitRegression({
      predictors: view('float32', rows * 2),
      predictorCount: 2,
      response: view('float32', rows),
      output: {
        coefficients: view('float32', 3),
        standardErrors: view('float32', 3),
        fitted: view('float32', rows),
        marginalEffects: view('float32', 2),
        summary: view('float32', 4),
        status: view('uint32', 1)
      }
    }).getCommandNodes(graph);
    getParsedKernelSources(nodes);
    return getGPUCommandNodeKernelSources(nodes);
  } finally {
    device.destroy();
  }
}

it('probit fit kernel passes the row count as a pipeline override, not baked into WGSL', () => {
  const small = getProbitKernels(40);
  const large = getProbitKernels(60000);
  expect(small.length).toBe(3);
  for (const [index, kernel] of small.entries()) {
    expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    if (kernel.id.endsWith('-fit') || kernel.id.endsWith('-effects')) {
      expect(kernel.constants.ROW_COUNT).toBe(40);
      expect(large[index].constants.ROW_COUNT).toBe(60000);
      // Only the override value differs: recompiling per row count is avoided.
      expect(large[index].source).toBe(kernel.source);
    }
  }
});

it('probit fit kernel sums the likelihood with compensation and damps overshoot before flagging separation', () => {
  const fit = getProbitKernels(40).find(kernel => kernel.id.endsWith('-fit'))!;
  // Compensated (Neumaier) log-likelihood and score sums.
  expect(fit.source).toContain('compensation');
  expect(fit.source).toContain('gradientCompensation');
  // The line search rejects unbounded trial steps by halving, instead of aborting on the first.
  expect(fit.source).toContain('lastLargestLinear <= 12.0');
  expect(fit.source).not.toContain('abs(linear) > 12.0');
  // Absolute 1e-6 acceptance tolerance is replaced by a likelihood-relative one.
  expect(fit.source).not.toContain('current - 1e-6');
  expect(fit.source).toContain('noiseScale * (abs(current) + 1.0)');
  // Separation is flagged only for unconverged fits pinned against the bound.
  expect(fit.source).toContain('code != 0u && largestLinear > 8.0');
});
