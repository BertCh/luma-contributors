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
    const probit = new GPUProbitRegression({
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
    });
    const nodes = probit.getCommandNodes(graph);
    expect(nodes.length).toBe(probit.estimatedCommandNodeCount);
    getParsedKernelSources(nodes);
    return getGPUCommandNodeKernelSources(nodes);
  } finally {
    device.destroy();
  }
}

// Behaviour change on purpose: the single-invocation fit kernel (which looped over every row in
// every iteration) is replaced by row tiles and unrolled iterations, so this pins the new shape.
it('probit row passes take the row count as a pipeline override and no single invocation loops over rows', () => {
  const small = getProbitKernels(40);
  const large = getProbitKernels(60000);
  expect(large.length).toBe(small.length);
  for (const [index, kernel] of small.entries()) {
    expect(auditWGSLModule(kernel.source, kernel.constants)).toEqual([]);
    if (kernel.id.endsWith('-tiles')) {
      expect(kernel.constants['ROW_COUNT']).toBe(40);
      expect(large[index].constants['ROW_COUNT']).toBe(60000);
      // Only the override value differs: recompiling per row count is avoided.
      expect(large[index].source).toBe(kernel.source);
    }
    // At 60,000 rows every row pass has many tiles; one-invocation kernels never see the rows.
    if (large[index].constants['INVOCATION_COUNT'] === 1) {
      expect(large[index].source, kernel.id).not.toContain('ROW_COUNT');
    }
  }
});

it('probit fit sums the likelihood with compensation and damps overshoot before flagging separation', () => {
  const source = getProbitKernels(40)
    .map(kernel => kernel.source)
    .join('\n');
  // Compensated (Neumaier) merges of the log-likelihood and score tile sums.
  expect(source).toContain('compensation');
  // The line search rejects unbounded trial steps by halving, instead of aborting on the first.
  expect(source).toContain('lastLargestLinear <= 12.0');
  expect(source).not.toContain('abs(linear) > 12.0');
  // Absolute 1e-6 acceptance tolerance is replaced by a likelihood-relative one.
  expect(source).not.toContain('current - 1e-6');
  expect(source).toContain('noiseScale * (abs(current) + 1.0)');
  // Separation is flagged only for unconverged fits pinned against the bound.
  expect(source).toContain('code != 0u && largestLinear > 8.0');
});
