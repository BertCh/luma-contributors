// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUCategoryPairCounts,
  GPUSpatialEntropy
} from '../../../src/gpu-spatial-analysis/spatial-entropy';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {executeSpatialAuditKernel} from '../spatial-validation/spatial-design-audit.test-utils';

it.each([
  'overflow',
  'candidateOverflow',
  'requiredCount'
] as const)('AMG-59 refuses even a zero-valued %s port', name => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `contract-${serial++}`, format, length);
  try {
    const complete = {
      offsets: view('uint32', 3),
      neighbors: view('uint32', 2),
      weights: view('float32', 2)
    };
    const weights = {...complete, status: {[name]: view('uint32', 1)}};
    const categories = view('uint32', 2);
    const output = {pairCounts: view('uint32', 4), categoryRowCounts: view('uint32', 2)};
    expect(
      () => new GPUCategoryPairCounts({weights, categories, categoryCount: 2, output})
    ).toThrow(/complete CSR/);
    expect(
      () =>
        new GPUSpatialEntropy({
          weights,
          categories,
          categoryCount: 2,
          output: {summary: view('float32', 8)}
        })
    ).toThrow(/complete CSR/);
    const nodes = new GPUCategoryPairCounts({
      weights: complete,
      categories,
      categoryCount: 2,
      output
    }).getCommandNodes(graph);
    expect(nodes.some(node => node.id.includes('suppress'))).toBe(false);
  } finally {
    device.destroy();
  }
});

it.each([
  [16, 5.545177444479562],
  [26, 6.516193076042964]
])('NUM-14 independent uniform %s-category table', (categoryCount, entropy) => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `uniform-${serial++}`, format, length);
  try {
    const contributor = new GPUSpatialEntropy({
      weights: {
        offsets: view('uint32', categoryCount * 2 + 1),
        neighbors: view('uint32', categoryCount ** 2),
        weights: view('float32', categoryCount ** 2)
      },
      categories: view('uint32', categoryCount * 2),
      categoryCount,
      output: {summary: view('float32', 8), counts: view('uint32', 2)}
    });
    const kernels = getGPUCommandNodeKernelSources(
      contributor
        .getCommandNodes(graph)
        .filter(node => node.id.startsWith('spatial-entropy-summary'))
    );
    const kernel = kernels[kernels.length - 1];
    expect(kernel.source).not.toContain('getTerm(');
    const summary = new Float32Array(8);
    const values = {
      pairCounts: new Uint32Array(categoryCount ** 2).fill(1),
      categoryRowCounts: new Uint32Array(categoryCount).fill(2),
      summary,
      totals: new Uint32Array(2),
      entropies: new Float32Array(4),
      counts: new Uint32Array(2)
    };
    for (const stage of kernels) executeSpatialAuditKernel(stage, values);
    expect(Array.from(values.counts)).toEqual([categoryCount * 2, categoryCount ** 2]);
    // Equiprobable independent endpoints: H(A,B)=ln(K²); I(A;B)=0 exactly.
    // Literals: /usr/bin/python3: math.log(16**2), math.log(26**2).
    expect(Number.isFinite(summary[3])).toBe(true);
    expect(Math.abs(summary[3] - entropy)).toBeLessThan(2e-6);
    expect(Number.isFinite(summary[4])).toBe(true);
    expect(Math.abs(summary[4])).toBeLessThan(2e-6);
  } finally {
    device.destroy();
  }
});
