// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUCategoryPairCounts} from '../../../src/gpu-spatial-analysis/spatial-entropy/gpu-category-pair-counts';
import {
  GPUSpatialEntropy,
  GPU_SPATIAL_ENTROPY_LAYOUT
} from '../../../src/gpu-spatial-analysis/spatial-entropy/gpu-spatial-entropy';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const producer of ['counts', 'entropy'] as const) {
  it.each([
    'overflow',
    'candidateOverflow',
    'requiredCount',
    'complete',
    'converged',
    'iterationLimitReached'
  ] as const)(`AMG-59 ${producer} refuses an unqualifiable %s certificate`, name => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `certificate-${serial++}`, format, length);
    try {
      const weights = {
        offsets: view('uint32', 3),
        neighbors: view('uint32', 2),
        weights: view('float32', 2)
      };
      const props = {weights, categories: view('uint32', 2), categoryCount: 2};
      const output = {
        pairCounts: view('uint32', 4),
        categoryRowCounts: view('uint32', 2)
      };
      const entropyOutput = {summary: view('float32', GPU_SPATIAL_ENTROPY_LAYOUT.length)};
      const construct = (status?: {[key: string]: GraphDataView<'uint32'>}) =>
        producer === 'counts'
          ? new GPUCategoryPairCounts({...props, weights: {...weights, status}, output})
          : new GPUSpatialEntropy({
              ...props,
              weights: {...weights, status},
              output: entropyOutput
            });
      expect(() => construct()).not.toThrow();
      // Neither output carries a loss certificate. Resident producer words cannot be
      // inspected at construction, so supplied evidence must be refused even if clean.
      expect(() => construct({[name]: view('uint32', 1)})).toThrow(/weights.status/);
    } finally {
      device.destroy();
    }
  });
}
