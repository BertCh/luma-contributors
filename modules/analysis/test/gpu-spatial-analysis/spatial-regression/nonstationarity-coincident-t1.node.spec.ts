// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeographicallyWeightedRegressionNonstationarityTest} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-geographically-weighted-regression-nonstationarity-test';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('t1 coincident focal queries share the same permuted design and response anchor', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `coincident-${serial++}`, format, length);
    const model = new GPUGeographicallyWeightedRegressionNonstationarityTest({
      positions: view('float32x2', 8),
      predictors: view('float32', 8),
      predictorCount: 1,
      response: view('float32', 8),
      coefficients: view('float32', 16),
      bandwidthParameters: view('float32', 8),
      selectedBandwidth: view('float32', 2),
      parameters: view('uint32', 4),
      maximumPermutations: 1,
      maximumBandwidthCount: 1,
      output: {table: view('float32', 12), summary: view('float32', 4)}
    });
    const kernel = getGPUCommandNodeKernelSources(model.getCommandNodes(graph)).find(candidate =>
      candidate.id.includes('permuted-tiles')
    )!;
    // Identical coordinates yield identical local weights; varying the algebraic anchor with
    // location creates a false f32 coefficient spread although the least-squares fit is identical.
    expect(kernel.source).toContain(
      'let anchorLocation = select(location, 0u, all(origin == firstOrigin));'
    );
    expect(kernel.source).toContain('getSwapOrNotPermutationIndex(anchorLocation, count)');
  } finally {
    device.destroy();
  }
});
