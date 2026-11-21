// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUHDBSCAN} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-hierarchical-density-clustering';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('caps omitted minimumSamples at 32 when minimumClusterSize is 40', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const model = new GPUHDBSCAN({
      minimumClusterSize: 40,
      positions: createTransientView(graph, 'positions', 'float32x2', 48),
      output: {labels: createTransientView(graph, 'labels', 'uint32', 48)}
    });
    expect(model.minimumSamples).toBe(32);
  } finally {
    device.destroy();
  }
});
