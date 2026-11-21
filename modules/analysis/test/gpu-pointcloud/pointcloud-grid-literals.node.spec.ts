// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createPointCloudGridIndex} from '../../src/pointcloud-analysis/operations/pointcloud-grid-index';
import {GPUPointCloudICP} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-icp';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';

it('S4: large finite float32 grid coordinates generate float literals in keys and ICP search windows', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const raster of [false, true]) {
      const graph = new GPUCommandGraph(device);
      const positions = createTransientView(graph, 'positions', 'float32x3', 3);
      const grid = {origin: [1e19, -1e19, 0] as const, dimensions: [2, 2, 2] as const, cellSize: 1};
      const {nodes} = createPointCloudGridIndex(graph, 'literal-grid', positions, grid, raster);
      const keys = getGPUCommandNodeKernelSources(nodes).find(
        kernel => kernel.id === 'literal-grid-keys'
      )!;
      expect(keys.source).toContain(
        raster
          ? 'vec2f(9999999980506448000.0,-9999999980506448000.0)'
          : 'vec3f(9999999980506448000.0,-9999999980506448000.0,0.0)'
      );
    }
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'float32x3' | 'uint32'>(
      id: string,
      format: Format,
      length: number
    ) => createTransientView(graph, id, format, length);
    const contributor = new GPUPointCloudICP({
      source: view('source', 'float32x3', 3),
      target: view('target', 'float32x3', 3),
      maximumDistance: 1,
      maximumIterations: 1,
      targetGrid: {origin: [1e19, -1e19, 0], dimensions: [2, 2, 2], cellSize: 1},
      output: {
        transform: view('transform', 'float32', 16),
        history: view('history', 'float32x3', 1),
        inlierMask: view('mask', 'uint32', 3),
        status: view('status', 'uint32', 1)
      }
    });
    const matches = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).filter(
      kernel => kernel.id.includes('-match-')
    );
    expect(matches).toHaveLength(2);
    for (const kernel of matches)
      expect(kernel.source).toContain(
        'position-vec3f(9999999980506448000.0,-9999999980506448000.0,0.0)'
      );
  } finally {
    device.destroy();
  }
});
