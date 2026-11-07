// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {addTrackQualityRecipe} from '../../../src/gpu-network/recipes/track-quality-recipe';
import {addNetworkDisruptionRecipe} from '../../../src/gpu-network/recipes/network-scenarios-recipe';
import {addInterpolationDiagnosticsRecipe} from '../../../src/gpu-spatial-analysis/recipes/interpolation-diagnostics-recipe';
import {addDrainageNetworkRecipe} from '../../../src/gpu-spatial-analysis/recipes/drainage-network-recipe';
import {
  getRecipeView,
  createRecipeKernelNode
} from '../../../src/gpu-spatial-analysis/recipes/expansion-recipe-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';

const grid: GPURasterMetadata = {
  width: 3,
  height: 2,
  affine: [1, 0, 0, 0, -1, 2],
  pixelInterpretation: 'area'
};
function context() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {device, graph, view};
}
it('new recipe kernels stay within core storage limits, including double-single timestamps', () => {
  const {device, graph, view} = context();
  try {
    addTrackQualityRecipe(graph, {
      trajectory: {
        positions: view('float32x2', 6),
        timestamps: view('float32', 6),
        timestampsLow: view('float32', 6),
        trackOffsets: view('uint32', 3)
      },
      thresholds: view('float32', 2),
      similarity: {pairA: view('uint32', 2), pairB: view('uint32', 2)}
    });
    addNetworkDisruptionRecipe(graph, {
      search: {
        topology: {
          offsets: view('uint32', 4),
          neighbors: view('uint32', 4),
          weights: view('float32', 4)
        },
        startNodes: view('uint32', 2),
        maximumIterations: 2
      },
      rowCount: 2,
      edgeMask: view('uint32', 4),
      parameters: view('float32', 4),
      opportunityWeights: view('float32', 3)
    });
    addInterpolationDiagnosticsRecipe(graph, {
      kriging: {
        positions: view('float32x2', 4),
        values: view('float32', 4),
        parameters: view('float32', 12),
        grid,
        indexGridSize: [2, 2],
        indexBounds: [0, 0, 3, 2],
        maximumNeighborCount: 4
      }
    });
    addDrainageNetworkRecipe(graph, {
      flow: {
        grid,
        elevation: {
          id: 'dem',
          format: 'float32',
          storage: {kind: 'buffer', values: view('float32', 6)}
        },
        parameters: view('float32', 8),
        maximumFillIterations: 2,
        maximumFlatIterations: 2,
        maximumAccumulationIterations: 2
      },
      maximumSteps: 6
    });
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.length).toBeGreaterThan(20);
    for (const kernel of kernels)
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
  } finally {
    device.destroy();
  }
});
it('recipe output overrides reject foreign graphs, wrong lengths and read/write aliases', () => {
  const first = context();
  const second = context();
  try {
    expect(() =>
      getRecipeView(first.graph, 'foreign', 'float32', 2, second.view('float32', 2))
    ).toThrow();
    expect(() =>
      getRecipeView(first.graph, 'size', 'float32', 3, first.view('float32', 2))
    ).toThrow();
    const shared: GraphDataView<'float32'> = first.view('float32', 2);
    expect(() =>
      createRecipeKernelNode(first.graph, {
        id: 'alias',
        operation: 'test',
        invocationCount: 2,
        bindings: [
          {name: 'input', view: shared, type: 'f32', access: 'read'},
          {name: 'output', view: shared, type: 'f32', access: 'read_write'}
        ],
        body: ''
      })
    ).toThrow(/share buffers/);
  } finally {
    first.device.destroy();
    second.device.destroy();
  }
});
