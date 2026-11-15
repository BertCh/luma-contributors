// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPURasterTileReprojection} from '../../src/raster-analysis/gpu-raster-tile-reprojection';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../utils/gpu-contributor-test-utils';

it('tile routing parses many tiles with default-device storage binding limits', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const sourceGrid = {
    width: 32,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const sources = Array.from({length: 32}, (_, index) => ({
    pixelBounds: [index, 0, index + 1, 1] as const,
    input: {
      id: `tile-${index}`,
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: createTransientView(graph, `tile-${index}`, 'float32', 1)
      }
    }
  }));
  const props = {
    sourceGrid,
    destinationGrid: sourceGrid,
    sources,
    output: {
      values: createTransientView(graph, 'values', 'float32', 32),
      validity: createTransientView(graph, 'validity', 'uint32', 32)
    }
  };
  const kernels = getParsedKernelSources(
    new GPURasterTileReprojection(props).getCommandNodes(graph)
  );
  expect(kernels.size).toBeGreaterThan(32);
  for (const [name, source] of kernels)
    expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
  expect(
    () =>
      new GPURasterTileReprojection({
        ...props,
        sources: [{...sources[0], pixelBounds: [0.5, 0, 1, 1]}]
      })
  ).toThrow();
  expect(
    () =>
      new GPURasterTileReprojection({
        ...props,
        sources: [{...sources[0], pixelBounds: [0, 0, 33, 1]}]
      })
  ).toThrow();
});

it('rejects foreign parameters before mosaic allocation and permits a corrected retry', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const foreignGraph = new GPUCommandGraph(device);
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const foreignParameters = createTransientView(foreignGraph, 'parameters', 'float32', 20);
  const localParameters = createTransientView(graph, 'parameters', 'float32', 20);
  const props = {
    id: 'tile-retry',
    sourceGrid: grid,
    destinationGrid: grid,
    sources: [],
    output: {
      values: createTransientView(graph, 'values', 'float32', 1),
      validity: createTransientView(graph, 'validity', 'uint32', 1)
    }
  };
  const invalid = new GPURasterTileReprojection({...props, parameters: foreignParameters});
  const corrected = new GPURasterTileReprojection({...props, parameters: localParameters});
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => invalid.getCommandNodes(graph)).toThrow(/target graph/);
    const failedAllocations = allocations.mock.calls.length;
    expect(() => {
      for (const node of corrected.getCommandNodes(graph)) graph.add(node);
    }).not.toThrow();
    expect(failedAllocations).toBe(0);
  } finally {
    allocations.mockRestore();
    invalid.destroy();
    corrected.destroy();
    device.destroy();
  }
});
