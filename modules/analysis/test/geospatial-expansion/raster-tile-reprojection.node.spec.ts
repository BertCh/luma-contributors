// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
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
