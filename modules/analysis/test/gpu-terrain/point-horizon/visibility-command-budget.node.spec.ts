// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  makeGPURasterMetadata,
  getGPURasterExtremaPyramidLayout
} from '@luma-contributors/analysis/raster-analysis';
import {GPUPointHorizonVisibility} from '../../../src/gpu-terrain/point-horizon/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';

function createVisibility(
  targetCount: number,
  targetsPerDispatch: number,
  route: 'march' | 'built-pyramid' | 'shared-pyramid'
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const grid = makeGPURasterMetadata({width: 64, height: 48, cellSize: 10});
  const layout = getGPURasterExtremaPyramidLayout(grid, {firstBlockSize: 4, footprint: 'bilinear'});
  const visibility = new GPUPointHorizonVisibility({
    id: 'budget-visibility',
    grid,
    elevation: createBand(graph, 'terrain', 64 * 48),
    maximumDistance: 1000,
    cellSize: 10,
    traversal: route === 'march' ? 'march' : 'pyramid',
    pyramid:
      route === 'shared-pyramid'
        ? {layout, combined: createTransientView(graph, 'pyramid', 'float32', 2 * layout.length)}
        : undefined,
    observers: createTransientView(graph, 'observers', 'float32x4', 1),
    targets: createTransientView(graph, 'targets', 'float32x4', targetCount),
    parameters: createTransientView(graph, 'parameters', 'float32', 12),
    output: {visibility: createTransientView(graph, 'visibility', 'uint32', targetCount)},
    targetsPerDispatch
  });
  return {graph, visibility};
}

it('rejects 40,000 one-target chunks before graph declaration', () => {
  expect(() => createVisibility(40000, 1, 'march')).toThrow(
    /budget-visibility estimated command node count must be at most 32768/
  );
});

it('bounds actual visibility nodes for march, built and shared pyramids including zero targets', () => {
  for (const route of ['march', 'built-pyramid', 'shared-pyramid'] as const) {
    for (const count of [0, 1, 50]) {
      const {graph, visibility} = createVisibility(count, 7, route);
      const nodes = visibility.getCommandNodes(graph);
      expect(nodes.filter(node => /-classify-\d+$/.test(String(node.id)))).toHaveLength(
        Math.max(1, Math.ceil(count / 7))
      );
      expect(nodes.length, route).toBeLessThanOrEqual(visibility.estimatedCommandNodeCount);
      expect(visibility.estimatedCommandNodeCount - nodes.length, route).toBeLessThanOrEqual(8);
    }
  }
});
