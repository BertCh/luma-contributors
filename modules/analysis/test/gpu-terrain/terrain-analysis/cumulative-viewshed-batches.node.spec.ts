// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_CUMULATIVE_VIEWSHED_SAMPLES_PER_DISPATCH,
  GPUTerrainCumulativeViewshed
} from '../../../src/gpu-terrain/terrain-analysis/index';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {countRasterPipelines, getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';
import {createBand} from '../terrain-test-utils';
import {AnalysisValidationError} from '../../../src/analysis-core/kernels/analysis-validation-error';

function createViewshed(
  observerCount: number,
  overrides: Record<string, unknown> = {},
  width = 9,
  height = 7
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const pixelCount = width * height;
  const viewshed = new GPUTerrainCumulativeViewshed({
    grid: makeGPURasterMetadata({width, height}),
    elevation: createBand(graph, 'elevation', pixelCount),
    observers: createTransientView(graph, 'observers', 'float32x2', observerCount),
    parameters: createTransientView(graph, 'settings', 'float32', 12),
    output: {visibleCount: createTransientView(graph, 'visible', 'uint32', pixelCount)},
    ...overrides
  });
  return {graph, viewshed};
}

/** CPU twin of the batch-table kernel: one thread walks the batches with a carry. */
function getBatchStarts(batchSize: number, pixelCount: number, batchCount: number): number[][] {
  const starts: number[][] = [];
  let firstObserver = 0;
  let firstPixel = 0;
  for (let batch = 0; batch < batchCount; batch++) {
    starts.push([firstObserver, firstPixel]);
    firstObserver += Math.floor(batchSize / pixelCount);
    firstPixel += batchSize % pixelCount;
    if (firstPixel >= pixelCount) {
      firstPixel -= pixelCount;
      firstObserver += 1;
    }
  }
  return starts;
}

it('review E-04: every observer batch shares one sight-line pipeline', () => {
  for (const traversal of ['march', 'pyramid'] as const) {
    const counts = [5, 6, 13, 40].map(observerCount => {
      const {graph, viewshed} = createViewshed(observerCount, {
        observersPerDispatch: 2,
        traversal
      });
      const nodes = viewshed.getCommandNodes(graph);
      const batches = getRasterPipelines(nodes).filter(pipeline => /-batch-\d+$/.test(pipeline.id));
      expect(batches.length).toBe(Math.ceil(observerCount / 2));
      expect(new Set(batches.map(pipeline => pipeline.key)).size).toBe(1);
      return countRasterPipelines(nodes);
    });
    expect(new Set(counts).size, traversal).toBe(1);
  }
});

it('review C-10: batches are sized by sight-line samples, splitting one observer when needed', () => {
  // 1024^2 pixels, max(w, h) = 1024 samples per pair: 2^28 / 1024 = 262144 pairs per dispatch, a
  // quarter of one observer's pixels. The old default floor(2^22 / pixels) = 4 observers per
  // dispatch was 4 * 2^20 * 1024 = 2^32 samples.
  const {viewshed} = createViewshed(3, {}, 1024, 1024);
  expect(viewshed.invocationsPerDispatch).toBe(
    GPU_TERRAIN_CUMULATIVE_VIEWSHED_SAMPLES_PER_DISPATCH / 1024
  );
  expect(viewshed.observersPerDispatch).toBe(0.25);
  expect(() => createViewshed(3, {observersPerDispatch: 1.5})).toThrow(/positive integer/);
});

it('the batch table covers every (observer, pixel) pair exactly once', () => {
  for (const [batchSize, pixelCount, observerCount] of [
    [7, 63, 5],
    [63, 63, 5],
    [100, 63, 5],
    [262144, 1048576, 3],
    [2 ** 31, 2 ** 20, 4096]
  ]) {
    const pairCount = observerCount * pixelCount;
    const batchCount = Math.ceil(pairCount / batchSize);
    const starts = getBatchStarts(batchSize, pixelCount, batchCount);
    for (const [batch, [firstObserver, firstPixel]] of starts.entries()) {
      // The carry walk equals the direct division, which the GPU cannot do in u32.
      expect(firstObserver * pixelCount + firstPixel).toBe(batch * batchSize);
      expect(firstPixel).toBeLessThan(pixelCount);
      // `pixelStart + index` stays a u32.
      expect(firstPixel + batchSize - 1).toBeLessThan(2 ** 32);
    }
  }
});

it('review LUMA-118: cumulative viewshed admits its command nodes at construction', () => {
  // One observer per dispatch: 40,000 observers are 40,000 batch kernels, over the shared 32,768
  // budget. Before the check, construction accepted it.
  expect(() => createViewshed(40000, {id: 'budget-viewshed', observersPerDispatch: 1})).toThrow(
    // 40,000 one-observer batches + six fixed-node allowances; no pyramid on this route.
    new AnalysisValidationError(
      'budget-viewshed',
      'estimated command node count must be at most 32768',
      {
        code: 'INTEGER_RANGE',
        path: 'estimated command node count',
        received: 40006,
        expected: {maximum: 32768}
      }
    )
  );
  for (const traversal of ['march', 'pyramid'] as const) {
    // 13 observers, 2 per dispatch: ceil(13 / 2) = 7 batch kernels.
    const {graph, viewshed} = createViewshed(13, {observersPerDispatch: 2, traversal});
    const nodes = viewshed.getCommandNodes(graph);
    expect(nodes.filter(node => /-batch-\d+$/.test(String(node.id)))).toHaveLength(7);
    expect(nodes.length, traversal).toBeLessThanOrEqual(viewshed.estimatedCommandNodeCount);
    expect(viewshed.estimatedCommandNodeCount - nodes.length, traversal).toBeLessThanOrEqual(6);
  }
});
