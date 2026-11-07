// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUGlobalSpatialStatistics,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT
} from '../../../src/gpu-spatial-analysis/global-spatial-statistics/index';
import type {GPUChunkedSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

const SLOTS_PER_ROW = 4;
let serial = 0;

function createGraph() {
  return new GPUCommandGraph(createNullWebGPUDevice(), {id: 'chunked-global-statistics'});
}

function createChunked(
  graph: GPUCommandGraph,
  chunkRowCounts: number[],
  transform: 'binary' | 'row' = 'binary'
): GPUChunkedSpatialWeights {
  const rowCount = chunkRowCounts.reduce((total, count) => total + count, 0);
  let rowStart = 0;
  return {
    rowCount,
    information: {rowCount, transform, symmetric: 'unknown'},
    chunks: chunkRowCounts.map((chunkRows, index) => {
      const slots = chunkRows * SLOTS_PER_ROW;
      const chunk = {
        rowStart,
        rowCount: chunkRows,
        weights: {
          offsets: createTransientView(graph, `offsets-${index}`, 'uint32', chunkRows + 1),
          neighbors: createTransientView(graph, `neighbors-${index}`, 'uint32', slots),
          weights: createTransientView(graph, `weights-${index}`, 'float32', slots)
        }
      };
      rowStart += chunkRows;
      return chunk;
    })
  };
}

function createNodes(
  graph: GPUCommandGraph,
  weights: GPUChunkedSpatialWeights,
  overrides: {values?: number; symmetryStrategy?: 'transpose' | 'pairwise'} = {}
) {
  const rows = overrides.values ?? weights.rowCount;
  const suffix = serial++;
  return new GPUGlobalSpatialStatistics({
    weights,
    symmetryStrategy: overrides.symmetryStrategy ?? 'pairwise',
    values: createTransientView(graph, `values-${suffix}`, 'float32', rows),
    secondValues: createTransientView(graph, `second-values-${suffix}`, 'float32', rows),
    statistics: ['moran', 'geary', 'getisOrdG', 'bivariateMoran', 'joinCount'],
    output: {
      summary: createTransientView(
        graph,
        `summary-${suffix}`,
        'float32',
        GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
      ),
      joinCounts: createTransientView(graph, `join-counts-${suffix}`, 'uint32', 3)
    }
  });
}

function getKernelIds(chunkRowCounts: number[], transform: 'binary' | 'row' = 'binary') {
  const graph = createGraph();
  const weights = createChunked(graph, chunkRowCounts, transform);
  const nodes = createNodes(graph, weights).getCommandNodes(graph);
  const kernels = getGPUCommandNodeKernelSources(nodes);
  graph.device.destroy();
  return {kernels, ids: kernels.map(kernel => kernel.id)};
}

it('generates per-chunk kernels with an (row chunk x target chunk) reverse lookup', () => {
  for (const transform of ['binary', 'row'] as const) {
    const {kernels, ids} = getKernelIds([25, 25, 10], transform);
    for (const family of ['islands', 'slot-rows', 'transpose-keys', 'pairs', 'columns']) {
      for (let chunk = 0; chunk < 3; chunk++) {
        expect(
          ids.some(id => id.includes(`${family}-chunk${chunk}`)),
          `${transform} ${family} chunk ${chunk}`
        ).toBe(true);
      }
    }
    expect(ids.filter(id => /reverse-chunk\d-target\d/.test(id))).toHaveLength(9);
    expect(ids.filter(id => id.includes('row-scale-chunk'))).toHaveLength(
      transform === 'row' ? 3 : 0
    );
    expect(ids.some(id => id.includes('columns-finalize'))).toBe(true);
    for (const {id, source} of kernels) {
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    }
  }
});

it('node count follows the chunk count only, not the tile size', () => {
  const countKernels = (chunkRowCounts: number[]) => getKernelIds(chunkRowCounts).ids.length;
  const perChunk = (count: number) => countKernels(Array.from({length: count}, () => 60 / count));
  const base = perChunk(1);
  // Per chunk: islands, slot-rows, transpose-keys, pairs, columns, plus its reverse kernels, and
  // the sort nodes of the grouping. The growth is quadratic in the chunk count at most.
  expect(perChunk(2)).toBeGreaterThan(base);
  expect(perChunk(3)).toBeGreaterThan(perChunk(2));
  // Same chunk count with different tile sizes generates the same structure.
  expect(countKernels([10, 40, 10])).toBe(countKernels([20, 20, 20]));
  expect(countKernels([1, 1, 58])).toBe(countKernels([30, 20, 10]));
});

it('keeps the whole-weights WGSL free of chunk constants', () => {
  const graph = createGraph();
  const view = (name: string, length: number) =>
    createTransientView(graph, name, 'float32', length);
  const nodes = new GPUGlobalSpatialStatistics({
    weights: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 61),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 240),
      weights: view('weights', 240)
    },
    values: view('values', 60),
    statistics: ['moran'],
    output: {summary: view('summary', GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length)}
  }).getCommandNodes(graph);
  const kernels = getGPUCommandNodeKernelSources(nodes);
  expect(kernels.filter(kernel => kernel.id.includes('-pairs'))).toHaveLength(1);
  expect(kernels.some(kernel => kernel.id.includes('chunk'))).toBe(false);
  expect(kernels.some(kernel => kernel.source.includes('WEIGHTS_ROW_START'))).toBe(false);
  graph.device.destroy();
});

it('validates chunk layout and per-row inputs', () => {
  const graph = createGraph();
  const weights = createChunked(graph, [25, 25, 10]);
  expect(() => createNodes(graph, weights)).not.toThrow();
  expect(() => createNodes(graph, weights, {values: 59})).toThrow(/one row per weights row/);
  const gap = {
    ...weights,
    chunks: [weights.chunks[0], {...weights.chunks[2], rowStart: 51}]
  };
  expect(() => createNodes(graph, gap)).toThrow(/must start at row/);
  expect(() => createNodes(graph, {...weights, rowCount: 61})).toThrow(/cover rowCount/);
  expect(() => createNodes(graph, {...weights, chunks: []})).toThrow();
  graph.device.destroy();
});
