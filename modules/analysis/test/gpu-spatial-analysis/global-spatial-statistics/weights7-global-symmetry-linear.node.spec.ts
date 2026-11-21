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
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function countNodes(chunkCount: number, symmetryStrategy: 'auto' | 'transpose' | 'pairwise') {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'weights7-symmetry'});
  const chunkRows = 4;
  const rowCount = chunkCount * chunkRows;
  const weights: GPUChunkedSpatialWeights = {
    rowCount,
    information: {rowCount, transform: 'binary', symmetric: 'unknown'},
    chunks: Array.from({length: chunkCount}, (_, index) => ({
      rowStart: index * chunkRows,
      rowCount: chunkRows,
      weights: {
        offsets: createTransientView(graph, `offsets-${serial}-${index}`, 'uint32', chunkRows + 1),
        neighbors: createTransientView(graph, `neighbors-${serial}-${index}`, 'uint32', 16),
        weights: createTransientView(graph, `weights-${serial}-${index}`, 'float32', 16)
      }
    }))
  };
  const nodes = new GPUGlobalSpatialStatistics({
    mode: 'reference',
    weights,
    symmetryStrategy,
    values: createTransientView(graph, `values-${serial}`, 'float32', rowCount),
    statistics: ['moran', 'geary'],
    output: {
      summary: createTransientView(
        graph,
        `summary-${serial}`,
        'float32',
        GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
      )
    }
  }).getCommandNodes(graph);
  serial++;
  // Geary's stable complete-support pass deliberately visits every chunk pair for three
  // fields (high, low, factor). It is independent of symmetryStrategy and is documented as
  // quadratic. Keep checking its exact combinatorial count, and measure the original symmetry
  // pipeline separately so a Geary request cannot invalidate the transpose growth contract.
  const gearyPairs = nodes.filter(node => node.id.includes('-geary-deviations-pairs-'));
  expect(gearyPairs).toHaveLength(3 * chunkCount ** 2);
  return nodes.filter(node => !node.id.includes('-geary-deviations-')).length;
}

it('the transpose symmetry strategy adds a constant number of kernels per chunk', () => {
  const counts = [1, 2, 4, 8, 16].map(chunkCount => countNodes(chunkCount, 'transpose'));
  // Exactly linear: equal increments per added chunk (second differences vanish).
  const perChunk = (counts[1] - counts[0]) / 1;
  expect(perChunk).toBeGreaterThan(0);
  [2, 4, 8, 16].forEach((chunkCount, index) => {
    expect(counts[index + 1] - counts[0]).toBe(perChunk * (chunkCount - 1));
  });
});

it('the pairwise fallback stays quadratic and auto picks the linear transpose strategy', () => {
  const pairwise = [2, 4, 8].map(chunkCount => countNodes(chunkCount, 'pairwise'));
  const transpose = [2, 4, 8].map(chunkCount => countNodes(chunkCount, 'transpose'));
  // Quadratic growth: second difference of the 2, 4, 8 sequence is not the linear one.
  expect(pairwise[2] - pairwise[1]).toBeGreaterThan(2 * (pairwise[1] - pairwise[0]));
  expect(transpose[2] - transpose[1]).toBe(2 * (transpose[1] - transpose[0]));
  // The null device reports the default 128 MiB binding limit: tiny weights select the transpose.
  expect([2, 4, 8].map(chunkCount => countNodes(chunkCount, 'auto'))).toEqual(transpose);
});
