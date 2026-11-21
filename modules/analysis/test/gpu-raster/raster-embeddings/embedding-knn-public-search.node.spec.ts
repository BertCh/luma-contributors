// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createPublicSearchClassifier,
  expectPublicSearchResults
} from './embedding-knn-public-search-fixture';

for (const metric of ['euclidean', 'cosine'] as const) {
  it(`declares both public-search comparison routes in one graph: ${metric}`, () => {
    const device = createNullWebGPUDevice();
    try {
      const graph = new GPUCommandGraph(device);
      const common = {
        pixelCount: 4,
        dimensions: 2,
        trainingCount: 258,
        classCount: 4,
        neighborCount: metric === 'euclidean' ? 4 : 2,
        metric,
        vectors: createTransientView(graph, 'vectors', 'float32', 8),
        mask: createTransientView(graph, 'mask', 'uint32', 4),
        trainingVectors: createTransientView(graph, 'training', 'float32', 516),
        trainingLabels: createTransientView(graph, 'training-labels', 'uint32', 258)
      };
      expect(() => {
        for (const route of ['brute-force', 'tiled'] as const) {
          graph.add(
            createPublicSearchClassifier(route, {
              ...common,
              output: {
                labels: createTransientView(graph, `${route}-labels`, 'uint32', 4),
                confidence: createTransientView(graph, `${route}-confidence`, 'float32', 4),
                probabilities: createTransientView(graph, `${route}-shares`, 'float32', 16)
              }
            })
          );
        }
      }).not.toThrow();
    } finally {
      device.destroy();
    }
  });
}

function cosineResult(neighborCount: 1 | 2): Record<string, number[]> {
  const result: Record<string, number[]> = {
    ids:
      neighborCount === 1
        ? [255, 255, 0xffffffff, 0xffffffff]
        : [255, 256, 255, 256, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff],
    scores:
      neighborCount === 1
        ? [1, 1, -Infinity, -Infinity]
        : [1, 1, 1, 1, -Infinity, -Infinity, -Infinity, -Infinity],
    counts: [neighborCount, neighborCount, 0, 0]
  };
  for (const route of ['brute-force', 'tiled']) {
    result[`${route}-labels`] =
      neighborCount === 1 ? [1, 1, 0xffffffff, 0xffffffff] : [0, 0, 0xffffffff, 0xffffffff];
    result[`${route}-confidence`] = neighborCount === 1 ? [1, 1, NaN, NaN] : [0.5, 0.5, NaN, NaN];
    result[`${route}-shares`] =
      neighborCount === 1
        ? [0, 1, 0, 0, 0, 1, 0, 0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN]
        : [0.5, 0.5, 0, 0, 0.5, 0.5, 0, 0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN];
  }
  return result;
}

it('pins a single cross-tile cosine winner and rejects swapping the tied rows', () => {
  // Both candidates lie on the positive x-axis: cosine = 1. Row 255 beats 256 by source order.
  const result = cosineResult(1);
  expect(() => expectPublicSearchResults(result, 'cosine', 1)).not.toThrow();
  for (const route of ['brute-force', 'tiled']) {
    const reversed = cosineResult(1);
    reversed[`${route}-labels`][0] = 0;
    expect(() => expectPublicSearchResults(reversed, 'cosine', 1)).toThrow();
  }
});

for (const port of [
  'scores',
  'brute-force-confidence',
  'tiled-confidence',
  'brute-force-shares',
  'tiled-shares'
]) {
  it(`rejects an extra readback element: ${port}`, () => {
    const result = cosineResult(2);
    result[port].push(99);
    expect(() => expectPublicSearchResults(result, 'cosine', 2)).toThrow();
  });
}
