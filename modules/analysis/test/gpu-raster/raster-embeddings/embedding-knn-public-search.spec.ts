// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUSimilaritySearch, type GraphEmbeddingMatrix} from '@luma.gl/graph/gpu-vector-search';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {it} from 'vitest';
import {
  createPublicSearchClassifier,
  expectPublicSearchResults
} from './embedding-knn-public-search-fixture';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createRasterTimeSeriesHarness} from '../raster-time-series/raster-time-series-test-utils';

function matrix(
  values: GraphDataView<'float32'>,
  rowCount: number,
  validity?: GraphDataView<'uint32'>
): GraphEmbeddingMatrix {
  return {
    dimensions: 2,
    rowCount,
    chunks: [
      {values, rowCount, rowStride: 2, byteOffset: values.byteOffset, sourceRowOffset: 0, validity}
    ]
  };
}

for (const [metric, neighborCount] of [
  ['euclidean', 4],
  ['cosine', 2],
  ['cosine', 1]
] as const) {
  it(`compares both classifier routes with public search: ${metric} k=${neighborCount}, cross-tile ties, overflow, masks and poisoned tails`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRasterTimeSeriesHarness(device);
    const training = new Array<number>(258 * 2).fill(NaN);
    const labels = new Array<number>(258).fill(3);
    // Euclidean: query [1,1] has distance 2 to rows 0,255,256; row 257 overflows.
    // Cosine: scaled axis vectors have exact similarities 0 (row 0), 1 (255/256), -1 (257).
    // The equal scores across the tile boundary keep row 255 before 256.
    const rows =
      metric === 'euclidean'
        ? [
            [0, 0, 0, 2],
            [255, 2, 0, 1],
            [256, 0, 2, 0],
            [257, 3e38, 0, 3]
          ]
        : [
            [0, 0, 2e38, 2],
            [255, 2e38, 0, 1],
            [256, 1e-30, 0, 0],
            [257, -2e38, 0, 3]
          ];
    for (const [row, first, second, label] of rows) {
      training[row * 2] = first;
      training[row * 2 + 1] = second;
      labels[row] = label;
    }
    const queries =
      metric === 'euclidean' ? [1, 1, 3e38, 3e38, 1, 1, NaN, 0] : [2e38, 0, 1e-30, 0, 1, 0, NaN, 0];
    const vectors = harness.floats(queries);
    const trainingVectors = harness.floats(training);
    const trainingLabels = harness.integers(labels);
    const mask = harness.integers([7, 2, 0, 1]);
    const outputIds = harness.output('ids', 'uint32', 4 * neighborCount);
    const outputScores = harness.output('scores', 'float32', 4 * neighborCount);
    const resultCounts = harness.output('counts', 'uint32', 4);
    const contributors = (['brute-force', 'tiled'] as const).map(searchStructure => {
      const output = {
        labels: harness.output(`${searchStructure}-labels`, 'uint32', 4),
        confidence: harness.output(`${searchStructure}-confidence`, 'float32', 4),
        probabilities: harness.output(`${searchStructure}-shares`, 'float32', 16)
      };
      // Seed every output, including missing-row tails, with a recognizable finite poison.
      harness.graph.add({
        getCommandNodes: graph => [
          createWGSLKernelNode(graph, {
            id: `poison-${searchStructure}`,
            operation: 'knn-test-poison',
            invocationCount: 16,
            bindings: [
              {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
              {name: 'confidence', view: output.confidence, type: 'f32', access: 'read_write'},
              {name: 'shares', view: output.probabilities, type: 'f32', access: 'read_write'}
            ],
            body: `if (index < 4u) { labels[labelsOffset + index] = 99u; confidence[confidenceOffset + index] = 99.0; }
shares[sharesOffset + index] = 99.0;`
          })
        ]
      });
      return createPublicSearchClassifier(searchStructure, {
        pixelCount: 4,
        dimensions: 2,
        vectors,
        mask,
        trainingVectors,
        trainingLabels,
        trainingCount: 258,
        classCount: 4,
        neighborCount,
        metric,
        output
      });
    });
    harness.graph.add({
      getCommandNodes: graph => [
        createWGSLKernelNode(graph, {
          id: 'poison-search',
          operation: 'knn-test-poison',
          invocationCount: 4 * neighborCount,
          bindings: [
            {name: 'ids', view: outputIds, type: 'u32', access: 'read_write'},
            {name: 'scores', view: outputScores, type: 'f32', access: 'read_write'},
            {name: 'counts', view: resultCounts, type: 'u32', access: 'read_write'}
          ],
          body: `ids[idsOffset + index] = 99u; scores[scoresOffset + index] = 99.0; if (index < 4u) { counts[countsOffset + index] = 99u; }`
        })
      ]
    });
    new GPUSimilaritySearch({
      dataset: matrix(trainingVectors, 258),
      queries: matrix(vectors, 4, mask),
      outputIds,
      outputScores,
      resultCounts,
      k: neighborCount,
      metric: metric === 'euclidean' ? 'squared-euclidean' : 'cosine',
      tileSize: 256
    }).addToGraph(harness.graph);
    const result = await harness.run(contributors);
    expectPublicSearchResults(result, metric, neighborCount);
    // Public search deliberately admits infinite squared distances from finite rows. Its four
    // returned IDs cannot be used directly for the classifier's finite-distance-only votes:
    // query 0 would give four shares of 1/4; query 1 must have no local votes at all.
  });
}
