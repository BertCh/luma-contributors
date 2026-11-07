// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUChunkedSpatialWeightsAlgebra combines k nearest neighbors with a distance band, one chunk of
// rows at a time. Rows are split into vertical bands; each chunk's output links get their own color,
// and the result is the same whatever the chunking.

import {GPUChunkedSpatialWeightsAlgebra} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createDistanceBandWeights, createKnnWeights, createUniformPoints} from '../../inputs';
import {parameters} from './chunked-weights-algebra.parameters';
import {
  addLinkSegments,
  allocateWeights,
  getChunkRowCounts,
  sortPointsByX,
  uploadChunkedWeights
} from './shared/weights';

const OPERATIONS = ['union', 'intersection', 'difference', 'symmetricDifference'] as const;
const CHUNK_COLORS = [
  [1, 0.55, 0.2, 0.9],
  [0.35, 0.8, 1, 0.9],
  [0.7, 0.9, 0.35, 0.9],
  [0.95, 0.45, 0.8, 0.9]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'Combined weights, colored by the chunk of rows that produced them.',
  createInputs: ({random, world}) => {
    const points = createUniformPoints(random, world, 90, 0.1);
    return sortPointsByX(points.positions, points.values);
  },
  build(builder, points, values) {
    const rowCount = points.values.length;
    const chunkRowCounts = getChunkRowCounts(rowCount, values.chunkCount);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const left = uploadChunkedWeights(
      builder,
      'left',
      createKnnWeights(points.positions, 3),
      chunkRowCounts
    );
    const right = uploadChunkedWeights(
      builder,
      'right',
      createDistanceBandWeights(points.positions, 0.1),
      chunkRowCounts
    );
    const outputs = left.chunks.map((chunk, index) =>
      allocateWeights(
        builder,
        `combined-${index}`,
        chunk.rowCount,
        chunk.capacity + right.chunks[index].capacity
      )
    );
    builder.graph.add(
      new GPUChunkedSpatialWeightsAlgebra({
        operation: OPERATIONS[values.operation],
        left: left.input,
        right: right.input,
        output: {
          chunks: left.chunks.map((chunk, index) => ({
            rowStart: chunk.rowStart,
            rowCount: chunk.rowCount,
            weights: outputs[index].port
          }))
        }
      })
    );
    return {
      outputs: {
        positions,
        chunkLinks: outputs.map((output, index) =>
          addLinkSegments(builder, `combined-${index}`, output, positions, {
            rowStart: left.chunks[index].rowStart
          })
        )
      }
    };
  },
  draw: ({outputs}) => [
    ...outputs.chunkLinks.map(
      (segments, index) =>
        ({
          layer: 'lines',
          segments,
          color: CHUNK_COLORS[index % CHUNK_COLORS.length],
          radius: 1.3
        }) as const
    ),
    {layer: 'dots', context: true, positions: outputs.positions, radius: 3}
  ]
});
