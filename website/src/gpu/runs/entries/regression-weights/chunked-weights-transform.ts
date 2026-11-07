// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUChunkedSpatialWeightsTransform applies one transform across chunks of rows. Points are colored
// by the sum of their transformed weights; link colors mark the chunks of rows, each transformed on its own
// but with global sums for the `double` and `variance` operations.

import {GPUChunkedSpatialWeightsTransform} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createDistanceBandWeights, createUniformPoints} from '../../inputs';
import {parameters} from './chunked-weights-transform.parameters';
import {
  addLinkSegments,
  addRowSums,
  getChunkRowCounts,
  sortPointsByX,
  uploadChunkedWeights
} from './shared/weights';

const CHUNK_COLORS = [
  [0.95, 0.55, 0.25, 0.4],
  [0.35, 0.75, 1, 0.4],
  [0.65, 0.9, 0.4, 0.4],
  [0.95, 0.45, 0.8, 0.4]
] as const;
const OPERATIONS = ['row', 'binary', 'kernel', 'double', 'variance'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Row sums of the transformed weights, computed chunk by chunk.',
  createInputs: ({random, world}) => {
    const points = createUniformPoints(random, world, 260, 0.1);
    return sortPointsByX(points.positions, points.values);
  },
  build(builder, points, values) {
    const rowCount = points.values.length;
    const chunkRowCounts = getChunkRowCounts(rowCount, values.chunkCount);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const band = uploadChunkedWeights(
      builder,
      'band',
      createDistanceBandWeights(points.positions, 0.1),
      chunkRowCounts
    );
    const transformed = band.chunks.map((chunk, index) =>
      builder.allocate(`transformed-${index}`, 'float32', chunk.capacity)
    );
    builder.graph.add(
      new GPUChunkedSpatialWeightsTransform({
        operation: OPERATIONS[values.operation],
        weights: band.input,
        output: transformed.map(buffer => buffer.view),
        kernel: 'triangular',
        bandwidth: values.bandwidth
      })
    );
    return {
      outputs: {
        chunks: band.chunks.map((chunk, index) => ({
          positions: builder.upload(
            `chunk-positions-${index}`,
            'float32x2',
            points.positions.slice(chunk.rowStart * 2, (chunk.rowStart + chunk.rowCount) * 2)
          ),
          rowSums: addRowSums(builder, `transformed-${index}`, chunk, transformed[index]),
          links: addLinkSegments(builder, `band-${index}`, chunk, positions, {
            rowStart: chunk.rowStart
          })
        }))
      }
    };
  },
  rebuildOn: ['operation', 'bandwidth', 'chunkCount'],
  draw: ({outputs}) => [
    ...outputs.chunks.map(
      (chunk, index) =>
        ({
          layer: 'lines',
          segments: chunk.links,
          color: CHUNK_COLORS[index % CHUNK_COLORS.length],
          radius: 1.2
        }) as const
    ),
    ...outputs.chunks.map(
      chunk =>
        ({
          layer: 'dots',
          positions: chunk.positions,
          values: chunk.rowSums,
          radius: 3.6,
          range: [0, 6]
        }) as const
    )
  ]
});
