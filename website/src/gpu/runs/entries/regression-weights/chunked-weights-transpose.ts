// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUChunkedSpatialWeightsTranspose of directed k-nearest-neighbor weights held in chunks of rows.
// The transpose is one whole CSR whatever the chunking. Points are colored by in-degree (their row
// length in the transpose); the faint links are the transposed weights.

import {GPUChunkedSpatialWeightsTranspose} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights, createUniformPoints} from '../../inputs';
import {parameters} from './chunked-weights-transpose.parameters';
import {
  addCardinality,
  addLinkSegments,
  allocateWeights,
  getChunkRowCounts,
  sortPointsByX,
  uploadChunkedWeights
} from './shared/weights';

const CHUNK_COLORS = [
  [0.95, 0.55, 0.25, 0.25],
  [0.35, 0.75, 1, 0.25],
  [0.65, 0.9, 0.4, 0.25],
  [0.95, 0.45, 0.8, 0.25]
] as const;

export default defineOperationRun({
  parameters,
  caption: 'In-degree of every point, from the transpose of chunked weights.',
  createInputs: ({random, world}) => {
    const points = createUniformPoints(random, world, 200, 0.1);
    return sortPointsByX(points.positions, points.values);
  },
  build(builder, points, values) {
    const rowCount = points.values.length;
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const directed = uploadChunkedWeights(
      builder,
      'directed',
      createKnnWeights(points.positions, values.k),
      getChunkRowCounts(rowCount, values.chunkCount)
    );
    const slotCapacity = directed.chunks.reduce((total, chunk) => total + chunk.capacity, 0);
    const transposed = allocateWeights(builder, 'transposed', rowCount, slotCapacity, true);
    builder.graph.add(
      new GPUChunkedSpatialWeightsTranspose({weights: directed.input, output: transposed.port})
    );
    return {
      outputs: {
        positions,
        inDegree: addCardinality(builder, 'transposed', transposed),
        links: addLinkSegments(builder, 'transposed', transposed, positions),
        chunkLinks: directed.chunks.map((chunk, index) =>
          addLinkSegments(builder, `directed-${index}`, chunk, positions, {
            rowStart: chunk.rowStart
          })
        )
      }
    };
  },
  draw: ({outputs}) => [
    // Each chunk of the input rows has its own faint color; the transposed links draw on top.
    ...outputs.chunkLinks.map(
      (segments, index) =>
        ({
          layer: 'lines',
          segments,
          color: CHUNK_COLORS[index % CHUNK_COLORS.length],
          radius: 2.4
        }) as const
    ),
    {layer: 'lines', segments: outputs.links, color: [0.8, 0.85, 1, 0.5], radius: 1},
    {layer: 'dots', positions: outputs.positions, values: outputs.inDegree, radius: 3.6}
  ]
});
