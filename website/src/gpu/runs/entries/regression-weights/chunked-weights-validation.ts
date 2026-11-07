// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUChunkedSpatialWeightsValidation over weights split into chunks of rows, with defects injected
// (a self slot, a neighbor id out of range, a negative weight, a duplicate neighbor). Points are
// colored by the violation flags written per row, in global row order whatever the chunking.

import {GPUChunkedSpatialWeightsValidation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights, createUniformPoints} from '../../inputs';
import {parameters} from './chunked-weights-validation.parameters';
import {addKernelPass} from './shared/kernel';
import {
  addLinkSegments,
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

export default defineOperationRun({
  parameters,
  caption: 'Rows colored by the violation flags the GPU wrote across chunks.',
  createInputs: ({random, world}) => {
    const points = createUniformPoints(random, world, 140, 0.1);
    return sortPointsByX(points.positions, points.values);
  },
  build(builder, points, values) {
    const rowCount = points.values.length;
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const csr = createKnnWeights(points.positions, 4);
    const weights = uploadChunkedWeights(
      builder,
      'uploaded',
      csr,
      getChunkRowCounts(rowCount, values.chunkCount)
    );
    const flags = builder.allocate('flags', 'uint32', rowCount);
    const invalidCount = builder.allocate('invalid-count', 'uint32', 1);
    builder.graph.add(
      new GPUChunkedSpatialWeightsValidation({
        weights: weights.input,
        output: {flags: flags.view, status: {invalidCount: invalidCount.view}}
      })
    );
    // Violation kind per row: 0 for a clean row, 2 plus the lowest violated bit otherwise.
    const kinds = builder.allocate('kinds', 'float32', rowCount);
    addKernelPass(builder, {
      id: 'violation-kinds',
      invocationCount: rowCount,
      bindings: [
        {name: 'flags', buffer: flags, access: 'read'},
        {name: 'kinds', buffer: kinds, access: 'read_write'}
      ],
      body: `let flag = flags[flagsOffset + index];
  kinds[kindsOffset + index] = select(0.0, 2.0 + f32(firstTrailingBit(flag)), flag != 0u);`
    });
    return {
      outputs: {
        positions,
        kinds,
        links: weights.chunks.map((chunk, index) =>
          addLinkSegments(builder, `uploaded-${index}`, chunk, positions, {
            rowStart: chunk.rowStart
          })
        )
      },
      update: current => {
        const neighbors = Uint32Array.from(csr.neighbors);
        const slotWeights = Float32Array.from(csr.weights);
        for (let row = 0; row < rowCount; row++) {
          const hash = Math.sin(row * 78.233) * 43758.5453;
          if (hash - Math.floor(hash) >= current.defects) continue;
          const first = csr.offsets[row];
          const kind = row % 4;
          if (kind === 0) neighbors[first] = row;
          else if (kind === 1) neighbors[first] = rowCount + 3;
          else if (kind === 2) slotWeights[first] = -1;
          else neighbors[first + 1] = neighbors[first];
        }
        for (const chunk of weights.chunks) {
          const begin = csr.offsets[chunk.rowStart];
          const end = csr.offsets[chunk.rowStart + chunk.rowCount];
          chunk.neighbors.write(neighbors.slice(begin, end));
          chunk.weights.write(slotWeights.slice(begin, end));
        }
      }
    };
  },
  rebuildOn: ['chunkCount'],
  draw: ({outputs}) => [
    ...outputs.links.map(
      (segments, index) =>
        ({
          layer: 'lines',
          segments,
          color: CHUNK_COLORS[index % CHUNK_COLORS.length],
          radius: 1.2
        }) as const
    ),
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.4},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.kinds,
      ramp: 'sequential',
      range: [0, 7],
      fade: true,
      radius: 4.2
    }
  ]
});
