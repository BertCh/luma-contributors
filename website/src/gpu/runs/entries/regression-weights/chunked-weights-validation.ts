// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUChunkedSpatialWeightsValidation over weights split into chunks of rows, with defects injected
// (a self slot, a neighbor id out of range, a negative weight, a duplicate neighbor). Points are
// colored by the violation flags written per row, in global row order whatever the chunking.

import {GPUChunkedSpatialWeightsValidation} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights, createUniformPoints} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {parameters} from './chunked-weights-validation.parameters';
import {addKernelPass} from './shared/kernel';
import {describeViolation, findNearestFlagged} from './shared/probes';
import {
  addLinkSegments,
  getChunkRowCounts,
  getSegmentTargets,
  sortPointsByX,
  uploadChunkBoundaries,
  uploadChunkedWeights,
  stretchPoints
} from './shared/weights';

export default defineOperationRun({
  parameters,
  caption: 'Rows colored by the violation flags the GPU wrote across chunks.',
  createInputs: ({random, world}) => {
    const points = stretchPoints(createUniformPoints(random, world, 140, 0.1), world);
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
    // Lowest violated bit per row (a category), 0xffffffff for a clean row.
    const kinds = builder.allocate('kinds', 'uint32', rowCount);
    addKernelPass(builder, {
      id: 'violation-kinds',
      invocationCount: rowCount,
      bindings: [
        {name: 'flags', buffer: flags, access: 'read'},
        {name: 'kinds', buffer: kinds, access: 'read_write'}
      ],
      body: `let flag = flags[flagsOffset + index];
  kinds[kindsOffset + index] = select(0xffffffffu, firstTrailingBit(flag), flag != 0u);`
    });
    return {
      outputs: {
        positions,
        kinds,
        flags,
        chunkStarts: weights.chunks.map(chunk => chunk.rowStart),
        boundaries: uploadChunkBoundaries(
          builder,
          points.positions,
          weights.chunks.map(chunk => chunk.rowStart)
        ),
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
  // The pointer finds the nearest flagged row; its ring, links and violation are named.
  interaction: {
    kind: 'probe',
    hint: 'hover a flagged point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      const flags = peek(outputs.flags);
      if (!flags) return {};
      const row = findNearestFlagged(flags, inputs.positions, pointer.x, pointer.y);
      if (row < 0) return {readout: 'no broken rows'};
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at: from, radius: 0.04},
        {shape: 'crosshair', at: from, size: 16}
      ];
      for (const links of outputs.links) {
        for (const to of getSegmentTargets(peek(links), from)) {
          glyphs.push({shape: 'line', from, to, dashed: true});
        }
      }
      return {focus: row, glyphs, readout: `row ${row} · ${describeViolation(flags[row])}`};
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.boundaries,
      count: outputs.chunkStarts.length - 1
    },
    ...outputs.links.map(
      segments =>
        ({layer: 'lines', context: true, segments, color: [0.6, 0.65, 0.85, 0.3]}) as const
    ),
    {layer: 'dots', context: true, positions: outputs.positions, radius: 2.4},
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.kinds,
      radius: 5.5
    }
  ]
});
