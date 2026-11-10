// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUChunkedSpatialWeightsTranspose of directed k-nearest-neighbor weights held in chunks of rows.
// The transpose is one whole CSR whatever the chunking. Points are colored by in-degree (their row
// length in the transpose); the faint links are the transposed weights.

import {GPUChunkedSpatialWeightsTranspose} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights, createUniformPoints} from '../../inputs';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './chunked-weights-transpose.parameters';
import {
  addCardinality,
  addLinkSegments,
  allocateWeights,
  getChunkNumber,
  getChunkRowCounts,
  getSegmentTargets,
  sortPointsByX,
  uploadChunkBoundaries,
  uploadChunkedWeights,
  stretchPoints
} from './shared/weights';
import {getArrowGlyphs} from './shared/probes';

export default defineOperationRun({
  parameters,
  caption: 'In-degree of every point, from the transpose of chunked weights.',
  createInputs: ({random, world}) => {
    const points = stretchPoints(createUniformPoints(random, world, 200, 0.1), world);
    const sorted = sortPointsByX(points.positions, points.values);
    return {...sorted, index: createPointIndex(sorted.positions), focus: {row: -1}};
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
        chunkStarts: directed.chunks.map(chunk => chunk.rowStart),
        boundaries: uploadChunkBoundaries(
          builder,
          points.positions,
          directed.chunks.map(chunk => chunk.rowStart)
        ),
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
  // The pointer picks a row: its own k outgoing links point away, the links the transpose lists for it point in.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const outgoing = outputs.chunkLinks.flatMap(links => getSegmentTargets(peek(links), from));
      const incoming = getSegmentTargets(peek(outputs.links), from);
      return {
        focus: row,
        glyphs: [
          ...getArrowGlyphs(from, outgoing, 'outgoing'),
          ...getArrowGlyphs(from, incoming, 'incoming'),
          {shape: 'ring', at: from, radius: 0.012}
        ],
        readout: `row ${row} (chunk ${getChunkNumber(outputs.chunkStarts, row)}) · out ${outgoing.length} · in ${incoming.length}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.boundaries,
      count: outputs.chunkStarts.length - 1
    },
    {layer: 'lines', segments: outputs.links, color: [0.8, 0.85, 1, 0.3], radius: 1},
    {layer: 'dots', positions: outputs.positions, values: outputs.inDegree, radius: 3.6}
  ]
});
