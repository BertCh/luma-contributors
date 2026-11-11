// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUChunkedSpatialWeightsTranspose of directed k-nearest-neighbor weights held in chunks of rows.
// The transpose is one whole CSR whatever the chunking. Points are colored by in-degree (their row
// length in the transpose); the faint links are the transposed weights.

import {GPUChunkedSpatialWeightsTranspose} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createKnnWeights, createUniformPoints} from '../../inputs';
import type {RunGlyph} from '../../interaction';
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
import {mergeText} from '../../text';

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
      // Outgoing links are solid arrows; the sources the transpose lists are dashed lines ending
      // in a dot, so the two directions differ in more than the arrow head.
      const sources: RunGlyph[] = incoming.flatMap(source => [
        {shape: 'line' as const, from: source, to: from, dashed: true},
        {shape: 'dot' as const, at: source}
      ]);
      return {
        focus: row,
        glyphs: [
          ...sources,
          ...getArrowGlyphs(from, outgoing, 'outgoing'),
          {shape: 'ring', at: from, radius: 0.012}
        ],
        readout: `row ${row} (chunk ${getChunkNumber(outputs.chunkStarts, row)}) · out ${outgoing.length} · in ${incoming.length}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.boundaries,
      count: outputs.chunkStarts.length - 1
    },
    {layer: 'lines', segments: outputs.links, color: [...theme.muted, 0.45], radius: 1},
    {layer: 'dots', positions: outputs.positions, values: outputs.inDegree, radius: 4.8},
    {
      layer: 'text',
      ...mergeText({
        labels: [
          {
            text: 'dot color: in-degree',
            at: [0, 0],
            space: 'canvas',
            anchor: 'left',
            offset: [8, -10],
            role: 'label',
            priority: 4
          },
          {
            text: 'hover: solid arrows out, dashed lines in',
            at: [0, 0],
            space: 'canvas',
            anchor: 'left',
            offset: [8, -23],
            role: 'tick',
            priority: 3
          }
        ]
      })
    }
  ]
});
