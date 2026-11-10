// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUChunkedSpatialWeightsAlgebra combines k nearest neighbors (blue) with a distance band (green),
// one chunk of rows at a time. Rows are split into vertical bands whose boundaries are drawn; the
// combined links (orange) are the same whatever the chunking.

import {GPUChunkedSpatialWeightsAlgebra} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createDistanceBandWeights, createKnnWeights, createUniformPoints} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './chunked-weights-algebra.parameters';
import {
  addLinkSegments,
  allocateWeights,
  getChunkNumber,
  getChunkRowCounts,
  getRowLinkGlyphs,
  getSegmentTargets,
  sortPointsByX,
  uploadChunkBoundaries,
  uploadChunkedWeights,
  stretchPoints
} from './shared/weights';

const OPERATIONS = ['union', 'intersection', 'difference', 'symmetricDifference'] as const;
const BAND_RADIUS = 0.1;

export default defineOperationRun({
  parameters,
  caption: 'Left (blue) and right (green) weights and their combination (orange), chunk by chunk.',
  createInputs: ({random, world}) => {
    const points = sortPointsByX(
      stretchPoints(createUniformPoints(random, world, 90, 0.1), world).positions,
      new Float32Array(90)
    );
    return {...points, index: createPointIndex(points.positions), focus: {row: -1}};
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
      createDistanceBandWeights(points.positions, BAND_RADIUS),
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
        chunkStarts: left.chunks.map(chunk => chunk.rowStart),
        boundaries: uploadChunkBoundaries(
          builder,
          points.positions,
          left.chunks.map(chunk => chunk.rowStart)
        ),
        combinedLinks: outputs.map((output, index) =>
          addLinkSegments(builder, `combined-${index}`, output, positions, {
            rowStart: left.chunks[index].rowStart
          })
        ),
        leftLinks: left.chunks.map((chunk, index) =>
          addLinkSegments(builder, `left-${index}`, chunk, positions, {rowStart: chunk.rowStart})
        ),
        rightLinks: right.chunks.map((chunk, index) =>
          addLinkSegments(builder, `right-${index}`, chunk, positions, {rowStart: chunk.rowStart})
        )
      }
    };
  },
  // The pointer picks a row: its kNN links (blue, dashed), its band (ring) and the combined links it keeps.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.4, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const combined = outputs.combinedLinks.flatMap(links => getSegmentTargets(peek(links), from));
      const knn = outputs.leftLinks.flatMap(links => getSegmentTargets(peek(links), from));
      const chunk = getChunkNumber(outputs.chunkStarts, row);
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at: from, radius: BAND_RADIUS, dashed: true},
        ...getRowLinkGlyphs(from, knn, {dashed: true, dots: false}),
        ...getRowLinkGlyphs(from, combined)
      ];
      return {
        focus: row,
        glyphs,
        readout: `${OPERATIONS[values.operation]} · row ${row} (chunk ${chunk}) ${combined.length} links, kNN ${knn.length}`
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
    ...outputs.leftLinks.map(
      segments => ({layer: 'lines', segments, color: [0.35, 0.6, 1, 0.3], radius: 1}) as const
    ),
    ...outputs.rightLinks.map(
      segments => ({layer: 'lines', segments, color: [0.35, 0.85, 0.5, 0.3], radius: 1}) as const
    ),
    ...outputs.combinedLinks.map(
      segments => ({layer: 'lines', segments, color: [1, 0.55, 0.2, 0.85], radius: 2}) as const
    ),
    {layer: 'dots', context: true, positions: outputs.positions, radius: 3}
  ]
});
