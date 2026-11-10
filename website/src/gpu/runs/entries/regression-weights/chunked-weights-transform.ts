// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUChunkedSpatialWeightsTransform applies one transform across chunks of rows. Points are colored
// by the sum of their transformed weights; link colors mark the chunks of rows, each transformed on its own
// but with global sums for the `double` and `variance` operations.

import {GPUChunkedSpatialWeightsTransform} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createDistanceBandWeights, createUniformPoints} from '../../inputs';
import type {RunGlyph} from '../../interaction';
import {chooseSticky, createPointIndex} from '../../pick';
import {parameters} from './chunked-weights-transform.parameters';
import {
  addLinkSegments,
  addRowSums,
  getChunkNumber,
  getChunkRowCounts,
  getRowLinkGlyphs,
  getSegmentTargets,
  sortPointsByX,
  uploadChunkBoundaries,
  uploadChunkedWeights,
  stretchPoints
} from './shared/weights';

const BAND_RADIUS = 0.1;
const OPERATIONS = ['row', 'binary', 'kernel', 'double', 'variance'] as const;

export default defineOperationRun({
  parameters,
  caption: 'Row sums of the transformed weights, computed chunk by chunk.',
  createInputs: ({random, world}) => {
    const points = stretchPoints(createUniformPoints(random, world, 260, 0.1), world);
    const sorted = sortPointsByX(points.positions, points.values);
    return {...sorted, index: createPointIndex(sorted.positions), focus: {row: -1}};
  },
  build(builder, points, values) {
    const rowCount = points.values.length;
    const chunkRowCounts = getChunkRowCounts(rowCount, values.chunkCount);
    const positions = builder.upload('positions', 'float32x2', points.positions);
    const band = uploadChunkedWeights(
      builder,
      'band',
      createDistanceBandWeights(points.positions, BAND_RADIUS),
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
        chunkStarts: band.chunks.map(chunk => chunk.rowStart),
        boundaries: uploadChunkBoundaries(
          builder,
          points.positions,
          band.chunks.map(chunk => chunk.rowStart)
        ),
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
  // The pointer picks a row: its band ring, its links, and the transformed weight sum of the row.
  interaction: {
    kind: 'probe',
    hint: 'hover a point',
    rest: [0.5, 0.5],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const row = chooseSticky(inputs.index, pointer.x, pointer.y, inputs.focus.row);
      inputs.focus.row = row;
      if (row < 0) return {};
      const from = [inputs.positions[row * 2], inputs.positions[row * 2 + 1]] as const;
      const links = outputs.chunks.flatMap(chunk => getSegmentTargets(peek(chunk.links), from));
      const chunkNumber = getChunkNumber(outputs.chunkStarts, row);
      const chunk = outputs.chunks[chunkNumber - 1];
      const rowSums = peek(chunk.rowSums);
      const rowSum = rowSums ? rowSums[row - outputs.chunkStarts[chunkNumber - 1]] : undefined;
      const glyphs: RunGlyph[] = [
        {shape: 'ring', at: from, radius: values.bandwidth, dashed: true},
        ...getRowLinkGlyphs(from, links)
      ];
      return {
        focus: row,
        glyphs,
        readout: `row ${row} (chunk ${chunkNumber}) · ${links.length} links · sum ${rowSum?.toFixed(2) ?? '-'}`
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
    ...outputs.chunks.map(
      chunk =>
        ({
          layer: 'lines',
          segments: chunk.links,
          color: [0.6, 0.65, 0.85, 0.3],
          radius: 1.1
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
