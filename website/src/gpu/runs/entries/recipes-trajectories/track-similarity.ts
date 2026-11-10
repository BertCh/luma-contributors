// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrackSimilarity over every pair of 24 tracks from three shape tracks. The matrix is the
// per-pair distance (row = track A, column = track B): similar tracks form dark diagonal blocks.

import {GPUTrackSimilarity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {getRasterCell, getRasterCellCenter} from '../../pick';
import {createTrackPairs, TRACK_COUNT, VERTEX_COUNT} from './lib/track-pairs';
import {parameters} from './track-similarity.parameters';

/** Both tracks of a matrix cell as polylines, with a crosshair at the cell. */
function getPairGlyphs(
  positions: Float32Array,
  row: number,
  column: number,
  center: readonly [number, number]
): RunGlyph[] {
  const vertices = (track: number) =>
    positions.subarray(track * VERTEX_COUNT * 2, (track + 1) * VERTEX_COUNT * 2);
  return [
    {shape: 'polyline', points: vertices(row)},
    {shape: 'polyline', points: vertices(column), dashed: true},
    {shape: 'crosshair', at: center}
  ];
}

export default defineOperationRun({
  parameters,
  caption: 'Pairwise distances between vessel tracks, ordered by starting longitude.',
  createInputs: ({random}) => createTrackPairs(random),
  build(builder, tracks, values) {
    const positions = builder.allocate('positions', 'float32x2', TRACK_COUNT * VERTEX_COUNT);
    const pairCount = tracks.pairA.length;
    const distances = builder.allocate('distances', 'float32', pairCount);
    const measure = ['hausdorff', 'frechet', 'dynamicTimeWarping'][values.measure] as
      | 'hausdorff'
      | 'frechet'
      | 'dynamicTimeWarping';
    builder.graph.add(
      new GPUTrackSimilarity({
        linesA: {
          kind: 'lines',
          positions: positions.view,
          lineOffsets: builder.upload('line-offsets', 'uint32', tracks.lineOffsets).view
        },
        pairA: builder.upload('pair-a', 'uint32', tracks.pairA).view,
        pairB: builder.upload('pair-b', 'uint32', tracks.pairB).view,
        output: {[measure]: distances.view}
      })
    );
    return {
      outputs: {distances},
      update: current => positions.write(tracks.getPositions(current.noise))
    };
  },
  rebuildOn: ['measure'],
  // The pointer picks a matrix cell: its two tracks are drawn over the dimmed matrix.
  interaction: {
    kind: 'probe',
    hint: 'hover a pair',
    rest: [0.4, 0.4],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const bounds = [0, 0, world.width, world.height] as const;
      const cell = getRasterCell(TRACK_COUNT, TRACK_COUNT, bounds, pointer.x, pointer.y);
      const distances = peek(outputs.distances);
      const measure = ['hausdorff', 'frechet', 'dynamic time warping'][values.measure];
      return {
        glyphs: getPairGlyphs(
          inputs.getPositions(values.noise),
          cell.row,
          cell.column,
          getRasterCellCenter(TRACK_COUNT, TRACK_COUNT, bounds, cell.column, cell.row)
        ),
        readout: `track ${cell.row} x track ${cell.column} · ${measure} ${distances ? distances[cell.index].toFixed(2) : '...'}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'raster',
      values: outputs.distances,
      columns: TRACK_COUNT,
      rows: TRACK_COUNT,
      alpha: 0.6
    }
  ]
});
