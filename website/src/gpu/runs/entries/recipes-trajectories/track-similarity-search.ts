// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrackSimilaritySearch over every pair of 24 tracks: pairs whose endpoint bound already
// exceeds the threshold are pruned (blank); the rest show their exact distance.

import {GPUTrackSimilaritySearch} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {findParameter, type RunGlyph, snapToParameter} from '../../interaction';
import {getRasterCell, getRasterCellCenter} from '../../pick';
import {createTrackPairs, TRACK_COUNT, VERTEX_COUNT} from './lib/track-pairs';
import {parameters} from './track-similarity-search.parameters';

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

/** Discrete Frechet distance between two tracks of `VERTEX_COUNT` vertices. */
function getDiscreteFrechet(positions: Float32Array, first: number, second: number): number {
  const table = new Float32Array(VERTEX_COUNT * VERTEX_COUNT);
  for (let i = 0; i < VERTEX_COUNT; i++) {
    for (let j = 0; j < VERTEX_COUNT; j++) {
      const distance = Math.hypot(
        positions[(first * VERTEX_COUNT + i) * 2] - positions[(second * VERTEX_COUNT + j) * 2],
        positions[(first * VERTEX_COUNT + i) * 2 + 1] -
          positions[(second * VERTEX_COUNT + j) * 2 + 1]
      );
      const previous =
        i === 0 && j === 0
          ? 0
          : Math.min(
              i > 0 ? table[(i - 1) * VERTEX_COUNT + j] : Infinity,
              j > 0 ? table[i * VERTEX_COUNT + j - 1] : Infinity,
              i > 0 && j > 0 ? table[(i - 1) * VERTEX_COUNT + j - 1] : Infinity
            );
      table[i * VERTEX_COUNT + j] = Math.max(distance, previous);
    }
  }
  return table[VERTEX_COUNT * VERTEX_COUNT - 1];
}

export default defineOperationRun({
  parameters,
  caption: 'Pairs within the threshold (bright); pruned pairs stay empty.',
  createInputs: ({random}) => createTrackPairs(random),
  build(builder, tracks, values) {
    const positions = builder.allocate('positions', 'float32x2', TRACK_COUNT * VERTEX_COUNT);
    const threshold = builder.allocate('threshold', 'float32', 1);
    const pairCount = tracks.pairA.length;
    const distance = builder.allocate('distance', 'float32', pairCount);
    const lowerBound = builder.allocate('lower-bound', 'float32', pairCount);
    const matched = builder.allocate('matched', 'uint32', pairCount);
    const status = builder.allocate('status', 'uint32', pairCount);
    builder.graph.add(
      new GPUTrackSimilaritySearch({
        linesA: {
          kind: 'lines',
          positions: positions.view,
          lineOffsets: builder.upload('line-offsets', 'uint32', tracks.lineOffsets).view
        },
        pairA: builder.upload('pair-a', 'uint32', tracks.pairA).view,
        pairB: builder.upload('pair-b', 'uint32', tracks.pairB).view,
        metric: values.metric === 1 ? 'dynamic-time-warping' : 'frechet',
        threshold: threshold.view,
        output: {
          distance: distance.view,
          lowerBound: lowerBound.view,
          matched: matched.view,
          status: status.view
        }
      })
    );
    return {
      outputs: {distance, matched},
      update: current => {
        positions.write(tracks.getPositions(current.noise));
        threshold.write(Float32Array.of(current.threshold));
      }
    };
  },
  rebuildOn: ['metric'],
  // The distance of the cell under the pointer is the threshold: every pair at least that similar
  // is matched, and the pair under the pointer is drawn over the matrix.
  interaction: {
    kind: 'level',
    hint: 'pick a distance',
    rest: ({toUnit, inputs, world}) => {
      // The cell whose discrete Frechet distance (CPU, default noise) snaps to the default
      // threshold, nearest the middle. Only CPU inputs, so the rest is known before any readback.
      const parameter = findParameter(parameters, 'threshold');
      const positions = inputs.getPositions(parameters[2].default);
      const bounds = [0, 0, world.width, world.height] as const;
      let best: [number, number] = [world.width / 2, world.height / 2];
      let bestAway = Infinity;
      for (let row = 0; row < TRACK_COUNT; row++) {
        for (let column = 0; column < TRACK_COUNT; column++) {
          if (row === column) continue;
          const distance = getDiscreteFrechet(positions, row, column);
          if (snapToParameter(parameter, distance) !== parameter.default) continue;
          const center = getRasterCellCenter(TRACK_COUNT, TRACK_COUNT, bounds, column, row);
          const away = Math.hypot(center[0] - world.width / 2, center[1] - world.height / 2);
          if (away < bestAway) {
            bestAway = away;
            best = center;
          }
        }
      }
      return toUnit(best);
    },
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const bounds = [0, 0, world.width, world.height] as const;
      const cell = getRasterCell(TRACK_COUNT, TRACK_COUNT, bounds, pointer.x, pointer.y);
      const distances = peek(outputs.distance);
      const matched = peek(outputs.matched);
      const hovered = distances ? distances[cell.index] : Number.NaN;
      const threshold =
        Number.isFinite(hovered) && hovered > 0
          ? snapToParameter(findParameter(parameters, 'threshold'), hovered)
          : values.threshold;
      const matchedCount = matched ? Array.from(matched).filter(value => value === 1).length : 0;
      return {
        values: {threshold},
        glyphs: getPairGlyphs(
          inputs.getPositions(values.noise),
          cell.row,
          cell.column,
          getRasterCellCenter(TRACK_COUNT, TRACK_COUNT, bounds, cell.column, cell.row)
        ),
        readout: `threshold ${threshold.toFixed(2)} · ${matchedCount} pairs matched`
      };
    }
  },
  // Small distance = similar = bright: a negative lower bound lifts the near-zero matches off the
  // background, and the diagonal and nearby matches read at thumbnail size.
  draw: ({outputs, values}) => [
    {
      layer: 'raster',
      values: outputs.distance,
      columns: TRACK_COUNT,
      rows: TRACK_COUNT,
      range: [-0.15, Math.max(0.3, values.threshold)]
    }
  ]
});
