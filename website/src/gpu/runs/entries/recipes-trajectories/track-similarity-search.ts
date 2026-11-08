// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrackSimilaritySearch over every pair of 24 tracks: pairs whose endpoint bound already
// exceeds the threshold are pruned (blank); the rest show their exact distance.

import {GPUTrackSimilaritySearch} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTrackFamilies, TRACK_COUNT, VERTEX_COUNT} from './lib/families';
import {parameters} from './track-similarity-search.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Pairs within the threshold (bright); pruned pairs stay empty.',
  createInputs: ({random}) => createTrackFamilies(random),
  build(builder, families, values) {
    const positions = builder.allocate('positions', 'float32x2', TRACK_COUNT * VERTEX_COUNT);
    const threshold = builder.allocate('threshold', 'float32', 1);
    const pairCount = families.pairA.length;
    const distance = builder.allocate('distance', 'float32', pairCount);
    const lowerBound = builder.allocate('lower-bound', 'float32', pairCount);
    const matched = builder.allocate('matched', 'uint32', pairCount);
    const status = builder.allocate('status', 'uint32', pairCount);
    builder.graph.add(
      new GPUTrackSimilaritySearch({
        linesA: {
          kind: 'lines',
          positions: positions.view,
          lineOffsets: builder.upload('line-offsets', 'uint32', families.lineOffsets).view
        },
        pairA: builder.upload('pair-a', 'uint32', families.pairA).view,
        pairB: builder.upload('pair-b', 'uint32', families.pairB).view,
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
        positions.write(families.getPositions(current.noise));
        threshold.write(Float32Array.of(current.threshold));
      }
    };
  },
  rebuildOn: ['metric'],
  // Small distance = similar = bright: a negative lower bound lifts the near-zero matches off the
  // background, and the diagonal and matched family blocks read at thumbnail size.
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
