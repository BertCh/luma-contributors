// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrackSimilarity over every pair of 24 tracks from three shape families. The matrix is the
// per-pair distance (row = track A, column = track B): similar tracks form dark diagonal blocks.

import {GPUTrackSimilarity} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createTrackFamilies, TRACK_COUNT, VERTEX_COUNT} from './lib/families';
import {parameters} from './track-similarity.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Pairwise track distance matrix: shape families appear as blocks.',
  createInputs: ({random}) => createTrackFamilies(random),
  build(builder, families, values) {
    const positions = builder.allocate('positions', 'float32x2', TRACK_COUNT * VERTEX_COUNT);
    const pairCount = families.pairA.length;
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
          lineOffsets: builder.upload('line-offsets', 'uint32', families.lineOffsets).view
        },
        pairA: builder.upload('pair-a', 'uint32', families.pairA).view,
        pairB: builder.upload('pair-b', 'uint32', families.pairB).view,
        output: {[measure]: distances.view}
      })
    );
    return {
      outputs: {distances},
      update: current => positions.write(families.getPositions(current.noise))
    };
  },
  rebuildOn: ['measure'],
  draw: ({outputs}) => [
    {layer: 'raster', values: outputs.distances, columns: TRACK_COUNT, rows: TRACK_COUNT}
  ]
});
