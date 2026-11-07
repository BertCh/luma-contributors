// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryClosestApproach between raw tracks. Each bar is one pair of tracks, as tall as the
// time they spent within `distance` of each other (solved exactly between fixes).

import {GPUTrajectoryClosestApproach} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createFleet, raiseFleet} from './lib/fleet';
import {parameters} from './trajectory-closest-approach.parameters';

const PAIR_CAPACITY = 40;

export default defineOperationRun({
  parameters,
  caption: 'Exact time spent within the distance, per pair of tracks.',
  createInputs: ({random, world}) =>
    raiseFleet(
      createFleet(random, world, {trackCount: 18, minimumFixes: 24, maximumFixes: 34}),
      0.42
    ),
  build(builder, fleet) {
    const distance = builder.allocate('distance', 'float32', 1);
    const ids = builder.allocate('pair-ids', 'uint32', PAIR_CAPACITY);
    const count = builder.allocate('pair-count', 'uint32', 1);
    const overflow = builder.allocate('pair-overflow', 'uint32', 1);
    const durations = builder.allocate('durations', 'float32', PAIR_CAPACITY);
    const entryOverflow = builder.allocate('entry-overflow', 'uint32', 1);
    const hitOverflow = builder.allocate('hit-overflow', 'uint32', 1);
    builder.graph.add(
      new GPUTrajectoryClosestApproach({
        positions: builder.upload('positions', 'float32x2', fleet.positions).view,
        timestamps: builder.upload('timestamps', 'float32', fleet.timestamps).view,
        trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        distance: distance.view,
        cellSize: 0.16,
        timeCellSize: 60,
        entryCapacity: 1 << 16,
        hitCapacity: 1 << 16,
        output: {
          pairs: {
            ids: ids.view,
            status: {count: count.view, overflow: overflow.view},
            durations: durations.view
          },
          status: {entryOverflow: entryOverflow.view, hitOverflow: hitOverflow.view}
        }
      })
    );
    return {
      outputs: {
        durations,
        nodes: builder.upload('nodes', 'float32x2', fleet.positions),
        segments: builder.upload('segments', 'float32x4', fleet.segments)
      },
      update: current => distance.write(Float32Array.of(current.distance))
    };
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', context: true, positions: outputs.nodes, radius: 3},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.durations,
      ramp: 'sequential',
      rectangle: [0.06, 0.05, 0.94, 0.38]
    }
  ]
});
