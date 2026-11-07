// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryEncounters over tracks resampled to a shared clock. The dots are the resampled
// table; each bar is one encounter pair (sorted by track) whose height is the number of time
// buckets the two tracks spent within `distance`.

import {
  GPUTrajectoryEncounters,
  GPUTrajectoryResample
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createFleet, raiseFleet} from './lib/fleet';
import {parameters} from './trajectory-encounters.parameters';

const PAIR_CAPACITY = 40;
const CELL_SIZE = 0.16;

export default defineOperationRun({
  parameters,
  caption: 'Track pairs within a distance of each other in the same time bucket.',
  createInputs: ({random, world}) =>
    raiseFleet(
      createFleet(random, world, {trackCount: 18, minimumFixes: 24, maximumFixes: 34}),
      0.42
    ),
  build(builder, fleet, values) {
    const bucketCount = values.bucketCount;
    const distance = builder.allocate('distance', 'float32', 1);
    const samples = builder.allocate('samples', 'float32x2', fleet.trackCount * bucketCount);
    const ids = builder.allocate('pair-ids', 'uint32', PAIR_CAPACITY);
    const count = builder.allocate('pair-count', 'uint32', 1);
    const overflow = builder.allocate('pair-overflow', 'uint32', 1);
    const bucketCounts = builder.allocate('bucket-counts', 'uint32', PAIR_CAPACITY);
    const candidateOverflow = builder.allocate('candidate-overflow', 'uint32', 1);
    builder.graph.add(
      new GPUTrajectoryResample({
        positions: builder.upload('positions', 'float32x2', fleet.positions).view,
        timestamps: builder.upload('timestamps', 'float32', fleet.timestamps).view,
        trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        sampleCount: bucketCount,
        output: {samples: samples.view}
      })
    );
    builder.graph.add(
      new GPUTrajectoryEncounters({
        samples: samples.view,
        trackCount: fleet.trackCount,
        bucketCount,
        distance: distance.view,
        cellSize: CELL_SIZE,
        bounds: [0, 0, fleet.worldWidth, 1],
        hitCapacity: 1 << 16,
        output: {
          pairs: {
            ids: ids.view,
            status: {count: count.view, overflow: overflow.view},
            bucketCounts: bucketCounts.view
          },
          status: {candidateOverflow: candidateOverflow.view}
        }
      })
    );
    return {
      outputs: {
        samples,
        bucketCounts,
        segments: builder.upload('segments', 'float32x4', fleet.segments)
      },
      update: current => distance.write(Float32Array.of(current.distance))
    };
  },
  rebuildOn: ['bucketCount'],
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', context: true, positions: outputs.samples, radius: 3.5},
    {
      layer: 'chart',
      kind: 'bars',
      values: outputs.bucketCounts,
      ramp: 'sequential',
      rectangle: [0.06, 0.05, 0.94, 0.38]
    }
  ]
});
