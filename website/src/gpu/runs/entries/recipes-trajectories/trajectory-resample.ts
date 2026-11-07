// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryResample over irregular GPS tracks. The raw fixes (muted) are replaced by the
// same number of evenly spaced samples per track: even in time, or even along the path.

import {GPUTrajectoryResample} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createFleet, createTrackEdges} from './lib/fleet';
import {parameters} from './trajectory-resample.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Every track resampled to the same number of evenly spaced points.',
  createInputs: ({random, world}) => createFleet(random, world, {trackCount: 10}),
  build(builder, fleet, values) {
    const sampleCount = values.sampleCount;
    const samples = builder.allocate('samples', 'float32x2', fleet.trackCount * sampleCount);
    const sampleTimes = builder.allocate('sample-times', 'float32', fleet.trackCount * sampleCount);
    builder.graph.add(
      new GPUTrajectoryResample({
        positions: builder.upload('positions', 'float32x2', fleet.positions).view,
        timestamps: builder.upload('timestamps', 'float32', fleet.timestamps).view,
        trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        sampleCount,
        spacing: values.spacing === 1 ? 'arc-length' : 'time',
        output: {samples: samples.view, sampleTimes: sampleTimes.view}
      })
    );
    return {
      outputs: {
        samples,
        sampleTimes,
        segments: builder.upload('segments', 'float32x4', fleet.segments),
        fixes: builder.upload('fixes', 'float32x2', fleet.positions),
        edges: builder.upload(
          'edges',
          'uint32x2',
          createTrackEdges(fleet.trackOffsets, sampleCount)
        )
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'dots', context: true, positions: outputs.fixes, radius: 2},
    {
      layer: 'segments',
      nodes: outputs.samples,
      edges: outputs.edges,
      values: outputs.sampleTimes,
      radius: 1.5
    },
    {layer: 'dots', positions: outputs.samples, values: outputs.sampleTimes, radius: 5}
  ]
});
