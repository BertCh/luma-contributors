// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryRestriction with a rectangle and a time window. Only the parts of each track that
// are inside the box during the window survive (bright); the rest of every track is muted.

import {GPUTrajectoryRestriction} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {createFleet} from './lib/fleet';
import {parameters} from './trajectory-restriction.parameters';

export default defineOperationRun({
  parameters,
  caption: 'Track parts inside the box and the time window.',
  createInputs: ({random, world}) =>
    createFleet(random, world, {trackCount: 14, minimumFixes: 22, maximumFixes: 34}),
  build(builder, fleet, _values) {
    const parameterBuffer = builder.allocate('parameters', 'float32', 7);
    const intervals = builder.allocate('intervals', 'float32x2', fleet.rowCount);
    const clipped = builder.allocate('clipped', 'float32x4', fleet.rowCount);
    const status = builder.allocate('status', 'uint32', fleet.rowCount);
    const relationships = builder.allocate('relationships', 'uint32x4', fleet.trackCount);
    const rectangle = builder.allocate('rectangle', 'float32x4', 4);
    builder.graph.add(
      new GPUTrajectoryRestriction({
        positions: builder.upload('positions', 'float32x2', fleet.positions).view,
        times: builder.upload('times', 'float32', fleet.timestamps).view,
        offsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        parameters: parameterBuffer.view,
        output: {
          intervals: intervals.view,
          positions: clipped.view,
          status: status.view,
          relationships: relationships.view
        }
      })
    );
    return {
      outputs: {
        clipped,
        rectangle,
        segments: builder.upload('segments', 'float32x4', fleet.segments)
      },
      update: current => {
        const minimumX = 0.2 * fleet.worldWidth;
        const maximumX = current.maximumX * fleet.worldWidth;
        parameterBuffer.write(
          Float32Array.of(
            current.minimumTime,
            current.maximumTime,
            minimumX,
            0.2,
            maximumX,
            0.8,
            1000
          )
        );
        rectangle.write(
          Float32Array.of(
            ...[minimumX, 0.2, maximumX, 0.2],
            ...[maximumX, 0.2, maximumX, 0.8],
            ...[maximumX, 0.8, minimumX, 0.8],
            ...[minimumX, 0.8, minimumX, 0.2]
          )
        );
      }
    };
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'lines', context: true, segments: outputs.rectangle, count: 4, radius: 1.5},
    {layer: 'lines', segments: outputs.clipped, count: inputs.rowCount, radius: 3}
  ]
});
