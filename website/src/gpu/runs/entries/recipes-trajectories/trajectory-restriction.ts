// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryRestriction with a rectangle and a time window. Only the parts of each track that
// are inside the box during the window survive (bright); the rest of every track is muted.

import {GPUTrajectoryRestriction} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {clamp} from '../../interaction';
import {createFleet} from './lib/fleet';
import {parameters} from './trajectory-restriction.parameters';

const BOX_START_X = 0.45;
const BOX_HALF_HEIGHT = 0.3;

/** The restriction rectangle: centered on the pointer-owned center, `maximumX` sets its width. */
function getBox(fleet: {worldWidth: number; center: readonly [number, number]}, maximumX: number) {
  const halfWidth = ((maximumX - 0.2) / 2) * fleet.worldWidth;
  return {
    minimumX: fleet.center[0] - halfWidth,
    maximumX: fleet.center[0] + halfWidth,
    minimumY: fleet.center[1] - BOX_HALF_HEIGHT,
    maximumY: fleet.center[1] + BOX_HALF_HEIGHT
  };
}

export default defineOperationRun({
  parameters,
  caption: 'Track parts inside the box and the time window.',
  createInputs: ({world}) => {
    const fleet = createFleet(world, {trackCount: 14, minimumFixes: 22, maximumFixes: 34});
    return {...fleet, center: [BOX_START_X * fleet.worldWidth, 0.5] as [number, number]};
  },
  build(builder, fleet) {
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
        const {minimumX, maximumX, minimumY, maximumY} = getBox(fleet, current.maximumX);
        parameterBuffer.write(
          Float32Array.of(
            current.minimumTime,
            current.maximumTime,
            minimumX,
            minimumY,
            maximumX,
            maximumY,
            1000
          )
        );
        rectangle.write(
          Float32Array.of(
            ...[minimumX, minimumY, maximumX, minimumY],
            ...[maximumX, minimumY, maximumX, maximumY],
            ...[maximumX, maximumY, minimumX, maximumY],
            ...[minimumX, maximumY, minimumX, minimumY]
          )
        );
      }
    };
  },
  // The pointer is the center of the box: parts of tracks inside it during the window survive.
  interaction: {
    kind: 'window',
    hint: 'move the box',
    rest: ({toUnit, inputs}) => toUnit(inputs.center),
    resolve(pointer, {inputs, values}) {
      const center: [number, number] = [
        clamp(Math.round(pointer.x * 50) / 50, 0, inputs.worldWidth),
        clamp(Math.round(pointer.y * 50) / 50, 0.1, 0.9)
      ];
      inputs.center = center;
      const box = getBox(inputs, values.maximumX);
      let trackCount = 0;
      for (let track = 0; track < inputs.trackCount; track++) {
        for (let row = inputs.trackOffsets[track]; row < inputs.trackOffsets[track + 1]; row++) {
          const x = inputs.positions[row * 2];
          const y = inputs.positions[row * 2 + 1];
          const time = inputs.timestamps[row];
          if (
            x >= box.minimumX &&
            x <= box.maximumX &&
            y >= box.minimumY &&
            y <= box.maximumY &&
            time >= values.minimumTime &&
            time <= values.maximumTime
          ) {
            trackCount++;
            break;
          }
        }
      }
      return {
        rerun: true,
        glyphs: [
          {
            shape: 'box',
            minimum: [box.minimumX, box.minimumY],
            maximum: [box.maximumX, box.maximumY],
            dashed: true
          }
        ],
        readout: `box (${center[0].toFixed(2)}, ${center[1].toFixed(2)}) · ${trackCount} tracks inside`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {layer: 'lines', context: true, segments: outputs.segments, count: inputs.segmentCount},
    {layer: 'lines', context: true, segments: outputs.rectangle, count: 4, radius: 1.5},
    {layer: 'lines', segments: outputs.clipped, count: inputs.rowCount, radius: 3}
  ]
});
