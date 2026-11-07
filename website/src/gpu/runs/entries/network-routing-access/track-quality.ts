// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addTrackQualityRecipe: flag raw track fixes that imply an impossible speed or follow a long time
// gap. Muted lines are the tracks; dots are the fixes, colored by their quality flag (clean,
// excessive speed, time gap, both, invalid).

import {addTrackQualityRecipe} from '@luma-contributors/analysis/gpu-network';
import {defineOperationRun} from '../../define-run';
import {createTrajectories} from '../../inputs';
import {addUint32Copy} from './shared/display';
import {parameters} from './track-quality.parameters';

const TRACK_COUNT = 14;
const STEP_COUNT = 40;

export default defineOperationRun({
  parameters,
  caption: 'Track fixes colored by quality flag: speed spikes and time gaps stand out.',
  createInputs: ({random, world}) => {
    const tracks = createTrajectories(random, world, TRACK_COUNT, STEP_COUNT);
    const fixCount = tracks.points.length / 2;
    return {
      tracks,
      jumpChance: Float32Array.from({length: fixCount}, () => random()),
      jumpOffset: Float32Array.from({length: fixCount * 2}, () => (random() - 0.5) * 0.5),
      gapChance: Float32Array.from({length: fixCount}, () => random()),
      points: new Float32Array(fixCount * 2),
      timestamps: new Float32Array(fixCount),
      segments: new Float32Array((fixCount - TRACK_COUNT) * 4)
    };
  },
  build(builder, inputs) {
    const {tracks} = inputs;
    const fixCount = tracks.points.length / 2;
    const positions = builder.allocate('positions', 'float32x2', fixCount);
    const timestamps = builder.allocate('timestamps', 'float32', fixCount);
    const thresholds = builder.allocate('thresholds', 'float32', 2);
    const result = addTrackQualityRecipe(builder.graph, {
      trajectory: {
        positions: positions.view,
        trackOffsets: builder.upload('track-offsets', 'uint32', tracks.trackOffsets).view,
        timestamps: timestamps.view
      },
      thresholds: thresholds.view
    });
    // The recipe owns its flag view; copy it to a buffer the layers can draw.
    const flags = addUint32Copy(builder, {
      id: 'flags',
      view: result.outputs.flags,
      // Bits: 1 invalid position, 2 invalid time, 4 excessive speed, 8 time gap. Class 0 is clean.
      transform:
        'result = select(select(select(select(0u, 1u, (value & 4u) != 0u), 2u, (value & 8u) != 0u), 3u, (value & 12u) == 12u), 4u, (value & 19u) != 0u);'
    });
    const lines = builder.upload('track-lines', 'float32x4', inputs.segments);
    return {
      outputs: {positions, flags, lines},
      update: values => {
        let elapsedExtra = 0;
        for (let fix = 0; fix < fixCount; fix++) {
          const isTrackStart = tracks.trackOffsets.includes(fix);
          if (isTrackStart) elapsedExtra = 0;
          const isGlitch = !isTrackStart && inputs.jumpChance[fix] < values.glitchRate;
          inputs.points[fix * 2] =
            tracks.points[fix * 2] + (isGlitch ? inputs.jumpOffset[fix * 2] : 0);
          inputs.points[fix * 2 + 1] =
            tracks.points[fix * 2 + 1] + (isGlitch ? inputs.jumpOffset[fix * 2 + 1] : 0);
          if (!isTrackStart && inputs.gapChance[fix] < values.glitchRate * 0.8) elapsedExtra += 6;
          inputs.timestamps[fix] = tracks.timestamps[fix] + elapsedExtra;
        }
        let segment = 0;
        for (let fix = 0; fix < fixCount - 1; fix++) {
          if (tracks.trackOffsets.includes(fix + 1)) continue;
          inputs.segments.set(
            [
              inputs.points[fix * 2],
              inputs.points[fix * 2 + 1],
              inputs.points[fix * 2 + 2],
              inputs.points[fix * 2 + 3]
            ],
            segment++ * 4
          );
        }
        positions.write(inputs.points);
        timestamps.write(inputs.timestamps);
        lines.write(inputs.segments);
        thresholds.write(Float32Array.of(values.maximumSpeed, values.maximumGap));
      }
    };
  },
  draw: ({outputs}) => [
    {layer: 'lines', context: true, segments: outputs.lines, radius: 1},
    {layer: 'dots', positions: outputs.positions, values: outputs.flags, radius: 4}
  ]
});
