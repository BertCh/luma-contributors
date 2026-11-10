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

const TRACK_COUNT = 7;
const STEP_COUNT = 40;

export default defineOperationRun({
  parameters,
  caption: 'Track fixes colored by quality flag: speed spikes and time gaps stand out.',
  createInputs: ({random, world}) => {
    const tracks = createTrajectories(random, world, TRACK_COUNT, STEP_COUNT);
    const fixCount = tracks.points.length / 2;
    const jumpChances = Float32Array.from({length: fixCount}, () => random());
    return {
      tracks,
      jumpChance: jumpChances,
      jumpOffset: Float32Array.from({length: fixCount * 2}, () => (random() - 0.5) * 0.5),
      // Each fix that follows a gap waits 1.5 to 10.5 time units, so `maximumGap` sorts them.
      gapChance: Float32Array.from({length: fixCount}, () => random()),
      // The fix the pointer drags (-1: none) and where it holds it.
      drag: {index: -1, x: 0, y: 0},
      // A fix that no default glitch moves, so the rest pointer reproduces the default picture.
      restFix: Array.from({length: fixCount}, (_, fix) => fix).find(
        fix => fix > 60 && !tracks.trackOffsets.includes(fix) && jumpChances[fix] >= 0.25
      ) as number,
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
          if (!isTrackStart && inputs.gapChance[fix] < values.glitchRate * 0.8) {
            elapsedExtra += 1.5 + 9 * (inputs.gapChance[fix] / (values.glitchRate * 0.8));
          }
          inputs.timestamps[fix] = tracks.timestamps[fix] + elapsedExtra;
        }
        if (inputs.drag.index >= 0) {
          inputs.points[inputs.drag.index * 2] = inputs.drag.x;
          inputs.points[inputs.drag.index * 2 + 1] = inputs.drag.y;
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
  // Drag: the fix nearest the pointer follows it; a fix pulled off its track breaks the speed limit.
  interaction: {
    kind: 'drag',
    hint: 'drag a fix',
    rest: ({inputs, toUnit}) =>
      toUnit([
        inputs.tracks.points[inputs.restFix * 2],
        inputs.tracks.points[inputs.restFix * 2 + 1]
      ]),
    resolve(pointer, {inputs, outputs, values, peek}) {
      let fix = inputs.drag.index;
      let nearest = Infinity;
      const distanceTo = (candidate: number) =>
        Math.hypot(
          inputs.tracks.points[candidate * 2] - pointer.x,
          inputs.tracks.points[candidate * 2 + 1] - pointer.y
        );
      for (let candidate = 0; candidate < inputs.tracks.points.length / 2; candidate++) {
        if (inputs.tracks.trackOffsets.includes(candidate)) continue;
        const distance = distanceTo(candidate);
        if (distance < nearest) {
          nearest = distance;
          fix = candidate;
        }
      }
      if (inputs.drag.index >= 0 && distanceTo(fix) > distanceTo(inputs.drag.index) * 0.85) {
        fix = inputs.drag.index;
      }
      const changed =
        fix !== inputs.drag.index || pointer.x !== inputs.drag.x || pointer.y !== inputs.drag.y;
      inputs.drag = {index: fix, x: pointer.x, y: pointer.y};
      const flags = peek(outputs.flags);
      let flagged = 0;
      if (flags) for (const flag of flags) if (flag !== 0) flagged++;
      return {
        rerun: changed,
        glyphs: [
          {
            shape: 'line',
            from: [inputs.tracks.points[fix * 2], inputs.tracks.points[fix * 2 + 1]],
            to: [pointer.x, pointer.y],
            dashed: true
          },
          {shape: 'crosshair', at: [pointer.x, pointer.y], size: 12}
        ],
        readout: `fix ${fix} · maximumSpeed ${values.maximumSpeed.toFixed(2)} · flagged ${flagged}`
      };
    }
  },
  draw: ({outputs, theme}) => [
    {layer: 'lines', context: true, segments: outputs.lines, radius: 1},
    // Clean fixes muted, flagged fixes bright with an ink ring.
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.flags,
      ramp: 'categorical',
      categories: [
        [...theme.muted, 0.35],
        ...theme.categories.slice(0, 4).map(color => [...color, 1] as const)
      ],
      radius: 6
    }
  ]
});
