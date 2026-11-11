// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryTrackRelativeTimes: Int64 epoch-millisecond words (low, high) to float32 ticks after
// each track's first fix. Map: fixes colored by that relative time. Chart: the error of every fix's
// time. A plain float32 of the epoch milliseconds (grey) is off by up to tens of seconds; the
// relative column (bright) is within a few milliseconds.

import {GPUTrajectoryTrackRelativeTimes} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp} from '../../interaction';
import {getChartWorldX} from '../../pick';
import {raiseFleet} from '../recipes-trajectories/lib/fleet';
import {createMovementFleet, getTrackOfRow, readFloat32} from './_lib/movement';
import {parameters} from './trajectory-track-relative-times.parameters';

const TRACK_COUNT = 12;
/** One playback second is 144 s of vessel time (a UTC day plays in ten minutes). */
const MILLISECONDS_PER_PLAYBACK_SECOND = 144000;
const MILLISECONDS_PER_YEAR = 31556952000;
const PLOT = [0.05, 0.06, 0.96, 0.36] as const;
const ERROR_RANGE = 70000;

type Outputs = {
  positions: RunBuffer;
  words: RunBuffer;
  relative: RunBuffer;
  naiveError: RunBuffer;
  rowIndex: RunBuffer;
};

/** Epoch milliseconds of every row for a calendar year. */
function getEpochMilliseconds(offsets: Float64Array, year: number): Float64Array {
  const base = Math.round((year - 1970) * MILLISECONDS_PER_YEAR);
  return Float64Array.from(offsets, offset => base + offset);
}

function getWords(milliseconds: Float64Array): Uint32Array {
  const words = new Uint32Array(milliseconds.length * 2);
  milliseconds.forEach((value, row) => {
    const exact = BigInt(Math.round(value));
    words[row * 2] = Number(exact & 0xffffffffn);
    words[row * 2 + 1] = Number(exact >> 32n);
  });
  return words;
}

export default defineOperationRun({
  parameters,
  caption:
    'Twelve Bay vessel tracks, times written as Int64 epoch milliseconds. Chart: time error of each fix, a plain float32 (grey) against the relative column (bright).',
  createInputs: ({world}) => {
    const fleet = raiseFleet(createMovementFleet(world, TRACK_COUNT), 0.42);
    const offsets = Float64Array.from(fleet.timestamps, time =>
      Math.round(time * MILLISECONDS_PER_PLAYBACK_SECOND)
    );
    // Exact milliseconds after each track's first fix.
    const exactRelative = new Float64Array(fleet.rowCount);
    for (let track = 0; track < fleet.trackCount; track++) {
      const first = fleet.trackOffsets[track];
      for (let row = first; row < fleet.trackOffsets[track + 1]; row++) {
        exactRelative[row] = offsets[row] - offsets[first];
      }
    }
    const rowIndex = Float32Array.from({length: fleet.rowCount}, (_, row) => row);
    const restRow = fleet.trackOffsets[2] + 6;
    return {world, fleet, offsets, exactRelative, rowIndex, restRow};
  },
  build(builder, inputs) {
    const {fleet} = inputs;
    const words = builder.allocate('words', 'uint32x2', fleet.rowCount);
    const relative = builder.allocate('relative', 'float32', fleet.rowCount);
    builder.graph.add(
      new GPUTrajectoryTrackRelativeTimes({
        timestamps: words.view,
        trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        output: relative.view
      })
    );
    const naiveError = builder.allocate('naive-error', 'float32', fleet.rowCount);
    const outputs: Outputs = {
      positions: builder.upload('positions', 'float32x2', fleet.positions),
      words,
      relative,
      naiveError,
      rowIndex: builder.upload('row-index', 'float32', inputs.rowIndex)
    };
    return {
      outputs,
      update: current => {
        const epoch = getEpochMilliseconds(inputs.offsets, current.year);
        words.write(getWords(epoch));
        naiveError.write(Float32Array.from(epoch, value => Math.fround(value) - value));
      }
    };
  },
  // The relative column is read back and compared with the exact milliseconds.
  async afterRun({outputs, inputs, upload}) {
    const relative = await readFloat32(outputs.relative);
    return {
      relativeError: upload(
        'relative-error',
        'float32',
        Float32Array.from(relative, (value, row) => value - inputs.exactRelative[row])
      ),
      relativeValues: relative
    };
  },
  // Probe: the fix under the pointer, its time since the track began, and both time errors.
  interaction: {
    kind: 'probe',
    hint: 'probe a fix',
    rest: ({toUnit, inputs}) =>
      toUnit([
        inputs.fleet.positions[inputs.restRow * 2],
        inputs.fleet.positions[inputs.restRow * 2 + 1]
      ]),
    resolve(pointer, {inputs, values, world}) {
      const {fleet} = inputs;
      let nearest = 0;
      let nearestDistance = Infinity;
      for (let row = 0; row < fleet.rowCount; row++) {
        const distance = Math.hypot(
          fleet.positions[row * 2] - pointer.x,
          fleet.positions[row * 2 + 1] - pointer.y
        );
        if (distance < nearestDistance) {
          nearest = row;
          nearestDistance = distance;
        }
      }
      const epoch = getEpochMilliseconds(inputs.offsets, values.year)[nearest];
      const naive = Math.fround(epoch) - epoch;
      const x = getChartWorldX(PLOT, world, (nearest + 0.5) / fleet.rowCount);
      const y =
        (PLOT[1] +
          ((clamp(naive, -ERROR_RANGE, ERROR_RANGE) + ERROR_RANGE) / (2 * ERROR_RANGE)) *
            (PLOT[3] - PLOT[1])) *
        world.height;
      const track = getTrackOfRow(fleet.trackOffsets, nearest);
      return {
        glyphs: [
          {
            shape: 'crosshair',
            at: [fleet.positions[nearest * 2], fleet.positions[nearest * 2 + 1]]
          },
          {shape: 'dot', at: [x, y]}
        ],
        readout: `track ${track} · +${(inputs.exactRelative[nearest] / 1000).toFixed(0)} s · float32 epoch off ${(naive / 1000).toFixed(1)} s`
      };
    }
  },
  draw: ({inputs, outputs, extra}) => [
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.relative,
      range: [0, 86400000],
      radius: 5
    },
    {
      layer: 'chart',
      kind: 'scatter',
      context: true,
      values: outputs.naiveError,
      xValues: outputs.rowIndex,
      xRange: [0, inputs.fleet.rowCount],
      range: [-ERROR_RANGE, ERROR_RANGE],
      rectangle: PLOT,
      radius: 2.5
    },
    ...(extra
      ? [
          {
            layer: 'chart' as const,
            kind: 'scatter' as const,
            values: extra.relativeError,
            xValues: outputs.rowIndex,
            xRange: [0, inputs.fleet.rowCount] as const,
            range: [-ERROR_RANGE, ERROR_RANGE] as const,
            rectangle: PLOT,
            radius: 3,
            color: [0.25, 0.8, 0.95, 1] as const
          }
        ]
      : [])
  ]
});
