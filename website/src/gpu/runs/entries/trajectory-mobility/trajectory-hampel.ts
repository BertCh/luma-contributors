// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTrajectoryHampel on the step speed along six San Francisco Bay vessel tracks, with twelve
// synthetic speed spikes added (labeled in the caption). Grey is the raw column, the bright line is
// the filtered column, and a dot marks every value the filter replaced by its window median.

import {
  getGPUTrajectoryHampelParameterValues,
  GPUTrajectoryHampel
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {getChartPosition, getChartWorldX} from '../../pick';
import {clamp} from '../../interaction';
import {createMovementFleet, getStepSpeeds, readUint32} from './_lib/movement';
import {formatSpeed} from './_lib/units';
import {parameters} from './trajectory-hampel.parameters';

const TRACK_COUNT = 6;
const SPIKE_COUNT = 12;
const PLOT = [0.05, 0.1, 0.96, 0.9] as const;

/** The window median and the scaled median absolute deviation of `values` around `row`. */
function getWindowStatistics(values: Float32Array, row: number, halfWindow: number) {
  const window = Array.from(values.slice(row - halfWindow, row + halfWindow + 1)).sort(
    (first, second) => first - second
  );
  const median = window[halfWindow];
  const deviations = window
    .map(value => Math.abs(value - median))
    .sort((first, second) => first - second);
  return {median, scaledDeviation: 1.4826 * deviations[halfWindow]};
}

export default defineOperationRun({
  parameters,
  caption:
    'Step speed along six Bay vessel tracks with 12 synthetic speed spikes added (the first fix of a track takes the speed of its second). Speeds are approximate: the frame stretches x and y differently. Grey: raw. Line: filtered. Dots: values replaced by the window median.',
  createInputs: ({world}) => {
    const fleet = createMovementFleet(world, TRACK_COUNT);
    // Step speed at every fix (a track's first fix takes its second's).
    const values = getStepSpeeds(fleet);
    for (let track = 0; track < fleet.trackCount; track++) {
      const first = fleet.trackOffsets[track];
      values[first] = values[first + 1];
    }
    // Synthetic spikes at evenly spread rows at least half of the largest window (7 rows) from a
    // track end, where the filter acts: a step speed 0.06 above the real one.
    const candidates: number[] = [];
    for (let track = 0; track < fleet.trackCount; track++) {
      for (
        let row = fleet.trackOffsets[track] + 7;
        row < fleet.trackOffsets[track + 1] - 7;
        row++
      ) {
        candidates.push(row);
      }
    }
    const spikeRows: number[] = [];
    for (let spike = 0; spike < SPIKE_COUNT; spike++) {
      const row = candidates[Math.floor(((spike + 0.5) * candidates.length) / SPIKE_COUNT)];
      spikeRows.push(row);
      values[row] += 0.06;
    }
    const top = values.reduce((maximum, value) => Math.max(maximum, value), 0) * 1.05;
    const bottom = 0;
    return {world, fleet, values, spikeRows, top, bottom};
  },
  build(builder, inputs, values) {
    const {fleet} = inputs;
    const raw = builder.upload('values', 'float32', inputs.values);
    const parameterBuffer = builder.allocate('parameters', 'float32', 4);
    const filtered = builder.allocate('filtered', 'float32', fleet.rowCount);
    const flags = builder.allocate('flags', 'uint32', fleet.rowCount);
    const outlierCount = builder.allocate('outlier-count', 'uint32', 1);
    builder.graph.add(
      new GPUTrajectoryHampel({
        values: raw.view,
        trackOffsets: builder.upload('offsets', 'uint32', fleet.trackOffsets).view,
        windowSize: values.windowSize,
        parameters: parameterBuffer.view,
        output: {filtered: filtered.view, flags: flags.view, outlierCount: outlierCount.view}
      })
    );
    const width = inputs.world.width;
    const toWorld = (row: number, value: number): [number, number] => [
      (PLOT[0] + ((row + 0.5) / fleet.rowCount) * (PLOT[2] - PLOT[0])) * width,
      (PLOT[1] + ((value - inputs.bottom) / (inputs.top - inputs.bottom)) * (PLOT[3] - PLOT[1])) *
        inputs.world.height
    ];
    const boundaries = new Float32Array((fleet.trackCount - 1) * 4);
    for (let track = 1; track < fleet.trackCount; track++) {
      const [x] = toWorld(fleet.trackOffsets[track] - 0.5, inputs.bottom);
      boundaries.set(
        [x, PLOT[1] * inputs.world.height, x, PLOT[3] * inputs.world.height],
        (track - 1) * 4
      );
    }
    const spikes = new Float32Array(inputs.spikeRows.length * 2);
    inputs.spikeRows.forEach((row, index) =>
      spikes.set(toWorld(row, inputs.values[row]), index * 2)
    );
    return {
      outputs: {
        raw,
        filtered,
        flags,
        outlierCount,
        boundaries: builder.upload('boundaries', 'float32x4', boundaries),
        spikes: builder.upload('spikes', 'float32x2', spikes)
      },
      update: current =>
        parameterBuffer.write(
          getGPUTrajectoryHampelParameterValues({numberOfSigmas: current.numberOfSigmas})
        )
    };
  },
  // The replaced values are read back to put a dot on each one, at its raw height.
  async afterRun({outputs, inputs, upload}) {
    const flags = await readUint32(outputs.flags);
    const positions: number[] = [];
    for (let row = 0; row < flags.length; row++) {
      if (!flags[row]) continue;
      positions.push(
        (PLOT[0] + ((row + 0.5) / flags.length) * (PLOT[2] - PLOT[0])) * inputs.world.width,
        (PLOT[1] +
          ((inputs.values[row] - inputs.bottom) / (inputs.top - inputs.bottom)) *
            (PLOT[3] - PLOT[1])) *
          inputs.world.height
      );
    }
    return {
      replacedCount: positions.length / 2,
      replaced: upload(
        'replaced',
        'float32x2',
        Float32Array.from(positions.length ? positions : [0, 0])
      )
    };
  },
  rebuildOn: ['windowSize'],
  buildCache: 6,
  // Probe: the row under the pointer, its window, and the band it is judged against.
  interaction: {
    kind: 'probe',
    hint: 'probe a value',
    rest: ({inputs}) => [
      PLOT[0] + ((inputs.spikeRows[3] + 0.5) / inputs.fleet.rowCount) * (PLOT[2] - PLOT[0]),
      0.5
    ],
    resolve(pointer, {inputs, values, world}) {
      const {fleet} = inputs;
      const position = getChartPosition(PLOT, world, pointer.x, pointer.y);
      const row = clamp(Math.floor(position.u * fleet.rowCount), 0, fleet.rowCount - 1);
      const halfWindow = Math.floor(values.windowSize / 2);
      let begin = 0;
      let end = fleet.rowCount;
      for (let track = 0; track < fleet.trackCount; track++) {
        if (row >= fleet.trackOffsets[track] && row < fleet.trackOffsets[track + 1]) {
          begin = fleet.trackOffsets[track];
          end = fleet.trackOffsets[track + 1];
        }
      }
      const left = getChartWorldX(PLOT, world, Math.max(row - halfWindow, begin) / fleet.rowCount);
      const right = getChartWorldX(
        PLOT,
        world,
        (Math.min(row + halfWindow, end - 1) + 1) / fleet.rowCount
      );
      const glyphs = [
        {
          shape: 'box' as const,
          minimum: [left, PLOT[1] * world.height] as [number, number],
          maximum: [right, PLOT[3] * world.height] as [number, number]
        }
      ];
      if (row < begin + halfWindow || row + halfWindow >= end) {
        return {
          glyphs,
          readout: `row ${row} · within half a window of a track end: passed through`
        };
      }
      const {median, scaledDeviation} = getWindowStatistics(inputs.values, row, halfWindow);
      const limit = values.numberOfSigmas * scaledDeviation;
      const deviation = Math.abs(inputs.values[row] - median);
      return {
        glyphs,
        readout: `row ${row} · speed off median ${formatSpeed(deviation, inputs.fleet.metersPerUnit)} vs ${values.numberOfSigmas}σ band ${formatSpeed(limit, inputs.fleet.metersPerUnit)} · ${deviation > limit ? 'replaced' : 'kept'}`
      };
    }
  },
  draw: ({outputs, inputs, extra}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.boundaries,
      count: inputs.fleet.trackCount - 1,
      alpha: 0.5
    },
    {
      layer: 'chart',
      kind: 'line',
      context: true,
      values: outputs.raw,
      range: [inputs.bottom, inputs.top],
      rectangle: PLOT,
      radius: 1.2
    },
    {
      layer: 'chart',
      kind: 'line',
      values: outputs.filtered,
      range: [inputs.bottom, inputs.top],
      rectangle: PLOT,
      radius: 2
    },
    ...(extra
      ? [
          {
            layer: 'dots' as const,
            positions: extra.replaced,
            count: extra.replacedCount,
            radius: 6,
            color: [1, 0.45, 0.2, 1] as const
          }
        ]
      : [])
  ]
});
