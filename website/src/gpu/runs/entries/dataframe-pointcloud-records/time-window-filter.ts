// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTimeWindowFilter over Midwest GLM flashes with measured times. Faint dots are
// every event; bright dots are the ones inside the window, faded in over startFadeDuration by the
// GPU-written fadeWeights.

import {
  GPUTimeWindowFilter,
  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
  getGPUTimeWindowFilterParameterValues
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp, findParameter, snapToParameter} from '../../interaction';
import {sampleFlashes} from '../../../data/lightning';
import {parameters} from './time-window-filter.parameters';

const EVENT_COUNT = 2500;
const HOURS = 18;
const STRIP_TOP = 0.24;
const STRIP = [0.07, 0.05, 0.95, STRIP_TOP] as const;

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  positions: RunBuffer;
  timestamps: RunBuffer;
  fadeWeights: RunBuffer;
  hourly: RunBuffer;
  hourlyInside: RunBuffer;
};

function createInputs({world}: {world: {width: number; height: number}}) {
  // The map sits above the time strip.
  const flashes = sampleFlashes({width: world.width, height: 1 - STRIP_TOP - 0.05}, EVENT_COUNT);
  for (let row = 0; row < EVENT_COUNT; row++) flashes.positions[row * 2 + 1] += STRIP_TOP + 0.05;
  const hourly = new Float32Array(HOURS);
  for (const hour of flashes.hours) hourly[Math.min(HOURS - 1, Math.floor(hour))]++;
  return {...flashes, world, hourly};
}

function getTimeX(inputs: Inputs, hour: number): number {
  return (STRIP[0] + (hour / HOURS) * (STRIP[2] - STRIP[0])) * inputs.world.width;
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Midwest flashes inside the window, in hours since May 21, 2024, 12:00 UTC; the strip is the hourly flash count with the window marked.',
  createInputs,
  build(builder, inputs) {
    const timestamps = builder.upload('timestamps', 'float32', inputs.hours);
    const positions = builder.upload('positions', 'float32x2', inputs.positions);
    const window = builder.allocate('window', 'float32', GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH);
    const fadeWeights = builder.allocate('fade-weights', 'float32', EVENT_COUNT);
    builder.graph.add(
      new GPUTimeWindowFilter({
        timestamps: timestamps.view,
        window: window.view,
        output: {
          rows: builder.allocate('rows', 'uint32', EVENT_COUNT).view,
          status: {
            count: builder.allocate('count', 'uint32', 1).view,
            overflow: builder.allocate('overflow', 'uint32', 1).view
          },
          fadeWeights: fadeWeights.view
        }
      })
    );
    const hourly = builder.upload('hourly', 'float32', inputs.hourly);
    const hourlyInside = builder.allocate('hourly-inside', 'float32', HOURS);
    return {
      outputs: {positions, timestamps, fadeWeights, hourly, hourlyInside},
      update: current => {
        const end = Math.max(current.end, current.start);
        window.write(
          getGPUTimeWindowFilterParameterValues({
            start: current.start,
            end,
            startFadeDuration: current.startFadeDuration
          })
        );
        // The hours whose flashes fall inside the window, for the bright bars of the strip.
        hourlyInside.write(
          Float32Array.from(inputs.hourly, (count, hour) =>
            hour + 1 > current.start && hour < end ? count : 0
          )
        );
      }
    };
  },
  rebuildOn: [],
  // The pointer is the window on the time strip: x is its center, y its length.
  interaction: {
    kind: 'window',
    hint: 'center and length of the window',
    rest: [0.55, 0.5],
    resolve(pointer, {inputs, values}) {
      const startParameter = findParameter(parameters, 'start');
      const endParameter = findParameter(parameters, 'end');
      const center = clamp((pointer.unitX - 0.05) / 0.9, 0, 1) * HOURS;
      const length = 2 + 12 * clamp((0.95 - pointer.unitY) / 0.9, 0, 1);
      const start = snapToParameter(startParameter, center - length / 2);
      const end = snapToParameter(endParameter, center + length / 2);
      let inside = 0;
      for (const hour of inputs.hours) if (hour >= start && hour <= end) inside++;
      return {
        values: {start, end},
        glyphs: [
          {
            shape: 'box',
            minimum: [getTimeX(inputs, start), STRIP[1] * inputs.world.height],
            maximum: [getTimeX(inputs, end), STRIP[3] * inputs.world.height]
          },
          {
            shape: 'line',
            from: [getTimeX(inputs, start), STRIP[3] * inputs.world.height],
            to: [
              getTimeX(inputs, Math.min(HOURS, start + values.startFadeDuration)),
              STRIP[3] * inputs.world.height * 0.55
            ]
          }
        ],
        readout: `window ${start}-${end} h · fade ${values.startFadeDuration} h · ${inside} flashes`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      color: [0.4, 0.45, 0.55, 0.45],
      radius: 2.4
    },
    {
      layer: 'dots',
      positions: outputs.positions,
      values: outputs.fadeWeights,
      ramp: 'sequential',
      range: [0, 1],
      fade: true,
      radius: 3.6
    },
    {
      layer: 'chart',
      kind: 'bars',
      context: true,
      values: outputs.hourly,
      range: [0, Math.max(...inputs.hourly)],
      rectangle: STRIP
    },
    {
      layer: 'chart',
      kind: 'bars',
      color: [0.2, 0.55, 0.9, 1],
      values: outputs.hourlyInside,
      range: [0, Math.max(...inputs.hourly)],
      rectangle: STRIP
    }
  ]
});
