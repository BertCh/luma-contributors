// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPeriodIndex: every flash gets the fixed-width period its Int64 time falls in. Dots are colored
// by period; muted dots fall outside the periods (before the origin or past the last period) and are
// counted in status.invalidCount. The strip is the hourly flash count with the period boundaries.

import {GPUPeriodIndex} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp} from '../../interaction';
import {addPeriodColors} from './_lib/period-colors';
import {
  createFlashEvents,
  FLASH_HOURS,
  FLASH_ORIGIN_MILLISECONDS,
  FLASH_TEMPORAL_CONTEXT,
  MILLISECONDS_PER_HOUR
} from './_lib/flash-events';
import {parameters} from './period-index.parameters';

const WIDTHS = [1, 2, 3, 4, 6];
const STRIP_TOP = 0.24;
const STRIP = [0.07, 0.06, 0.95, STRIP_TOP] as const;
const ORIGIN_STEPS = [0, 1, 2, 3, 4];

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {
  positions: RunBuffer;
  colors: RunBuffer;
  invalidCount: RunBuffer;
  hourly: RunBuffer;
  hourlyInside: RunBuffer;
  boundaries: RunBuffer;
  centers: RunBuffer;
  centerColors: RunBuffer;
};

function createInputs({world}: {world: {width: number; height: number}}) {
  const events = createFlashEvents();
  // The map sits above the time strip.
  const positions = new Float32Array(events.positions);
  for (let row = 0; row < events.count; row++) {
    positions[row * 2] *= world.width;
    positions[row * 2 + 1] = STRIP_TOP + 0.05 + positions[row * 2 + 1] * (1 - STRIP_TOP - 0.08);
  }
  const hourly = new Float32Array(FLASH_HOURS);
  for (const hour of events.hours) hourly[Math.min(FLASH_HOURS - 1, Math.floor(hour))]++;
  return {world, events, positions, hourly};
}

function getTimeX(world: {width: number}, hour: number): number {
  return (STRIP[0] + (hour / FLASH_HOURS) * (STRIP[2] - STRIP[0])) * world.width;
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption:
    'Flashes colored by fixed-width period from the origin; muted flashes fall outside the periods.',
  createInputs,
  build(builder, inputs, values) {
    const width = WIDTHS[values.width];
    const {events, world} = inputs;
    const timestamps = builder.upload('timestamps', 'uint32x2', events.timestampWords);
    const periods = builder.allocate('periods', 'uint32', events.count);
    const invalidCount = builder.allocate('invalid-count', 'uint32', 1);
    builder.graph.add(
      new GPUPeriodIndex({
        temporalContext: FLASH_TEMPORAL_CONTEXT,
        periods: {
          kind: 'fixed',
          width: BigInt(width * MILLISECONDS_PER_HOUR),
          origin:
            FLASH_ORIGIN_MILLISECONDS + BigInt(Math.round(values.origin * MILLISECONDS_PER_HOUR)),
          count: values.count
        },
        timestamps: timestamps.view,
        output: {periods: periods.view, status: {invalidCount: invalidCount.view}}
      })
    );
    // Strip geometry for this axis: boundaries between periods and one marker per period.
    const start = values.origin;
    const end = start + values.count * width;
    const boundaries = new Float32Array((values.count + 1) * 4);
    for (let boundary = 0; boundary <= values.count; boundary++) {
      const x = getTimeX(world, Math.min(FLASH_HOURS, start + boundary * width));
      boundaries.set([x, STRIP[1] * 0.5, x, STRIP[3]], boundary * 4);
    }
    const centers = new Float32Array(values.count * 2);
    for (let period = 0; period < values.count; period++) {
      centers[period * 2] = getTimeX(world, Math.min(FLASH_HOURS, start + (period + 0.5) * width));
      centers[period * 2 + 1] = STRIP[1] * 0.5;
    }
    const hourlyInside = new Float32Array(FLASH_HOURS);
    for (const hour of events.hours) {
      if (hour >= start && hour < end) hourlyInside[Math.min(FLASH_HOURS - 1, Math.floor(hour))]++;
    }
    return {
      outputs: {
        positions: builder.upload('positions', 'float32x2', inputs.positions),
        colors: addPeriodColors(builder, {id: 'flash', periods, periodCount: values.count}),
        invalidCount,
        hourly: builder.upload('hourly', 'float32', inputs.hourly),
        hourlyInside: builder.upload('hourly-inside', 'float32', hourlyInside),
        boundaries: builder.upload('boundaries', 'float32x4', boundaries),
        centers: builder.upload('centers', 'float32x2', centers),
        centerColors: builder.upload(
          'center-colors',
          'uint32',
          Uint32Array.from({length: values.count}, (_, period) => period)
        )
      }
    };
  },
  rebuildOn: ['width', 'count', 'origin'],
  buildCache: 5,
  // The pointer chooses the axis: x its period width, y its origin; both apply once it rests.
  interaction: {
    kind: 'pad',
    hint: 'width and origin of the periods',
    rest: [0.5, 0.99],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const widthIndex = clamp(Math.floor(pointer.unitX * WIDTHS.length), 0, WIDTHS.length - 1);
      const origin =
        ORIGIN_STEPS[
          clamp(
            Math.round((1 - pointer.unitY) * (ORIGIN_STEPS.length - 1)),
            0,
            ORIGIN_STEPS.length - 1
          )
        ];
      const width = WIDTHS[widthIndex];
      const glyphs = [];
      for (let boundary = 0; boundary <= values.count; boundary++) {
        const hour = origin + boundary * width;
        if (hour > FLASH_HOURS) break;
        glyphs.push({
          shape: 'line' as const,
          from: [getTimeX(world, hour), STRIP[1] * 0.4] as [number, number],
          to: [getTimeX(world, hour), STRIP[3]] as [number, number]
        });
      }
      const invalid = peek(outputs.invalidCount)?.[0];
      void inputs;
      return {
        settle: {width: widthIndex, origin},
        glyphs,
        spotlight: null,
        readout: `periods.width ${width} h · count ${values.count} · origin ${origin} h${
          invalid === undefined ? '' : ` · out of range ${invalid}`
        }`
      };
    }
  },
  draw: ({inputs, outputs}) => [
    {
      layer: 'dots',
      context: true,
      positions: outputs.positions,
      color: [0.4, 0.45, 0.55, 0.4],
      radius: 2.2
    },
    {layer: 'dots', positions: outputs.positions, values: outputs.colors, radius: 3.4},
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
      values: outputs.hourlyInside,
      color: [0.35, 0.6, 0.95, 1],
      range: [0, Math.max(...inputs.hourly)],
      rectangle: STRIP
    },
    {layer: 'lines', segments: outputs.boundaries, radius: 1, color: [1, 1, 1, 0.6]},
    {layer: 'dots', positions: outputs.centers, values: outputs.centerColors, radius: 7}
  ]
});
