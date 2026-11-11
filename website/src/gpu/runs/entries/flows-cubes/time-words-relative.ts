// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUTimeWordsRelative: Int64 flash times become float32 ticks after an origin. Each dot is one
// flash: x is its time, y the error of the float32 result against the exact Int64 difference, in
// milliseconds. The words are subtracted exactly and rounded once, so the error is zero within 2^24
// ticks of the origin and grows with the distance beyond it; a float32 of the absolute epoch time
// would be off by minutes.

import {
  GPUTimeWordsRelative,
  getInt64TimeWords
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {clamp} from '../../interaction';
import {
  createFlashEvents,
  FLASH_HOURS,
  FLASH_ORIGIN_MILLISECONDS,
  MILLISECONDS_PER_HOUR
} from './_lib/flash-events';
import {parameters} from './time-words-relative.parameters';

const UNITS = ['second', 'millisecond', 'microsecond'] as const;
/** Milliseconds per tick of each unit. */
const UNIT_MILLISECONDS = [1000, 1, 0.001];
const EXACT_TICKS = 2 ** 24;
const ORIGIN_CHOICES = [0, 4.5, 9, 13.5, 18];
const CHART = [0.08, 0.14, 0.96, 0.9] as const;
const ERROR_RANGE = 6;
const ORIGIN_MILLISECONDS = Number(FLASH_ORIGIN_MILLISECONDS);

type Inputs = ReturnType<typeof createInputs>;
type Outputs = {relative: RunBuffer; hours: RunBuffer; guides: RunBuffer};

function createInputs({world}: {world: {width: number; height: number}}) {
  const events = createFlashEvents();
  const milliseconds = Float64Array.from(
    events.hours,
    hours => ORIGIN_MILLISECONDS + Math.round(hours * MILLISECONDS_PER_HOUR)
  );
  // The Int64 words of each unit, the exact tick of every flash, and what a float32 of the absolute
  // tick would be off by.
  const perUnit = UNIT_MILLISECONDS.map(unitMilliseconds => {
    const ticks = Float64Array.from(milliseconds, value => Math.round(value / unitMilliseconds));
    let naiveError = 0;
    for (const tick of ticks) naiveError = Math.max(naiveError, Math.abs(Math.fround(tick) - tick));
    return {
      ticks,
      words: getInt64TimeWords(BigInt64Array.from(ticks, tick => BigInt(tick))),
      naiveErrorMilliseconds: naiveError * unitMilliseconds
    };
  });
  return {world, events, perUnit};
}

const getOriginTicks = (unit: number, origin: number) =>
  Math.round((ORIGIN_MILLISECONDS + origin * MILLISECONDS_PER_HOUR) / UNIT_MILLISECONDS[unit]);

/** World position of an hour on the chart's time axis and of an error on its value axis. */
const getChartX = (world: {width: number}, hour: number) =>
  (CHART[0] + (hour / FLASH_HOURS) * (CHART[2] - CHART[0])) * world.width;
const getChartY = (errorMilliseconds: number) =>
  CHART[1] + ((errorMilliseconds + ERROR_RANGE) / (2 * ERROR_RANGE)) * (CHART[3] - CHART[1]);

export default defineOperationRun<Inputs, Outputs, {errors: RunBuffer}>({
  parameters,
  caption:
    'Error of the float32 relative time against the exact Int64 difference for each flash, in milliseconds. The origin is where the error is zero.',
  createInputs,
  build(builder, inputs, values) {
    const unit = values.unit;
    const timestamps = builder.upload('timestamps', 'uint32x2', inputs.perUnit[unit].words);
    const relative = builder.allocate('relative', 'float32', inputs.events.count);
    builder.graph.add(
      new GPUTimeWordsRelative({
        timestamps: timestamps.view,
        temporalContext: {
          unit: UNITS[unit],
          epoch: 'unix',
          encoding: 'int64-words',
          axis: {kind: 'irregular'}
        },
        origin: BigInt(getOriginTicks(unit, values.origin)),
        output: relative.view
      })
    );
    // Guides: the zero line, and where the exact range of 2^24 ticks ends on either side of the origin.
    const limitHours = (EXACT_TICKS * UNIT_MILLISECONDS[unit]) / MILLISECONDS_PER_HOUR;
    const guides: number[] = [
      getChartX(inputs.world, 0),
      getChartY(0),
      getChartX(inputs.world, FLASH_HOURS),
      getChartY(0)
    ];
    for (const hour of [values.origin - limitHours, values.origin + limitHours]) {
      const at = hour >= 0 && hour <= FLASH_HOURS ? hour : -100;
      guides.push(getChartX(inputs.world, at), CHART[1], getChartX(inputs.world, at), CHART[3]);
    }
    return {
      outputs: {
        relative,
        hours: builder.upload('hours', 'float32', inputs.events.hours),
        guides: builder.upload('guides', 'float32x4', Float32Array.from(guides))
      }
    };
  },
  rebuildOn: ['unit', 'origin'],
  buildCache: 5,
  async afterRun({inputs, outputs, values, upload}) {
    const bytes = await outputs.relative.buffer.readAsync();
    const relative = new Float32Array(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
    );
    const {ticks} = inputs.perUnit[values.unit];
    const originTicks = getOriginTicks(values.unit, values.origin);
    const errors = Float32Array.from(
      relative,
      (value, row) => (value - (ticks[row] - originTicks)) * UNIT_MILLISECONDS[values.unit]
    );
    return {errors: upload('errors', 'float32', errors)};
  },
  // The pointer chooses the origin along the time axis and the unit by height; both apply at rest.
  interaction: {
    kind: 'pad',
    hint: 'origin and unit',
    rest: [CHART[0], 0.5],
    resolve(pointer, {inputs, values, world, peek, outputs}) {
      const hour = clamp((pointer.unitX - CHART[0]) / (CHART[2] - CHART[0]), 0, 1) * FLASH_HOURS;
      const origin = ORIGIN_CHOICES.reduce((best, choice) =>
        Math.abs(choice - hour) < Math.abs(best - hour) ? choice : best
      );
      const unit = clamp(Math.floor((1 - pointer.unitY) * 3), 0, 2);
      const relative = peek(outputs.relative);
      let maximumError = 0;
      if (relative) {
        const {ticks} = inputs.perUnit[values.unit];
        const originTicks = getOriginTicks(values.unit, values.origin);
        for (let row = 0; row < relative.length; row++) {
          maximumError = Math.max(
            maximumError,
            Math.abs(relative[row] - (ticks[row] - originTicks))
          );
        }
        maximumError *= UNIT_MILLISECONDS[values.unit];
      }
      const exactHours = (EXACT_TICKS * UNIT_MILLISECONDS[values.unit]) / MILLISECONDS_PER_HOUR;
      const naive = inputs.perUnit[values.unit].naiveErrorMilliseconds / 1000;
      return {
        settle: {unit, origin},
        spotlight: null,
        glyphs: [
          {
            shape: 'line',
            from: [getChartX(world, origin), CHART[1]],
            to: [getChartX(world, origin), CHART[3]]
          },
          // The exact zone: within 2^24 ticks of the origin the float32 result has no rounding error.
          {
            shape: 'box',
            minimum: [getChartX(world, clamp(origin - exactHours, 0, FLASH_HOURS)), CHART[1]],
            maximum: [getChartX(world, clamp(origin + exactHours, 0, FLASH_HOURS)), CHART[3]],
            dashed: true
          }
        ],
        readout: `${UNITS[values.unit]} · origin ${String((12 + values.origin) % 24).padStart(2, '0')}:00 · exact within ±${
          exactHours < 100
            ? exactHours.toPrecision(3)
            : Math.round(exactHours).toLocaleString('en-US')
        } h · worst error ${maximumError.toFixed(maximumError < 10 ? 1 : 0)} ms · epoch float32 off by ${
          naive < 1 ? '<1' : naive.toFixed(0)
        } s`
      };
    }
  },
  draw: ({outputs, extra}) => [
    {
      layer: 'lines',
      context: true,
      segments: outputs.guides,
      radius: 1,
      color: [0.6, 0.65, 0.75, 0.6]
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: extra.errors,
      xValues: outputs.hours,
      xRange: [0, FLASH_HOURS],
      range: [-ERROR_RANGE, ERROR_RANGE],
      rectangle: CHART,
      color: [0.35, 0.65, 0.95, 0.9],
      radius: 2.6
    }
  ]
});
