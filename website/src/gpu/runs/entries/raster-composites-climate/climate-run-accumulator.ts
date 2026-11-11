// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPURasterClimateRunAccumulator streams a daily series in chunks and keeps, per cell, the longest
// run of days that meet a condition against a threshold, the number of spells of at least
// `minimumSpellLength` days and how many days they cover (the run and threshold indices of xclim).
// The year is synthetic daily rainfall (see support/climate-stack.ts) whose wetness follows the
// interpolated Hurricane Helene station rainfall, run in two 183-day chunks. The pointer sets the
// threshold along x; `condition` and `minimumSpellLength` are construction props, and the map shows
// the chosen field of the per-cell run state.

import {
  GPURasterClimateRunAccumulator,
  getGPURasterClimateRunAccumulatorParameterValues
} from '@luma-contributors/analysis/raster-analysis';
import type {GPURasterClimateCondition} from '@luma-contributors/analysis/raster-analysis';
import {defineOperationRun} from '../../define-run';
import {bindAxes, formatNumber} from '../../interaction';
import {parameters} from './climate-run-accumulator.parameters';
import {createClimateStack, createDailyYear} from './support/climate-stack';

const CONDITIONS: GPURasterClimateCondition[] = ['above', 'at-or-above', 'below', 'at-or-below'];
const CHUNK = 183;
const DAYS = 366;
/** State field per `statistic` option: longestRun, spellCount, matchingSampleCount, spellSampleCount. */
const STATE_FIELDS = [1, 3, 2, 4] as const;
/** Fixed color range per state field, so the threshold moves the picture rather than the scale. */
const STATE_RANGES = [
  [0, 20],
  [0, 12],
  [0, 120],
  [0, 120]
] as const;
const STATISTIC_NAMES = ['longest run', 'spells', 'matching days', 'days in spells'] as const;

export default defineOperationRun({
  parameters,
  rebuildOn: ['condition', 'minimumSpellLength'],
  buildCache: 6,
  caption: 'A synthetic daily year, Helene-interpolated wetness, in; runs and spells out.',
  createInputs: ({random, world}) => {
    const stack = createClimateStack(random, world);
    return {...stack, daily: createDailyYear(random, stack)};
  },
  build(builder, inputs, values) {
    const {cellCount} = inputs;
    const runs = builder.allocate('runs', 'uint32', cellCount * 6);
    const parameterBuffers = [0, 1].map(chunk =>
      builder.allocate(`parameters-${chunk}`, 'float32', 4)
    );
    for (const chunk of [0, 1]) {
      const days = new Float32Array(cellCount * CHUNK).fill(Number.NaN);
      for (let cell = 0; cell < cellCount; cell++) {
        for (let day = 0; day < CHUNK; day++) {
          days[cell * CHUNK + day] = inputs.daily[cell * DAYS + chunk * CHUNK + day];
        }
      }
      builder.graph.add(
        new GPURasterClimateRunAccumulator({
          id: `chunk-${chunk}`,
          cellCount,
          chunkSliceCount: CHUNK,
          condition: CONDITIONS[Math.round(values.condition)],
          minimumSpellLength: Math.round(values.minimumSpellLength),
          values: builder.upload(`days-${chunk}`, 'float32', days).view,
          parameters: parameterBuffers[chunk].view,
          state: {runs: runs.view}
        })
      );
    }
    return {
      outputs: {runs},
      update: current =>
        [0, 1].forEach(chunk =>
          parameterBuffers[chunk].write(
            getGPURasterClimateRunAccumulatorParameterValues(
              {reset: chunk === 0, threshold: current.threshold, maximumIncrement: 200},
              CHUNK
            )
          )
        )
    };
  },
  // One field of the six-word run state per cell, for the map.
  async afterRun({inputs, outputs, values, upload}) {
    const bytes = await outputs.runs.buffer.readAsync();
    const state = new Uint32Array(bytes.buffer, bytes.byteOffset, inputs.cellCount * 6);
    const field = STATE_FIELDS[Math.round(values.statistic)];
    const shown = Float32Array.from(
      {length: inputs.cellCount},
      (_, cell) => state[cell * 6 + field]
    );
    return {shown: upload('shown', 'float32', shown)};
  },
  interaction: bindAxes(parameters, {
    kind: 'level',
    hint: 'set the threshold',
    x: 'threshold',
    decorate: values => ({
      readout: `${STATISTIC_NAMES[Math.round(values.statistic)]}: ${CONDITIONS[Math.round(values.condition)]} ${formatNumber(
        values.threshold,
        1
      )} mm/day, spells of ${Math.round(values.minimumSpellLength)}+ days`
    })
  }),
  draw: ({inputs, extra, values}) => [
    {
      layer: 'raster',
      values: extra.shown,
      columns: inputs.columns,
      rows: inputs.rows,
      ramp: 'sequential',
      range: STATE_RANGES[Math.round(values.statistic)]
    }
  ]
});
