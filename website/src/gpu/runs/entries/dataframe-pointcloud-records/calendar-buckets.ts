// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCalendarBuckets on July 2023 Atlanta departure timestamps (Int64 epoch milliseconds). The grid is the
// GPU's hour-by-weekday count matrix: columns are local hours, rows are weekdays from the first day
// of the week. Moving the UTC offset slides the pattern sideways; the first day rotates the rows.

import {
  GPUCalendarBuckets,
  getGPUCalendarBucketsParameterValues,
  getInt64TimeWords,
  GPU_CALENDAR_BUCKETS_MATRIX_LENGTH
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {addIntegerToFloat} from './_lib/convert';
import {bindAxes} from '../../interaction';
import {getRasterCell} from '../../pick';
import {parameters} from './calendar-buckets.parameters';

import departures from '../../../data/atlantadepartures.json';

const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
// July 1 midnight at Atlanta, in Eastern Daylight Time (UTC−4).
const ORIGIN_MILLISECONDS = Date.UTC(2023, 6, 1, 4);

type Inputs = ReturnType<typeof createCalendarInputs>;
type Outputs = {counts: RunBuffer; tickSegments: RunBuffer};

function createCalendarInputs(world: {width: number; height: number}) {
  return {
    world,
    timestamps: getInt64TimeWords(
      BigInt64Array.from(departures, minutes => BigInt(ORIGIN_MILLISECONDS + minutes * 60000))
    )
  };
}

export default defineOperationRun<Inputs, Outputs>({
  parameters,
  caption: 'Atlanta departures by local hour and weekday, July 2023.',
  createInputs: ({world}) => createCalendarInputs(world),
  build(builder, inputs) {
    const timestamps = builder.upload('timestamps', 'uint32x2', inputs.timestamps);
    const parameterBuffer = builder.allocate('parameters', 'sint32', 2);
    const matrix = builder.allocate('matrix', 'uint32', GPU_CALENDAR_BUCKETS_MATRIX_LENGTH);
    builder.graph.add(
      new GPUCalendarBuckets({
        timestamps: timestamps.view,
        parameters: parameterBuffer.view,
        output: {hourWeekdayCounts: matrix.view}
      })
    );
    const counts = addIntegerToFloat(builder, 'counts', matrix);
    // Hour ticks along the top edge and a tick per weekday row on the left edge.
    const ticks: number[] = [];
    for (let hour = 0; hour < 24; hour += 6) {
      const x = (hour / 24) * inputs.world.width;
      ticks.push(x, 1, x, 0.965);
    }
    for (let row = 1; row < 7; row++) ticks.push(0, 1 - row / 7, 0.012, 1 - row / 7);
    const tickSegments = builder.upload('ticks', 'float32x4', Float32Array.from(ticks));
    return {
      outputs: {counts, tickSegments},
      update: current =>
        parameterBuffer.write(
          getGPUCalendarBucketsParameterValues(current.utcOffsetHours * 60, current.firstDayOfWeek)
        )
    };
  },
  rebuildOn: [],
  // The pointer is a pad: x slides the local clock (utcOffset), y rotates the week (firstDayOfWeek).
  interaction: bindAxes<Inputs, Outputs>(parameters, {
    kind: 'pad',
    hint: 'slide the clock, rotate the week',
    x: 'utcOffsetHours',
    y: 'firstDayOfWeek',
    decorate: (values, pointer, {inputs, outputs, peek}) => {
      const bounds = [0, 0, inputs.world.width, 1] as const;
      const cell = getRasterCell(24, 7, bounds, pointer.x, pointer.y);
      const column = cell.column;
      const row = cell.row;
      const counts = peek(outputs.counts);
      const weekday = WEEKDAYS[(row + values.firstDayOfWeek) % 7];
      const left = (column / 24) * inputs.world.width;
      const right = ((column + 1) / 24) * inputs.world.width;
      return {
        glyphs: [
          {shape: 'box', minimum: [left, 1 - (row + 1) / 7], maximum: [right, 1 - row / 7]},
          {shape: 'box', minimum: [0, 1 - 1 / 7], maximum: [inputs.world.width, 1], dashed: true}
        ],
        readout: `utc ${values.utcOffsetHours} h · ${WEEKDAYS[values.firstDayOfWeek]} first · ${weekday} ${String(column).padStart(2, '0')}:00 = ${counts ? counts[row * 24 + column] : '…'}`
      };
    }
  }),
  draw: ({outputs}) => [
    {layer: 'raster', values: outputs.counts, columns: 24, rows: 7, range: 'auto'},
    {
      layer: 'lines',
      context: true,
      segments: outputs.tickSegments,
      count: 10,
      color: [1, 1, 1, 0.8]
    }
  ]
});
