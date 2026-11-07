// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUCalendarBuckets on four weeks of trip timestamps (Int64 epoch milliseconds). The grid is the
// GPU's hour-by-weekday count matrix: columns are local hours, rows are weekdays from the first day
// of the week. Moving the UTC offset slides the pattern sideways; the first day rotates the rows.

import {
  GPUCalendarBuckets,
  getGPUCalendarBucketsParameterValues,
  getInt64TimeWords,
  GPU_CALENDAR_BUCKETS_MATRIX_LENGTH
} from '@luma-contributors/analysis/dataframe-analysis';
import {defineOperationRun} from '../../define-run';
import {addIntegerToFloat} from './_lib/convert';
import {parameters} from './calendar-buckets.parameters';

const EVENT_COUNT = 4000;
// Monday 2024-01-01 00:00 UTC.
const ORIGIN_MILLISECONDS = 1704067200000;
const HOUR_MILLISECONDS = 3600000;

export default defineOperationRun({
  parameters,
  caption: 'Trips per local hour (columns) and weekday (rows).',
  createInputs: ({random}) => {
    const times = new BigInt64Array(EVENT_COUNT);
    for (let index = 0; index < EVENT_COUNT; index++) {
      const day = Math.floor(random() * 28);
      const isWeekend = day % 7 >= 5;
      const peak = isWeekend ? 13 : random() < 0.5 ? 8 : 17;
      const hour = peak + (random() + random() + random() - 1.5) * (isWeekend ? 4 : 2);
      times[index] = BigInt(
        ORIGIN_MILLISECONDS + Math.round((day * 24 + hour) * HOUR_MILLISECONDS)
      );
    }
    return {timestamps: getInt64TimeWords(times)};
  },
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
    return {
      outputs: {counts},
      update: current =>
        parameterBuffer.write(
          getGPUCalendarBucketsParameterValues(current.utcOffsetHours * 60, current.firstDayOfWeek)
        )
    };
  },
  rebuildOn: [],
  draw: ({outputs}) => [
    {layer: 'raster', values: outputs.counts, columns: 24, rows: 7, range: [0, 120]}
  ]
});
