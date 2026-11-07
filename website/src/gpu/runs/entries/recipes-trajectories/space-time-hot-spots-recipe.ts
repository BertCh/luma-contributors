// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSpaceTimeHotSpotsRecipe: timestamped events counted into a cell by month cube, Gi* per slice
// and a Mann-Kendall trend per cell. The map shows the trend z-score: red cells are intensifying
// hot spots (the block whose counts grow), blue cells are fading.

import {getGPUCalendarBucketsParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
import {
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  GPUParameterBuffer,
  addSpaceTimeHotSpotsRecipe
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import {parameters} from './space-time-hot-spots-recipe.parameters';

const WIDTH = 8;
const HEIGHT = 5;
const SLICES = 12;
const CELLS = WIDTH * HEIGHT;

/** Events on the lattice over twelve months of 2023: a hot block that grows and a cold corner. */
function createEvents(growth: number) {
  let state = 31;
  const random = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const positions: number[] = [];
  const words: number[] = [];
  for (let cell = 0; cell < CELLS; cell++) {
    const column = cell % WIDTH;
    const row = Math.floor(cell / WIDTH);
    for (let month = 0; month < SLICES; month++) {
      let eventCount = 3 + Math.floor(random() * 3);
      if (column >= 1 && column <= 3 && row >= 1 && row <= 2) eventCount += growth * month;
      else if (column >= 5 && row >= 3)
        eventCount = Math.max(0, eventCount - 3 + Math.floor(growth * (SLICES - month) * 0.4));
      for (let event = 0; event < eventCount; event++) {
        positions.push(column + 0.05 + 0.9 * random(), row + 0.05 + 0.9 * random());
        const milliseconds = Date.UTC(2023, month, 1 + Math.floor(random() * 28), 12);
        words.push(milliseconds % 2 ** 32, Math.floor(milliseconds / 2 ** 32));
      }
    }
  }
  return {positions: Float32Array.from(positions), timestamps: Uint32Array.from(words)};
}

export default defineOperationRun({
  parameters,
  caption: 'Trend of Gi* hot spots through twelve months: intensifying red, fading blue.',
  createInputs: () => ({}),
  build(builder, _inputs, values) {
    const events = createEvents(values.growth);
    const trendZ = builder.allocate('trend-z', 'float32', CELLS);
    const category = builder.allocate('category', 'uint32', CELLS);
    const recipe = addSpaceTimeHotSpotsRecipe(builder.graph, {
      timestamps: builder.upload('timestamps', 'uint32x2', events.timestamps).view,
      calendarParameters: builder.upload(
        'calendar-parameters',
        'sint32',
        getGPUCalendarBucketsParameterValues(0)
      ).view,
      slices: {step: 'month', origin: {year: 2023, month: 1}, count: SLICES},
      cells: {
        kind: 'lattice',
        positions: builder.upload('positions', 'float32x2', events.positions).view,
        width: WIDTH,
        height: HEIGHT,
        bounds: [0, 0, WIDTH, HEIGHT]
      },
      settings: {radius: values.radius, temporalWindow: values.temporalWindow},
      parameters: new GPUParameterBuffer(builder.device, {
        id: 'emerging-parameters',
        format: 'float32',
        length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
      }),
      outputs: {trendZ: trendZ.view, category: category.view}
    });
    return {
      outputs: {trendZ},
      update: current =>
        recipe.setSettings({radius: current.radius, temporalWindow: current.temporalWindow})
    };
  },
  rebuildOn: ['growth'],
  draw: ({outputs}) => [
    {layer: 'raster', values: outputs.trendZ, columns: WIDTH, rows: HEIGHT, ramp: 'diverging'}
  ]
});
