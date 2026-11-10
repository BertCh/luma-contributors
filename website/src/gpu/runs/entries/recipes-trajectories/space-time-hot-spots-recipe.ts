// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// addSpaceTimeHotSpotsRecipe: timestamped events counted into a cell by month cube, Gi* per slice
// and a Mann-Kendall trend per cell. The map shows the trend z-score: red cells are intensifying
// hot spots, blue cells are fading.

import {getGPUCalendarBucketsParameterValues} from '@luma-contributors/analysis/dataframe-analysis';
import {
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  GPUParameterBuffer,
  addSpaceTimeHotSpotsRecipe
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {earthquakeEvents, getEventDots, getEventPosition} from './lib/earthquakes';
import {defineOperationRun} from '../../define-run';
import {getRasterCell, getRasterCellCenter} from '../../pick';
import {parameters} from './space-time-hot-spots-recipe.parameters';

const WIDTH = 16;
const HEIGHT = 10;
const SLICES = 12;
const CELLS = WIDTH * HEIGHT;

function selectEvents(minimumMagnitude: number) {
  const events = earthquakeEvents.filter(event => event[3] >= minimumMagnitude);
  // The GPU lattice numbers rows from its minimum y. Raster row zero is drawn at the top.
  const positions = Float32Array.from(
    events.flatMap(event => {
      const [column, northing] = getEventPosition(event, WIDTH, HEIGHT);
      return [column, HEIGHT - northing];
    })
  );
  const timestamps = Uint32Array.from(
    events.flatMap(event => {
      const milliseconds = event[2] * 1000;
      return [milliseconds % 2 ** 32, Math.floor(milliseconds / 2 ** 32)];
    })
  );
  return {positions, timestamps};
}

export default defineOperationRun({
  parameters,
  caption: 'Trend of Gi* hot spots among 2023 earthquakes in the Türkiye–Syria region.',
  createInputs: ({world}) => ({world, dots: getEventDots(world.width, world.height)}),
  build(builder, {dots}, values) {
    const events = selectEvents(values.minimumMagnitude);
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
      outputs: {trendZ, dots: builder.upload('events', 'float32x2', dots)},
      update: current =>
        recipe.setSettings({radius: current.radius, temporalWindow: current.temporalWindow})
    };
  },
  rebuildOn: ['minimumMagnitude'],
  // The pointer picks a cell: the ring is the spatial neighborhood (radius cells) its Gi* uses.
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: ({toUnit, world}) =>
      toUnit(getRasterCellCenter(WIDTH, HEIGHT, [0, 0, world.width, world.height], 4, 6)),
    resolve(pointer, {outputs, values, world, peek}) {
      const bounds = [0, 0, world.width, world.height] as const;
      const cell = getRasterCell(WIDTH, HEIGHT, bounds, pointer.x, pointer.y);
      const [centerX, centerY] = getRasterCellCenter(WIDTH, HEIGHT, bounds, cell.column, cell.row);
      const trend = peek(outputs.trendZ);
      return {
        glyphs: [
          {shape: 'ring', at: [centerX, centerY], radius: values.radius / HEIGHT},
          {shape: 'crosshair', at: [centerX, centerY]}
        ],
        readout: `cell (${cell.column}, ${cell.row}) · trend z ${trend ? trend[cell.index].toFixed(1) : '...'} · radius ${values.radius}`
      };
    }
  },
  draw: ({outputs}) => [
    {
      layer: 'raster',
      values: outputs.trendZ,
      columns: WIDTH,
      rows: HEIGHT,
      ramp: 'diverging',
      range: [-3, 3],
      alpha: 0.85
    },
    {layer: 'dots', context: true, positions: outputs.dots, radius: 2.2}
  ]
});
