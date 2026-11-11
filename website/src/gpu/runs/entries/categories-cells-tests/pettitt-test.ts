// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUPettittTest on the Dixie fire NDVI stack (Sentinel-2, 16 real scenes, 2017-2024): one rank-based
// change point test per cell. The map colors each cell by the date of its most likely change (early to
// late) where the p-value is under the cutoff; the fire of 2021 shows up as the burn scar. Hover a cell:
// its NDVI at the true scene dates, and a dashed rule at the change. Statistics are uncorrected.

import {GPUPettittTest} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {
  CUBE_CELL_COUNT,
  CUBE_COLUMNS,
  CUBE_ROWS,
  createDixieCube,
  DISPLAY_COLUMNS,
  getCellBox,
  getDataCell,
  getSceneFractions,
  getSeriesX,
  NDVI_RANGE,
  padToDisplay,
  SCENE_DATES,
  SCENE_DAYS,
  SERIES_RECTANGLE
} from './_lib/dixie-cube';
import {readFloats, readUints} from './_lib/readback';
import {parameters} from './pettitt-test.parameters';

const NO_CHANGE = 0xffffffff;

export default defineOperationRun({
  parameters,
  caption:
    'Dixie fire NDVI, 16 real Sentinel-2 scenes (2017-2024). Rank test per cell, uncorrected.',
  createInputs: () => ({focus: {cell: 0}, centered: false}),
  build(builder, inputs, values) {
    const cube = createDixieCube(values.sliceCount);
    const statistic = builder.allocate('statistic', 'uint32', CUBE_CELL_COUNT);
    const changeSlice = builder.allocate('change-slice', 'uint32', CUBE_CELL_COUNT);
    const pValues = builder.allocate('p-values', 'float32', CUBE_CELL_COUNT);
    builder.graph.add(
      new GPUPettittTest({
        values: builder.upload('cube', 'float32', cube).view,
        layout: 'cell-major',
        cellCount: CUBE_CELL_COUNT,
        sliceCount: values.sliceCount,
        output: {statistic: statistic.view, changeSlice: changeSlice.view, pValues: pValues.view}
      })
    );
    const firstScene = Float32Array.from(
      {length: CUBE_CELL_COUNT},
      (_, cell) => cube[cell * values.sliceCount]
    );
    return {
      outputs: {
        statistic,
        changeSlice,
        pValues,
        cube,
        sliceCount: values.sliceCount,
        fractions: builder.upload('fractions', 'float32', getSceneFractions(values.sliceCount)),
        series: builder.allocate('series', 'float32', values.sliceCount),
        context: builder.upload('context', 'float32', padToDisplay(firstScene))
      }
    };
  },
  rebuildOn: ['sliceCount'],
  buildCache: 4,
  async afterRun({outputs, values, upload}) {
    const changes = await readUints(outputs.changeSlice);
    const p = await readFloats(outputs.pValues);
    const when = Float32Array.from({length: CUBE_CELL_COUNT}, (_, cell) =>
      changes[cell] !== NO_CHANGE && p[cell] <= values.alpha
        ? changes[cell] / (outputs.sliceCount - 1)
        : Number.NaN
    );
    return {when: upload('when', 'float32', padToDisplay(when))};
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.27, 0.45],
    resolve(pointer, {inputs, outputs, world, peek}) {
      const cell = getDataCell(pointer.x, pointer.y, world);
      if (cell >= 0) inputs.focus.cell = cell;
      const focus = inputs.focus.cell;
      const sliceCount = outputs.sliceCount;
      outputs.series.write(outputs.cube.slice(focus * sliceCount, (focus + 1) * sliceCount));
      const glyphs: RunGlyph[] = [{shape: 'box', ...getCellBox(focus, world)}];
      const changeSlice = peek(outputs.changeSlice)?.[focus];
      const statistic = peek(outputs.statistic)?.[focus];
      const p = peek(outputs.pValues)?.[focus];
      let readout = `cell [${focus % CUBE_COLUMNS}, ${Math.floor(focus / CUBE_COLUMNS)}]`;
      if (
        changeSlice !== undefined &&
        changeSlice !== NO_CHANGE &&
        changeSlice > 0 &&
        p !== undefined
      ) {
        // The rule sits between the last scene before the change and the first after it.
        const day = (SCENE_DAYS[changeSlice - 1] + SCENE_DAYS[changeSlice]) / 2;
        const x = getSeriesX(day, sliceCount, world);
        glyphs.push({
          shape: 'line',
          from: [x, SERIES_RECTANGLE[1] * world.height],
          to: [x, SERIES_RECTANGLE[3] * world.height],
          dashed: true
        });
        readout = `change at ${SCENE_DATES[changeSlice]} · K ${statistic} · p ${p.toPrecision(2)}`;
      }
      return {glyphs, readout, spotlight: null};
    }
  },
  draw: ({outputs, extra}) => [
    {
      layer: 'raster',
      context: true,
      values: outputs.context,
      columns: DISPLAY_COLUMNS,
      rows: CUBE_ROWS,
      ramp: 'grey',
      range: [0, 1]
    },
    {
      layer: 'raster',
      values: extra.when,
      columns: DISPLAY_COLUMNS,
      rows: CUBE_ROWS,
      ramp: 'sequential',
      range: [0, 1]
    },
    {
      layer: 'chart',
      kind: 'scatter',
      values: outputs.series,
      xValues: outputs.fractions,
      xRange: [0, 1],
      ramp: 'sequential',
      range: [NDVI_RANGE[0], NDVI_RANGE[1]],
      rectangle: SERIES_RECTANGLE,
      radius: 4
    }
  ]
});
