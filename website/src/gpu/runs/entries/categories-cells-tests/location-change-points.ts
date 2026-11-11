// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPULocationChangePoints on the Dixie fire NDVI stack (16 real Sentinel-2 scenes, 2017-2024): optimal
// partitioning of every cell's series by a penalized cost. The map colors each cell by how many change
// points it has. Hover a cell: its NDVI at the true scene dates, the fitted segment means and a dashed
// rule at each change. Raise `penalty` from "every wiggle" to "only the fire". The `mean` cost assumes
// unit variance, so the penalty is on the scale of squared NDVI.

import {
  GPU_LOCATION_CHANGE_POINTS_NONE,
  GPULocationChangePoints,
  type GPULocationChangePointCost
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun} from '../../define-run';
import type {RunGlyph} from '../../interaction';
import {
  CUBE_CELL_COUNT,
  CUBE_COLUMNS,
  createDixieCube,
  CUBE_ROWS,
  DISPLAY_COLUMNS,
  getCellBox,
  getDataCell,
  getSceneFractions,
  getSeriesX,
  getSeriesY,
  NDVI_RANGE,
  padToDisplay,
  SCENE_DATES,
  SCENE_DAYS,
  SERIES_RECTANGLE
} from './_lib/dixie-cube';
import {readUints} from './_lib/readback';
import {parameters} from './location-change-points.parameters';

const SLICE_COUNT = 16;
const MAXIMUM_CHANGE_POINTS = 8;
const COSTS: GPULocationChangePointCost[] = ['mean', 'slope'];
const SMALLEST_SEGMENT = [1, 3];

export default defineOperationRun({
  parameters,
  caption:
    'Dixie fire NDVI, 16 real Sentinel-2 scenes (2017-2024), segmented per cell by a penalized cost.',
  createInputs: () => ({cube: createDixieCube(SLICE_COUNT), focus: {cell: 0}}),
  build(builder, inputs, values) {
    const parameterBuffer = builder.allocate('parameters', 'float32', 2);
    const changePointCount = builder.allocate('change-point-count', 'uint32', CUBE_CELL_COUNT);
    const changePoints = builder.allocate(
      'change-points',
      'uint32',
      CUBE_CELL_COUNT * MAXIMUM_CHANGE_POINTS
    );
    const contributor = new GPULocationChangePoints({
      values: builder.upload('cube', 'float32', inputs.cube).view,
      layout: 'cell-major',
      cellCount: CUBE_CELL_COUNT,
      sliceCount: SLICE_COUNT,
      cost: COSTS[values.cost],
      maximumChangePoints: MAXIMUM_CHANGE_POINTS,
      parameters: parameterBuffer.view,
      output: {changePointCount: changePointCount.view, changePoints: changePoints.view}
    });
    builder.graph.add(contributor);
    const firstScene = Float32Array.from(
      {length: CUBE_CELL_COUNT},
      (_, cell) => inputs.cube[cell * SLICE_COUNT]
    );
    return {
      outputs: {
        changePointCount,
        changePoints,
        fractions: builder.upload('fractions', 'float32', getSceneFractions(SLICE_COUNT)),
        series: builder.allocate('series', 'float32', SLICE_COUNT),
        context: builder.upload('context', 'float32', padToDisplay(firstScene))
      },
      update: current =>
        parameterBuffer.write(
          contributor.getParameterValues({
            penalty: current.penalty,
            minimumSegmentLength: Math.max(
              current.minimumSegmentLength,
              SMALLEST_SEGMENT[current.cost]
            )
          })
        )
    };
  },
  rebuildOn: ['cost'],
  buildCache: 2,
  async afterRun({outputs, upload}) {
    const counts = await readUints(outputs.changePointCount);
    return {counts: upload('counts', 'float32', padToDisplay(Float32Array.from(counts)))};
  },
  interaction: {
    kind: 'probe',
    hint: 'hover a cell',
    rest: [0.27, 0.45],
    resolve(pointer, {inputs, outputs, values, world, peek}) {
      const cell = getDataCell(pointer.x, pointer.y, world);
      if (cell >= 0) inputs.focus.cell = cell;
      const focus = inputs.focus.cell;
      const series = inputs.cube.slice(focus * SLICE_COUNT, (focus + 1) * SLICE_COUNT);
      outputs.series.write(series);
      const glyphs: RunGlyph[] = [{shape: 'box', ...getCellBox(focus, world)}];
      const count = peek(outputs.changePointCount)?.[focus] ?? 0;
      const points = peek(outputs.changePoints);
      const found: number[] = [];
      if (points) {
        for (let index = 0; index < Math.min(count, MAXIMUM_CHANGE_POINTS); index++) {
          const slice = points[focus * MAXIMUM_CHANGE_POINTS + index];
          if (slice !== GPU_LOCATION_CHANGE_POINTS_NONE) found.push(slice);
        }
      }
      // Segment means (the fit of the mean cost) as a step line through the scenes.
      const bounds = [0, ...found, SLICE_COUNT];
      const steps: number[] = [];
      for (let segment = 0; segment < bounds.length - 1; segment++) {
        let sum = 0;
        let used = 0;
        for (let slice = bounds[segment]; slice < bounds[segment + 1]; slice++) {
          if (Number.isFinite(series[slice])) {
            sum += series[slice];
            used++;
          }
        }
        if (!used) continue;
        const y = getSeriesY(Math.min(NDVI_RANGE[1], Math.max(NDVI_RANGE[0], sum / used)), world);
        steps.push(
          getSeriesX(SCENE_DAYS[bounds[segment]], SLICE_COUNT, world),
          y,
          getSeriesX(SCENE_DAYS[bounds[segment + 1] - 1], SLICE_COUNT, world),
          y
        );
      }
      // The step line is the fit of the mean cost only.
      for (let index = 0; values.cost === 0 && index < steps.length; index += 4) {
        glyphs.push({
          shape: 'line',
          from: [steps[index], steps[index + 1]],
          to: [steps[index + 2], steps[index + 3]]
        });
      }
      for (const slice of found) {
        const x = getSeriesX((SCENE_DAYS[slice - 1] + SCENE_DAYS[slice]) / 2, SLICE_COUNT, world);
        glyphs.push({
          shape: 'line',
          from: [x, SERIES_RECTANGLE[1] * world.height],
          to: [x, SERIES_RECTANGLE[3] * world.height],
          dashed: true
        });
      }
      return {
        glyphs,
        spotlight: null,
        readout: `${count} change points${
          found.length ? `: ${found.map(slice => SCENE_DATES[slice]).join(', ')}` : ''
        } · cell [${focus % CUBE_COLUMNS}, ${Math.floor(focus / CUBE_COLUMNS)}]`
      };
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
      values: extra.counts,
      columns: DISPLAY_COLUMNS,
      rows: CUBE_ROWS,
      ramp: 'sequential',
      range: [0, 3],
      alpha: 0.85
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
