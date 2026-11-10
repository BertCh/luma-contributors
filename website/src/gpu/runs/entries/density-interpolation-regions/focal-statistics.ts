// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// GPUFocalStatistics summarizes a moving window around every cell. The operation writes mean,
// min, max, range and standard deviation in one pass; `output` picks which one is drawn.

import {
  getGPUFocalStatisticsParameterValues,
  GPUFocalStatistics
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {defineOperationRun, type RunBuffer} from '../../define-run';
import {createElevationRaster} from '../../inputs';
import {findParameter, getFractionOf, getParameterAt} from '../../interaction';
import {getRasterCell, getRasterCellCenter} from '../../pick';
import {parameters} from './focal-statistics.parameters';
import {createRunGrid} from './support/grid';

// Fixed ranges, so that moving the radius changes the colors and not the scale.
const STATISTIC_RANGES = [
  [0, 1],
  [0, 1],
  [0, 1],
  [0, 0.5],
  [0, 0.2]
] as const;

type Statistics = Record<
  'mean' | 'min' | 'max' | 'range' | 'standardDeviation',
  RunBuffer<'float32'>
>;

function getStatistic(outputs: Statistics, index: number): RunBuffer<'float32'> {
  return [outputs.mean, outputs.min, outputs.max, outputs.range, outputs.standardDeviation][index];
}

export default defineOperationRun({
  parameters,
  caption: 'Window statistic of a terrain-like raster; the muted raster is the input.',
  createInputs: ({world}) => {
    const grid = createRunGrid(world, 72);
    const raster = createElevationRaster(world, 72);
    return {
      grid,
      raster,
      windowCells: new Float32Array(grid.columns * grid.rows),
      rasterRange: [Math.min(...raster.values), Math.max(...raster.values)] as [number, number]
    };
  },
  build(builder, {grid, raster}) {
    const cellCount = grid.columns * grid.rows;
    const input = builder.upload('input', 'float32', raster.values);
    const parameterBuffer = builder.allocate('parameters', 'float32', 8);
    const statistics = {
      mean: builder.allocate('mean', 'float32', cellCount),
      min: builder.allocate('min', 'float32', cellCount),
      max: builder.allocate('max', 'float32', cellCount),
      range: builder.allocate('range', 'float32', cellCount),
      standardDeviation: builder.allocate('standard-deviation', 'float32', cellCount)
    };
    builder.graph.add(
      new GPUFocalStatistics({
        values: input.view,
        grid: grid.grid,
        maximumRadius: 8,
        parameters: parameterBuffer.view,
        output: {
          mean: statistics.mean.view,
          min: statistics.min.view,
          max: statistics.max.view,
          range: statistics.range.view,
          standardDeviation: statistics.standardDeviation.view
        }
      })
    );
    return {
      outputs: {
        input,
        ...statistics,
        window: builder.upload('window', 'float32', new Float32Array(cellCount).fill(Number.NaN))
      },
      update: values =>
        parameterBuffer.write(
          getGPUFocalStatisticsParameterValues({
            radius: values.radius,
            shape: values.shape === 0 ? 'square' : 'circle'
          })
        )
    };
  },
  // The pointer is the center of the window (x, y snapped to a cell) and its height sets the
  // radius: the glyph is the footprint the statistic of that cell was computed over.
  interaction: {
    kind: 'window',
    hint: 'move and size the window',
    rest: [
      0.5,
      1 -
        getFractionOf(
          findParameter(parameters, 'radius'),
          findParameter(parameters, 'radius').default
        )
    ],
    resolve(pointer, {inputs, outputs, values, peek}) {
      const {grid} = inputs;
      const radius = getParameterAt(findParameter(parameters, 'radius'), 1 - pointer.unitY);
      const cell = getRasterCell(grid.columns, grid.rows, grid.window, pointer.x, pointer.y);
      const center = getRasterCellCenter(
        grid.columns,
        grid.rows,
        grid.window,
        cell.column,
        cell.row
      );
      const reach = (radius + 0.5) * grid.cellSize;
      // The raw cells under the footprint are shown in grey over the statistic.
      fillWindow(inputs, cell, radius, values.shape === 0, inputs.windowCells);
      outputs.window.write(inputs.windowCells);
      const result = peek(getStatistic(outputs, values.statistic))?.[cell.index];
      const name = String(parameters[2].options?.[values.statistic]);
      return {
        values: {radius},
        glyphs:
          values.shape === 0
            ? [
                {
                  shape: 'box',
                  minimum: [center[0] - reach, center[1] - reach],
                  maximum: [center[0] + reach, center[1] + reach]
                }
              ]
            : [{shape: 'ring', at: center, radius: reach}],
        readout: `radius ${radius} cells · ${name} ${result === undefined ? '...' : result.toFixed(2)}`
      };
    }
  },
  draw: ({inputs, outputs, values}) => {
    const statistic = getStatistic(outputs, values.statistic);
    return [
      {
        layer: 'raster',
        context: true,
        values: outputs.input,
        columns: inputs.grid.columns,
        rows: inputs.grid.rows
      },
      {
        layer: 'raster',
        values: statistic,
        columns: inputs.grid.columns,
        rows: inputs.grid.rows,
        range: STATISTIC_RANGES[values.statistic]
      },
      {
        layer: 'raster',
        values: outputs.window,
        columns: inputs.grid.columns,
        rows: inputs.grid.rows,
        ramp: 'grey',
        range: inputs.rasterRange
      }
    ];
  }
});

/** Copies the raw input cells under the footprint into `target` (NaN elsewhere). */
function fillWindow(
  inputs: {grid: {columns: number; rows: number}; raster: {values: Float32Array}},
  cell: {column: number; row: number},
  radius: number,
  isSquare: boolean,
  target: Float32Array
): void {
  const {columns, rows} = inputs.grid;
  target.fill(Number.NaN);
  for (
    let row = Math.max(0, cell.row - radius);
    row <= Math.min(rows - 1, cell.row + radius);
    row++
  ) {
    for (
      let column = Math.max(0, cell.column - radius);
      column <= Math.min(columns - 1, cell.column + radius);
      column++
    ) {
      if (isSquare || Math.hypot(column - cell.column, row - cell.row) <= radius + 0.5) {
        target[row * columns + column] = inputs.raster.values[row * columns + column];
      }
    }
  }
}
